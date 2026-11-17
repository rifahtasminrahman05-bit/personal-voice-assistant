# -*- coding: utf-8 -*-
"""Advanced Language Model Intelligence — OctoV2 transformer + model router.

OctoV2 upgrades the base Octo transformer with:
  - Rotary Position Embeddings (RoPE)
  - Grouped-Query Attention (GQA, shared KV heads)
  - KV cache for incremental autoregressive decoding
  - sliding-window attention option
  - quantization-ready linears (pair with octo.quantize)

ModelRouter picks the cheapest capable backend per task: NumPy intent net,
frozen-embedding classifier, Octo LLM (RAG-grounded), or Octo-Pro (LoRA).
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from octo.model import OctoConfig


class AdvancedConfig(OctoConfig):
    def __init__(self, vocab_size, block_size=128, n_layer=2, n_head=2, n_embd=64,
                 dropout=0.0, n_kv_heads=None, sliding_window=None, rope=True):
        super().__init__(vocab_size, block_size, n_layer, n_head, n_embd, dropout)
        self.n_kv_heads = n_kv_heads or n_head
        self.sliding_window = sliding_window
        self.rope = rope


def rope_cache(seq_len, head_dim, device, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    """x: (B, H, T, D) with even D; rotate half-pairs."""
    T = x.size(2)
    cos, sin = cos[:T].unsqueeze(0).unsqueeze(0), sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class AttentionV2(nn.Module):
    """Causal self-attention with GQA, optional sliding window and KV cache."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_heads == 0
        self.n_head, self.n_kv_heads = cfg.n_head, cfg.n_kv_heads
        self.head_dim = cfg.n_embd // cfg.n_head
        self.group = self.n_head // self.n_kv_heads
        self.window = cfg.sliding_window
        self.use_rope = cfg.rope
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.k_proj = nn.Linear(cfg.n_embd, self.n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(cfg.n_embd, self.n_kv_heads * self.head_dim)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x, cache=None):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.use_rope:
            cos, sin = rope_cache(T + (cache["k"].size(2) if cache else 0),
                                  self.head_dim, x.device)
            offset = cache["k"].size(2) if cache else 0
            cos, sin = cos[offset:offset + T], sin[offset:offset + T]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            cache["k"], cache["v"] = torch.cat([cache["k"], k], dim=2), \
                torch.cat([cache["v"], v], dim=2)
            k_full, v_full = cache["k"], cache["v"]
        else:
            k_full, v_full = k, v
        k_full = k_full.repeat_interleave(self.group, dim=1)
        v_full = v_full.repeat_interleave(self.group, dim=1)
        T_total = k_full.size(2)
        att = (q @ k_full.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T_total, T_total, device=x.device, dtype=torch.bool))
        keep = mask[:, T_total - T:]
        if cache is not None:
            keep = torch.ones_like(keep, dtype=torch.bool)
        if self.window is not None and cache is None:
            pos = torch.arange(T_total, device=x.device)
            dist = pos.view(1, -1) - pos.view(-1, 1)
            window_mask = (dist >= -self.window) & (dist <= 0)
            keep = keep & window_mask
        att = att.masked_fill(~keep, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v_full).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class BlockV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(cfg.n_embd), AttentionV2(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln1(x), cache=cache)
        return x + self.mlp(self.ln2(x))


class OctoV2(nn.Module):
    """RoPE + GQA + KV-cache decoder-only transformer."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(BlockV2(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, caches=None):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = None
            if caches is not None:
                cache = caches[i]
                if cache is None:
                    d = self.cfg.n_embd // self.cfg.n_head
                    zero = torch.empty(B, self.cfg.n_kv_heads, 0, d, device=idx.device)
                    cache = {"k": zero.clone(), "v": zero.clone()}
                    new_caches.append(cache)
            x = block(x, cache=cache)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, (new_caches if new_caches else caches)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=True):
        self.eval()
        caches = None
        cache_on = use_cache
        if cache_on:
            logits, _, caches = self(idx)
        for _ in range(max_new_tokens):
            if cache_on:
                logits_next = logits[:, -1, :]
            else:
                logits, _, _ = self(idx)
                logits_next = logits[:, -1, :]
            logits_next = logits_next / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits_next, min(top_k, logits_next.size(-1)))
                logits_next[logits_next < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits_next, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat([idx, idx_next], dim=1)
            if cache_on:
                logits, _, caches = self(idx_next, caches=caches)
            if idx.size(1) > self.cfg.block_size:
                idx = idx[:, -self.cfg.block_size:]
                caches = None
                cache_on = False
        return idx


class ModelRouter:
    """Model Routing System: cheapest capable backend per task.

    Supports a quantized runtime: pass `quantized=True` to load the int8
    checkpoint produced by `python -m octo.quantize` (falls back to the
    full checkpoint when it does not exist).
    """

    def __init__(self, checkpoint="octo_checkpoints/octo.pt", device="cpu",
                 quantized=False):
        self.checkpoint = checkpoint
        self.device = device
        self.quantized = quantized
        self._octo = None

    def _checkpoint_path(self):
        if self.quantized:
            qpath = os.path.splitext(self.checkpoint)[0] + "-int8.pt"
            if os.path.exists(qpath):
                return qpath
        return self.checkpoint

    def _octo_ready(self):
        if self._octo is None and os.path.exists(self._checkpoint_path()):
            from octo.chat import load_model
            self._octo = load_model(self._checkpoint_path(), self.device)
        return self._octo

    def route(self, task_type):
        if task_type in ("intent", "classify"):
            return ("nnet", None)
        if task_type in ("grounded_answer", "rag"):
            return ("octo", self._octo_ready())
        if task_type in ("long_form", "fine_tune"):
            return ("octo-pro", None)
        return ("octo", self._octo_ready())

    def answer(self, prompt, steps=100):
        ready = self._octo_ready()
        if ready is None:
            return "Octo is not trained yet; run: python -m octo.train --corpus data/octo"
        model, tokenizer = ready
        from octo.chat import complete
        return complete(model, tokenizer, prompt, steps, 0.8, 40, self.device).strip()


@torch.no_grad()
def speculative_generate(target, draft, idx, max_new_tokens, k=4,
                         temperature=1.0, top_k=None):
    """Speculative Decoding: a small draft model proposes k tokens at a time;
    the target model verifies them in one batched forward pass and corrects
    the first mismatch. Returns the full sequence including the prompt.

    Both models must share the tokenizer/vocab. Greedy verification keeps
    output quality equal to target-only decoding.
    """
    target.eval(), draft.eval()
    produced = idx
    generated = 0
    while generated < max_new_tokens:
        n_draft = min(k, max_new_tokens - generated)
        draft_seq = draft.generate(produced, n_draft, temperature, top_k,
                                   use_cache=True)[:, produced.size(1):]
        candidate = torch.cat([produced, draft_seq], dim=1)
        logits, _, _ = target(candidate[:, :-1])
        pred = logits[:, -draft_seq.size(1):, :].argmax(-1)
        accepted = 0
        for i in range(n_draft):
            if pred[0, i].item() == draft_seq[0, i].item():
                accepted += 1
            else:
                break
        if accepted < n_draft:
            new_tokens = torch.cat([draft_seq[:, :accepted],
                                    pred[:, accepted:accepted + 1]], dim=1)
        else:
            new_tokens = draft_seq
        produced = torch.cat([produced, new_tokens], dim=1)
        generated += new_tokens.size(1)
        if produced.size(1) > target.cfg.block_size:
            produced = produced[:, -target.cfg.block_size:]
    return produced
