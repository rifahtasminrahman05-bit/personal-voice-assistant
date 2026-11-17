# -*- coding: utf-8 -*-
"""OctoV2 training pipeline — trains the advanced transformer (RoPE + GQA + KV cache).

Usage:
    python -m octo_core.train_v2 --corpus data/octo --steps 2000
"""
import argparse
import glob
import os
import time

import torch

from octo.data import encode_corpus, get_batch
from octo.tokenizer import CharTokenizer, load_tokenizer

from .models import AdvancedConfig, OctoV2


def read_corpus(path):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    parts = []
    for p in sorted(glob.glob(os.path.join(path, "**", "*.txt"), recursive=True)):
        with open(p, "r", encoding="utf-8") as f:
            parts.append(f.read())
    return "\n".join(parts)


@torch.no_grad()
def estimate_loss(model, data, cfg, device, iters=10, batch_size=8):
    model.eval()
    losses = []
    for _ in range(iters):
        xb, yb = get_batch(data, cfg.block_size, batch_size, device)
        _, loss, _ = model(xb, targets=yb)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_v2(corpus="data/octo", out="octo_checkpoints", steps=2000, batch_size=16,
             block_size=128, n_layer=2, n_head=4, n_embd=96, n_kv_heads=2,
             sliding_window=None, lr=3e-4, dropout=0.1, eval_every=200,
             eval_iters=10, seed=42, resume=None, tokenizer_file=None,
             device=None, progress=None):
    """Train OctoV2; returns the checkpoint path. `progress` is a callback
    (step, total, loss) invoked at eval intervals — used by the desktop UI."""
    device = device or ("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(seed)
    text = read_corpus(corpus)
    if not text.strip():
        raise SystemExit(f"no corpus text found at: {corpus}")

    if tokenizer_file:
        tokenizer = load_tokenizer(tokenizer_file)
    else:
        tokenizer = CharTokenizer.from_text(text)
    data = encode_corpus(text, tokenizer).to(device)

    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        cfg = AdvancedConfig(**{k: v for k, v in ckpt["config"].items()})
        model = OctoV2(cfg).to(device)
        model.load_state_dict(ckpt["model"])
    else:
        cfg = AdvancedConfig(vocab_size=len(tokenizer), block_size=block_size,
                             n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                             n_kv_heads=n_kv_heads, sliding_window=sliding_window,
                             dropout=dropout)
        model = OctoV2(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    os.makedirs(out, exist_ok=True)
    ckpt_path = os.path.join(out, "octo_v2.pt")
    for step in range(1, steps + 1):
        xb, yb = get_batch(data, cfg.block_size, batch_size, device)
        _, loss, _ = model(xb, targets=yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_every == 0 or step == steps:
            val = estimate_loss(model, data, cfg, device, iters=eval_iters,
                                batch_size=min(batch_size, 8))
            if progress:
                progress(step, steps, val)
            torch.save({"model": model.state_dict(),
                        "config": {"vocab_size": cfg.vocab_size,
                                   "block_size": cfg.block_size,
                                   "n_layer": cfg.n_layer, "n_head": cfg.n_head,
                                   "n_embd": cfg.n_embd, "dropout": cfg.dropout,
                                   "n_kv_heads": cfg.n_kv_heads,
                                   "sliding_window": cfg.sliding_window,
                                   "rope": cfg.rope},
                        "tokenizer_path": None},
                       ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
            tokenizer.save(os.path.join(out, "tokenizer_v2.json"))
    return ckpt_path


def load_v2(checkpoint, device="cpu"):
    """Load a trained OctoV2 checkpoint + its tokenizer."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = AdvancedConfig(**{k: v for k, v in ckpt["config"].items()})
    model = OctoV2(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok_path = os.path.join(os.path.dirname(checkpoint), "tokenizer_v2.json")
    tokenizer = load_tokenizer(tok_path) if os.path.exists(tok_path) else None
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser(description="Train OctoV2 (RoPE + GQA transformer).")
    ap.add_argument("--corpus", default="data/octo")
    ap.add_argument("--out", default="octo_checkpoints")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=96)
    ap.add_argument("--n-kv-heads", type=int, default=2)
    ap.add_argument("--sliding-window", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    def progress(step, total, loss):
        print(f"step {step}/{total} | val loss {loss:.4f}")

    started = time.time()
    path = train_v2(corpus=args.corpus, out=args.out, steps=args.steps,
                    batch_size=args.batch_size, block_size=args.block_size,
                    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                    n_kv_heads=args.n_kv_heads, sliding_window=args.sliding_window,
                    lr=args.lr, dropout=args.dropout, eval_every=args.eval_every,
                    seed=args.seed, resume=args.resume, device=args.device,
                    progress=progress)
    print(f"saved: {path} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
