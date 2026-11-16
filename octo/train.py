# -*- coding: utf-8 -*-
"""Train the Octo language model on a text corpus.

Usage:
    python -m octo.train --corpus data/octo/corpus.txt --steps 2000
"""
import argparse
import glob
import math
import os

import torch

from .data import encode_corpus, get_batch
from .model import OctoConfig, OctoModel
from .tokenizer import BPETokenizer, CharTokenizer, load_tokenizer


def main():
    ap = argparse.ArgumentParser(description="Train Octo, a small offline LLM.")
    ap.add_argument("--corpus", default="data/octo", help="text file or directory of .txt files")
    ap.add_argument("--out", default="octo_checkpoints")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=192)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None, metavar="CKPT", help="continue training from a checkpoint")
    ap.add_argument("--tokenizer", choices=("char", "bpe"), default="char")
    ap.add_argument("--vocab-size", type=int, default=512, help="BPE vocab size")
    ap.add_argument("--warmup", type=int, default=100, help="LR warmup steps")
    ap.add_argument("--grad-accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--bf16", action="store_true", help="bfloat16 autocast (CPU/GPU where supported)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.isdir(args.corpus):
        paths = sorted(glob.glob(os.path.join(args.corpus, "**", "*.txt"), recursive=True))
        if not paths:
            raise SystemExit(f"no .txt files found in {args.corpus}")
        texts = []
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                texts.append(f.read())
            print(f"corpus: {p}")
        text = "\n".join(texts)
    else:
        with open(args.corpus, "r", encoding="utf-8") as f:
            text = f.read()
    if len(text) < args.block_size + 2:
        raise SystemExit(f"corpus too small ({len(text)} chars) for block_size {args.block_size}")

    if args.tokenizer == "bpe":
        print("training BPE tokenizer ...")
        tokenizer = BPETokenizer.train(text, vocab_size=args.vocab_size)
    else:
        tokenizer = CharTokenizer.from_text(text)
    data = encode_corpus(text, tokenizer)
    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]

    cfg = OctoConfig(
        vocab_size=len(tokenizer),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = OctoModel(cfg).to(device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        saved_cfg = ckpt.get("config", {})
        for key in ("block_size", "n_layer", "n_head", "n_embd"):
            if key in saved_cfg and saved_cfg[key] != getattr(cfg, key):
                setattr(cfg, key, saved_cfg[key])
        model = OctoModel(cfg).to(device)
        state = ckpt["model_state"]
        model_sd = model.state_dict()
        compatible = {k: v for k, v in state.items()
                      if k in model_sd and model_sd[k].shape == v.shape}
        skipped = set(state) - set(compatible)
        if skipped:
            print(f"resumed {len(compatible)}/{len(state)} tensors (shape changed: {sorted(skipped)})")
        model.load_state_dict(compatible, strict=False)
    print(f"Octo | device={device} vocab={len(tokenizer)} params={model.num_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def lr_lambda(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    autocast_ctx = (torch.autocast(device_type=device, dtype=torch.bfloat16)
                    if args.bf16 else torch.autocast(device_type=device, enabled=False))

    def eval_loss(split_data):
        model.eval()
        losses = []
        for _ in range(args.eval_iters):
            x, y = get_batch(split_data, cfg.block_size, args.batch_size, device)
            with torch.no_grad():
                _, loss = model(x, y)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    model.train()
    best_val = float("inf")
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, cfg.block_size, args.batch_size, device)
        with autocast_ctx:
            _, loss = model(x, y)
            (loss / args.grad_accum).backward()
        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            val_src = val_data if len(val_data) > cfg.block_size + 1 else train_data
            val = eval_loss(val_src)
            print(f"step {step:5d} | train loss {loss.item():.4f} | val loss {val:.4f}")
            if val < best_val:
                best_val = val
                torch.save({"model_state": model.state_dict(), "config": cfg.__dict__},
                           os.path.join(args.out, "octo-best.pt"))

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "octo.pt")
    torch.save({"model_state": model.state_dict(), "config": cfg.__dict__}, ckpt_path)
    tokenizer.save(os.path.join(args.out, "tokenizer.json"))
    print(f"saved: {ckpt_path} (tokenizer={args.tokenizer}, vocab={len(tokenizer)})")


if __name__ == "__main__":
    main()
