# -*- coding: utf-8 -*-
"""Octo quantizer — int8 dynamic quantization for smaller, faster CPU inference.

Usage:
    python -m octo.quantize --checkpoint octo_checkpoints/octo-best.pt
    python -m octo.chat --checkpoint octo_checkpoints/octo-int8.pt
"""
import argparse
import os

import torch
from torch import nn

from .chat import load_model


def quantize(checkpoint, out=None):
    device = "cpu"
    model, tokenizer = load_model(checkpoint, device)
    qmodel = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    out = out or os.path.join(os.path.dirname(checkpoint), "octo-int8.pt")
    torch.save(qmodel, out)
    tokenizer.save(os.path.join(os.path.dirname(out), "tokenizer.json"))

    base_mb = os.path.getsize(checkpoint) / 1e6
    q_mb = os.path.getsize(out) / 1e6
    print(f"quantized: {out}")
    print(f"size: {base_mb:.1f} MB -> {q_mb:.1f} MB ({100 * (1 - q_mb / base_mb):.0f}% smaller)")
    return out


def main():
    ap = argparse.ArgumentParser(description="int8-quantize a trained Octo checkpoint.")
    ap.add_argument("--checkpoint", default="octo_checkpoints/octo-best.pt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    quantize(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
