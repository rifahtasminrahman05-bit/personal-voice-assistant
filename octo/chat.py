# -*- coding: utf-8 -*-
"""Generate text with a trained Octo model.

Usage:
    python -m octo.chat --checkpoint octo_checkpoints/octo.pt --prompt "hi" --tokens 200
    python -m octo.chat --checkpoint octo_checkpoints/octo.pt   # interactive
"""
import argparse
import os

import torch

from .model import OctoConfig, OctoModel
from .rag import context_for_prompt
from .tokenizer import load_tokenizer


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = OctoConfig(**ckpt["config"])
    model = OctoModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tokenizer = load_tokenizer(os.path.join(os.path.dirname(checkpoint), "tokenizer.json"))
    return model, tokenizer


def complete(model, tokenizer, prompt, max_new_tokens, temperature, top_k, device, rag=False):
    if rag:
        context = context_for_prompt(prompt)
        if context:
            prompt = context + "\n\n" + prompt
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())


def main():
    ap = argparse.ArgumentParser(description="Chat with Octo.")
    ap.add_argument("--checkpoint", default="octo_checkpoints/octo.pt")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--rag", action="store_true", help="prepend retrieved corpus context to prompts")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(args.checkpoint, device)
    print(f"Octo loaded from {args.checkpoint} (device={device}). Type 'bye' to exit.")

    if args.prompt is not None:
        print(complete(model, tokenizer, args.prompt, args.tokens, args.temperature, args.top_k, device, rag=args.rag))
        return

    while True:
        try:
            prompt = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not prompt or prompt.lower() in ("bye", "exit", "quit"):
            break
        print("octo:", complete(model, tokenizer, prompt, args.tokens, args.temperature, args.top_k, device, rag=args.rag))


if __name__ == "__main__":
    main()
