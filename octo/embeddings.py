# -*- coding: utf-8 -*-
"""Octo deep-learning embeddings — semantic search using the trained model itself.

Mean-pooled final hidden states of Octo act as sentence embeddings.

Usage:
    python -m octo.embeddings --query "how to prevent sql injection"
"""
import argparse
import os

import numpy as np
import torch

from .chat import load_model
from .rag import _chunks, _gather

EMBED_DIM_SLICE = None


class OctoEmbedder:
    def __init__(self, checkpoint="octo_checkpoints/octo-best.pt", device="cpu",
                 max_len=128, batch_size=16):
        if not os.path.exists(checkpoint):
            raise SystemExit(f"no checkpoint at {checkpoint}; train first (or pass --checkpoint)")
        self.model, self.tokenizer = load_model(checkpoint, device)
        self.device = device
        self.max_len = max_len
        self.batch_size = batch_size

    @torch.no_grad()
    def encode(self, texts):
        vecs = []
        self.model.eval()
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            ids = [t[:self.max_len] for t in (self.tokenizer.encode(x) for x in batch)]
            width = max(1, max(len(t) for t in ids))
            x = torch.zeros(len(ids), width, dtype=torch.long)
            mask = torch.zeros(len(ids), width, dtype=torch.bool)
            for i, t in enumerate(ids):
                x[i, :len(t)] = torch.tensor(t)
                mask[i, :len(t)] = True
            x, mask = x.to(self.device), mask.to(self.device)
            pos = torch.arange(width, device=self.device)
            h = self.model.drop(self.model.tok_emb(x) + self.model.pos_emb(pos))
            for block in self.model.blocks:
                h = block(h)
            h = self.model.ln_f(h)
            m = mask.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            vecs.append(pooled.cpu().numpy())
        vecs = np.concatenate(vecs).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


class NeuralIndex:
    def __init__(self, chunks, matrix, embedder):
        self.chunks = chunks
        self.matrix = matrix
        self.embedder = embedder

    @classmethod
    def build(cls, roots=("data/octo",), max_chunks=1500, checkpoint="octo_checkpoints/octo-best.pt"):
        from .rag import RagIndex
        chunks = []
        for root in roots:
            for path in _gather(root):
                with open(path, "r", encoding="utf-8") as f:
                    chunks += list(_chunks(f.read()))
        chunks = chunks[:max_chunks]
        if not chunks:
            raise SystemExit("no corpus text found")
        embedder = OctoEmbedder(checkpoint=checkpoint)
        return cls(chunks, embedder.encode(chunks), embedder)

    def search(self, query, k=3):
        q = self.embedder.encode([query])[0]
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


def main():
    ap = argparse.ArgumentParser(description="Semantic search with Octo embeddings.")
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--checkpoint", default="octo_checkpoints/octo-best.pt")
    args = ap.parse_args()
    index = NeuralIndex.build(checkpoint=args.checkpoint)
    for i, (chunk, score) in enumerate(index.search(args.query, k=args.k), 1):
        print(f"--- match {i} (score {score:.3f}) ---\n{chunk[:300]}\n")


if __name__ == "__main__":
    main()
