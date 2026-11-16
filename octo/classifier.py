# -*- coding: utf-8 -*-
"""Octo intent classifier — transfer learning on top of Octo embeddings.

Trains a linear head over frozen Octo sentence embeddings to classify intents
(e.g. from data/octo/intents.json). Small data friendly.

Usage:
    python -m octo.classifier --train --data data/octo/intents.json
    python -m octo.classifier --predict "আবহাওয়া কেমন"
    python -m octo.classifier --predict "নোট" --topk 2
"""
import argparse
import json
import os

import numpy as np
import torch

from .embeddings import OctoEmbedder

HEAD_PATH = "octo_checkpoints/intent_head.pt"


class IntentClassifier:
    def __init__(self, embedder, labels, head):
        self.embedder = embedder
        self.labels = labels
        self.head = head

    @classmethod
    def train(cls, data_path, checkpoint="octo_checkpoints/octo-best.pt", epochs=120, lr=0.05):
        with open(data_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        texts = [s["text"] for s in samples]
        labels = sorted({s["intent"] for s in samples})
        y = torch.tensor([labels.index(s["intent"]) for s in samples])
        embedder = OctoEmbedder(checkpoint=checkpoint)
        X = torch.tensor(embedder.encode(texts))
        head = torch.nn.Linear(X.shape[1], len(labels))
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        head.train()
        for epoch in range(epochs):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(head(X), y)
            loss.backward()
            opt.step()
            sched.step()
        head.eval()
        with torch.no_grad():
            acc = (head(X).argmax(1) == y).float().mean().item()
        print(f"trained intent head: {len(labels)} classes, train acc {acc:.2f}")
        return cls(embedder, labels, head)

    def predict(self, text, topk=1):
        with torch.no_grad():
            x = torch.tensor(self.embedder.encode([text]))
            probs = torch.softmax(self.head(x), dim=-1)[0]
        top = torch.topk(probs, k=min(topk, len(self.labels)))
        return [(self.labels[i], float(p)) for p, i in zip(top.values, top.indices)]

    def save(self, path=HEAD_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"head": self.head.state_dict(), "labels": self.labels}, path)

    @classmethod
    def load(cls, path=HEAD_PATH, checkpoint="octo_checkpoints/octo-best.pt"):
        if not os.path.exists(path):
            raise SystemExit(f"no intent head at {path}; run: python -m octo.classifier --train")
        blob = torch.load(path, map_location="cpu")
        n_embd = blob["head"]["weight"].shape[1]
        head = torch.nn.Linear(n_embd, len(blob["labels"]))
        head.load_state_dict(blob["head"])
        head.eval()
        return cls(OctoEmbedder(checkpoint=checkpoint), blob["labels"], head)


def main():
    ap = argparse.ArgumentParser(description="Octo intent classifier (transfer learning).")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--predict", default=None)
    ap.add_argument("--data", default="data/octo/intents.json")
    ap.add_argument("--checkpoint", default="octo_checkpoints/octo-best.pt")
    ap.add_argument("--topk", type=int, default=1)
    args = ap.parse_args()

    if args.train:
        clf = IntentClassifier.train(args.data, checkpoint=args.checkpoint)
        clf.save()
        return
    if args.predict is not None:
        clf = IntentClassifier.load(checkpoint=args.checkpoint)
        for label, p in clf.predict(args.predict, topk=args.topk):
            print(f"{label}\t{p:.3f}")


if __name__ == "__main__":
    main()
