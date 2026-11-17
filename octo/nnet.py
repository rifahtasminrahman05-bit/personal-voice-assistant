# -*- coding: utf-8 -*-
"""Pure-NumPy neural network — hand-written MLP with manual backprop and Adam.

Fast, dependency-light intent classifier that works without PyTorch.
Train:  python -m octo.nnet --train --data data/octo/intents.json
Predict: python -m octo.nnet --predict "remember my password"
"""
import argparse
import json
import os
import re
from collections import Counter

import numpy as np

NET_PATH = "octo_checkpoints/nnet.npz"


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def featurize(texts, vocab):
    X = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in tokenize(text):
            if word in vocab:
                X[row, vocab[word]] += 1.0
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


class NumPyMLP:
    """2-layer MLP: input -> tanh hidden -> softmax output, trained with Adam."""

    def __init__(self, n_in, n_hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, (n_in, n_hidden)).astype(np.float32)
        self.b1 = np.zeros(n_hidden, dtype=np.float32)
        self.W2 = rng.normal(0, 0.1, (n_hidden, n_out)).astype(np.float32)
        self.b2 = np.zeros(n_out, dtype=np.float32)
        self.m = [np.zeros_like(p) for p in self._params()]
        self.v = [np.zeros_like(p) for p in self._params()]
        self.t = 0

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        e = np.exp(self.z2 - self.z2.max(axis=1, keepdims=True))
        self.probs = e / e.sum(axis=1, keepdims=True)
        return self.probs

    def backward(self, X, y):
        n = X.shape[0]
        dz2 = self.probs.copy()
        dz2[np.arange(n), y] -= 1.0
        dz2 /= n
        gW2 = self.a1.T @ dz2
        gb2 = dz2.sum(0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - np.tanh(self.z1) ** 2)
        gW1 = X.T @ dz1
        gb1 = dz1.sum(0)
        return [gW1, gb1, gW2, gb2]

    def step(self, grads, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for p, g, m, v in zip(self._params(), grads, self.m, self.v):
            m *= beta1
            m += (1 - beta1) * g
            v *= beta2
            v += (1 - beta2) * g * g
            mhat = m / (1 - beta1 ** self.t)
            vhat = v / (1 - beta2 ** self.t)
            p -= lr * mhat / (np.sqrt(vhat) + eps)

    def loss(self, X, y):
        p = self.forward(X)
        return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-9))


def train(data_path, n_hidden=32, epochs=300, lr=0.02):
    with open(data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    texts = [s["text"] for s in samples]
    labels = sorted({s["intent"] for s in samples})
    y = np.array([labels.index(s["intent"]) for s in samples])

    counts = Counter(w for t in texts for w in tokenize(t))
    vocab = {w: i for i, (w, _) in enumerate(sorted(counts.items()))}
    X = featurize(texts, vocab)

    net = NumPyMLP(len(vocab), n_hidden, len(labels))
    for epoch in range(epochs):
        net.forward(X)
        grads = net.backward(X, y)
        net.step(grads, lr=lr * (1 - epoch / epochs) + 1e-3)
    acc = (net.forward(X).argmax(1) == y).mean()
    print(f"NumPy MLP trained: {len(vocab)} features, {len(labels)} classes, train acc {acc:.2f}")
    return {"vocab": vocab, "labels": labels, "net": net}


def predict(text, blob, topk=1):
    X = featurize([text], blob["vocab"])
    probs = blob["net"].forward(X)[0]
    top = np.argsort(-probs)[:topk]
    return [(blob["labels"][i], float(probs[i])) for i in top]


def save(blob, path=NET_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path,
             vocab=json.dumps(blob["vocab"], ensure_ascii=False),
             labels=json.dumps(blob["labels"], ensure_ascii=False),
             **{f"W{i}": w for i, w in enumerate(blob["net"]._params())})


def load(path=NET_PATH):
    if not os.path.exists(path):
        raise SystemExit(f"no NumPy net at {path}; run: python -m octo.nnet --train")
    data = np.load(path, allow_pickle=False)
    weights = [data[k] for k in ("W0", "W1", "W2", "W3")]
    net = NumPyMLP(weights[0].shape[0], weights[0].shape[1], weights[2].shape[1])
    net.W1, net.b1, net.W2, net.b2 = weights
    return {"vocab": json.loads(str(data["vocab"])), "labels": json.loads(str(data["labels"])), "net": net}


def main():
    ap = argparse.ArgumentParser(description="Pure-NumPy intent neural network.")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--predict", default=None)
    ap.add_argument("--data", default="data/octo/intents.json")
    ap.add_argument("--topk", type=int, default=1)
    args = ap.parse_args()

    if args.train:
        blob = train(args.data)
        save(blob)
        return
    if args.predict is not None:
        for label, p in predict(args.predict, load(), topk=args.topk):
            print(f"{label}\t{p:.3f}")


if __name__ == "__main__":
    main()
