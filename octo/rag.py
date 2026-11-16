# -*- coding: utf-8 -*-
"""Octo RAG — retrieve relevant corpus chunks to ground Octo's answers.

Usage:
    python -m octo.rag --query "how to prevent sql injection"
"""
import argparse
import glob
import math
import os
import re
from collections import Counter

import numpy as np

CHUNK_CHARS = 500
OVERLAP_CHARS = 80


def _chunks(text, size=CHUNK_CHARS, overlap=OVERLAP_CHARS):
    step = max(1, size - overlap)
    for start in range(0, max(1, len(text) - overlap), step):
        piece = text[start:start + size].strip()
        if len(piece) > 40:
            yield piece
        if start + size >= len(text):
            break


def _tokens(text):
    return re.findall(r"[\w'>=-]+", text.lower())


class RagIndex:
    def __init__(self, chunks, matrix, idf, vocab):
        self.chunks = chunks
        self.matrix = matrix
        self.idf = idf
        self.vocab = vocab

    @classmethod
    def build(cls, roots=("data/octo",), max_chunks=4000):
        chunks = []
        for root in roots:
            for path in sorted(glob.glob(os.path.join(root, "**", "*.txt"), recursive=True)):
                if os.path.basename(path) == "agi_memory.txt":
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    chunks += list(_chunks(f.read()))
        chunks = chunks[:max_chunks]
        if not chunks:
            raise SystemExit("no corpus text found for RAG")

        doc_freq = Counter()
        doc_tokens = []
        for ch in chunks:
            toks = set(_tokens(ch))
            doc_tokens.append(toks)
            doc_freq.update(toks)

        vocab = {w: i for i, w in enumerate(sorted(doc_freq))}
        n = len(chunks)
        idf = np.zeros(len(vocab))
        for w, i in vocab.items():
            idf[i] = math.log((n + 1) / (doc_freq[w] + 1)) + 1.0

        matrix = np.zeros((n, len(vocab)), dtype=np.float32)
        for row, toks in enumerate(doc_tokens):
            for w in toks:
                matrix[row, vocab[w]] += 1.0
        matrix *= idf
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix /= norms
        return cls(chunks, matrix, idf, vocab)

    def search(self, query, k=3):
        vec = np.zeros((1, len(self.vocab)), dtype=np.float32)
        for w in _tokens(query):
            i = self.vocab.get(w)
            if i is not None:
                vec[0, i] += 1.0
        vec *= self.idf
        norm = np.linalg.norm(vec)
        if norm == 0:
            return []
        vec /= norm
        scores = (self.matrix @ vec.T).ravel()
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]


def context_for_prompt(query, k=2, max_chars=600, roots=("data/octo",)):
    """Return retrieved context text (may be empty) to prepend to a prompt."""
    try:
        index = RagIndex.build(roots=roots)
    except SystemExit:
        return ""
    hits = index.search(query, k=k)
    return "\n".join(chunk for chunk, _ in hits)[:max_chars]


def main():
    ap = argparse.ArgumentParser(description="Octo RAG: search the corpus.")
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    index = RagIndex.build()
    for i, (chunk, score) in enumerate(index.search(args.query, k=args.k), 1):
        print(f"--- match {i} (score {score:.3f}) ---\n{chunk}\n")


if __name__ == "__main__":
    main()
