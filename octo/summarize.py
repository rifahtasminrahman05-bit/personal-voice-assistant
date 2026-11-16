# -*- coding: utf-8 -*-
"""Octo summarizer — extractive summaries of research documents via TF scoring.

Usage:
    python -m octo.summarize --file data/octo/research/agi-firewall.txt --sentences 3
    python -m octo.summarize --query "sql injection" --sentences 3   # summarize RAG hits
"""
import argparse
import math
import re
from collections import Counter

_SENT_SPLIT = re.compile(r"(?<=[.!?।])\s+")


def split_sentences(text):
    return [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 20]


def score_sentences(sentences):
    counts = Counter(w.lower() for s in sentences for w in re.findall(r"\w+", s))
    if not counts:
        return []
    max_count = max(counts.values())
    scores = []
    for i, s in enumerate(sentences):
        words = re.findall(r"\w+", s)
        if not words:
            scores.append(0.0)
            continue
        density = sum(counts[w.lower()] / max_count for w in words) / len(words)
        position = 1.0 if i < len(sentences) * 0.2 else 0.6
        scores.append(density * position)
    return scores


def summarize(text, n=3):
    sentences = split_sentences(text)
    if not sentences:
        return text[:400]
    scores = score_sentences(sentences)
    top = sorted(range(len(sentences)), key=lambda i: -scores[i])[:n]
    top.sort()
    return " ".join(sentences[i] for i in top)


def main():
    ap = argparse.ArgumentParser(description="Extractive text summarizer.")
    ap.add_argument("--file", default=None)
    ap.add_argument("--query", default=None, help="summarize RAG search results for this query")
    ap.add_argument("--sentences", type=int, default=3)
    args = ap.parse_args()

    if args.query:
        from .rag import RagIndex
        text = "\n".join(chunk for chunk, _ in RagIndex.build().search(args.query, k=3))
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        ap.error("pass --file or --query")

    print(summarize(text, n=args.sentences))


if __name__ == "__main__":
    main()
