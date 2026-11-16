# -*- coding: utf-8 -*-
"""Tokenizers for Octo: char-level and BPE (byte pair encoding), no external deps."""
import json
from collections import Counter

UNK = "<unk>"


class CharTokenizer:
    type = "char"

    def __init__(self, vocab=None):
        self.stoi = {}
        self.itos = []
        if vocab is not None:
            self.itos = list(vocab)
            self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    def __len__(self):
        return len(self.itos)

    def encode(self, text):
        unk = self.stoi.get(UNK, 0)
        return [self.stoi.get(ch, unk) for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if 0 <= i < len(self.itos))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": self.type, "vocab": self.itos}, f, ensure_ascii=False)


def _stats(symbols):
    return Counter(zip(symbols, symbols[1:]))


def _merge(symbols, pair, merged):
    out, i = [], 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


class BPETokenizer:
    """Word-boundary BPE with an end-of-word marker, Karpathy-style."""
    type = "bpe"
    END = "</w>"

    def __init__(self, vocab=None, merges=None):
        self.merges = [tuple(m) for m in (merges or [])]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.itos = list(vocab or [])
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self._cache = {}

    @classmethod
    def train(cls, text, vocab_size=512, max_chars=500_000):
        text = text[:max_chars]
        words = [w + cls.END for w in text.split()]
        base = sorted({ch for w in words for ch in w})
        num_merges = max(0, vocab_size - len(base))
        symbols_per_word = [list(w) for w in words]
        merges = []
        for _ in range(num_merges):
            counts = Counter()
            for sym in symbols_per_word:
                counts.update(_stats(sym))
            if not counts:
                break
            pair, _ = max(counts.items(), key=lambda kv: kv[1])
            merges.append(pair)
            merged = pair[0] + pair[1]
            symbols_per_word = [_merge(sym, pair, merged) for sym in symbols_per_word]
        vocab = base + [a + b for a, b in merges] + [UNK]
        return cls(vocab, merges)

    def __len__(self):
        return len(self.itos)

    def _encode_word(self, word):
        if word in self._cache:
            return self._cache[word]
        symbols = list(word) + [self.END]
        for pair in self.merges:
            if len(symbols) < 2:
                break
            symbols = _merge(symbols, pair, pair[0] + pair[1])
        unk = self.stoi.get(UNK, 0)
        ids = [self.stoi.get(s, unk) for s in symbols]
        self._cache[word] = ids
        return ids

    def encode(self, text):
        ids = []
        for word in text.split():
            ids.extend(self._encode_word(word))
        return ids

    def decode(self, ids):
        toks = [self.itos[i] for i in ids if 0 <= i < len(self.itos)]
        return "".join(toks).replace(self.END, " ").strip()

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": self.type, "vocab": self.itos,
                       "merges": [list(m) for m in self.merges]}, f, ensure_ascii=False)


def load_tokenizer(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("type") == "bpe":
        return BPETokenizer(data["vocab"], data.get("merges"))
    return CharTokenizer(data["vocab"])
