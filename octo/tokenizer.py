# -*- coding: utf-8 -*-
"""Character-level tokenizer for Octo. Supports Bengali and any Unicode text."""
import json


class CharTokenizer:
    def __init__(self, vocab=None):
        self.stoi = {}
        self.itos = []
        if vocab is not None:
            self.itos = list(vocab)
            self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    @classmethod
    def from_text(cls, text):
        tok = cls()
        tok.itos = sorted(set(text))
        tok.stoi = {ch: i for i, ch in enumerate(tok.itos)}
        return tok

    def __len__(self):
        return len(self.itos)

    def encode(self, text):
        unk = self.stoi.get("<unk>")
        return [self.stoi.get(ch, unk if unk is not None else 0) for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if 0 <= i < len(self.itos))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.itos}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f)["vocab"])
