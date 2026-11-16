# -*- coding: utf-8 -*-
"""Batching helpers for Octo training."""
import torch


def encode_corpus(text, tokenizer):
    ids = tokenizer.encode(text)
    return torch.tensor(ids, dtype=torch.long)


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
