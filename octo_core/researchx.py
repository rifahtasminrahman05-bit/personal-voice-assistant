# -*- coding: utf-8 -*-
"""Research cross-check: multi-source verification with a citation list.

Runs several web searches for a query, reads pages, and reports which
claims/sources agree — the Autonomous Research Agent's verification step.
"""
import re
from collections import Counter

from octo.search import search_and_read


def _sentences(text):
    return [s.strip() for s in re.split(r"[.।\n]", text) if len(s.strip()) > 30]


def cross_check(query, fetch=3, min_agreement=2):
    """Search, read multiple pages, and find statements supported by 2+ sources.

    Returns {"query", "sources": [...], "agreed": [...], "unique": [...]}.
    """
    body, sources = search_and_read(query, k=fetch + 2, fetch=fetch, save=False)
    if not body:
        return {"query": query, "sources": sources, "agreed": [], "unique": []}
    chunks = _sentences(body)
    tokenized = [frozenset(w for w in re.findall(r"\w{4,}", s.lower())) for s in chunks]
    counts = Counter()
    for toks in tokenized:
        counts.update(toks)
    agreed, unique = [], []
    for s, toks in zip(chunks, tokenized):
        support = sum(1 for w in toks if counts[w] >= min_agreement)
        if toks and support >= max(1, len(toks) // 2):
            agreed.append(s)
        else:
            unique.append(s)
    return {"query": query, "sources": sources[:fetch],
            "agreed": agreed[:8], "unique": unique[:8]}
