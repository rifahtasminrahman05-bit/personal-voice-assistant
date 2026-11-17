# -*- coding: utf-8 -*-
"""Advanced Memory & Knowledge Intelligence (features 13-18).

AssociativeMemoryNetwork, MemoryConsolidationEngine, MemoryDecayManager,
KnowledgeEvolutionTracker, SemanticCompressor, CrossDomainTransfer.
"""
import datetime
import re
from collections import Counter


class AssociativeMemoryNetwork:
    """13. Associative Memory Network: spread-activation recall."""

    def __init__(self):
        self.nodes = {}

    def store(self, key, content, links=()):
        self.nodes[key] = {"content": content, "links": set(links)}

    def link(self, a, b):
        self.nodes[a]["links"].add(b)
        self.nodes[b]["links"].add(a)

    def associate(self, key, depth=2, decay=0.5):
        activation, frontier = {}, {key: 1.0}
        for _ in range(depth):
            nxt = {}
            for node, act in frontier.items():
                for nb in self.nodes.get(node, {}).get("links", ()):
                    a = act * decay
                    if nb != key and a > activation.get(nb, 0.0):
                        nxt[nb] = a
            activation.update(nxt)
            frontier = nxt
        return sorted(((k, round(v, 3), self.nodes[k]["content"])
                       for k, v in activation.items()), key=lambda t: -t[1])


class MemoryConsolidationEngine:
    """14. Memory Consolidation: promote repeated items to long-term."""

    def __init__(self, threshold=3):
        self.threshold = threshold
        self.short_term = Counter()
        self.long_term = {}

    def touch(self, item, importance=1):
        self.short_term[item] += importance
        if self.short_term[item] >= self.threshold:
            self.long_term[item] = self.short_term.pop(item)
            return "consolidated"
        return "short-term"

    def recall(self, item):
        if item in self.long_term:
            return {"item": item, "store": "long-term",
                    "strength": self.long_term[item]}
        if item in self.short_term:
            return {"item": item, "store": "short-term",
                    "strength": self.short_term[item]}
        return {"item": item, "store": "unknown", "strength": 0}


class MemoryDecayManager:
    """15. Memory Decay Manager: importance fades; forget below floor."""

    def __init__(self, rate=0.2, floor=0.05):
        self.rate, self.floor = rate, floor
        self.scores = {}

    def learn(self, item, strength=1.0):
        self.scores[item] = max(self.scores.get(item, 0.0), strength)

    def access(self, item):
        if item in self.scores:
            self.scores[item] = min(1.0, self.scores[item] + 0.2)

    def decay(self, cycles=1):
        forgotten = []
        for _ in range(cycles):
            for k in list(self.scores):
                self.scores[k] = round(self.scores[k] * (1 - self.rate), 4)
        forgotten = [k for k, v in self.scores.items() if v < self.floor]
        for k in forgotten:
            del self.scores[k]
        return forgotten


class KnowledgeEvolutionTracker:
    """16. Knowledge Evolution Tracker: belief changes over time."""

    def __init__(self):
        self.history = {}

    def record(self, subject, belief, source="unknown", ts=None):
        self.history.setdefault(subject, []).append({
            "ts": ts or datetime.datetime.now().isoformat(timespec="seconds"),
            "belief": belief, "source": source})

    def evolution(self, subject):
        return self.history.get(subject, [])

    def changed(self, subject):
        beliefs = [h["belief"] for h in self.history.get(subject, [])]
        return len(set(beliefs)) > 1

    def current(self, subject):
        h = self.history.get(subject)
        return h[-1]["belief"] if h else None


class SemanticCompressor:
    """17. Semantic Knowledge Compression: keep top-information sentences."""

    def compress(self, text, keep=2):
        sentences = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", text)
                     if s.strip()]
        if len(sentences) <= keep:
            return {"summary": " ".join(sentences), "ratio": 1.0}
        freq = Counter(w.lower() for w in re.findall(r"\w{3,}", text))
        scored = []
        for i, s in enumerate(sentences):
            words = re.findall(r"\w{3,}", s.lower())
            score = sum(freq[w] for w in words) / max(len(words), 1)
            scored.append((score, -i, s))
        top = sorted(sorted(scored, reverse=True)[:keep], key=lambda t: -t[1])
        summary = " ".join(s for _, _, s in top)
        return {"summary": summary,
                "ratio": round(len(summary) / max(len(text), 1), 2)}


class CrossDomainTransfer:
    """18. Cross-Domain Knowledge Transfer: apply relation patterns across
    domains via a concept mapping."""

    def __init__(self):
        self.domains = {}

    def learn(self, domain, triples):
        self.domains[domain] = set(map(tuple, triples))

    def transfer(self, source_domain, target_domain, mapping):
        out = []
        for s, r, t in self.domains.get(source_domain, ()):
            if s in mapping and t in mapping:
                out.append((mapping[s], r, mapping[t]))
        self.domains.setdefault(target_domain, set()).update(out)
        return out

    def shared_relations(self, a, b):
        ra = {r for _, r, _ in self.domains.get(a, ())}
        rb = {r for _, r, _ in self.domains.get(b, ())}
        return sorted(ra & rb)
