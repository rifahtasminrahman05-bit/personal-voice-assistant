# -*- coding: utf-8 -*-
"""Knowledge Intelligence — gap detection and ontology learning.

KnowledgeGapDetector: finds sparse entities, contradictions and missing
common relations in the knowledge graph.
OntologyLearner: extracts is-a / part-of / used-for relations from text and
builds a concept hierarchy.
"""
import re
from collections import Counter

ISA_PATTERNS = [
    re.compile(r"\b([A-Za-z\u0980-\u09FF][\w\u0980-\u09FF -]{1,40}?)\s+is\s+(?:a|an|the)\s+"
               r"([A-Za-z\u0980-\u09FF][\w\u0980-\u09FF -]{1,40})", re.I),
    re.compile(r"\b([A-Za-z\u0980-\u09FF][\w\u0980-\u09FF -]{1,40}?)\s+(?:হলো|একটি|একটা)\s+"
               r"([A-Za-z\u0980-\u09FF][\w\u0980-\u09FF -]{1,40})"),
]
PART_PATTERNS = [
    re.compile(r"\b([\w-]{2,30}?)\s+(?:is )?part of\s+(?:the\s+)?([\w-]{2,30})", re.I),
    re.compile(r"\b([\w-]{2,30}?)\s+of\s+(?:the\s+)?([\w-]{2,30})\b"),
]
USED_FOR = re.compile(r"\b([\w-]{2,30}?)\s+(?:is )?used (?:for|to)\s+([\w -]{3,60})", re.I)


class KnowledgeGapDetector:
    """Knowledge Gap Detector: find what the graph does not know."""

    def __init__(self, graph, min_facts=2):
        self.graph = graph
        self.min_facts = min_facts

    def gaps(self):
        facts = self.graph.all_facts()
        by_subject = Counter(f["subject"] for f in facts)
        sparse = [{"subject": s, "facts": n, "gap": "sparse entity"}
                  for s, n in by_subject.items() if n < self.min_facts]
        contradictions = [{"subject": f["subject"], "relation": f["relation"],
                           "object": f["object"], "gap": "conflicting fact"}
                          for f in facts if f.get("conflicts", 0) > 0]
        common = Counter(f["relation"] for f in facts)
        missing = []
        subjects = set(by_subject)
        for s in subjects:
            have = {f["relation"] for f in self.graph.query(subject=s)}
            for rel in ("is", "used_for", "part_of"):
                if rel in common and rel not in have:
                    missing.append({"subject": s, "missing_relation": rel,
                                    "gap": "missing common relation"})
        return {"sparse": sparse, "contradictions": contradictions,
                "missing_relations": missing[:20],
                "total_facts": len(facts),
                "coverage": round(1 - (len(sparse) + len(missing))
                                  / max(len(subjects) * 3, 1), 2)}

    def questions(self, limit=5):
        """Turn gaps into research questions the agent can pursue."""
        g = self.gaps()
        qs = [f"{s['subject']} এর কী বৈশিষ্ট্য? (sparse)" for s in g["sparse"]]
        qs += [f"{c['subject']} এর {c['relation']} আসলে কী? (conflict)"
               for c in g["contradictions"]]
        qs += [f"{m['subject']} কি {m['missing_relation']}? (missing)"
               for m in g["missing_relations"]]
        return qs[:limit]


class OntologyLearner:
    """Ontology Learning Engine: concepts, relations, hierarchy from text."""

    def __init__(self):
        self.concepts = set()
        self.parents = {}
        self.part_of = {}
        self.used_for = {}

    @staticmethod
    def _clean(term):
        return re.sub(r"^(?:a|an|the)\s+", "", term.strip().lower())

    def learn(self, text):
        added = 0
        for pattern in ISA_PATTERNS:
            for child, parent in pattern.findall(text):
                child, parent = self._clean(child), self._clean(parent)
                if child != parent and len(child.split()) <= 4 and len(parent.split()) <= 4:
                    self.concepts.update((child, parent))
                    self.parents.setdefault(child, set()).add(parent)
                    added += 1
        for child, whole in PART_PATTERNS[0].findall(text):
            self.part_of.setdefault(child.lower(), set()).add(whole.lower())
            self.concepts.update((child.lower(), whole.lower()))
            added += 1
        for tool, purpose in USED_FOR.findall(text):
            self.used_for.setdefault(tool.lower(), set()).add(purpose.strip().lower())
            self.concepts.add(tool.lower())
            added += 1
        return added

    def ancestors(self, concept, max_depth=5):
        seen, frontier, depth = set(), [concept.lower()], 0
        while frontier and depth < max_depth:
            nxt = []
            for c in frontier:
                for p in self.parents.get(c, ()):
                    if p not in seen:
                        seen.add(p)
                        nxt.append(p)
            frontier = nxt
            depth += 1
        return sorted(seen)

    def hierarchy(self, root=None, depth=0, max_depth=4, out=None):
        out = [] if out is None else out
        roots = [root] if root else sorted(
            c for c in self.concepts if c not in self.parents)
        for c in roots:
            out.append("  " * depth + c)
            if depth < max_depth:
                children = sorted(k for k, ps in self.parents.items() if c in ps)
                for ch in children:
                    self.hierarchy(ch, depth + 1, max_depth, out)
        return out

    def summary(self):
        return {"concepts": len(self.concepts),
                "is_a_links": sum(len(v) for v in self.parents.values()),
                "part_of_links": sum(len(v) for v in self.part_of.values()),
                "used_for_links": sum(len(v) for v in self.used_for.values())}
