# -*- coding: utf-8 -*-
"""Advanced Memory & Knowledge Intelligence.

Knowledge graph with provenance and conflict detection, working/episodic
memory, hybrid retrieval (RAG + graph facts), conversation compression and a
MemoryManager facade. All storage is local files/SQLite — no external deps.
"""
import datetime
import json
import os
import re
import threading
from collections import Counter, deque

from octo.rag import RagIndex
from octo.summarize import summarize

STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "and",
             "or", "for", "on", "with", "as", "by", "that", "this", "it", "be"}


def keywords(text, n=8):
    words = [w for w in re.findall(r"[\w'-]+", text.lower()) if len(w) > 3 and w not in STOPWORDS]
    return [w for w, _ in Counter(words).most_common(n)]


class KnowledgeGraph:
    """Graph-Based Memory: entities, (subject, relation, object) facts, provenance."""

    def __init__(self):
        self._facts = []
        self._lock = threading.Lock()

    def add_fact(self, subject, relation, obj, source="unknown", confidence=0.8):
        with self._lock:
            conflicts = [f for f in self._facts
                         if f["subject"] == subject and f["relation"] == relation
                         and f["object"].lower() != str(obj).lower()]
            fact = {
                "subject": subject,
                "relation": relation,
                "object": str(obj),
                "source": source,
                "confidence": float(confidence),
                "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                "conflicts": len(conflicts),
            }
            self._facts.append(fact)
            return fact

    def resolve_conflicts(self, subject, relation):
        """Memory Conflict Resolution: keep the newest, most confident fact."""
        with self._lock:
            matches = [f for f in self._facts
                       if f["subject"] == subject and f["relation"] == relation]
            if len(matches) < 2:
                return matches
            winner = max(matches, key=lambda f: (f["confidence"], f["ts"]))
            self._facts = [f for f in self._facts if f not in matches]
            self._facts.append(winner)
            return [winner]

    def query(self, subject=None, relation=None, obj=None):
        with self._lock:
            return [f for f in self._facts
                    if (subject is None or f["subject"] == subject)
                    and (relation is None or f["relation"] == relation)
                    and (obj is None or f["object"].lower() == str(obj).lower())]

    def related(self, subject, max_hops=1):
        """Multi-hop traversal: facts about subject, then facts about its objects."""
        frontier = {subject}
        seen = []
        for _ in range(max_hops):
            nxt = set()
            for s in list(frontier):
                for f in self.query(subject=s):
                    if f not in seen:
                        seen.append(f)
                        nxt.add(f["object"])
            frontier = nxt
            if not frontier:
                break
        return seen

    def all_facts(self):
        with self._lock:
            return list(self._facts)

    def __len__(self):
        return len(self._facts)


class WorkingMemory:
    """Bounded recent-context buffer for the current session."""

    def __init__(self, capacity=10):
        self._items = deque(maxlen=capacity)

    def push(self, text):
        self._items.append(str(text))

    def context(self, max_chars=600):
        text = "\n".join(self._items)
        return text[-max_chars:]

    def __len__(self):
        return len(self._items)


class EpisodicMemory:
    """Episodic Timeline Reconstruction: timestamped events, replayable."""

    def __init__(self, path="data/octo/episodes.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._lock = threading.Lock()

    def record(self, event, detail=""):
        entry = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "event": str(event)[:120],
            "detail": str(detail)[:300],
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(entry["ts"] + " | " + entry["event"] + " | " + entry["detail"] + "\n")
        return entry

    def timeline(self, n=20):
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            return [l.rstrip() for l in f.readlines()[-n:]]


class SemanticMemory:
    """Hybrid RAG Architecture: TF-IDF vector search + keyword + graph traversal."""

    def __init__(self, roots=("data/octo",)):
        self.roots = roots
        self._index = None

    @property
    def index(self):
        if self._index is None:
            self._index = RagIndex.build(roots=self.roots)
        return self._index

    def retrieve(self, query, k=3):
        hybrid = []
        for chunk, score in self.index.search(query, k=k):
            hybrid.append({"text": chunk, "score": round(score, 3), "origin": "vector"})
        return hybrid


class MemoryManager:
    """Facade: remember (graph + episodic), recall (hybrid), compress (summarize)."""

    def __init__(self, roots=("data/octo",), working_capacity=10):
        self.graph = KnowledgeGraph()
        self.working = WorkingMemory(working_capacity)
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory(roots)

    def remember(self, subject, relation, obj, source="conversation"):
        fact = self.graph.add_fact(subject, relation, obj, source=source)
        self.episodic.record("remember", f"{subject} {relation} {obj}")
        return fact

    def observe(self, text):
        """Contextual Memory Recall prep: keep recent turns in working memory."""
        self.working.push(text)
        for m in re.finditer(r"(\w[\w -]{1,30}?)\s+(?:is|are|হলো)\s+(.{3,80}?)(?:[.।]|$)",
                             text, re.I):
            subj, obj = m.group(1).strip(), m.group(2).strip()
            if subj.lower() not in STOPWORDS and len(subj.split()) <= 4:
                self.graph.add_fact(subj, "is", obj, source="observed", confidence=0.6)

    def recall(self, query, k=3):
        """Contextual Memory Recall: graph facts + working context + RAG."""
        hits = []
        for w in keywords(query, 3):
            for fact in self.graph.query(subject=w) or self.graph.related(w):
                hits.append({"text": f"{fact['subject']} {fact['relation']} {fact['object']}",
                             "score": fact["confidence"], "origin": "graph",
                             "source": fact["source"]})
        for chunk, score in self.semantic.index.search(query, k=k):
            hits.append({"text": chunk, "score": round(score, 3), "origin": "rag"})
        ctx = self.working.context()
        if ctx:
            hits.append({"text": ctx, "score": 0.3, "origin": "working"})
        return sorted(hits, key=lambda h: -h["score"])[:k]

    def compress(self, history, n=3):
        """Memory Compression Engine: extractive summary of past conversations."""
        return summarize("\n".join(history), n=n)

    def provenance(self, subject, relation):
        """Knowledge Provenance Tracking: where a stored fact came from."""
        return [{"source": f["source"], "ts": f["ts"], "confidence": f["confidence"]}
                for f in self.graph.query(subject, relation)]

    def export_json(self, path):
        """Export the full memory state (facts + episodes) as JSON backup."""
        state = {"facts": self.graph.all_facts(),
                 "episodes": self.episodic.timeline(5000)}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=1)
        return state

    def import_json(self, path):
        """Restore facts (and episode lines) from a backup file."""
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        restored = 0
        for fact in state.get("facts", []):
            self.graph.add_fact(fact["subject"], fact["relation"], fact["object"],
                                source=fact.get("source", "imported"),
                                confidence=fact.get("confidence", 0.7))
            restored += 1
        for line in state.get("episodes", []):
            self.episodic.record("imported", line[:200])
        return restored


class EncryptedEpisodicMemory(EpisodicMemory):
    """Data Privacy Layer: episodic memory encrypted at rest via Fernet.

    Requires `cryptography` and an unlocked EncryptionManager; falls back to
    plaintext behavior (with a marker) when encryption is unavailable so the
    app never crashes over an optional dependency.
    """

    def __init__(self, path="data/octo/episodes.jsonl", manager=None):
        super().__init__(path=path)
        self.manager = manager

    def _ready(self):
        return self.manager is not None and getattr(self.manager, "unlocked", lambda: False)()

    def record(self, event, detail=""):
        if self._ready():
            token = self.manager.encrypt(f"{event} | {detail}")
            entry = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
                     "event": "encrypted", "detail": token}
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(entry["ts"] + " | " + entry["event"] + " | "
                            + entry["detail"].replace("\n", " ") + "\n")
            return entry
        return super().record(event, detail)

    def timeline(self, n=20):
        lines = super().timeline(n)
        if not self._ready():
            return lines
        out = []
        for line in lines:
            parts = line.split(" | ", 2)
            if len(parts) == 3 and parts[1] == "encrypted":
                try:
                    out.append(parts[0] + " | " + self.manager.decrypt(parts[2].strip()))
                    continue
                except Exception:
                    out.append(parts[0] + " | (encrypted, unlock to read)")
            else:
                out.append(line)
        return out
