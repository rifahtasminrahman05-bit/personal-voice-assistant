# -*- coding: utf-8 -*-
"""World Model — rich internal representation of reality (Components Deep Dive).

Entities, typed/weighted relationships, properties & states, temporal model,
causal graph, spatial awareness, and a query/inference engine (traversal,
path finding, pattern matching). Pure stdlib, fully offline.
"""
import itertools
from collections import deque

REL_KINDS = ("causal", "hierarchic", "associative", "temporal")


class WorldModel:
    def __init__(self):
        self.entities = {}
        self.relations = []
        self.states = {}
        self.timeline = []
        self.causal = {}
        self.positions = {}

    def add_entity(self, name, kind="concept"):
        self.entities.setdefault(name, {"kind": kind})
        return name

    def relate(self, a, kind, b, weight=1.0):
        if kind not in REL_KINDS:
            raise ValueError(f"relation kind must be one of {REL_KINDS}")
        rel = {"from": a, "kind": kind, "to": b, "weight": weight}
        self.relations.append(rel)
        self.add_entity(a)
        self.add_entity(b)
        if kind == "causal":
            self.causal.setdefault(a, set()).add(b)
        return rel

    def set_state(self, entity, **props):
        st = self.states.setdefault(entity, {})
        st.update(props)
        self.timeline.append({"entity": entity, "state": dict(st)})
        return st

    def get_state(self, entity):
        return dict(self.states.get(entity, {}))

    def history(self, entity):
        return [t for t in self.timeline if t["entity"] == entity]

    def project(self, entity, trend=0.0):
        """Temporal model: naive projection of numeric state forward."""
        cur = dict(self.states.get(entity, {}))
        return {k: round(v + trend, 3) if isinstance(v, (int, float)) else v
                for k, v in cur.items()}

    def neighbors(self, entity, kind=None):
        return [r for r in self.relations
                if r["from"] == entity and (kind is None or r["kind"] == kind)]

    def causes(self, effect, transitive=False):
        direct = {r["from"] for r in self.relations
                  if r["kind"] == "causal" and r["to"] == effect}
        if not transitive:
            return sorted(direct)
        seen, queue = set(), deque(direct)
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            for r in self.relations:
                if r["kind"] == "causal" and r["to"] == node and r["from"] not in seen:
                    queue.append(r["from"])
        return sorted(seen)

    def effects(self, cause):
        return sorted(self.causal.get(cause, set()))

    def place(self, entity, x, y):
        self.positions[entity] = (x, y)

    def distance(self, a, b):
        ax, ay = self.positions[a]
        bx, by = self.positions[b]
        return round(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5, 2)

    def nearby(self, entity, radius):
        return sorted(o for o in self.positions
                      if o != entity and self.distance(entity, o) <= radius)

    def path(self, start, goal):
        """Shortest associative/causal path via BFS."""
        queue = deque([(start, [start])])
        seen = {start}
        while queue:
            node, path = queue.popleft()
            if node == goal:
                return path
            for r in self.relations:
                if r["from"] == node and r["to"] not in seen:
                    seen.add(r["to"])
                    queue.append((r["to"], path + [r["to"]]))
        return None

    def pattern_match(self, pattern):
        """Find (a, b) pairs matching {'from': f, 'kind': k, 'to': t} where
        None means wildcard."""
        out = []
        for r in self.relations:
            if ((pattern.get("from") is None or r["from"] == pattern["from"])
                    and (pattern.get("kind") is None or r["kind"] == pattern["kind"])
                    and (pattern.get("to") is None or r["to"] == pattern["to"])):
                out.append(r)
        return out

    def deduce(self, subject):
        """Logical deduction summary: direct facts + inferred closure."""
        facts = [f"{r['from']} -{r['kind']}-> {r['to']}"
                 for r in self.neighbors(subject)]
        inferred = [f"{subject} causes {e}" for e in self.effects(subject)]
        return {"facts": facts, "inferred": inferred}

    def snapshot(self):
        return {"entities": sorted(self.entities),
                "relations": len(self.relations),
                "states": {k: dict(v) for k, v in self.states.items()},
                "timeline": len(self.timeline)}

    def merge(self, other_relations):
        """Bulk import of relation triples [from, kind, to, weight]."""
        added = 0
        for a, kind, b, *w in other_relations:
            self.relate(a, kind, b, weight=w[0] if w else 1.0)
            added += 1
        return added


def causal_closure(model, seeds):
    """All effects reachable from any of the seed causes."""
    reach = set()
    frontier = list(seeds)
    while frontier:
        node = frontier.pop()
        for nxt in model.effects(node):
            if nxt not in reach:
                reach.add(nxt)
                frontier.append(nxt)
    return sorted(reach)


def conflicting_states(model, entity, key):
    """Detect contradictory property values across the timeline."""
    vals = [t["state"].get(key) for t in model.history(entity) if key in t["state"]]
    return {"entity": entity, "key": key, "values": vals,
            "conflict": len(set(map(str, vals))) > 1}
