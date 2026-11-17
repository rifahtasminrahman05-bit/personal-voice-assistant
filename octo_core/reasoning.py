# -*- coding: utf-8 -*-
"""Reasoning Engine — multi-strategy cognitive core (Components Deep Dive).

Chain-of-Thought, Tree-of-Thoughts, Reflection, Meta-Cognition (confidence
calibration), Analogical Reasoning, Abductive Reasoning. Pure heuristics,
fully offline and testable.
"""
import re
from difflib import SequenceMatcher

STEP_SPLIT = re.compile(r"\n+|(?<=[.!?;])\s+|,\s+(?:then|and then)\s+", re.I)
CAUSE_WORDS = re.compile(r"\b(because|since|therefore|so that|due to|as a result)\b", re.I)


class ChainOfThought:
    """Linear, step-by-step logical decomposition of a problem."""

    def decompose(self, problem):
        raw = [s.strip() for s in STEP_SPLIT.split(problem) if s.strip()]
        steps = [f"step {i + 1}: {s}" for i, s in enumerate(raw)]
        return {"problem": problem[:120], "steps": steps,
                "n_steps": len(steps)}

    def verify(self, steps, check):
        """Check each step against a verification callable (step -> bool)."""
        results = [{"step": s, "ok": bool(check(s))} for s in steps]
        return {"results": results,
                "all_ok": all(r["ok"] for r in results),
                "failed": [r["step"] for r in results if not r["ok"]]}


class TreeOfThoughts:
    """Parallel exploration of reasoning branches with pruning."""

    def __init__(self, beam_width=3, max_depth=4):
        self.beam_width = beam_width
        self.max_depth = max_depth

    def explore(self, problem, brancher, scorer, depth=None):
        """brancher(node) -> list of child strings; scorer(node) -> float."""
        depth = self.max_depth if depth is None else depth
        frontier = [(problem, [problem], 0.0)]
        explored = 0
        for _ in range(depth):
            candidates = []
            for node, path, score in frontier:
                for child in brancher(node):
                    explored += 1
                    candidates.append((child, path + [child], scorer(child)))
            if not candidates:
                break
            candidates.sort(key=lambda c: -c[2])
            frontier = candidates[:self.beam_width]
        best = frontier[0]
        return {"path": best[1], "score": round(best[2], 3),
                "explored": explored, "pruned": explored - len(frontier)}


class ReflectionEngine:
    """Self-critique loop: draft -> critique -> refine, bounded rounds."""

    def __init__(self, max_rounds=3):
        self.max_rounds = max_rounds

    def refine(self, draft, critic, improver):
        """critic(text) -> list of issues; improver(text, issues) -> text."""
        issues = critic(draft)
        history = [{"round": 0, "text": draft, "issues": issues}]
        text = draft
        for r in range(1, self.max_rounds + 1):
            if not issues:
                break
            text = improver(text, issues)
            issues = critic(text)
            history.append({"round": r, "text": text, "issues": issues})
        return {"final": text, "rounds": len(history) - 1,
                "resolved": not issues, "history": history}


class ConfidenceCalibrator:
    """Meta-Cognition: confidence calibration + knowledge-gap report."""

    EVIDENCE = re.compile(r"\b(data|study|source|benchmark|test|measured|log)\b", re.I)

    def calibrate(self, claim, evidence=""):
        hedge = len(re.findall(r"\b(maybe|possibly|likely|unclear|might)\b",
                               claim.lower()))
        support = len(self.EVIDENCE.findall((evidence or "") + " " + claim))
        confidence = max(0.0, min(1.0, 0.35 + 0.2 * support - 0.15 * hedge))
        band = "low" if confidence < 0.4 else \
               "medium" if confidence < 0.7 else "high"
        gaps = [m for m in ("definition", "mechanism", "measurement")
                if m not in (claim + " " + evidence).lower()]
        return {"confidence": round(confidence, 2), "band": band,
                "evidence_hits": support, "hedges": hedge,
                "knowledge_gaps": gaps}


class AnalogicalReasoning:
    """Map solutions from past cases onto novel situations."""

    def __init__(self):
        self.cases = []

    def add_case(self, situation, features, solution):
        self.cases.append({"situation": situation,
                           "features": set(f.lower() for f in features),
                           "solution": solution})

    def _similarity(self, feats, other):
        if not feats or not other:
            return 0.0
        return len(feats & other) / len(feats | other)

    def map(self, target_features, min_sim=0.3):
        target = set(f.lower() for f in target_features)
        ranked = []
        for case in self.cases:
            sim = self._similarity(target, case["features"])
            if sim >= min_sim:
                ranked.append({"case": case["situation"],
                               "solution": case["solution"],
                               "similarity": round(sim, 2)})
        ranked.sort(key=lambda r: -r["similarity"])
        return {"transfer": ranked[0] if ranked else None,
                "candidates": ranked[:5]}


class AbductiveReasoning:
    """Rank the most plausible explanations for observations."""

    def __init__(self):
        self.hypotheses = []

    def add_hypothesis(self, name, explains, prior=0.5, cost=0.0):
        self.hypotheses.append({"name": name,
                                "explains": set(e.lower() for e in explains),
                                "prior": prior, "cost": cost})

    def best_explanation(self, observations):
        obs = set(o.lower() for o in observations)
        ranked = []
        for h in self.hypotheses:
            covered = len(obs & h["explains"])
            score = (covered / max(len(obs), 1)) * h["prior"] - h["cost"]
            ranked.append({"hypothesis": h["name"],
                           "covers": sorted(obs & h["explains"]),
                           "coverage": round(covered / max(len(obs), 1), 2),
                           "score": round(score, 3)})
        ranked.sort(key=lambda r: -r["score"])
        return {"best": ranked[0]["hypothesis"] if ranked else None,
                "ranking": ranked}


def analogous_text(a, b):
    """Cheap surface-level textual analogy ratio (0..1)."""
    return round(SequenceMatcher(None, a.lower(), b.lower()).ratio(), 2)
