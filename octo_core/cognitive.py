# -*- coding: utf-8 -*-
"""Cognitive Intelligence Core — reasoning, planning, reflection, introspection.

Pure-stdlib heuristics that wrap Octo's small-model limits into structured
thinking: task decomposition, hypothesis generation/ranking, causal chains,
uncertainty quantification, knowledge synthesis, constraint checks and a
reflection loop over execution traces.
"""
import re
from collections import Counter

CAUSE_WORDS = ("because", "caused by", "due to", "leads to", "results in",
               "কারণ", "ফলে", "জনিত")
HEDGES = ("maybe", "possibly", "may", "might", "unclear", "unknown",
          "সম্ভবত", "হয়তো", "অজানা")


def decompose(goal):
    """Dynamic task decomposition: split a goal into executable subtasks."""
    goal = goal.strip()
    subtasks = []
    parts = re.split(r"\band then\b|\bthen\b|,|।", goal, flags=re.I)
    parts = [p.strip(" .?!") for p in parts if len(p.strip(" .?!")) > 2]
    if len(parts) > 1:
        for p in parts:
            subtasks.append({"task": p, "kind": _classify(p)})
        return subtasks
    kind = _classify(goal)
    if kind == "question":
        subtasks.append({"task": f"gather facts about: {goal}", "kind": "research"})
        subtasks.append({"task": f"answer: {goal}", "kind": "answer"})
    elif kind == "command":
        subtasks.append({"task": goal, "kind": "execute"})
        subtasks.append({"task": f"verify result of: {goal}", "kind": "verify"})
    else:
        subtasks.append({"task": goal, "kind": kind})
    return subtasks


def _classify(text):
    if text.endswith("?") or re.match(r"^(what|how|why|when|who|which|ki|কী|কিভাবে|কেন)", text, re.I):
        return "question"
    if re.match(r"^(research|find|search|scan|analyze|audit|খোঁজ|গবেষণা)", text, re.I):
        return "research"
    if re.match(r"^(write|build|create|fix|run|make|বানাও|ঠিক)", text, re.I):
        return "execute"
    return "command"


def generate_hypotheses(problem, evidence=()):
    """Hypothesis Generation Engine: propose, score and rank explanations."""
    problem_l = problem.lower()
    hyps = []
    for item in evidence:
        item = str(item).strip()
        if not item:
            continue
        score = 0.5
        for w in problem_l.split():
            if len(w) > 3 and w in item.lower():
                score += 0.1
        if any(cw in item.lower() for cw in CAUSE_WORDS):
            score += 0.2
        hyps.append({"hypothesis": item[:200], "score": round(min(score, 0.99), 2)})
    if not hyps:
        hyps.append({"hypothesis": f"insufficient evidence for: {problem}", "score": 0.1})
    return sorted(hyps, key=lambda h: h["score"], reverse=True)


def causal_chain(text):
    """Causal Inference Engine: extract cause -> effect pairs from text."""
    chains = []
    for sentence in re.split(r"[.।\n]", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        for cw in CAUSE_WORDS:
            if cw in sentence.lower():
                left, right = re.split(cw, sentence, maxsplit=1, flags=re.I)
                left, right = left.strip(" ,"), right.strip(" ,")
                if 5 < len(left) < 160 and 5 < len(right) < 160:
                    chains.append({"cause": left, "effect": right})
                break
    return chains


def quantify_uncertainty(text, evidence_hits=0):
    """Uncertainty Quantification: confidence estimate for a conclusion."""
    words = text.lower().split()
    if not words:
        return {"confidence": 0.0, "level": "none", "hedges": 0}
    hedge_hits = sum(1 for w in words if w.strip(".,!?") in HEDGES)
    confidence = 0.5 + 0.1 * min(evidence_hits, 4) - 0.15 * hedge_hits
    confidence = max(0.0, min(1.0, confidence))
    level = "high" if confidence >= 0.75 else "medium" if confidence >= 0.45 else "low"
    return {"confidence": round(confidence, 2), "level": level, "hedges": hedge_hits}


def synthesize(sources, max_points=6):
    """Knowledge Synthesis Engine: merge multiple texts into coherent points."""
    sentences = []
    for src in sources:
        for s in re.split(r"[.।\n]", str(src)):
            s = s.strip()
            if len(s) > 10:
                sentences.append(s)
    seen, points = set(), []
    for s in sentences:
        key = " ".join(sorted(w for w in s.lower().split() if len(w) > 3))
        if key and key not in seen:
            seen.add(key)
            points.append(s)
    ranked = sorted(points, key=lambda s: -len(set(s.lower().split())))[:max_points]
    return ranked


def check_constraints(plan_steps, constraints):
    """Constraint-Aware Reasoning: verify a plan against allowed/forbidden rules."""
    violations = []
    for rule in constraints:
        rule = str(rule)
        if rule.startswith("!"):
            forbidden = rule[1:].lower()
            for step in plan_steps:
                if forbidden in str(step).lower():
                    violations.append(f"forbidden '{forbidden}' in: {str(step)[:60]}")
        else:
            if not any(rule.lower() in str(step).lower() for step in plan_steps):
                violations.append(f"required '{rule}' missing from plan")
    return violations


def reflect(trace):
    """Introspection Framework + Strategy Reflection Loop.

    Analyzes an execution trace [(tool, output), ...], finds failures and
    proposes an alternative strategy for the next attempt.
    """
    failures = [(t, o) for t, o in trace if _is_failure(o)]
    reflection = {
        "steps": len(trace),
        "failures": len(failures),
        "success_rate": round(1 - len(failures) / len(trace), 2) if trace else 0.0,
        "failed_tools": sorted({t for t, _ in failures}),
        "next_strategy": None,
    }
    if failures:
        reflection["next_strategy"] = (
            f"retry without failed tools {reflection['failed_tools']}; "
            "use narrower subtasks and verify each result before proceeding")
    return reflection


def _is_failure(output):
    o = str(output).lower()
    return ("failed" in o or "error" in o or "nothing" in o or "not trained" in o
            or "could not" in o or "no matches" in o)


class DecisionEngine:
    """Adaptive Strategy Selection: pick a strategy based on task complexity."""

    def __init__(self, strategies=None):
        self.strategies = strategies or {
            "simple": {"max_steps": 3, "tools": ("search", "answer")},
            "standard": {"max_steps": 6, "tools": ("research", "search", "answer")},
            "deep": {"max_steps": 10, "tools": ("deep", "research", "search", "answer")},
        }

    def choose(self, goal):
        g = goal.lower()
        if re.search(r"deep|গভীর|comprehensive|exhaustive", g):
            name = "deep"
        elif len(decompose(goal)) > 1 or len(g.split()) > 10:
            name = "standard"
        else:
            name = "simple"
        return name, self.strategies[name]
