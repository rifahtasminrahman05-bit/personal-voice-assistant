# -*- coding: utf-8 -*-
"""Experimental Frontier Intelligence (features 57-60).

CognitiveArchitectureEvolutionEngine, ArtificialSocietySimulator,
EmergenceAnalyzer, AutonomousResearchPlanner.
"""
import random
from collections import Counter, defaultdict

LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


class CognitiveArchitectureEvolutionEngine:
    """57. Cognitive Architecture Evolution: mutate + select vectors."""

    def __init__(self, capabilities=("attention", "memory", "tools",
                                     "reflection"), seed=7):
        self.capabilities = tuple(capabilities)
        self.rng = random.Random(seed)

    def evolve(self, score_fn, generations=6, population=6):
        pop = [{c: self.rng.choice(LEVELS) for c in self.capabilities}
               for _ in range(population)]
        best, best_score = None, -1e9
        for _ in range(generations):
            scored = sorted(((score_fn(a), a) for a in pop),
                            key=lambda p: -p[0])
            if scored[0][0] > best_score:
                best_score, best = scored[0][0], dict(scored[0][1])
            survivors = [dict(a) for _, a in scored[:max(2, population // 2)]]
            pop = [dict(survivors[i % len(survivors)])
                   for i in range(population)]
            for arch in pop[1:]:
                cap = self.rng.choice(self.capabilities)
                arch[cap] = self.rng.choice(LEVELS)
        return {"best": best, "score": round(best_score, 3)}


class ArtificialSocietySimulator:
    """58. Artificial Society Simulator: agents exchange opinions."""

    def __init__(self, agents, seed=3):
        self.opinions = dict(agents)
        self.rng = random.Random(seed)
        self.history = []

    def tick(self, interactions=10, influence=0.2):
        names = sorted(self.opinions)
        for _ in range(interactions):
            if len(names) < 2:
                break
            a, b = self.rng.sample(names, 2)
            diff = self.opinions[b] - self.opinions[a]
            self.opinions[a] = round(self.opinions[a] + influence * diff, 3)
            self.opinions[b] = round(self.opinions[b] - influence * diff * 0.5, 3)
        self.history.append(dict(self.opinions))
        return dict(self.opinions)

    def polarization(self):
        vals = list(self.opinions.values())
        if not vals:
            return 0.0
        mean = sum(vals) / len(vals)
        return round((sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5, 3)


class EmergenceAnalyzer:
    """59. Emergent Behavior Analyzer: patterns in multi-agent logs."""

    def analyze(self, logs):
        by_tick = defaultdict(Counter)
        for entry in logs:
            by_tick[entry["tick"]][entry["action"]] += 1
        ticks = sorted(by_tick)
        consensus = None
        for t in ticks:
            counts = by_tick[t]
            top_action, n = counts.most_common(1)[0]
            if n / sum(counts.values()) >= 0.8:
                consensus = top_action
        actions = Counter(entry["action"] for entry in logs)
        agents = {entry["agent"] for entry in logs}
        dominant = actions.most_common(1)[0][0] if actions else None
        emergent = consensus is not None or bool(
            actions and actions[dominant] > len(logs) / 2)
        return {"ticks": len(ticks), "agents": len(agents),
                "action_distribution": dict(actions),
                "dominant_action": dominant, "late_consensus": consensus,
                "emergent": emergent}


class AutonomousResearchPlanner:
    """60. Autonomous Research Planner: question -> plan + experiments."""

    def plan(self, question, known_facts=()):
        sub_questions = [q.strip() for q in question.split("?") if q.strip()]
        experiments = [{"name": f"experiment-{i + 1}", "tests": q}
                       for i, q in enumerate(sub_questions)]
        if not experiments:
            experiments = [{"name": "experiment-1", "tests": question}]
        return {"question": question,
                "sub_questions": sub_questions or [question],
                "steps": ["literature scan", "define variables",
                          "collect data", "run experiments",
                          "analyze + cross-check", "write up"],
                "experiments": experiments,
                "success_criteria": "results replicated across >=2 independent checks",
                "known_facts": list(known_facts)}
