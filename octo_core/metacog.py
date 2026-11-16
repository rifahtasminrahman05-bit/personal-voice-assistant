# -*- coding: utf-8 -*-
"""Meta-Cognitive Intelligence — cognitive architecture simulation.

CognitiveArchitectureSimulator: define architectures as capability-weight
vectors (attention, memory, tools, reflection, ...), simulate them against
task suites with capability requirements + resource costs, compare, and
grid-search optimal configurations.
"""
from itertools import product


DEFAULT_CAPABILITIES = ("attention", "memory", "tools", "reflection")


class CognitiveArchitectureSimulator:
    """Cognitive Architecture Simulator: test, compare, optimize."""

    def __init__(self, capabilities=DEFAULT_CAPABILITIES, overhead=0.05):
        self.capabilities = tuple(capabilities)
        self.overhead = overhead

    def simulate(self, architecture, tasks):
        """architecture: {capability: weight 0-1}; tasks: list of
        {"name", "needs": {capability: required 0-1}}.
        Score per task: sum of min(weight, need) minus overhead*unused weights.
        """
        arch = {c: float(architecture.get(c, 0.0)) for c in self.capabilities}
        results = []
        for task in tasks:
            needs = task.get("needs", {})
            covered = sum(min(arch.get(c, 0.0), needs.get(c, 0.0))
                          for c in self.capabilities)
            required = sum(needs.get(c, 0.0) for c in self.capabilities)
            unused = sum(arch.values()) - covered
            score = covered - self.overhead * unused
            success = required == 0 or covered / required >= 0.6
            results.append({"task": task["name"], "score": round(score, 3),
                            "covered": round(covered, 3),
                            "required": round(required, 3),
                            "success": bool(success)})
        avg = sum(r["score"] for r in results) / max(len(results), 1)
        return {"architecture": dict(arch),
                "avg_score": round(avg, 3),
                "success_rate": round(sum(r["success"] for r in results)
                                      / max(len(results), 1), 2),
                "results": results}

    def compare(self, architectures, tasks):
        ranked = []
        for name, arch in architectures.items():
            sim = self.simulate(arch, tasks)
            ranked.append({"name": name, "avg_score": sim["avg_score"],
                           "success_rate": sim["success_rate"]})
        ranked.sort(key=lambda r: (-r["avg_score"], -r["success_rate"]))
        return {"ranking": ranked, "best": ranked[0]["name"] if ranked else None}

    def optimize(self, tasks, levels=(0.0, 0.25, 0.5, 0.75, 1.0)):
        """Exhaustive grid search over capability weight levels."""
        best, best_score = None, -1e9
        for combo in product(levels, repeat=len(self.capabilities)):
            arch = dict(zip(self.capabilities, combo))
            score = self.simulate(arch, tasks)["avg_score"]
            if score > best_score:
                best, best_score = arch, score
        return {"best_architecture": best, "score": round(best_score, 3)}
