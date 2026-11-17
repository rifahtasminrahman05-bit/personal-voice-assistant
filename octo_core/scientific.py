# -*- coding: utf-8 -*-
"""Scientific Intelligence — hypothesis generation/testing and a controlled
autonomous discovery loop.

ScientificHypothesisEngine: extracts variable correlations from observations,
turns them into testable hypotheses, and verifies against held-out data.
DiscoveryLab: propose -> design experiment -> run (optionally sandboxed code)
-> record -> rank, in bounded rounds.
"""
import re


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0


class ScientificHypothesisEngine:
    """Scientific Hypothesis Engine: generate, test and rank hypotheses."""

    def generate(self, observations):
        """observations: list of dicts with numeric fields. Returns
        correlation-based hypotheses sorted by |r|."""
        if len(observations) < 3:
            return []
        fields = [k for k, v in observations[0].items() if isinstance(v, (int, float))]
        hyps = []
        for i, a in enumerate(fields):
            for b in fields[i + 1:]:
                xs = [o[a] for o in observations if a in o and b in o]
                ys = [o[b] for o in observations if a in o and b in o]
                r = _pearson(xs, ys)
                if abs(r) >= 0.5:
                    direction = "increases" if r > 0 else "decreases"
                    hyps.append({
                        "hypothesis": f"{a} {direction} with {b}",
                        "variables": (a, b), "r": round(r, 3),
                        "strength": "strong" if abs(r) >= 0.8 else "moderate",
                    })
        return sorted(hyps, key=lambda h: -abs(h["r"]))

    def test(self, hypothesis, data, threshold=0.4):
        """Verify a hypothesis' prediction against new data."""
        a, b = hypothesis["variables"]
        xs = [o[a] for o in data if a in o and b in o]
        ys = [o[b] for o in data if a in o and b in o]
        r = _pearson(xs, ys)
        predicted_positive = hypothesis["r"] > 0
        supported = (r > threshold) == predicted_positive and abs(r) >= threshold
        return {"hypothesis": hypothesis["hypothesis"], "r_new": round(r, 3),
                "verdict": "supported" if supported else "refuted"}


class DiscoveryLab:
    """Autonomous Discovery Laboratory: controlled experiment cycles.

    `runner(experiment)` executes an experiment and returns a result dict
    {"ok": bool, "value": float, "log": str} — e.g. sandboxed code via
    octo_core.sandbox_code. The lab iterates: hypothesis -> experiment ->
    evidence -> next hypothesis, never exceeding max_rounds.
    """

    def __init__(self, runner, audit=None):
        self.runner = runner
        self.audit = audit
        self.logbook = []

    def run_cycle(self, hypotheses, max_rounds=5):
        findings = []
        for round_no, hyp in enumerate(hypotheses[:max_rounds], 1):
            experiment = {"round": round_no, "hypothesis": hyp,
                          "method": "measure variables and compare"}
            result = self.runner(experiment)
            entry = {"round": round_no, "hypothesis": hyp,
                     "ok": bool(result.get("ok")),
                     "value": result.get("value"),
                     "log": str(result.get("log", ""))[:200]}
            self.logbook.append(entry)
            if self.audit:
                self.audit.log("discovery_experiment", "lab",
                               f"round {round_no}: {hyp[:80]}")
            findings.append(entry)
        supported = [f for f in findings if f["ok"]]
        next_steps = []
        if supported:
            next_steps.append("replicate supported experiments with more samples")
        if len(supported) < len(findings):
            next_steps.append("refine hypotheses that failed measurement")
        return {"findings": findings, "supported": len(supported),
                "next_steps": next_steps}

    def experiment_report(self):
        return {"experiments": len(self.logbook),
                "supported": sum(1 for f in self.logbook if f["ok"])}


def extract_variables(text):
    """Pull numeric observations out of free text: 'accuracy=0.9 loss=0.3'."""
    pairs = re.findall(r"([A-Za-z_]\w*)\s*[=:]\s*([0-9]*\.?[0-9]+)", text)
    return dict(pairs) if pairs else {}
