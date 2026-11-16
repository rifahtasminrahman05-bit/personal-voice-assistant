# -*- coding: utf-8 -*-
"""Predictive Intelligence — forecasting and future-state simulation.

PredictiveEngine: least-squares trend forecasting over a numeric series with
residual-based confidence. FutureStateSimulator: projects decision branches
into ranked probable futures.
"""
import random


class PredictiveEngine:
    """Predictive Intelligence Engine: estimate future states and outcomes."""

    @staticmethod
    def forecast(series, horizon=3):
        """Linear-regression forecast; returns {"forecast", "slope",
        "confidence", "direction"}. Confidence falls with residual noise."""
        if len(series) < 2:
            return {"forecast": list(series), "slope": 0.0,
                    "confidence": 0.0, "direction": "unknown"}
        n = len(series)
        xs = list(range(n))
        mean_x, mean_y = sum(xs) / n, sum(series) / n
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, series)) \
            / max(sum((x - mean_x) ** 2 for x in xs), 1e-9)
        intercept = mean_y - slope * mean_x
        residuals = [y - (slope * x + intercept) for x, y in zip(xs, series)]
        spread = (sum(r * r for r in residuals) / n) ** 0.5
        scale = max(abs(mean_y), 1e-9)
        confidence = max(0.0, min(1.0, 1.0 - min(spread / scale, 1.0)))
        forecast = [round(slope * (n + i) + intercept, 4) for i in range(horizon)]
        direction = "rising" if slope > 1e-9 else "falling" if slope < -1e-9 else "flat"
        return {"forecast": forecast, "slope": round(slope, 4),
                "confidence": round(confidence, 2), "direction": direction}

    def predict_outcome(self, history_entries, metric_path=("math", "score"),
                        horizon=3):
        """Forecast a metric from BenchmarkHistory entries."""
        values = []
        for e in history_entries:
            node = e["report"]
            for key in metric_path:
                if not isinstance(node, dict) or key not in node:
                    node = None
                    break
                node = node[key]
            if isinstance(node, (int, float)):
                values.append(node)
        return self.forecast(values, horizon=horizon)


class FutureStateSimulator:
    """Future State Simulator: project decision branches into ranked futures.

    Each decision: {"name", "cost" (0-1), "gain" (0-1), "risk" (0-1)}.
    Simulation rolls probabilistic futures per decision and ranks by
    expected value; low-risk/high-gain branches win.
    """

    def __init__(self, seed=42, rounds=200):
        self.seed = seed
        self.rounds = rounds

    def simulate(self, state, decisions, steps=3):
        rng = random.Random(self.seed)
        futures = []
        for d in decisions:
            outcomes = []
            for _ in range(self.rounds):
                value = 0.0
                for _ in range(steps):
                    success = rng.random() > d.get("risk", 0.5)
                    value += d.get("gain", 0.0) if success else -d.get("cost", 0.0)
                outcomes.append(value)
            outcomes.sort()
            expected = sum(outcomes) / len(outcomes)
            p10 = outcomes[len(outcomes) // 10]
            p90 = outcomes[-1 - len(outcomes) // 10]
            futures.append({
                "decision": d["name"],
                "expected_value": round(expected, 3),
                "p10": round(p10, 3),
                "p90": round(p90, 3),
                "risk_of_loss": round(sum(1 for o in outcomes if o < 0) / len(outcomes), 2),
            })
        futures.sort(key=lambda f: -f["expected_value"])
        return {"initial_state": dict(state), "steps": steps, "futures": futures,
                "recommended": futures[0]["decision"] if futures else None}
