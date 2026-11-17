# -*- coding: utf-8 -*-
"""Advanced Cognitive Intelligence (features 1-6).

CognitiveLoadManager, AdaptiveReasoningEngine, CausalReasoningEngine,
CounterfactualReasoning, MetacognitiveController, UncertaintyAwareness.
"""
import re
from collections import Counter

COMPLEX_SIGNALS = re.compile(
    r"\b(prove|optimize|design|architecture|multi-step|derive|compare|"
    r"evaluate|why|trade-?off)\b", re.I)
SIMPLE_SIGNALS = re.compile(r"\b(what is|who|when|list|define|hello|hi)\b", re.I)
HEDGE_WORDS = ("maybe", "possibly", "might", "could", "unclear", "uncertain",
               "probably", "সম্ভবত", "হয়তো")


class CognitiveLoadManager:
    """1. Cognitive Load Manager: allocate compute by task complexity."""

    def estimate_load(self, task):
        score = 0.2
        score += min(len(task.split()) / 60.0, 0.3)
        score += 0.3 if COMPLEX_SIGNALS.search(task) else 0.0
        score -= 0.15 if SIMPLE_SIGNALS.search(task) else 0.0
        load = max(0.0, min(1.0, score))
        level = "low" if load < 0.35 else "medium" if load < 0.7 else "high"
        return {"load": round(load, 2), "level": level}

    def allocate(self, task, max_steps=512):
        est = self.estimate_load(task)
        share = {"low": 0.25, "medium": 0.5, "high": 1.0}[est["level"]]
        return {"load": est["load"], "level": est["level"],
                "steps": max(16, int(max_steps * share)),
                "samples": 3 if est["level"] == "high" else 1}


class AdaptiveReasoningEngine:
    """2. Adaptive Reasoning Engine: strategy per problem type."""

    STRATEGIES = (
        (re.compile(r"\b(math|calculate|compute|sum|equation|solve)\b", re.I),
         "stepwise-arithmetic"),
        (re.compile(r"\b(code|function|bug|compile|program|script)\b", re.I),
         "decompose-and-test"),
        (re.compile(r"\b(research|sources|evidence|literature)\b", re.I),
         "gather-then-synthesize"),
        (re.compile(r"\b(ethic|should|moral|policy)\b", re.I),
         "deliberate-multiple-perspectives"),
        (re.compile(r"\b(prove|derive|theorem|logic)\b", re.I),
         "formal-deduction"),
    )

    def select_strategy(self, problem):
        for pattern, strategy in self.STRATEGIES:
            if pattern.search(problem):
                return {"strategy": strategy, "reason": "matched problem type"}
        return {"strategy": "chain-of-thought",
                "reason": "default general reasoning"}

    def reason(self, problem):
        sel = self.select_strategy(problem)
        sel["steps"] = [f"frame: {problem[:60]}",
                        f"apply {sel['strategy']}",
                        "verify result against the question"]
        return sel


class CausalReasoningEngine:
    """3. Causal Reasoning Engine: temporal precedence + co-occurrence."""

    def infer(self, episodes, outcome, min_support=2):
        prior_counts, cause_totals, outcome_total = Counter(), Counter(), 0
        for events in episodes:
            if outcome in events:
                outcome_total += 1
                idx = events.index(outcome)
                for e in set(events[:idx]):
                    prior_counts[e] += 1
            for e in set(events):
                cause_totals[e] += 1
        causes = []
        for cause, co in prior_counts.items():
            if co >= min_support:
                causes.append({"cause": cause, "co_occurrences": co,
                               "p_outcome_if_cause":
                                   round(co / max(cause_totals[cause], 1), 2)})
        causes.sort(key=lambda c: (-c["co_occurrences"],
                                   -c["p_outcome_if_cause"]))
        return {"outcome": outcome, "outcome_episodes": outcome_total,
                "causes": causes[:5]}

    def is_plausible_cause(self, cause_event, outcome_event, timeline):
        return (cause_event in timeline and outcome_event in timeline
                and timeline.index(cause_event) < timeline.index(outcome_event))


class CounterfactualReasoning:
    """4. Counterfactual Reasoning: what-if analysis over decisions."""

    def analyze(self, actual, alternatives, baseline_value):
        rows = [{"decision": actual["name"], "value": baseline_value,
                 "delta": 0.0, "better": False}]
        for alt in alternatives:
            delta = alt["value"] - baseline_value
            rows.append({"decision": alt["name"], "value": alt["value"],
                         "delta": round(delta, 3), "better": delta > 0})
        rows.sort(key=lambda r: -r["value"])
        best = rows[0]["decision"]
        return {"scenarios": rows,
                "regret": round(max(0.0, rows[0]["value"] - baseline_value), 3),
                "would_have_chosen": best}


class MetacognitiveController:
    """5. Metacognitive Controller: monitor and evaluate reasoning traces."""

    def monitor(self, trace):
        total = max(len(trace), 1)
        evidenced = sum(1 for t in trace if t.get("has_evidence"))
        verified = sum(1 for t in trace if t.get("verifies"))
        gaps = [t["step"] for t in trace if not t.get("has_evidence")]
        return {"steps": len(trace),
                "evidence_ratio": round(evidenced / total, 2),
                "verification_ratio": round(verified / total, 2),
                "ungrounded_steps": gaps,
                "verdict": "sound" if not gaps and verified >= len(trace) - 1
                           else "needs-review"}

    def should_escalate(self, report, threshold=0.6):
        return report["evidence_ratio"] < threshold


class UncertaintyAwareness:
    """6. Uncertainty Awareness Engine: how uncertain is this answer."""

    def quantify(self, answer, evidence=""):
        ans_words = set(re.findall(r"\w{3,}", answer.lower()))
        ev_words = set(re.findall(r"\w{3,}", str(evidence).lower()))
        support = len(ans_words & ev_words) / max(len(ans_words), 1)
        hedges = sum(1 for w in HEDGE_WORDS if w in answer.lower())
        vague = 0.05 * min(len(re.findall(r"\b(some|many|several|few)\b",
                                          answer.lower())), 3)
        uncertainty = max(0.0, min(1.0, 0.6 * (1 - support) + 0.1 * hedges + vague))
        label = ("low" if uncertainty < 0.35
                 else "medium" if uncertainty < 0.7 else "high")
        return {"uncertainty": round(uncertainty, 2),
                "evidence_support": round(support, 2), "hedges": hedges,
                "label": label}
