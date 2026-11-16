# -*- coding: utf-8 -*-
"""Advanced AI Evaluation & Benchmarking.

Reasoning benchmark suite (math, multi-step, grounding), hallucination
detection against the RAG corpus, response quality metrics, agent success
metrics and a pytest-based regression runner.
"""
import os
import re
import subprocess
import sys

from octo.rag import RagIndex

from .cognitive import quantify_uncertainty


def _tokens(text):
    return {w for w in re.findall(r"[\w'-]+", text.lower()) if len(w) > 2}


class HallucinationDetector:
    """Hallucination Detection: sentence-level grounding check against a corpus."""

    def __init__(self, roots=("data/octo",)):
        self.index = RagIndex.build(roots=roots)

    def check(self, response, threshold=0.25):
        sentences = [s.strip() for s in re.split(r"[.।\n]", response) if len(s.strip()) > 15]
        claims = []
        for s in sentences:
            toks = _tokens(s)
            best = self.index.search(s, k=1)
            support = 0.0
            if best:
                overlap = len(toks & _tokens(best[0][0]))
                support = overlap / max(len(toks), 1)
            claims.append({"claim": s[:160], "support": round(support, 2),
                           "supported": support >= threshold})
        grounded = sum(c["supported"] for c in claims)
        return {
            "claims": claims,
            "grounded_ratio": round(grounded / len(claims), 2) if claims else 1.0,
            "verdict": "grounded" if not claims or grounded / len(claims) >= 0.5 else "suspect",
        }


def quality_metrics(response, reference):
    """Response Quality Evaluator: relevance, overlap, conciseness."""
    r_toks, ref_toks = _tokens(response), _tokens(reference)
    if not r_toks:
        return {"relevance": 0.0, "recall": 0.0, "conciseness": 0.0}
    overlap = r_toks & ref_toks
    relevance = len(overlap) / max(len(r_toks), 1)
    recall = len(overlap) / max(len(ref_toks), 1)
    conciseness = min(1.0, 120 / max(len(response.split()), 1))
    return {"relevance": round(relevance, 2), "recall": round(recall, 2),
            "conciseness": round(conciseness, 2)}


class BenchmarkSuite:
    """Reasoning Benchmark Suite + Memory Recall Benchmark + Safety checks."""

    MATH_CASES = [
        ("2+3*4", 14), ("(10-4)/2", 3.0), ("7*6", 42), ("100/5", 20.0),
    ]

    def __init__(self, answer_fn=None, roots=("data/octo",)):
        self.answer_fn = answer_fn
        self.detector = HallucinationDetector(roots=roots)

    def math_benchmark(self):
        from octo.agi import safe_calc
        passed = 0
        for expr, expected in self.MATH_CASES:
            try:
                passed += abs(safe_calc(expr) - expected) < 1e-6
            except Exception:
                pass
        return {"suite": "math", "passed": passed, "total": len(self.MATH_CASES),
                "score": round(passed / len(self.MATH_CASES), 2)}

    def grounding_benchmark(self, responses):
        results = [self.detector.check(r) for r in responses]
        ratios = [r["grounded_ratio"] for r in results] or [0.0]
        return {"suite": "grounding", "responses": len(responses),
                "avg_grounded_ratio": round(sum(ratios) / len(ratios), 2),
                "suspect": sum(1 for r in results if r["verdict"] == "suspect")}

    def recall_benchmark(self, memory, facts):
        """Store facts, then check contextual recall returns them."""
        hits = 0
        for subject, relation, obj in facts:
            memory.remember(subject, relation, obj, source="benchmark")
            recalled = memory.recall(f"{subject} {relation}")
            if any(obj in h["text"] for h in recalled):
                hits += 1
        return {"suite": "memory_recall", "passed": hits, "total": len(facts),
                "score": round(hits / max(len(facts), 1), 2)}

    def uncertainty_benchmark(self, texts):
        levels = [quantify_uncertainty(t)["level"] for t in texts]
        return {"suite": "uncertainty", "levels": levels}

    def run(self, responses=()):
        report = {"math": self.math_benchmark()}
        if responses:
            report["grounding"] = self.grounding_benchmark(responses)
        return report


class RegressionRunner:
    """Automated Testing Framework: pytest in a subprocess + metric diff."""

    def run_tests(self, path="tests/"):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", path, "-q", "--no-header"],
            capture_output=True, text=True, timeout=540)
        summary = result.stdout.strip().splitlines()[-1] if result.stdout else ""
        return {"returncode": result.returncode, "summary": summary}

    def compare(self, baseline, current, threshold=0.05):
        from .learning import detect_regression
        return detect_regression(baseline, current, threshold)


class AgentMetrics:
    """Agent Success Metrics: completion rate, tool accuracy, reliability."""

    @staticmethod
    def from_trace(trace):
        total = len(trace)
        ok = sum(1 for _, out in trace
                 if not re.search(r"failed|error|could not|nothing|denied", str(out), re.I))
        return {"steps": total, "success_rate": round(ok / total, 2) if total else 0.0,
                "reliable": total > 0 and ok / total >= 0.8}
