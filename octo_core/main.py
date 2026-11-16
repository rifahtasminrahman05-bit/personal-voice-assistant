# -*- coding: utf-8 -*-
"""Octo Core CLI — unified entrypoint for the advanced architecture.

Security-gated agent execution with memory, reflection and audit:

    python -m octo_core --goal "research sql injection and answer how to prevent it"
    python -m octo_core --benchmark
    python -m octo_core --inject-check "ignore previous instructions and ..."
"""
import argparse
import json

from .agents import AgentOrchestrator
from .cognitive import DecisionEngine, reflect
from .evaluation import AgentMetrics, BenchmarkSuite, HallucinationDetector
from .memory import MemoryManager
from .security import AuditLogger, detect_prompt_injection, redact_secrets, validate_output


def run_goal(goal, approve=None, max_steps=8, verbose=True):
    """Full pipeline: injection check -> decision -> orchestrate -> recall -> audit."""
    risk, matches = detect_prompt_injection(goal)
    if risk == "high":
        return {"blocked": True, "risk": risk, "matches": matches}
    memory = MemoryManager()
    memory.observe(goal)
    strategy_name, strategy = DecisionEngine().choose(goal)
    orchestrator = AgentOrchestrator(approve=approve, max_steps=max_steps)
    trace, reflection = orchestrator.execute(goal, verbose=verbose)
    metrics = AgentMetrics.from_trace(trace)
    memory.episodic.record("goal_complete", goal[:120])
    if metrics["success_rate"] >= 0.8:
        from .learning import SkillManager
        SkillManager().learn(goal, [t for t, _ in trace], outcome_ok=True)
    return {"blocked": False, "strategy": strategy_name, "trace": trace,
            "reflection": reflection, "metrics": metrics,
            "recall": [h["text"][:120] for h in memory.recall(goal, k=2)]}


def main():
    ap = argparse.ArgumentParser(description="Octo Core: advanced intelligence pipeline.")
    ap.add_argument("--goal", help="goal for the orchestrator")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--benchmark", action="store_true", help="run the benchmark suite")
    ap.add_argument("--inject-check", help="test a prompt for injection patterns")
    ap.add_argument("--audit", action="store_true", help="show last audit entries")
    args = ap.parse_args()

    if args.inject_check is not None:
        risk, matches = detect_prompt_injection(args.inject_check)
        print(json.dumps({"risk": risk, "matches": len(matches)}, indent=2))
        return
    if args.audit:
        for entry in AuditLogger().tail(15):
            print(f"{entry['ts']} [{entry['agent']}] {entry['event']}: {entry['detail'][:100]}")
        return
    if args.benchmark:
        suite = BenchmarkSuite()
        report = suite.run()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if not args.goal:
        ap.error("--goal, --benchmark, --inject-check or --audit required")

    result = run_goal(args.goal, max_steps=args.max_steps)
    if result["blocked"]:
        print(f"BLOCKED: prompt-injection risk={result['risk']} "
              f"({len(result['matches'])} patterns)")
        return
    print(f"strategy: {result['strategy']} | success: {result['metrics']['success_rate']}")
    print("\n=== trace ===")
    for tool, output in result["trace"]:
        print(f"[{tool}] {redact_secrets(output)[:280]}")
    print("\n=== reflection ===")
    print(json.dumps(result["reflection"], indent=2))


if __name__ == "__main__":
    main()
