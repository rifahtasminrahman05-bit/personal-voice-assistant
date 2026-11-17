# -*- coding: utf-8 -*-
"""Advanced Autonomous Agent Intelligence (features 7-12).

CollaborationNetwork, NegotiationEngine, DynamicAgentFactory,
ReputationSystem, WorkflowComposer, LongTermGoalManager.
"""
import re
from collections import Counter


class CollaborationNetwork:
    """7. Multi-Agent Collaboration Network: shared blackboard + handoff."""

    def __init__(self):
        self.agents = {}
        self.blackboard = []

    def register(self, name, specialties=()):
        self.agents[name] = {"specialties": set(specialties), "messages": []}
        return name

    def post(self, sender, topic, content):
        msg = {"from": sender, "topic": topic, "content": content}
        self.blackboard.append(msg)
        for name, agent in self.agents.items():
            if name != sender and topic in agent["specialties"]:
                agent["messages"].append(msg)
        return msg

    def inbox(self, name):
        return self.agents[name]["messages"]

    def collaborators(self, topic):
        return sorted(n for n, a in self.agents.items()
                      if topic in a["specialties"])


class NegotiationEngine:
    """8. Agent Negotiation Engine: merge competing plans by weighted votes."""

    def negotiate(self, proposals, weights=None):
        weights = weights or {}
        by_plan = {}
        for p in proposals:
            entry = by_plan.setdefault(p["plan"], {
                "plan": p["plan"], "votes": 0, "weight": 0.0,
                "total_score": 0.0, "agents": []})
            entry["votes"] += 1
            entry["weight"] += weights.get(p["agent"], 1.0)
            entry["total_score"] += p.get("score", 0.0)
            entry["agents"].append(p["agent"])
        ranked = sorted(by_plan.values(),
                        key=lambda e: (-e["weight"], -e["total_score"]))
        return {"ranking": ranked,
                "agreed": ranked[0]["plan"] if ranked else None,
                "consensus": bool(ranked) and ranked[0]["votes"] > len(proposals) / 2}

    def resolve_conflict(self, conflicts):
        ordered = sorted(conflicts, key=lambda c: -c.get("confidence", 0))
        return {"claim": ordered[0]["claim"], "by": ordered[0]["agent"],
                "runner_up": ordered[1]["claim"] if len(ordered) > 1 else None}


class DynamicAgentFactory:
    """9. Dynamic Agent Creation: spin up a specialist agent per task."""

    ROLE_MAP = (
        (re.compile(r"\b(security|vuln|attack|audit)\b", re.I),
         ("security-analyst", ("scan", "audit_log"))),
        (re.compile(r"\b(code|bug|test|refactor)\b", re.I),
         ("coder", ("code", "run_tests"))),
        (re.compile(r"\b(research|paper|source|cite)\b", re.I),
         ("researcher", ("search", "cross_check"))),
        (re.compile(r"\b(data|train|model|benchmark)\b", re.I),
         ("data-scientist", ("train", "evaluate"))),
    )

    def create(self, task, registry=None):
        for pattern, (role, tools) in self.ROLE_MAP:
            if pattern.search(task):
                break
        else:
            role, tools = "generalist", ("chat",)
        spec = {"name": f"{role}-{sum(map(ord, task)) % 1000}", "role": role,
                "tools": list(tools), "task": task}
        if registry is not None:
            registry[spec["name"]] = spec
        return spec


class ReputationSystem:
    """10. Agent Reputation System: EMA reliability per agent."""

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.scores = {}

    def record(self, agent, success):
        s = self.scores.get(agent, 0.5)
        self.scores[agent] = round(s + self.alpha * ((1.0 if success else 0.0) - s), 3)
        return self.scores[agent]

    def score(self, agent):
        return self.scores.get(agent, 0.5)

    def rank(self):
        return sorted(self.scores.items(), key=lambda kv: -kv[1])

    def select(self, candidates):
        return max(candidates, key=lambda a: self.score(a))


class WorkflowComposer:
    """11. Autonomous Workflow Composer: goal -> ordered executable steps."""

    RULES = (
        (re.compile(r"\bresearch\b", re.I),
         ["search sources", "cross-check facts", "synthesize report"]),
        (re.compile(r"\b(code|feature|bug)\b", re.I),
         ["understand codebase", "implement change", "run tests",
          "summarize diff"]),
        (re.compile(r"\b(security|audit)\b", re.I),
         ["enumerate attack surface", "prioritize findings",
          "recommend fixes"]),
    )

    def compose(self, goal):
        for pattern, steps in self.RULES:
            if pattern.search(goal):
                return {"goal": goal, "steps": list(steps)}
        return {"goal": goal,
                "steps": ["clarify objective", "gather context", "execute",
                          "verify"]}


class LongTermGoalManager:
    """12. Long-Term Goal Manager: goals with milestones + progress."""

    def __init__(self):
        self.goals = {}

    def add(self, goal_id, title, milestones):
        self.goals[goal_id] = {"title": title,
                               "milestones": [{"name": m, "done": False}
                                              for m in milestones]}

    def complete(self, goal_id, milestone):
        for m in self.goals[goal_id]["milestones"]:
            if m["name"] == milestone:
                m["done"] = True

    def progress(self, goal_id):
        ms = self.goals[goal_id]["milestones"]
        done = sum(1 for m in ms if m["done"])
        return {"goal": self.goals[goal_id]["title"], "done": done,
                "total": len(ms),
                "percent": round(done / max(len(ms), 1), 2)}

    def next_action(self, goal_id):
        for m in self.goals[goal_id]["milestones"]:
            if not m["done"]:
                return m["name"]
        return None
