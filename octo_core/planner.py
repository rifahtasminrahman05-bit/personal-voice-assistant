# -*- coding: utf-8 -*-
"""Planning & Goal System — hierarchical, adaptive planning framework
(Components Deep Dive).

Goal creation/decomposition, sequential/parallel action planning,
execution monitoring with dynamic replanning, multi-criteria decisions,
contingency fallbacks, and long-horizon scheduling. Pure stdlib.
"""
import datetime

from .cognitive import decompose


class GoalManager:
    """Goal creation with priority, constraints, deadlines, success metrics."""

    def __init__(self):
        self.goals = {}

    def create(self, goal_id, description, priority=3, deadline=None,
               metrics=None, constraints=None):
        g = {"description": description, "priority": priority,
             "deadline": deadline, "metrics": metrics or {},
             "constraints": constraints or [], "subgoals": [],
             "status": "open", "created": _now()}
        self.goals[goal_id] = g
        return g

    def decompose(self, goal_id, subgoals):
        """Break a high-level goal into sub-goals and atomic tasks."""
        g = self.goals[goal_id]
        atoms = decompose(g["description"])
        g["subgoals"] = [{"id": s, "tasks": atoms} for s in subgoals]
        g["status"] = "planned"
        return g["subgoals"]

    def progress(self, goal_id, done_subgoals):
        g = self.goals[goal_id]
        total = max(len(g["subgoals"]), 1)
        g["status"] = "done" if done_subgoals >= total else "in-progress"
        return {"subgoals_done": done_subgoals, "subgoals_total": total,
                "ratio": round(done_subgoals / total, 2),
                "status": g["status"]}

    def overdue(self, now=None):
        now = now or datetime.datetime.now()
        out = []
        for gid, g in self.goals.items():
            if g["deadline"] and g["status"] != "done":
                try:
                    dl = datetime.datetime.fromisoformat(g["deadline"])
                except ValueError:
                    continue
                if dl < now:
                    out.append(gid)
        return out


class ActionPlanner:
    """Sequential or parallel executable plans from tasks with deps."""

    def plan(self, tasks):
        """tasks: [{id, deps: [ids], parallel_safe: bool}]."""
        by_id = {t["id"]: t for t in tasks}
        stages, placed = [], set()
        pending = set(by_id)
        while pending:
            ready = sorted(t for t in pending
                           if all(d in placed for d in by_id[t].get("deps", [])))
            if not ready:
                raise ValueError("cyclic dependencies in task set")
            stage = [by_id[t] for t in ready]
            parallel = all(by_id[t].get("parallel_safe", False) for t in ready)
            stages.append({"tasks": [t["id"] for t in stage],
                           "parallel": parallel})
            placed |= set(ready)
            pending -= set(ready)
        return {"stages": stages, "n_stages": len(stages)}

    def critical_path(self, tasks, durations):
        """Longest dependency chain by duration (approx greedy)."""
        by_id = {t["id"]: t for t in tasks}
        memo = {}

        def longest(tid):
            if tid in memo:
                return memo[tid]
            deps = by_id[tid].get("deps", [])
            best = max((longest(d) for d in deps if d in by_id),
                       default=(0, []))
            memo[tid] = (best[0] + durations.get(tid, 0), best[1] + [tid])
            return memo[tid]

        best = max((longest(t) for t in by_id), key=lambda p: p[0])
        return {"length": best[0], "path": best[1]}


class ExecutionMonitor:
    """Real-time progress tracking with dynamic replanning."""

    def __init__(self, plan):
        self.plan = plan
        self.done = set()
        self.failed = {}

    def mark(self, task_id, ok, error=""):
        if ok:
            self.done.add(task_id)
        else:
            self.failed[task_id] = error
        return self.report()

    def report(self):
        total = sum(len(s["tasks"]) for s in self.plan["stages"])
        return {"done": len(self.done), "total": total,
                "ratio": round(len(self.done) / max(total, 1), 2),
                "failures": dict(self.failed)}

    def replan(self, fallbacks):
        """Dynamic replanning: swap failed tasks for fallbacks."""
        swaps = {}
        for task, err in self.failed.items():
            if task in fallbacks:
                swaps[task] = fallbacks[task]
        return {"swaps": swaps, "unrecoverable":
                sorted(t for t in self.failed if t not in swaps)}


class MultiCriteriaDecisionMaker:
    """Utility, risk, cost and value-alignment weighted evaluation."""

    def __init__(self, weights=None):
        self.weights = weights or {"utility": 0.4, "risk": 0.25,
                                   "cost": 0.2, "alignment": 0.15}

    def score(self, options):
        """options: [{name, utility, risk, cost, alignment}] (0..1 each)."""
        w = self.weights
        ranked = []
        for o in options:
            s = (w["utility"] * o.get("utility", 0)
                 - w["risk"] * o.get("risk", 0)
                 - w["cost"] * o.get("cost", 0)
                 + w["alignment"] * o.get("alignment", 0))
            ranked.append({"name": o["name"], "score": round(s, 3)})
        ranked.sort(key=lambda r: -r["score"])
        return {"choice": ranked[0]["name"] if ranked else None,
                "ranking": ranked}


class ContingencyPlanner:
    """Pre-computed fallback strategies for likely failure points."""

    def __init__(self):
        self.fallbacks = {}

    def register(self, failure_point, fallback_action):
        self.fallbacks[failure_point] = fallback_action

    def for_plan(self, tasks):
        return [{"task": t, "fallback": self.fallbacks.get(t, "abort-and-report")}
                for t in tasks]

    def likely_failures(self, tasks, risky_markers):
        """Flag tasks whose name suggests a registered/likely failure point."""
        return [t for t in tasks
                if any(m in t.lower() for m in risky_markers)]


class LongHorizonPlanner:
    """Multi-day / multi-week goal tracking and scheduling."""

    def __init__(self):
        self.schedule = []

    def schedule_goal(self, goal_id, start_day, days, weekly=False):
        item = {"goal": goal_id, "day": start_day, "days": days,
                "weekly": weekly}
        self.schedule.append(item)
        self.schedule.sort(key=lambda s: s["day"])
        return item

    def horizon(self, day_from, day_to):
        return [s for s in self.schedule
                if day_from <= s["day"] <= day_to]

    def workload(self, day):
        """Total scheduled days of goals active on the given day."""
        return sum(s["days"] for s in self.schedule
                   if s["day"] <= day <= s["day"] + s["days"] - 1)


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")
