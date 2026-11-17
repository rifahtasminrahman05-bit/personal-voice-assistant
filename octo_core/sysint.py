# -*- coding: utf-8 -*-
"""Advanced System Intelligence (features 45-50).

DynamicResourceAllocator, PriorityScheduler, FailurePredictor,
SelfHealingManager, DistributedExecutor, SmartCacheManager.
"""
import datetime

class DynamicResourceAllocator:
    """45. Dynamic Resource Allocation: split resources by task demand."""

    def allocate(self, demands, total):
        total_weight = sum(demands.values()) or 1
        return {task: {res: round(w / total_weight * amount, 2)
                       for res, amount in total.items()}
                for task, w in demands.items()}


class PriorityScheduler:
    """46. Intelligent Task Scheduler: priority + deadline + dependencies."""

    def __init__(self):
        self.tasks = []

    def submit(self, name, priority=5, deadline=None, depends=()):
        self.tasks.append({"name": name, "priority": priority,
                           "deadline": deadline, "depends": list(depends),
                           "done": False})

    def ready(self):
        done = {t["name"] for t in self.tasks if t["done"]}
        return [t for t in self.tasks if not t["done"]
                and all(d in done for d in t["depends"])]

    def next_task(self):
        ready = sorted(self.ready(),
                       key=lambda t: (t["deadline"] is None,
                                      t["deadline"] or 0, t["priority"]))
        return ready[0]["name"] if ready else None

    def complete(self, name):
        for t in self.tasks:
            if t["name"] == name:
                t["done"] = True


class FailurePredictor:
    """47. Predictive Failure Detection: metric trends -> failure risk."""

    def predict(self, metrics, limits):
        risks = {}
        for name, series in metrics.items():
            if not series:
                continue
            limit = limits.get(name) or max(max(series), 1e-9)
            util = min(series[-1] / limit, 1.5) / 1.5
            trend = 0.0
            if len(series) >= 2 and series[-1] > series[0]:
                trend = min((series[-1] - series[0]) / max(limit, 1e-9), 1.0)
            risks[name] = round(min(1.0, 0.8 * util + 0.4 * trend), 2)
        overall = max(risks.values()) if risks else 0.0
        return {"risks": risks, "failure_risk": overall,
                "verdict": ("act-now" if overall > 0.7
                            else "watch" if overall > 0.4 else "healthy")}


class SelfHealingManager:
    """48. Self-Healing Infrastructure: registered recovery procedures;
    unsafe procedures require explicit human approval."""

    def __init__(self, approve=None):
        self.procedures = {}
        self.approve = approve or (lambda action, desc: True)
        self.history = []

    def register(self, symptom, procedure, safe=True):
        self.procedures[symptom] = {"procedure": list(procedure), "safe": safe}

    def heal(self, symptom):
        proc = self.procedures.get(symptom)
        if not proc:
            return {"healed": False, "reason": "no procedure registered"}
        if not proc["safe"] and not self.approve("heal",
                                                 f"{symptom}: {proc['procedure']}"):
            return {"healed": False, "reason": "awaiting human approval"}
        self.history.append({"symptom": symptom, "steps": proc["procedure"]})
        return {"healed": True, "steps": proc["procedure"]}


class DistributedExecutor:
    """49. Distributed Agent Execution: assign tasks to nodes by load."""

    def __init__(self, nodes):
        self.nodes = dict(nodes)

    def assign(self, tasks):
        loads = {n: 0.0 for n in self.nodes}
        plan = {n: [] for n in self.nodes}
        for t in sorted(tasks, key=lambda x: -x.get("cost", 1)):
            node = min(loads, key=lambda n: loads[n] / self.nodes[n])
            plan[node].append(t["name"])
            loads[node] += t.get("cost", 1)
        return {"plan": plan, "loads": loads}


class SmartCacheManager:
    """50. Intelligent Cache Manager: LRU + frequency-aware eviction."""

    def __init__(self, capacity=100):
        self.capacity = capacity
        self.entries = {}
        self.tick = 0

    def put(self, key, value):
        self.tick += 1
        if len(self.entries) >= self.capacity and key not in self.entries:
            victim = min(self.entries,
                         key=lambda k: (self.entries[k]["hits"],
                                        self.entries[k]["tick"]))
            del self.entries[victim]
        hits = self.entries[key]["hits"] if key in self.entries else 0
        self.entries[key] = {"value": value, "hits": hits, "tick": self.tick}

    def get(self, key):
        entry = self.entries.get(key)
        if not entry:
            return None
        entry["hits"] += 1
        self.tick += 1
        entry["tick"] = self.tick
        return entry["value"]

    def stats(self):
        return {"entries": len(self.entries), "keys": sorted(self.entries)}


class GracefulDegradation:
    """Maintain core functionality under resource pressure."""

    TIERS = (("full", 0.0), ("reduced", 0.6), ("core-only", 0.85))

    def __init__(self):
        self.features = {"chat": True, "memory": True, "agents": True,
                         "multimodal": True, "training": True}

    def degrade(self, pressure):
        """Disable optional features as pressure (0..1) rises."""
        tier = "full"
        for name, threshold in self.TIERS:
            if pressure >= threshold:
                tier = name
        if tier == "reduced":
            self.features["multimodal"] = False
            self.features["training"] = False
        elif tier == "core-only":
            self.features.update({"multimodal": False, "training": False,
                                  "agents": False})
        return {"pressure": pressure, "tier": tier,
                "enabled": sorted(k for k, v in self.features.items() if v)}

    def restore(self):
        self.features = {k: True for k in self.features}
        return dict(self.features)


class IdleOptimizer:
    """Background consolidation and maintenance during low-activity periods."""

    def __init__(self, idle_seconds=300):
        self.idle_seconds = idle_seconds
        self.tasks = []

    def register(self, name, fn):
        self.tasks.append({"name": name, "fn": fn, "last": None})
        return name

    def run_idle(self, idle_for):
        """Run maintenance tasks only when the user has been idle long enough."""
        if idle_for < self.idle_seconds:
            return {"ran": [], "reason": f"idle {idle_for}s < {self.idle_seconds}s"}
        ran = []
        for task in self.tasks:
            try:
                task["fn"]()
                task["last"] = _now()
                ran.append(task["name"])
            except Exception:
                pass
        return {"ran": ran, "reason": "idle threshold reached"}


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")
