# -*- coding: utf-8 -*-
"""Self-Learning & Model Evolution (features 19-24).

ContinualLearningEngine, ExperienceReplaySystem, AdaptiveCurriculum,
SkillAcquisitionEngine, ModelAdaptationController, SelfImprovementEvalLoop.
"""
import random
from collections import Counter, deque


class ContinualLearningEngine:
    """19. Continual Learning Engine: new data + rehearsal replay."""

    def __init__(self, buffer_size=50, seed=11):
        self.buffer = deque(maxlen=buffer_size)
        self.rng = random.Random(seed)
        self.rounds = 0

    def add_experience(self, example):
        self.buffer.append(example)

    def incorporate(self, new_batch, replay_ratio=0.3):
        replay_n = int(len(new_batch) * replay_ratio)
        pool = list(self.buffer)
        replay = self.rng.sample(pool, min(replay_n, len(pool)))
        self.rounds += 1
        self.buffer.extend(new_batch)
        return {"round": self.rounds, "trained_on": len(new_batch) + len(replay),
                "new": len(new_batch), "replayed": len(replay)}


class ExperienceReplaySystem:
    """20. Experience Replay System: prioritized sampling of experiences."""

    def __init__(self, capacity=100):
        self.capacity = capacity
        self.items = []

    def add(self, experience, reward=0.0):
        self.items.append({"experience": experience, "reward": reward,
                           "priority": abs(reward) + 0.1})
        if len(self.items) > self.capacity:
            self.items.pop(0)

    def sample(self, k=3):
        if not self.items:
            return []
        weights = [i["priority"] for i in self.items]
        chosen = random.choices(self.items, weights=weights,
                                k=min(k, len(self.items)))
        return [c["experience"] for c in chosen]

    def best(self):
        return max(self.items, key=lambda i: i["reward"])["experience"] \
            if self.items else None


class AdaptiveCurriculum:
    """21. Adaptive Curriculum Generator: next-task difficulty from
    recent performance."""

    def __init__(self, window=3):
        self.window = window
        self.history = []

    def record(self, score):
        self.history.append(score)

    def next_task(self, tasks_by_level):
        recent = self.history[-self.window:]
        avg = sum(recent) / len(recent) if recent else 0.0
        level = "hard" if avg >= 0.8 else "medium" if avg >= 0.5 else "easy"
        pool = tasks_by_level.get(level) or []
        return {"level": level, "task": pool[0] if pool else None,
                "avg_recent": round(avg, 2)}


class SkillAcquisitionEngine:
    """22. Skill Acquisition Engine: validated workflows -> reusable skills."""

    def __init__(self):
        self.skills = {}

    def acquire(self, trigger, workflow, validated=True):
        if not validated:
            return None
        self.skills[trigger] = list(workflow)
        return trigger

    def match(self, goal):
        for trigger, workflow in self.skills.items():
            if trigger.lower() in goal.lower():
                return {"trigger": trigger, "workflow": list(workflow)}
        return None


class ModelAdaptationController:
    """23. Model Adaptation Controller: adjust inference params from
    feedback signals."""

    def __init__(self, settings=None):
        self.settings = dict(settings or {"temperature": 0.7, "steps": 120})

    def adapt(self, feedback):
        if feedback.get("too_random"):
            self.settings["temperature"] = round(
                max(0.1, self.settings["temperature"] - 0.1), 2)
        if feedback.get("too_deterministic"):
            self.settings["temperature"] = round(
                min(1.5, self.settings["temperature"] + 0.1), 2)
        if feedback.get("too_slow"):
            self.settings["steps"] = max(20, int(self.settings["steps"] * 0.8))
        if feedback.get("low_quality"):
            self.settings["steps"] = int(self.settings["steps"] * 1.2)
        return dict(self.settings)


class SelfImprovementEvalLoop:
    """24. Self-Improvement Evaluation Loop: before/after benchmark
    comparison; apply only on measured improvement + explicit approval."""

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def evaluate(self, before_scores, after_scores, approved=False):
        if not before_scores or len(before_scores) != len(after_scores):
            return {"approved": False, "delta": 0.0,
                    "reason": "incomparable benchmark sets"}
        delta = sum(a - b for b, a in zip(before_scores, after_scores)) \
            / len(before_scores)
        improved = delta > self.threshold
        return {"delta": round(delta, 4), "improved": improved,
                "approved": bool(improved and approved),
                "reason": ("improvement + approval" if improved and approved
                           else "improvement but awaiting approval" if improved
                           else "no improvement")}
