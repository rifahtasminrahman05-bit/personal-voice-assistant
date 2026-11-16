# -*- coding: utf-8 -*-
"""Advanced Model Intelligence — test-time compute scaling and ensembles.

TestTimeCompute: allocates reasoning budget by difficulty — easy prompts get
one short pass, hard prompts get N samples with self-consistency voting.
Ensemble: runs several member generators and returns the consensus answer
with an agreement score.
"""
import re


HARD_SIGNALS = re.compile(
    r"\b(prove|derive|optimize|why|explain|design|architecture|algorithm|"
    r"complex|multi-step|কেন|ব্যাখ্যা|প্রমাণ)\b", re.I)


def _tokens(text):
    return {w for w in re.findall(r"\w{3,}", text.lower())}


def similarity(a, b):
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta | tb), 1)


class TestTimeCompute:
    """Test-Time Compute Scaling: more computation for harder tasks."""

    def __init__(self, generate_fn, samples_hard=5, steps_easy=60, steps_hard=180):
        self.generate_fn = generate_fn
        self.samples_hard = samples_hard
        self.steps_easy = steps_easy
        self.steps_hard = steps_hard

    def difficulty(self, prompt):
        score = 0
        score += min(len(prompt.split()) // 12, 2)
        score += 2 if HARD_SIGNALS.search(prompt) else 0
        score += 1 if "?" in prompt and len(prompt) > 80 else 0
        return "hard" if score >= 2 else "easy"

    def solve(self, prompt, difficulty=None):
        difficulty = difficulty or self.difficulty(prompt)
        if difficulty == "easy":
            return {"answer": self.generate_fn(prompt, self.steps_easy),
                    "difficulty": difficulty, "samples": 1, "agreement": 1.0}
        answers = [self.generate_fn(prompt, self.steps_hard)
                   for _ in range(self.samples_hard)]
        consensus, agreement = self._vote(answers)
        return {"answer": consensus, "difficulty": difficulty,
                "samples": len(answers), "agreement": round(agreement, 2)}

    @staticmethod
    def _vote(answers):
        if not answers:
            return "", 0.0
        clusters = []
        for a in answers:
            for c in clusters:
                if similarity(a, c[0]) >= 0.5:
                    c.append(a)
                    break
            else:
                clusters.append([a])
        best = max(clusters, key=len)
        return best[0], len(best) / len(answers)


class Ensemble:
    """Ensemble Model Intelligence: combine multiple model outputs."""

    def __init__(self, members):
        """members: list of (name, callable(prompt) -> str)."""
        self.members = list(members)

    def answer(self, prompt):
        outputs = []
        for name, fn in self.members:
            try:
                outputs.append((name, str(fn(prompt))))
            except Exception as e:
                outputs.append((name, f"error: {e}"))
        valid = [text for _, text in outputs if not text.startswith("error")]
        if not valid:
            return {"answer": outputs[0][1] if outputs else "", "agreement": 0.0,
                    "outputs": dict(outputs), "consensus": False}
        clusters = []
        for text in valid:
            for c in clusters:
                if similarity(text, c[0]) >= 0.5:
                    c.append(text)
                    break
            else:
                clusters.append([text])
        best = max(clusters, key=len)
        return {"answer": best[0], "agreement": round(len(best) / len(valid), 2),
                "outputs": dict(outputs),
                "consensus": len(best) > len(valid) / 2}

    def weighted_answer(self, prompt, weights):
        """Agreement-weighted pick: prefer the highest-weight member's output
        when a consensus cluster exists."""
        result = self.answer(prompt)
        if result["agreement"] == 1.0 or not result["consensus"]:
            return result
        best_name = max(self.members, key=lambda m: weights.get(m[0], 0))[0]
        if not result["outputs"].get(best_name, "").startswith("error"):
            result["answer"] = result["outputs"][best_name]
            result["picked_by"] = "weight"
        return result
