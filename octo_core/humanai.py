# -*- coding: utf-8 -*-
"""Advanced Human-AI Interaction (features 51-56).

PersonalizedInteractionEngine, IntentPredictor, DialogueManager,
ExplainableDecisionEngine, CollaborativeProblemSolver, AdaptiveTeachingEngine.
"""
import re
from collections import Counter, defaultdict, deque

REFERENCE_WORDS = re.compile(r"\b(it|that|this|এটা|ওটা)\b", re.I)


class PersonalizedInteractionEngine:
    """51. Personalized Interaction Engine: style per user profile."""

    def __init__(self):
        self.profiles = {}

    def set_profile(self, user, expertise="beginner", language="bn"):
        self.profiles[user] = {"expertise": expertise, "language": language}

    def adapt(self, user, answer):
        profile = self.profiles.get(user, {"expertise": "beginner",
                                           "language": "bn"})
        if profile["expertise"] == "expert":
            return {"answer": answer, "style": "concise-technical",
                    "glossary": False}
        return {"answer": answer, "style": "explained-simply",
                "glossary": profile["expertise"] == "beginner"}


class IntentPredictor:
    """52. Intent Prediction System: next intent from transition history."""

    def __init__(self):
        self.transitions = defaultdict(Counter)

    def observe(self, prev_intent, next_intent):
        self.transitions[prev_intent][next_intent] += 1

    def predict(self, current_intent):
        options = self.transitions.get(current_intent)
        if not options:
            return {"intent": None, "confidence": 0.0}
        intent, count = options.most_common(1)[0]
        return {"intent": intent,
                "confidence": round(count / sum(options.values()), 2)}


class DialogueManager:
    """53. Context-Aware Dialogue Manager: context + reference resolution."""

    def __init__(self, max_turns=10):
        self.turns = deque(maxlen=max_turns)
        self.last_entity = None

    def push(self, speaker, text, entity=None):
        self.turns.append({"speaker": speaker, "text": text})
        if entity:
            self.last_entity = entity

    def resolve(self, text):
        if self.last_entity and REFERENCE_WORDS.search(text):
            return REFERENCE_WORDS.sub(self.last_entity, text)
        return text

    def context(self, n=3):
        return list(self.turns)[-n:]


class ExplainableDecisionEngine:
    """54. Explainable Decision Engine: decision + reasons + evidence."""

    def choose(self, options, score_fn, evidence_fn=None):
        scored = sorted(((opt, score_fn(opt)) for opt in options),
                        key=lambda p: -p[1])
        best, score = scored[0]
        return {"decision": best, "score": round(score, 3),
                "why": f"highest score ({round(score, 3)}) among {len(options)} options",
                "alternatives": [(o, round(s, 3)) for o, s in scored[1:]],
                "evidence": evidence_fn(best) if evidence_fn else []}


class CollaborativeProblemSolver:
    """55. Collaborative Problem-Solving: propose -> feedback -> refine."""

    def __init__(self, max_rounds=5):
        self.max_rounds = max_rounds

    def solve(self, problem, propose, feedback, satisfied):
        history = []
        for round_no in range(1, self.max_rounds + 1):
            solution = propose(problem, round_no)
            notes = feedback(solution)
            history.append({"round": round_no, "solution": solution,
                            "notes": notes})
            if satisfied(solution, notes):
                return {"solved": True, "rounds": round_no,
                        "solution": solution, "history": history}
        return {"solved": False, "rounds": self.max_rounds,
                "solution": history[-1]["solution"], "history": history}


class AdaptiveTeachingEngine:
    """56. Adaptive Teaching Engine: explanation style per learner level."""

    def __init__(self):
        self.mastery = {}

    def record(self, learner, topic, correct):
        key = (learner, topic)
        level = self.mastery.get(key, 0.5)
        self.mastery[key] = round(max(0.0, min(1.0,
            level + (0.2 if correct else -0.2))), 2)

    def teach(self, learner, topic, content):
        level = self.mastery.get((learner, topic), 0.5)
        if level < 0.4:
            style = "step-by-step with analogies"
        elif level < 0.75:
            style = "examples + practice"
        else:
            style = "advanced summary"
        return {"topic": topic, "style": style, "content": content,
                "mastery": level}
