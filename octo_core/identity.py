# -*- coding: utf-8 -*-
"""Personality, Ethics and Communication layers (Components Deep Dive).

PersonalityLayer: core identity, emotional model, relationship memory,
consistency engine, persona modulation. EthicsEngine: value system, harm
assessment, transparency, boundary enforcement, user value learning.
CommunicationLayer: NLG, conversation management, clarification strategies,
explanation generation, multi-turn coherence. Pure stdlib, offline.
"""
import re


class PersonalityLayer:
    CORE_IDENTITY = {
        "name": "Octo",
        "principles": ["helpful", "honest", "safe", "local-first"],
        "style": "concise-technical",
    }
    EMOTION_WORDS = {
        "joy": re.compile(r"\b(thanks|great|awesome|love|happy)\b", re.I),
        "stress": re.compile(r"\b(urgent|asap|broken|failing|deadline)\b", re.I),
        "curiosity": re.compile(r"\b(why|how|what if|explore|research)\b", re.I),
    }

    def __init__(self):
        self.identity = dict(self.CORE_IDENTITY)
        self.emotion = {"valence": 0.0, "arousal": 0.2, "label": "neutral"}
        self.relationship = {"interactions": 0, "preferences": {}, "topics": {}}
        self.formality = 0.5

    def observe_user(self, text):
        self.relationship["interactions"] += 1
        for topic in re.findall(r"\b(python|security|network|model|linux|macos|windows)\b",
                                text.lower()):
            self.relationship["topics"][topic] = \
                self.relationship["topics"].get(topic, 0) + 1
        for pref, pat in (("short answers", re.compile(r"\b(brief|short|tl;?dr)\b", re.I)),
                          ("detail", re.compile(r"\b(explain|detail|deep)\b", re.I))):
            if pat.search(text):
                self.relationship["preferences"][pref] = \
                    self.relationship["preferences"].get(pref, 0) + 1

    def feel(self, text):
        hits = {k: bool(p.search(text)) for k, p in self.EMOTION_WORDS.items()}
        if hits["stress"]:
            self.emotion.update(valence=-0.3, arousal=0.8, label="stressed-concerned")
        elif hits["joy"]:
            self.emotion.update(valence=0.5, arousal=0.5, label="pleased")
        elif hits["curiosity"]:
            self.emotion.update(valence=0.2, arousal=0.6, label="curious")
        else:
            self.emotion.update(valence=0.0, arousal=0.3, label="neutral")
        self.emotion["valence"] = round(self.emotion["valence"], 2)
        return dict(self.emotion)

    def modulate(self, context):
        """Persona modulation: adjust tone without breaking core identity."""
        formal = re.search(r"\b(report|documentation|production|client)\b",
                           context, re.I)
        casual = re.search(r"\b(chat|quick|hey|casual)\b", context, re.I)
        if formal:
            self.formality = 0.9
        elif casual:
            self.formality = 0.3
        return {"formality": self.formality,
                "style": self.identity["style"],
                "principles": self.identity["principles"]}

    def consistency(self, response):
        """Consistency engine: score a response against the identity."""
        checks = {"honest": not re.search(r"\b(guarantee|100% certain|never fails)\b",
                                          response, re.I),
                  "safe": not re.search(r"\b(rm -rf|format c:|disable antivirus)\b",
                                        response, re.I),
                  "local": "cloud" not in response.lower()}
        return {"checks": checks,
                "consistent": all(checks.values()),
                "violations": [k for k, v in checks.items() if not v]}


class EthicsEngine:
    HARM_PATTERNS = (
        (re.compile(r"\b(delete|wipe|overwrite|drop table|rm -rf)\b", re.I), 0.8, "irreversible-action"),
        (re.compile(r"\b(exploit|brute.?force|ddos|keylog|backdoor)\b", re.I), 0.9, "offensive-security"),
        (re.compile(r"\b(steal|exfiltrate|phish|impersonate)\b", re.I), 0.95, "deception/theft"),
        (re.compile(r"\b(password|token|api[- ]?key)\b.*\b(send|upload|share)\b", re.I), 0.85, "secret-exfiltration"),
    )
    DEFAULT_VALUES = ("no-harm", "privacy", "transparency", "user-sovereignty")

    def __init__(self, values=None):
        self.values = list(values or self.DEFAULT_VALUES)
        self.user_values = []
        self.boundaries = []

    def learn_user_value(self, value):
        if value not in self.user_values:
            self.user_values.append(value)
        return self.user_values

    def add_boundary(self, rule):
        self.boundaries.append(rule)

    def assess(self, action_text):
        harms, score = [], 0.0
        for pat, weight, kind in self.HARM_PATTERNS:
            if pat.search(action_text):
                harms.append(kind)
                score = max(score, weight)
        return {"action": action_text[:120], "harm_score": round(score, 2),
                "harm_categories": harms,
                "verdict": "allow" if score == 0 else
                           "review" if score < 0.8 else "refuse"}

    def enforce(self, action_text):
        """Boundary enforcement: refuse if harm or explicit boundary hit."""
        a = self.assess(action_text)
        hit = next((b for b in self.boundaries
                    if b.lower() in action_text.lower()), None)
        if a["verdict"] == "refuse" or hit:
            return {"allowed": False, "reason": hit or a["harm_categories"]}
        return {"allowed": True, "reason": None}

    def transparency(self, decision, uncertainty=0.0, limits=()):
        return {"decision": decision[:120],
                "uncertainty": round(uncertainty, 2),
                "limitations": list(limits),
                "disclosure": ("decision, uncertainty and limits "
                               "communicated to the user")}


class CommunicationLayer:
    OPEN_QUESTION = re.compile(r"\?\s*$")

    def __init__(self, personality=None):
        self.personality = personality or PersonalityLayer()
        self.turns = []
        self.topics = []
        self.open_questions = []

    def respond(self, content, context=""):
        self.personality.modulate(context or content)
        emo = self.personality.feel(content)
        prefix = {"stressed-concerned": "Understood — treating this as urgent. ",
                  "pleased": "Glad it helped! ",
                  "curious": "Good question. "}.get(emo["label"], "")
        formality = self.personality.formality
        if formality > 0.7 and not prefix:
            prefix = "Summary: "
        reply = prefix + content
        self.turns.append({"user": content[:200], "reply": reply[:200]})
        self._track_topics(content)
        return {"reply": reply, "emotion": emo, "formality": formality}

    def _track_topics(self, text):
        m = re.search(r"\b(about|regarding)\s+([\w -]{3,40})", text, re.I)
        if m:
            topic = m.group(2).strip().lower()
            if topic not in self.topics:
                self.topics.append(topic)

    def manage_conversation(self):
        return {"turns": len(self.turns), "topics": self.topics,
                "open_questions": self.open_questions}

    def needs_clarification(self, text):
        """Clarification strategies: detect incomplete requests."""
        issues = []
        if len(text.split()) < 3:
            issues.append("request too short — ask what exactly is needed")
        if re.search(r"\b(it|that|this)\b", text.lower()) and len(self.turns) == 0:
            issues.append("unresolved reference with no prior context")
        if not text.strip().endswith("?") and \
                re.search(r"\b(help|fix|do)\b", text.lower()):
            issues.append("goal stated without specifics — ask for target and constraints")
        return {"clarify": bool(issues), "questions": issues}

    def clarify(self, text):
        check = self.needs_clarification(text)
        if check["clarify"]:
            self.open_questions.extend(check["questions"])
        return check

    def explain(self, decision, reasons, evidence=()):
        parts = [f"Decision: {decision}"]
        parts += [f"{i + 1}. {r}" for i, r in enumerate(reasons)]
        if evidence:
            parts.append("Evidence: " + "; ".join(str(e) for e in evidence))
        return "\n".join(parts)

    def coherence(self, window=5):
        """Multi-turn coherence: repeated-entity continuity ratio."""
        recent = self.turns[-window:]
        if len(recent) < 2:
            return {"coherence": 1.0, "turns": len(recent)}
        def toks(t):
            return set(re.findall(r"\w{4,}", t.lower()))
        links = 0
        for a, b in zip(recent, recent[1:]):
            if toks(a["user"]) & toks(b["user"]):
                links += 1
        return {"coherence": round(links / (len(recent) - 1), 2),
                "turns": len(recent)}
