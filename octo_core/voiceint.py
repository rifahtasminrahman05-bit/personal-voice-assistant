# -*- coding: utf-8 -*-
"""Voice Intelligence (features 19-24, 27) — local audio signal heuristics.

Voice authentication, speaker recognition, emotion from voice, voice
activity detection, noise cancellation, multilingual voice support,
conversational voice memory. Works on raw feature vectors (RMS, pitch,
zero-crossing-rate) so it never needs heavy audio dependencies.
"""
import math
import re


class VoiceAuthenticator:
    """19. Voice Authentication: enroll a voiceprint, verify later."""

    def __init__(self, threshold=0.82):
        self.threshold = threshold
        self.prints = {}

    def enroll(self, user, features):
        self.prints[user] = list(features)
        return {"user": user, "dims": len(features)}

    def verify(self, user, features):
        ref = self.prints.get(user)
        if ref is None:
            return {"ok": False, "reason": "user not enrolled"}
        sim = _cosine(ref, features)
        return {"ok": sim >= self.threshold, "similarity": round(sim, 3),
                "threshold": self.threshold}


class SpeakerRecognizer:
    """20. Speaker Recognition: identify who is speaking."""

    def __init__(self):
        self.prints = {}

    def add_speaker(self, name, features):
        self.prints[name] = list(features)

    def identify(self, features):
        best, score = None, -1.0
        for name, ref in self.prints.items():
            sim = _cosine(ref, features)
            if sim > score:
                best, score = name, sim
        return {"speaker": best, "confidence": round(max(score, 0.0), 3)}


class VoiceEmotionDetector:
    """21. Emotion Detection from Voice: pitch + energy heuristics."""

    def detect(self, pitch_hz, energy, speech_rate):
        if pitch_hz > 220 and energy > 0.6:
            label = "excited" if speech_rate > 1.2 else "angry"
        elif pitch_hz < 120 and energy < 0.3:
            label = "sad" if speech_rate < 0.9 else "calm"
        elif energy > 0.7 and speech_rate > 1.4:
            label = "stressed"
        else:
            label = "neutral"
        return {"emotion": label, "pitch": pitch_hz, "energy": energy,
                "rate": speech_rate}


class VoiceActivityDetector:
    """22. Voice Activity Detection: RMS threshold over frames."""

    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def detect(self, frame_levels):
        voiced = [lvl >= self.threshold for lvl in frame_levels]
        segments, start = [], None
        for i, v in enumerate(voiced):
            if v and start is None:
                start = i
            if not v and start is not None:
                segments.append((start, i - 1))
                start = None
        if start is not None:
            segments.append((start, len(voiced) - 1))
        return {"voiced_ratio": round(sum(voiced) / max(len(voiced), 1), 2),
                "segments": segments}


class NoiseCanceller:
    """23. Noise Cancellation: spectral-floor style level gating."""

    def __init__(self, floor_db=-40.0):
        self.floor_db = floor_db

    def reduce(self, frame_levels_db):
        out = [max(lvl, self.floor_db) for lvl in frame_levels_db]
        removed = sum(1 for l in frame_levels_db if l < self.floor_db)
        return {"levels": out, "frames_gated": removed,
                "snr_gain_db": round(max(out) - self.floor_db, 1)}


class MultilingualVoice:
    """24. Multilingual Voice Support: language detect + TTS voice pick."""

    LANGS = ((re.compile(r"[\u0980-\u09FF]"), "bn"),
             (re.compile(r"[\u0900-\u097F]"), "hi"),
             (re.compile(r"[\u0600-\u06FF]"), "ar"),
             (re.compile(r"[\u4E00-\u9FFF]"), "zh"))

    def detect_language(self, text):
        for pat, code in self.LANGS:
            if pat.search(text):
                return code
        return "en"

    def voice_for(self, lang):
        return {"bn": "bn-IN-Standard-A", "hi": "hi-IN-Standard-B",
                "en": "en-US-Standard-C"}.get(lang, "en-US-Standard-C")


class ConversationalVoiceMemory:
    """27. Conversational Voice Memory: voice turns with speaker tags."""

    def __init__(self):
        self.turns = []

    def add(self, speaker, text):
        self.turns.append({"speaker": speaker, "text": text[:300]})
        return len(self.turns)

    def context(self, n=5):
        return " ".join(t["text"] for t in self.turns[-n:])

    def by_speaker(self, speaker):
        return [t for t in self.turns if t["speaker"] == speaker]


def _cosine(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
