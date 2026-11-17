# -*- coding: utf-8 -*-
"""Vision & Perception extensions (features 57-58, 60-68).

Face detection, face/person recognition, facial expression analysis,
document recognition, visual anomaly detection and video analysis —
all local heuristics over frames (feature vectors / bounding boxes),
no cloud, no new hard dependencies.
"""
import re


class FaceDetector:
    """57. Face Detection: bounding boxes over frame features."""

    SKIN_RANGE = ((0.25, 0.65), (0.15, 0.55))

    def detect(self, regions):
        """regions: [{x, y, w, h, skin_ratio}] -> faces above skin ratio."""
        faces = [r for r in regions
                 if r.get("skin_ratio", 0) >= 0.4
                 and r["w"] * r["h"] >= 100]
        return {"faces": faces, "count": len(faces)}


class PersonRecognizer:
    """60. Person Recognition: face-print enrollment + identification."""

    def __init__(self):
        self.known = {}

    def enroll(self, name, face_print):
        self.known[name] = list(face_print)
        return name

    def recognize(self, face_print, threshold=0.75):
        best, score = None, 0.0
        for name, ref in self.known.items():
            sim = _cosine(ref, face_print)
            if sim > score:
                best, score = name, sim
        if best is None or score < threshold:
            return {"person": None, "known": False}
        return {"person": best, "known": True, "confidence": round(score, 3)}


class ExpressionAnalyzer:
    """62. Facial Expression Analysis: landmark-ratio heuristics."""

    def analyze(self, mouth_open, brow_raise, smile_ratio):
        if smile_ratio > 0.6:
            label = "happy"
        elif brow_raise > 0.7 and mouth_open:
            label = "surprised"
        elif smile_ratio < 0.2 and brow_raise < 0.3:
            label = "sad" if mouth_open else "neutral"
        else:
            label = "neutral"
        return {"expression": label, "smile_ratio": smile_ratio,
                "brow_raise": brow_raise}


class DocumentRecognizer:
    """65. Document Recognition: classify a document from OCR text."""

    TYPES = ((re.compile(r"invoice|amount due|total payable", re.I), "invoice"),
             (re.compile(r"receipt|paid|transaction", re.I), "receipt"),
             (re.compile(r"passport|national id|date of birth", re.I), "id"),
             (re.compile(r"curriculum vitae|resume|experience", re.I), "resume"),
             (re.compile(r"dear|sincerely|regards", re.I), "letter"))

    def classify(self, ocr_text):
        for pat, kind in self.TYPES:
            if pat.search(ocr_text):
                fields = self._fields(ocr_text)
                return {"type": kind, "fields": fields}
        return {"type": "generic", "fields": self._fields(ocr_text)}

    def _fields(self, text):
        import datetime
        dates = re.findall(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}", text)
        amounts = re.findall(r"(?:\$|৳|USD\s?)\s?\d[\d,]*", text)
        return {"dates": dates[:3], "amounts": amounts[:3],
                "words": len(text.split())}


class VisualAnomalyDetector:
    """67. Visual Anomaly Detection: deviation from the baseline frame."""

    def __init__(self, sensitivity=0.25):
        self.sensitivity = sensitivity
        self.baseline = None

    def learn_baseline(self, frame_features):
        self.baseline = list(frame_features)

    def detect(self, frame_features):
        if self.baseline is None:
            return {"anomaly": False, "reason": "no baseline"}
        deltas = [abs(a - b) for a, b in zip(self.baseline, frame_features)]
        max_delta = max(deltas, default=0.0)
        return {"anomaly": max_delta > self.sensitivity,
                "max_delta": round(max_delta, 3),
                "sensitivity": self.sensitivity}


class VideoAnalyzer:
    """68. Video Analysis: frame-sequence events and motion timeline."""

    def analyze(self, frames):
        """frames: [{ts, motion, objects: [names]}]."""
        events, prev_objects = [], set()
        for i, frame in enumerate(frames):
            new = set(frame.get("objects", [])) - prev_objects
            if new:
                events.append({"ts": frame.get("ts", i), "event": "appear",
                               "objects": sorted(new)})
            gone = prev_objects - set(frame.get("objects", []))
            if gone:
                events.append({"ts": frame.get("ts", i), "event": "leave",
                               "objects": sorted(gone)})
            prev_objects = set(frame.get("objects", []))
        motion_total = sum(f.get("motion", 0.0) for f in frames)
        return {"frames": len(frames), "events": events,
                "avg_motion": round(motion_total / max(len(frames), 1), 3)}


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0
