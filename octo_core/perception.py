# -*- coding: utf-8 -*-
"""Advanced Multimodal Intelligence (features 25-30).

SceneUnderstandingEngine, ObjectTracker, SpatialReasoningEngine,
GestureRecognizer, AudioSceneAnalyzer, MultimodalContextFusion.
"""
import re
from collections import Counter


class SceneUnderstandingEngine:
    """25. Real-Time Scene Understanding: summarize a frame's objects."""

    def understand(self, frame):
        objs = frame.get("objects", [])
        counts = Counter(o["label"] for o in objs)
        dominant = counts.most_common(1)[0][0] if counts else None
        parts = ", ".join(f"{c} {label}" for label, c in counts.most_common())
        return {"object_count": len(objs), "labels": dict(counts),
                "dominant": dominant,
                "description": f"scene with {parts}" if parts else "empty scene"}


class ObjectTracker:
    """26. Visual Object Tracking: nearest-neighbor ID across frames."""

    def __init__(self, max_distance=40, max_missed=2):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks = {}

    def update(self, detections):
        assigned = []
        for det in detections:
            best_id, best_d = None, self.max_distance
            for tid, tr in self.tracks.items():
                if tr["label"] != det["label"]:
                    continue
                d = ((tr["x"] - det["x"]) ** 2 + (tr["y"] - det["y"]) ** 2) ** 0.5
                if d < best_d:
                    best_id, best_d = tid, d
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            self.tracks[best_id] = {"label": det["label"], "x": det["x"],
                                    "y": det["y"], "missed": 0}
            assigned.append({**det, "id": best_id})
        seen = {a["id"] for a in assigned}
        for tid in list(self.tracks):
            if tid not in seen:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]
        return assigned

    def active(self):
        return {tid: dict(tr) for tid, tr in self.tracks.items()}


class SpatialReasoningEngine:
    """27. Spatial Reasoning Engine: relative positions and distances."""

    def relations(self, a, b):
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        rels = []
        if abs(dx) > abs(dy) * 0.5:
            rels.append("right of" if dx > 0 else "left of")
        if abs(dy) > abs(dx) * 0.5:
            rels.append("above" if dy > 0 else "below")
        if not rels:
            rels.append("near")
        return {"relations": rels,
                "distance": round((dx * dx + dy * dy) ** 0.5, 2)}

    def path_length(self, points):
        total = 0.0
        for p, q in zip(points, points[1:]):
            total += ((q["x"] - p["x"]) ** 2 + (q["y"] - p["y"]) ** 2) ** 0.5
        return round(total, 2)


class GestureRecognizer:
    """28. Gesture Recognition System: hand motion from landmark sequence."""

    def classify(self, frames):
        if len(frames) < 2:
            return {"gesture": "static", "confidence": 0.5}
        dx = frames[-1]["x"] - frames[0]["x"]
        dy = frames[-1]["y"] - frames[0]["y"]
        if max(abs(dx), abs(dy)) < 10:
            return {"gesture": "static", "confidence": 0.9}
        if abs(dx) > abs(dy):
            return {"gesture": "swipe_right" if dx > 0 else "swipe_left",
                    "confidence": 0.8}
        return {"gesture": "swipe_up" if dy > 0 else "swipe_down",
                "confidence": 0.8}


class AudioSceneAnalyzer:
    """29. Audio Scene Understanding: environment from acoustic features."""

    def classify(self, features):
        if features.get("speech_ratio", 0) > 0.5:
            env = "conversation"
        elif features.get("periodic") and features.get("volume", 0) > 0.6:
            env = "machinery"
        elif features.get("variety", 0) > 0.6:
            env = "busy-street"
        elif features.get("volume", 1) < 0.2:
            env = "quiet-room"
        else:
            env = "ambient"
        return {"environment": env, "volume": features.get("volume")}


class MultimodalContextFusion:
    """30. Multimodal Context Fusion: merge text/image/audio/video context."""

    def fuse(self, text="", image_labels=(), audio=None, video=None):
        text_entities = set(re.findall(r"\w{3,}", text.lower())) if text else set()
        visual = {str(l).lower() for l in image_labels}
        present = [m for m, v in (("text", text), ("image", image_labels),
                                  ("audio", audio), ("video", video)) if v]
        return {"modalities": present,
                "entities": sorted(text_entities | visual),
                "cross_modal_overlap": sorted(text_entities & visual),
                "confidence": round(min(1.0, 0.4 * bool(text)
                                        + 0.3 * bool(image_labels)
                                        + 0.2 * bool(audio)
                                        + 0.2 * bool(video)), 2)}
