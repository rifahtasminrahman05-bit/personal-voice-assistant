# -*- coding: utf-8 -*-
"""Security & Surveillance (features 96-107).

Intrusion detection, unauthorized-access detection, face-based access
control, suspicious-activity detection, network security monitoring,
signature-based local malware scanning, emergency alerts and camera
frame analysis. All local and defensive; nothing is ever sent anywhere.
"""
import hashlib
import os
import re
import time


class IntrusionDetector:
    """96. Intrusion Detection: failed attempts + unusual hours."""

    def __init__(self, max_failures=5, window=600.0):
        self.max_failures = max_failures
        self.window = window
        self.failures = {}

    def record_failure(self, source, now=None):
        now = now if now is not None else time.time()
        recent = [t for t in self.failures.get(source, [])
                  if now - t <= self.window]
        recent.append(now)
        self.failures[source] = recent
        return {"source": source, "failures": len(recent),
                "intrusion": len(recent) >= self.max_failures}

    def check_time(self, hour, allowed=(6, 23)):
        return {"unusual_hour": not (allowed[0] <= hour < allowed[1])}


class UnauthorizedAccessDetector:
    """97. Unauthorized Access Detection: ACL checks on resources."""

    def __init__(self):
        self.acl = {}

    def grant(self, user, resource):
        self.acl.setdefault(resource, set()).add(user)

    def check(self, user, resource):
        allowed = self.acl.get(resource, set())
        ok = user in allowed
        return {"allowed": ok, "resource": resource, "user": user,
                "violation": not ok}


class FaceAccessController:
    """98. Face-Based Access Control via visionx.PersonRecognizer."""

    def __init__(self, recognizer=None, threshold=0.75):
        from .visionx import PersonRecognizer
        self.recognizer = recognizer or PersonRecognizer()
        self.threshold = threshold
        self.authorized = set()

    def enroll(self, name, face_print, authorized=True):
        self.recognizer.enroll(name, face_print)
        if authorized:
            self.authorized.add(name)

    def request_entry(self, face_print):
        match = self.recognizer.recognize(face_print,
                                          threshold=self.threshold)
        name = match.get("person")
        granted = match.get("known") and name in self.authorized
        return {"granted": bool(granted), "match": match}


class SuspiciousActivityDetector:
    """99. Suspicious Activity Detection: rate + pattern rules."""

    RULES = (
        (re.compile(r"delete|rm -rf|format|wipe", re.I), "destructive"),
        (re.compile(r"password|token|secret|key", re.I), "secret-access"),
        (re.compile(r"download|upload|transfer", re.I), "data-transfer"),
    )

    def __init__(self, max_events_per_minute=30):
        self.limit = max_events_per_minute
        self.events = []

    def inspect(self, text, now=None):
        now = now if now is not None else time.time()
        self.events = [t for t in self.events if now - t <= 60]
        self.events.append(now)
        tags = [label for pattern, label in self.RULES
                if pattern.search(text or "")]
        return {"tags": tags, "rate_burst": len(self.events) > self.limit,
                "suspicious": bool(tags) or len(self.events) > self.limit}


class NetworkSecurityMonitor:
    """100. Network Security Monitoring: local connection-list rules."""

    RISKY_PORTS = {23, 445, 3389, 4444, 5555}

    def audit_connections(self, connections):
        """connections: [{remote, port}] -> flagged entries."""
        flagged = [{"remote": c.get("remote"), "port": c.get("port"),
                    "reason": "risky port"}
                   for c in connections
                   if int(c.get("port", 0)) in self.RISKY_PORTS]
        return {"connections": len(connections), "flagged": flagged,
                "clean": not flagged}


class MalwareScanner:
    """101. Malware Detection: local signature scan (no quarantine)."""

    SIGNATURES = (
        b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR",  # EICAR test string
        b"eval(base64_decode(",
        b"/tmp/.hidden_payload",
    )

    def hash_file(self, path):
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def scan_file(self, path):
        try:
            with open(path, "rb") as fh:
                blob = fh.read(1 << 20)
        except OSError as exc:
            return {"ok": False, "error": str(exc)[:120]}
        hits = [sig[:16].decode("utf-8", "replace")
                for sig in self.SIGNATURES if sig in blob]
        return {"path": path, "clean": not hits, "signatures": hits}

    def scan_dir(self, root, limit=200):
        results, scanned = [], 0
        for base, _dirs, files in os.walk(root):
            for name in files:
                if scanned >= limit:
                    break
                path = os.path.join(base, name)
                result = self.scan_file(path)
                if not result.get("clean", True):
                    results.append(result)
                scanned += 1
        return {"scanned": scanned, "threats": results}


class EmergencyAlerts:
    """105. Emergency Alerts: local alert queue with escalation."""

    LEVELS = ("info", "warning", "critical")

    def __init__(self):
        self.alerts = []

    def raise_alert(self, message, level="warning"):
        if level not in self.LEVELS:
            level = "warning"
        alert = {"message": message[:200], "level": level,
                 "at": time.time()}
        self.alerts.append(alert)
        return alert

    def critical(self):
        return [a for a in self.alerts if a["level"] == "critical"]


class CameraAnalyzer:
    """107. Security Camera Analysis: motion over frame hashes."""

    def __init__(self, motion_threshold=0.90):
        self.threshold = motion_threshold
        self.last_signature = None

    def analyze_frame(self, frame_vector):
        if self.last_signature is None:
            self.last_signature = list(frame_vector)
            return {"motion": False, "first_frame": True}
        sim = _cosine(self.last_signature, frame_vector)
        self.last_signature = list(frame_vector)
        return {"motion": sim < self.threshold, "similarity": round(sim, 3)}


def _cosine(a, b):
    size = min(len(a), len(b))
    if size == 0:
        return 0.0
    va, vb = a[:size], b[:size]
    dot = sum(x * y for x, y in zip(va, vb))
    na = sum(x * x for x in va) ** 0.5 or 1.0
    nb = sum(y * y for y in vb) ** 0.5 or 1.0
    return dot / (na * nb)
