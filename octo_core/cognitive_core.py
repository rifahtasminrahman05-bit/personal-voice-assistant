# -*- coding: utf-8 -*-
"""Cognitive Core — মস্তিষ্ক ও সিদ্ধান্ত গ্রহণ (brain & decision-making).

Unifies the four embodied components behind one perceive → decide → act
loop:

    1. VoiceInterface  — কথা বলা ও শোনা (speech engine, graceful fallback)
    2. VisionSystem    — ক্যামেরা ও স্ক্রিন বোঝা (image/OCR/screen analysis)
    3. AutomationEngine— কম্পিউটার, স্মার্ট হোম ও ডিভাইস নিয়ন্ত্রণ (guarded)
    4. SecurityCore    — অনুমতি, নিরাপত্তা ও কার্যক্রম পর্যবেক্ষণ

Everything is local-only; every device action is permission-checked,
approval-gated (fail-closed) and audit-logged.
"""
import re

from .security import ApprovalGateway, AuditLogger, PermissionManager


class VoiceInterface:
    """1. Voice Interface: listen and speak via the shared SpeechEngine.

    Works without any audio hardware: reports availability and degrades to
    text-only mode instead of crashing.
    """

    def __init__(self, engine=None):
        self._engine = engine
        self._checked = engine is not None

    def _get(self):
        if not self._checked:
            self._checked = True
            try:
                from speech.engine import SpeechEngine
                self._engine = SpeechEngine()
            except Exception:
                self._engine = None
        return self._engine

    def status(self):
        eng = self._get()
        if eng is None:
            return {"stt": False, "tts": False, "wakeword": False,
                    "mode": "text-only"}
        return {"stt": eng.is_stt_ready(), "tts": eng.is_tts_ready(),
                "wakeword": eng.is_wakeword_ready(),
                "mode": "voice" if eng.is_stt_ready() else "text-only"}

    def listen(self, timeout=None):
        eng = self._get()
        if eng is None or not eng.is_stt_ready():
            return {"ok": False, "text": "",
                    "reason": "speech recognition unavailable (text mode)"}
        return {"ok": True, "text": eng.listen(timeout=timeout) or ""}

    def speak(self, text):
        eng = self._get()
        if eng is None or not eng.is_tts_ready():
            return {"ok": False, "reason": "speech synthesis unavailable"}
        eng.speak(text)
        return {"ok": True}


class VisionSystem:
    """2. Vision System: understand camera frames and screen captures."""

    def __init__(self):
        self.seen = []

    def camera_available(self):
        try:
            import cv2
            return True
        except ImportError:
            return False

    def analyze_image(self, path):
        """Image/document understanding (metadata + OCR when available)."""
        from .multimodal import describe_image, ocr_image
        desc = describe_image(path)
        text = ocr_image(path)
        entry = {"path": path, "description": desc, "text": str(text)[:300]}
        self.seen.append(entry)
        return entry

    def understand_screen(self, screenshot_path):
        """Screen understanding: same pipeline, labeled as screen."""
        entry = self.analyze_image(screenshot_path)
        entry["kind"] = "screen"
        return entry

    def capture_camera(self):
        """Capture one frame from the camera (needs opencv, local only)."""
        if not self.camera_available():
            return {"ok": False, "reason": "opencv not installed"}
        import cv2
        cam = cv2.VideoCapture(0)
        ok, frame = cam.read()
        cam.release()
        if not ok:
            return {"ok": False, "reason": "camera read failed"}
        path = "data/octo/camera_frame.png"
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, frame)
        return {"ok": True, "path": path}


class AutomationEngine:
    """3. Automation Engine: guarded control of computer, smart home and
    devices. Every control() is permission-checked, approval-gated for
    risky/lock actions, and audit-logged."""

    ALWAYS_APPROVE = {"lock", "unlock", "power", "delete"}

    def __init__(self, permissions=None, audit=None, approval=None):
        self.permissions = permissions or PermissionManager()
        self.audit = audit or AuditLogger()
        self.approval = approval
        self.devices = {}
        self.history = []

    def register_device(self, name, kind, action_fn, risky=False):
        self.devices[name] = {"kind": kind, "fn": action_fn, "risky": risky}
        granted = self.permissions._grants.get("octo-agent", set())
        self.permissions.grant("octo-agent", granted | {name})
        return name

    def devices_by_kind(self, kind):
        return sorted(n for n, d in self.devices.items() if d["kind"] == kind)

    def control(self, device, command="", approve=None):
        dev = self.devices.get(device)
        if dev is None:
            return {"ok": False, "error": f"unknown device: {device}"}
        grants = self.permissions._grants.get("octo-agent", set())
        if device not in grants:
            self.audit.log("automation_denied", device, "no permission")
            return {"ok": False, "error": "permission denied (least privilege)"}
        needs_approval = dev["risky"] or any(
            k in command.lower() for k in self.ALWAYS_APPROVE)
        if needs_approval:
            gate = ApprovalGateway(approve=approve or self.approval)
            if not gate.request("exec", f"{device}: {command}"):
                self.audit.log("automation_unapproved", device, command)
                return {"ok": False, "error": "requires explicit user approval"}
        try:
            output = str(dev["fn"](command))
        except Exception as exc:
            self.history.append({"device": device, "command": command,
                                 "ok": False, "error": str(exc)[:120]})
            return {"ok": False, "error": str(exc)[:200]}
        self.history.append({"device": device, "command": command, "ok": True,
                             "output": output[:120]})
        self.audit.log("automation_ok", device, f"{command} -> {output[:80]}")
        return {"ok": True, "output": output[:200]}

    def log(self, n=20):
        return self.history[-n:]


class SecurityCore:
    """4. Security Core: permissions, safety and activity monitoring."""

    ANOMALY_EVENTS = {"action_blocked", "action_denied", "automation_denied",
                      "action_unapproved", "automation_unapproved",
                      "action_error"}

    def __init__(self, permissions=None, audit=None, approval=None):
        self.permissions = permissions or PermissionManager()
        self.audit = audit or AuditLogger()
        self.approval = approval

    def grant(self, agent, tools):
        self.permissions.grant(agent, tools)
        self.audit.log("permission_grant", agent, ",".join(sorted(tools)))

    def revoke(self, agent):
        self.permissions.revoke(agent)
        self.audit.log("permission_revoke", agent, "")

    def approve_gateway(self):
        return ApprovalGateway(approve=self.approval)

    def monitor(self, n=100):
        """Activity monitoring: recent events + anomaly summary."""
        events = self.audit.tail(n)
        anomalies = [e for e in events if e.get("event") in self.ANOMALY_EVENTS]
        by_agent = {}
        for e in events:
            by_agent[e.get("agent", "?")] = by_agent.get(e.get("agent", "?"), 0) + 1
        return {"events": len(events),
                "anomalies": len(anomalies),
                "anomaly_detail": [{"ts": a["ts"], "event": a["event"],
                                    "agent": a["agent"]} for a in anomalies[-5:]],
                "by_agent": by_agent,
                "verdict": "clean" if not anomalies else "review"}

    def require_approval(self, action, description=""):
        return self.approve_gateway().request(action, description)


class CognitiveCore:
    """মস্তিষ্ক: perceive → decide → act loop over the four components."""

    DEVICE_PATTERNS = re.compile(
        r"\b(light|lights|lamp|thermostat|ac|fan|lock|door|speaker|tv|"
        r"computer|volume|brightness)\b", re.I)
    VISION_PATTERNS = re.compile(r"\b(screen|screenshot|camera|image|photo|"
                                 r"picture|scan this|ছবি|স্ক্রিন)\b", re.I)
    VOICE_PATTERNS = re.compile(r"\b(listen|speak|say|voice|শোনো|বলো)\b", re.I)

    def __init__(self, voice=None, vision=None, automation=None,
                 security=None, audit=None):
        self.audit = audit or AuditLogger()
        self.security = security or SecurityCore(audit=self.audit)
        self.voice = voice or VoiceInterface()
        self.vision = vision or VisionSystem()
        self.automation = automation or AutomationEngine(
            permissions=self.security.permissions, audit=self.audit)

    def status(self):
        return {"voice": self.voice.status(),
                "vision": {"camera": self.vision.camera_available(),
                           "images_seen": len(self.vision.seen)},
                "automation": {"devices": sorted(self.automation.devices),
                               "actions": len(self.automation.history)},
                "security": self.security.monitor(50)["verdict"]}

    def decide(self, user_input):
        """Route an input to the right component (brain & decision-making)."""
        if self.DEVICE_PATTERNS.search(user_input):
            device = next((d for d in self.automation.devices
                           if d.lower() in user_input.lower()), None)
            return {"route": "automation", "device": device,
                    "input": user_input[:120]}
        if self.VISION_PATTERNS.search(user_input):
            return {"route": "vision", "input": user_input[:120]}
        if self.VOICE_PATTERNS.search(user_input):
            return {"route": "voice", "input": user_input[:120]}
        return {"route": "reason", "input": user_input[:120]}

    def act(self, decision, approve=None, path=None):
        """Execute a decision from decide()."""
        route = decision["route"]
        if route == "automation":
            if not decision["device"]:
                return {"ok": False,
                        "error": "no matching registered device"}
            return self.automation.control(decision["device"],
                                           decision["input"], approve=approve)
        if route == "vision":
            if not path:
                return {"ok": False,
                        "error": "vision needs an image/screenshot path"}
            entry = self.vision.analyze_image(path)
            return {"ok": True, "vision": entry}
        if route == "voice":
            st = self.voice.status()
            return {"ok": True, "voice": st}
        return {"ok": True, "note": "routed to reasoning/chat layer"}

    def cycle(self, user_input, approve=None, path=None):
        """Full perceive → decide → act round with audit."""
        decision = self.decide(user_input)
        result = self.act(decision, approve=approve, path=path)
        self.audit.log("core_cycle", "octo",
                       f"{decision['route']} -> ok={result.get('ok')}")
        return {"decision": decision, "result": result}
