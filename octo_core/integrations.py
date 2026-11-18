# -*- coding: utf-8 -*-
"""Advanced Communication & Integration (features 176-185).

Smartphone/smartwatch/IoT/API integration, a local vault instead of
cloud services (offline constraint), cross-device sync, guarded remote
access, encrypted local messaging, notifications and multi-platform
support. Everything runs through a local message bus; nothing ever
leaves the machine. Cloud sync is intentionally replaced by an
encrypted local vault.
"""
import base64
import hashlib
import json
import os
import time

try:
    from cryptography.fernet import Fernet
    _HAVE_FERNET = True
except ImportError:
    Fernet = None
    _HAVE_FERNET = False


class _XorCipher:
    """Stdlib fallback cipher (SHA-256 keystream) used when the optional
    `cryptography` package is not installed. Weaker than Fernet but keeps
    the vault and messenger working with zero hard dependencies."""

    def __init__(self, key=b"octo-local-key"):
        self.key = key

    @staticmethod
    def generate_key():
        return os.urandom(32)

    def _keystream(self, nonce, length):
        out = b""
        counter = 0
        while len(out) < length:
            out += hashlib.sha256(
                self.key + nonce + counter.to_bytes(4, "big")).digest()
            counter += 1
        return out[:length]

    def encrypt(self, data):
        nonce = os.urandom(8)
        stream = self._keystream(nonce, len(data))
        return base64.urlsafe_b64encode(
            nonce + bytes(a ^ b for a, b in zip(data, stream)))

    def decrypt(self, token):
        raw = base64.urlsafe_b64decode(token)
        nonce, body = raw[:8], raw[8:]
        stream = self._keystream(nonce, len(body))
        return bytes(a ^ b for a, b in zip(body, stream))


class LocalMessageBus:
    """Shared local bus used by phone/watch/IoT/API integrations."""

    def __init__(self):
        self.subscribers = {}
        self.history = []

    def subscribe(self, device, handler):
        self.subscribers.setdefault(device, []).append(handler)

    def publish(self, topic, payload, sender="local"):
        message = {"topic": topic, "payload": payload, "sender": sender,
                   "t": time.time()}
        self.history.append(message)
        del self.history[:-200]
        delivered = []
        for device, handlers in self.subscribers.items():
            for handler in handlers:
                try:
                    handler(message)
                    delivered.append(device)
                except Exception:
                    continue
        return {"delivered_to": sorted(set(delivered)), **message}


class SmartphoneIntegration:
    """176. Smartphone Integration: contacts/notifications relay."""

    def __init__(self, bus=None):
        self.bus = bus or LocalMessageBus()
        self.paired = False

    def pair(self, device_name):
        self.paired = True
        self.bus.publish("phone.paired", {"device": device_name})
        return {"paired": True, "device": device_name}

    def relay_notification(self, app, text):
        if not self.paired:
            return {"ok": False, "reason": "phone not paired"}
        return self.bus.publish("phone.notification",
                                {"app": app, "text": text[:160]},
                                sender="phone")


class SmartwatchIntegration:
    """177. Smartwatch Integration: health ticks + haptics out."""

    def __init__(self, bus=None):
        self.bus = bus or LocalMessageBus()
        self.health = []

    def ingest(self, heart_rate, steps):
        sample = {"heart_rate": heart_rate, "steps": steps,
                  "t": time.time()}
        self.health.append(sample)
        del self.health[:-500]
        self.bus.publish("watch.health", sample, sender="watch")
        return sample

    def haptic(self, pattern="short"):
        return self.bus.publish("watch.haptic", {"pattern": pattern})


class IoTIntegration:
    """178. IoT Integration: local device registry over the bus."""

    def __init__(self, bus=None):
        self.bus = bus or LocalMessageBus()
        self.devices = {}

    def register(self, device_id, kind, room=None):
        self.devices[device_id] = {"kind": kind, "room": room,
                                   "online": True}
        self.bus.publish("iot.registered", {"device": device_id,
                                            "kind": kind})
        return device_id

    def send(self, device_id, command):
        if device_id not in self.devices:
            return {"ok": False, "reason": "unknown device"}
        if not self.devices[device_id]["online"]:
            return {"ok": False, "reason": "device offline"}
        return self.bus.publish("iot.command",
                                {"device": device_id,
                                 "command": command}, sender="octo")


class APIIntegration:
    """179. API Integration: registry of LOCAL endpoints only."""

    def __init__(self):
        self.endpoints = {}

    def register(self, name, handler, local=True):
        if not local:
            raise ValueError("only local endpoints are allowed")
        self.endpoints[name] = handler
        return name

    def call(self, name, *args, **kwargs):
        handler = self.endpoints.get(name)
        if handler is None:
            return {"ok": False, "error": "unknown endpoint"}
        try:
            return {"ok": True, "result": handler(*args, **kwargs)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:120]}


class LocalVault:
    """180. Cloud Service Integration -> encrypted LOCAL vault.

    Octo is 100% offline: instead of any cloud provider, "sync" data
    goes into a Fernet-encrypted local vault file.
    """

    def __init__(self, path="data/octo/vault.bin",
                 key_path="data/octo/vault.key"):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(key_path):
            with open(key_path, "rb") as fh:
                key = fh.read().strip()
            self._fernet = Fernet(key) if _HAVE_FERNET else _XorCipher(key)
        elif _HAVE_FERNET:
            key = Fernet.generate_key()
            with open(key_path, "wb") as fh:
                fh.write(key)
            self._fernet = Fernet(key)
        else:
            with open(key_path, "wb") as fh:
                fh.write(b"octo-local-key")
            self._fernet = _XorCipher()
        self.items = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as fh:
                    blob = json.loads(self._fernet.decrypt(fh.read()))
                self.items = blob
            except Exception:
                self.items = {}

    def put(self, key, value):
        self.items[key] = value
        self._flush()
        return {"stored": key}

    def get(self, key):
        return self.items.get(key)

    def _flush(self):
        blob = self._fernet.encrypt(
            json.dumps(self.items).encode("utf-8"))
        with open(self.path, "wb") as fh:
            fh.write(blob)


class CrossDeviceSync:
    """181. Cross-Device Synchronization: local state merge."""

    def __init__(self, vault=None):
        self.vault = vault or LocalVault()

    def push(self, device, state):
        stored = self.vault.get("sync") or {}
        stored[device] = {"state": state, "t": time.time()}
        self.vault.put("sync", stored)
        return {"pushed": device, "devices": sorted(stored)}

    def pull(self, device):
        stored = self.vault.get("sync") or {}
        entry = stored.get(device)
        return entry["state"] if entry else None

    def conflicts(self):
        stored = self.vault.get("sync") or {}
        timestamps = [v["t"] for v in stored.values()]
        return {"devices": sorted(stored),
                "single_source": len(set(timestamps)) <= 1}


class RemoteAccessGuard:
    """182. Remote Access: default-deny, session-scoped, local only."""

    def __init__(self):
        self.allowed = False
        self.sessions = []

    def enable(self, token, expected):
        if token != expected:
            return {"ok": False, "error": "token mismatch"}
        self.allowed = True
        return {"ok": True, "scope": "localhost-only"}

    def open_session(self, user):
        if not self.allowed:
            return {"ok": False, "reason": "remote access disabled"}
        session = {"user": user, "t": time.time()}
        self.sessions.append(session)
        return {"ok": True, "session": session}

    def disable(self):
        self.allowed = False
        self.sessions = []
        return {"ok": True, "sessions_closed": True}


class SecureMessenger:
    """183. Secure Messaging: Fernet-encrypted local threads."""

    def __init__(self, fernet=None):
        if fernet is not None:
            self._fernet = fernet
        elif _HAVE_FERNET:
            self._fernet = Fernet(Fernet.generate_key())
        else:
            self._fernet = _XorCipher(os.urandom(32))
        self.threads = {}

    def send(self, thread, sender, text):
        token = self._fernet.encrypt(
            json.dumps({"sender": sender, "text": text}).encode("utf-8"))
        self.threads.setdefault(thread, []).append(token)
        return {"ok": True, "thread": thread, "encrypted": True}

    def read(self, thread):
        messages = []
        for token in self.threads.get(thread, []):
            try:
                messages.append(json.loads(
                    self._fernet.decrypt(token).decode("utf-8")))
            except Exception:
                continue
        return messages


class NotificationCenter:
    """184. Notification Management: dedupe + priority queue."""

    def __init__(self):
        self.queue = []
        self._seen = set()

    def notify(self, title, priority=1, dedupe=True):
        key = title.lower()
        if dedupe and key in self._seen:
            return {"ok": False, "reason": "duplicate suppressed"}
        self._seen.add(key)
        item = {"title": title, "priority": priority, "t": time.time()}
        self.queue.append(item)
        self.queue.sort(key=lambda n: n["priority"])
        return {"ok": True, **item}

    def pending(self):
        return list(self.queue)

    def dismiss(self, title):
        before = len(self.queue)
        self.queue = [n for n in self.queue
                      if n["title"].lower() != title.lower()]
        return {"dismissed": before - len(self.queue)}


class MultiPlatformSupport:
    """185. Multi-Platform Support: platform profile detection."""

    PROFILE = {
        "linux": {"ui": "tkinter", "audio": "alsa/pulse", "paths": "/"},
        "darwin": {"ui": "tkinter", "audio": "coreaudio",
                   "paths": "/Users"},
        "windows": {"ui": "tkinter", "audio": "wasapi",
                    "paths": "C:/Users"},
    }

    def profile(self, system=None):
        system = system or os.name
        key = ("windows" if system == "nt"
               else "darwin" if system == "darwin" else "linux")
        return {"system": key, **self.PROFILE[key]}
