# -*- coding: utf-8 -*-
"""Data Persistence & State Management (Components Deep Dive).

agi_state.json — full conversation history, active goals and runtime state.
agent.save() auto-persists memory + state. Versioned snapshots for rollback.
Optional encrypted local backups for long-term archival. Local-only, offline.
"""
import datetime
import json
import os
import shutil


class AgentState:
    """Single source of truth for the agent's persisted runtime state."""

    def __init__(self, path="data/octo/agi_state.json",
                 snapshot_dir="data/octo/snapshots"):
        self.path = path
        self.snapshot_dir = snapshot_dir
        self.conversation = []
        self.goals = {}
        self.runtime = {}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        os.makedirs(snapshot_dir, exist_ok=True)
        self.load()

    def add_turn(self, role, text):
        self.conversation.append({
            "role": role, "text": text[:2000],
            "ts": datetime.datetime.now().isoformat(timespec="seconds")})
        return len(self.conversation)

    def set_goal(self, goal_id, description, status="open"):
        self.goals[goal_id] = {"description": description[:300],
                               "status": status,
                               "updated": _now()}
        return self.goals[goal_id]

    def set_runtime(self, key, value):
        self.runtime[key] = value

    def as_dict(self):
        return {"version": 1, "saved": _now(),
                "conversation": self.conversation,
                "goals": self.goals, "runtime": self.runtime}

    def save(self):
        """agent.save() — persist full state to agi_state.json."""
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.as_dict(), f, ensure_ascii=False, indent=1)
        return self.path

    def load(self):
        if not os.path.exists(self.path):
            return False
        with open(self.path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.conversation = state.get("conversation", [])
        self.goals = state.get("goals", {})
        self.runtime = state.get("runtime", {})
        return True

    def snapshot(self, label=None):
        """Versioned state snapshot for rollback capability."""
        self.save()
        stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        name = f"{stamp}_{label or 'manual'}.json"
        dest = os.path.join(self.snapshot_dir, name)
        shutil.copy2(self.path, dest)
        return dest

    def snapshots(self):
        return sorted(os.listdir(self.snapshot_dir))

    def rollback(self, name=None):
        """Restore the latest (or named) snapshot; returns restored version."""
        if not self.snapshots():
            return {"ok": False, "error": "no snapshots available"}
        src = os.path.join(self.snapshot_dir,
                           name or self.snapshots()[-1])
        shutil.copy2(src, self.path)
        self.load()
        return {"ok": True, "restored": os.path.basename(src)}

    def encrypted_backup(self, manager, out="data/octo/agi_state.enc"):
        """Optional encrypted local backup via an EncryptionManager."""
        if manager is None or not manager.unlocked():
            return {"ok": False, "error": "encryption manager unavailable"}
        token = manager.encrypt(json.dumps(self.as_dict(), ensure_ascii=False))
        with open(out, "w", encoding="utf-8") as f:
            f.write(token)
        return {"ok": True, "path": out}

    def summary(self):
        return {"turns": len(self.conversation), "goals": len(self.goals),
                "runtime_keys": sorted(self.runtime),
                "snapshots": len(self.snapshots())}


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


class EncryptionManager:
    """Local-only encryption for backups (Fernet, key stays on device).

    Gracefully reports locked/unavailable state instead of crashing when the
    optional `cryptography` dependency is missing.
    """

    def __init__(self, key_path="data/octo/state.key"):
        self.key_path = key_path
        self._fernet = None
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            return
        os.makedirs(os.path.dirname(key_path) or ".", exist_ok=True)
        if os.path.exists(key_path):
            with open(key_path, "rb") as f:
                self._fernet = Fernet(f.read())
        else:
            key = Fernet.generate_key()
            with open(key_path, "wb") as f:
                f.write(key)
            self._fernet = Fernet(key)

    def unlocked(self):
        return self._fernet is not None

    def encrypt(self, text):
        if self._fernet is None:
            raise RuntimeError("encryption unavailable")
        return self._fernet.encrypt(text.encode("utf-8")).decode("ascii")

    def decrypt(self, token):
        if self._fernet is None:
            raise RuntimeError("encryption unavailable")
        return self._fernet.decrypt(token.encode("ascii")).decode("utf-8")
