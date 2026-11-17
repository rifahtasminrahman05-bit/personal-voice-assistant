# -*- coding: utf-8 -*-
"""Settings for the Octo desktop app — persisted to data/octo/settings.json."""
import json
import os
import threading

SETTINGS_FILE = os.path.join("data", "octo", "settings.json")

DEFAULTS = {
    "language": "bn",
    "theme": "system",
    "temperature": 0.8,
    "top_k": 40,
    "steps": 100,
    "device": "auto",
    "quantized": False,
    "max_steps": 8,
    "voice_enabled": False,
}


class Settings:
    def __init__(self, path=SETTINGS_FILE):
        self.path = path
        self._lock = threading.Lock()
        self._data = dict(DEFAULTS)
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data.update(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass
        return self._data

    def save(self):
        with self._lock:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=1)
        return self._data

    def get(self, key):
        return self._data.get(key, DEFAULTS.get(key))

    def set(self, key, value):
        with self._lock:
            self._data[key] = value
        self.save()
        return value

    def reset(self):
        with self._lock:
            self._data = dict(DEFAULTS)
        self.save()
        return self._data

    def as_dict(self):
        return dict(self._data)
