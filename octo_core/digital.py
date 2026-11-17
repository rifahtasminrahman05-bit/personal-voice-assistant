# -*- coding: utf-8 -*-
"""Digital World Intelligence — digital twins and embodied interfaces.

DigitalTwin: a virtual replica of a system that syncs observations, detects
drift against reality and projects forward with a dynamics function.
EmbodiedInterface: a device protocol (connect/act/observe) with a built-in
grid-world robot simulator so the loop is testable without hardware.
"""
import datetime


class DigitalTwin:
    """Digital Twin Engine: virtual replica of a real/digital system."""

    def __init__(self, name):
        self.name = name
        self.state = {}
        self.history = []
        self.last_sync = None

    def sync(self, observations):
        """Merge real-world observations into the twin (timestamped)."""
        self.state.update(observations)
        self.last_sync = datetime.datetime.now().isoformat(timespec="seconds")
        self.history.append({"ts": self.last_sync, "state": dict(self.state)})
        return dict(self.state)

    def drift(self, reality):
        """Compare twin state against a fresh reality snapshot."""
        diffs = {}
        for key in set(self.state) | set(reality):
            twin_v, real_v = self.state.get(key), reality.get(key)
            if twin_v != real_v:
                diffs[key] = {"twin": twin_v, "real": real_v}
        return {"keys": len(diffs), "in_sync": not diffs, "diffs": diffs}

    def project(self, steps, dynamics):
        """Simulate the twin forward: dynamics(state_dict) -> new state_dict."""
        projected = []
        state = dict(self.state)
        for _ in range(steps):
            state = dict(dynamics(state))
            projected.append(state)
        return projected


class LoopbackRobot:
    """A simple simulated robot on a grid — for tests and demos."""

    def __init__(self, x=0, y=0, size=5):
        self.x, self.y, self.size = x, y, size
        self.battery = 100.0

    def act(self, action):
        moves = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
        if action in moves:
            dx, dy = moves[action]
            self.x = max(0, min(self.size - 1, self.x + dx))
            self.y = max(0, min(self.size - 1, self.y + dy))
            self.battery = max(0.0, self.battery - 1.0)
        elif action == "charge":
            self.battery = 100.0
        return {"x": self.x, "y": self.y, "battery": self.battery}

    def observe(self):
        return {"x": self.x, "y": self.y, "battery": self.battery}


class EmbodiedInterface:
    """Embodied Intelligence Interface: talk to robots/simulators/environments."""

    def __init__(self):
        self.devices = {}

    def register(self, name, device):
        """device must implement act(action) and observe()."""
        self.devices[name] = device
        return name

    def connect(self, name):
        if name not in self.devices:
            raise KeyError(f"unknown device: {name}")
        return self.devices[name].observe()

    def act_and_observe(self, name, action):
        if name not in self.devices:
            raise KeyError(f"unknown device: {name}")
        self.devices[name].act(action)
        return self.devices[name].observe()

    def sense_all(self):
        return {name: device.observe() for name, device in self.devices.items()}
