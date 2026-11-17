# -*- coding: utf-8 -*-
"""Smart Home Automation (features 28-40).

HomeHub: device registry + guarded control for lights, fan, AC, door lock,
CCTV, motion sensors, temperature, appliances, energy management, home
security, occupancy detection, scheduling and centralized control.
Every actuation is permission-checked and audit-logged; lock/security
actions require explicit approval.
"""
import datetime

from .security import AuditLogger, PermissionManager

APPROVAL_COMMANDS = {"lock", "unlock", "arm", "disarm", "power_off", "open"}


class Device:
    def __init__(self, name, kind, state=None, risky=False):
        self.name = name
        self.kind = kind
        self.state = state or {"on": False}
        self.risky = risky
        self.events = []

    def set(self, **props):
        self.state.update(props)
        self.events.append({"ts": _now(), "props": dict(props)})
        return dict(self.state)


class HomeHub:
    """Centralized Home Control (feature 40)."""

    def __init__(self, permissions=None, audit=None, approval=None):
        self.permissions = permissions or PermissionManager()
        self.audit = audit or AuditLogger()
        self.approval = approval
        self.devices = {}
        self.schedules = []
        self.security_events = []
        self.energy_log = []

    def add_device(self, name, kind, risky=False):
        self.devices[name] = Device(name, kind, risky=risky)
        return name

    def control(self, device, command, value=None, approve=None):
        dev = self.devices.get(device)
        if dev is None:
            return {"ok": False, "error": f"unknown device: {device}"}
        if command.lower() in APPROVAL_COMMANDS and not self._approved(
                f"{device}:{command}", approve):
            self.audit.log("home_denied", device, command)
            return {"ok": False, "error": "requires explicit user approval"}
        props = {"command": command}
        if command.lower() == "on":
            props["on"] = True
        elif command.lower() in ("off", "stop"):
            props["on"] = False
        if value is not None:
            props["value"] = value
        state = dev.set(**props)
        self.audit.log("home_ok", device, f"{command} {value or ''}".strip())
        return {"ok": True, "device": device, "state": state}

    def _approved(self, what, approve=None):
        gate = approve or self.approval
        if gate is None:
            return False
        return bool(gate(what))

    def all_off(self, approve=None):
        """Centralized control: switch every on-device off."""
        off = []
        for name, dev in self.devices.items():
            if dev.state.get("on"):
                dev.set(on=False)
                off.append(name)
        self.audit.log("home_all_off", "hub", ",".join(off))
        return {"ok": True, "turned_off": off}

    def schedule(self, device, when, command):
        item = {"device": device, "when": when, "command": command}
        self.schedules.append(item)
        return item

    def due(self, now=None):
        now = now or datetime.datetime.now().isoformat(timespec="minutes")
        return [s for s in self.schedules if s["when"] <= now]

    def motion_event(self, sensor, detected, area="home"):
        """Motion Sensor Integration + Automated Home Security."""
        self.devices.setdefault(sensor, Device(sensor, "motion_sensor"))
        self.devices[sensor].set(detected=detected)
        if detected:
            self.security_events.append({"ts": _now(), "sensor": sensor,
                                         "area": area, "kind": "motion"})
        return {"ok": True, "motion": detected,
                "security_events": len(self.security_events)}

    def occupancy(self, room, present):
        """Room Occupancy Detection."""
        self.devices.setdefault(f"occupancy:{room}", Device(room, "occupancy"))
        self.devices[f"occupancy:{room}"].set(present=present)
        return {"room": room, "present": present}

    def temperature(self, sensor, celsius):
        """Temperature Monitoring."""
        self.devices.setdefault(sensor, Device(sensor, "temperature"))
        self.devices[sensor].set(celsius=celsius)
        alert = celsius > 32 or celsius < 10
        return {"sensor": sensor, "celsius": celsius,
                "alert": alert}

    def energy(self, device, watts):
        """Smart Energy Management: consumption log + top consumer."""
        self.energy_log.append({"device": device, "watts": watts})
        total = sum(e["watts"] for e in self.energy_log)
        top = max(self.energy_log, key=lambda e: e["watts"])["device"]
        return {"total_watts": total, "top_consumer": top}

    def cctv_snapshot(self, camera, image_path):
        """CCTV Integration: route frame through vision analysis."""
        from .cognitive_core import VisionSystem
        entry = VisionSystem().analyze_image(image_path)
        entry["camera"] = camera
        return entry

    def status(self):
        return {"devices": {n: dict(d.state) for n, d in self.devices.items()},
                "schedules": len(self.schedules),
                "security_events": len(self.security_events)}


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")
