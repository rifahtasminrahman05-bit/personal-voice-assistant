# -*- coding: utf-8 -*-
"""Rifah-Inspired Advanced Features (features 151-165).

A sci-fi style command layer: holographic/AR/HUD interfaces, suit and
armor diagnostics, flight telemetry, tactical analysis, target
tracking, environmental threat detection, emergency response
coordination, hardware diagnostics and predictive maintenance.
Everything is local simulation and analysis — no network, no hardware
writes without explicit user action.
"""
import time


class HolographicInterface:
    """151. Holographic Interface + 152. 3D Visualization."""

    def __init__(self):
        self.objects = {}

    def project(self, name, points, color="cyan", scale=1.0):
        self.objects[name] = {"points": [tuple(p) for p in points],
                              "color": color, "scale": scale}
        return {"projected": name, "vertices": len(points)}

    def rotate(self, name, axis="z", degrees=90):
        import math
        obj = self.objects.get(name)
        if obj is None:
            return {"ok": False, "error": "not projected"}
        rad = math.radians(degrees)
        rotated = []
        for x, y, z in obj["points"]:
            if axis == "z":
                x, y = (x * math.cos(rad) - y * math.sin(rad),
                        x * math.sin(rad) + y * math.cos(rad))
            elif axis == "y":
                x, z = (x * math.cos(rad) + z * math.sin(rad),
                        -x * math.sin(rad) + z * math.cos(rad))
            else:
                y, z = (y * math.cos(rad) - z * math.sin(rad),
                        y * math.sin(rad) + z * math.cos(rad))
            rotated.append((round(x, 3), round(y, 3), round(z, 3)))
        obj["points"] = rotated
        return {"ok": True, "object": name, "axis": axis,
                "vertices": len(rotated)}


class ARInterface:
    """153. Augmented Reality Interface: overlay plan over camera frame."""

    def overlay(self, frame_meta, widgets):
        return {"frame": frame_meta, "overlays": list(widgets),
                "anchored": len(widgets)}


class HUD:
    """154. Heads-Up Display: priority-sorted status lines."""

    def __init__(self):
        self.panels = {}

    def update(self, panel, text, priority=1):
        self.panels[panel] = {"text": text[:120], "priority": priority}

    def render(self):
        ordered = sorted(self.panels.items(),
                         key=lambda kv: kv[1]["priority"])
        return {"hud": [{"panel": name, **data} for name, data in ordered]}


class SuitDiagnostics:
    """155. Real-Time Suit Diagnostics + 156. Armor Monitoring."""

    def __init__(self):
        self.subsystems = {
            "power": 100.0, "life_support": 100.0,
            "thrusters": 100.0, "comms": 100.0,
        }
        self.armor_plates = {f"plate_{i}": 100.0 for i in range(6)}

    def wear(self, subsystem, amount):
        if subsystem in self.subsystems:
            self.subsystems[subsystem] = max(
                0.0, self.subsystems[subsystem] - abs(amount))

    def impact(self, plate, force):
        if plate in self.armor_plates:
            self.armor_plates[plate] = max(
                0.0, self.armor_plates[plate] - abs(force) * 2)

    def report(self):
        critical = [s for s, v in self.subsystems.items() if v < 30]
        breached = [p for p, v in self.armor_plates.items() if v < 25]
        return {"subsystems": dict(self.subsystems),
                "armor": dict(self.armor_plates),
                "critical": critical, "breached": breached,
                "integrity": round(sum(self.armor_plates.values())
                                   / len(self.armor_plates), 1)}


class FlightTelemetry:
    """157. Flight Telemetry Analysis."""

    def __init__(self):
        self.samples = []

    def record(self, altitude_m, speed_mps, heading_deg):
        sample = {"altitude_m": altitude_m, "speed_mps": speed_mps,
                  "heading_deg": heading_deg, "t": time.time()}
        self.samples.append(sample)
        return sample

    def analyze(self):
        if len(self.samples) < 2:
            return {"ok": False, "reason": "need >= 2 samples"}
        climb = self.samples[-1]["altitude_m"] - self.samples[0]["altitude_m"]
        speeds = [s["speed_mps"] for s in self.samples]
        headings = [s["heading_deg"] for s in self.samples]
        turn_rate = min(
            abs(headings[i + 1] - headings[i]) for i in range(len(headings) - 1))
        return {"ok": True, "climb_m": round(climb, 1),
                "avg_speed_mps": round(sum(speeds) / len(speeds), 1),
                "max_speed_mps": max(speeds),
                "turn_rate_deg": round(turn_rate, 1),
                "stable": max(speeds) - min(speeds) < 50}


class TacticalAnalyzer:
    """158. Threat Assessment + 159. Tactical Analysis + 161. Env Threats."""

    ENVIRONMENTAL = ("fire", "smoke", "gas", "flood", "radiation",
                     "extreme heat", "extreme cold")

    def threat_level(self, contacts):
        """contacts: [{distance, speed, hostile}] -> overall level."""
        score = 0
        for c in contacts:
            proximity = 1.0 / (1.0 + max(0.0, c.get("distance", 100)) / 50)
            score += proximity * (3 if c.get("hostile") else 0.5)
        level = ("none" if score < 1 else "low" if score < 4 else
                 "elevated" if score < 8 else "high")
        return {"score": round(score, 2), "level": level,
                "hostile_contacts": sum(1 for c in contacts
                                        if c.get("hostile"))}

    def tactical_picture(self, contacts, threats):
        assessment = self.threat_level(contacts)
        env = [t for t in self.ENVIRONMENTAL
               if any(t in str(e).lower() for e in threats)]
        return {"assessment": assessment, "environmental": env,
                "recommended": "evade" if assessment["level"] in
                ("elevated", "high") else "monitor"}


class TargetTracker:
    """160. Target Tracking: nearest-hostile lock with history."""

    def __init__(self):
        self.locked = None
        self.trail = []

    def update(self, contacts):
        candidates = [c for c in contacts if c.get("hostile")]
        if not candidates:
            self.locked = None
            return {"locked": None}
        target = min(candidates,
                     key=lambda c: c.get("distance", float("inf")))
        self.locked = target.get("id")
        self.trail.append({"id": self.locked,
                           "distance": target.get("distance")})
        del self.trail[:-50]
        return {"locked": self.locked,
                "distance": target.get("distance"),
                "approaching": len(self.trail) > 1
                and target.get("distance", 0)
                < self.trail[-2]["distance"]}


class EmergencyResponseCoordinator:
    """163. Emergency Response Coordination."""

    PROTOCOLS = {
        "fire": ["seal zone", "ventilate", "extinguish"],
        "hull breach": ["seal bulkheads", "suit up", "patch"],
        "medical": ["stabilize", "diagnose", "call for aid"],
        "power loss": ["switch to reserve", "shed load", "reroute"],
    }

    def coordinate(self, emergency):
        key = next((k for k in self.PROTOCOLS
                    if k in emergency.lower()), None)
        steps = self.PROTOCOLS.get(key, ["assess", "contain", "report"])
        return {"emergency": emergency[:120], "protocol": key,
                "steps": steps, "coordinated": True}


class HardwareDiagnostics:
    """164. Hardware Diagnostics + 165. Predictive Maintenance."""

    def __init__(self):
        self.components = {}

    def register(self, name, hours_used, rated_hours, health=100.0):
        self.components[name] = {"hours": hours_used,
                                 "rated": rated_hours,
                                 "health": health}

    def diagnose(self):
        results = []
        for name, c in self.components.items():
            wear = c["hours"] / c["rated"] if c["rated"] else 0.0
            remaining = max(0.0, c["rated"] - c["hours"])
            results.append({
                "component": name, "wear": round(min(wear, 1.0), 2),
                "remaining_hours": round(remaining, 1),
                "health": c["health"],
                "service_soon": wear > 0.8 or c["health"] < 40,
            })
        return {"components": results,
                "attention": [r["component"] for r in results
                              if r["service_soon"]]}

    def predictive_maintenance(self, horizon_hours=100):
        plan = []
        for name, c in self.components.items():
            remaining = c["rated"] - c["hours"]
            if remaining <= horizon_hours:
                plan.append({"component": name,
                             "within_hours": round(max(remaining, 0), 1),
                             "action": "inspect and service"})
        return {"horizon_hours": horizon_hours, "schedule": plan}
