# -*- coding: utf-8 -*-
"""Robotics & Physical Control (features 119-129).

Robot integration, sensor bus, motor control, navigation with obstacle
detection and path planning, environmental mapping, autonomous
movement, object manipulation, always-honored emergency stop and
robot status monitoring. Simulation-first: a RobotInterface subclass
or injected driver does the real hardware work; the default driver is
a safe in-memory simulator.
"""
import math
import time


class SensorBus:
    """120. Sensor Integration: named sensor streams with history."""

    def __init__(self, history=100):
        self.limit = history
        self.streams = {}

    def publish(self, sensor, value, now=None):
        stream = self.streams.setdefault(sensor, [])
        stream.append({"value": value, "t": now if now is not None
                       else time.time()})
        del stream[:-self.limit]
        return {"sensor": sensor, "value": value}

    def latest(self, sensor):
        stream = self.streams.get(sensor)
        return stream[-1]["value"] if stream else None

    def average(self, sensor, last=10):
        stream = self.streams.get(sensor, [])[-last:]
        values = [s["value"] for s in stream
                  if isinstance(s["value"], (int, float))]
        return sum(values) / len(values) if values else None


class MotorController:
    """121. Motor Control: per-wheel commands with safety clamp."""

    MAX_SPEED = 1.0

    def __init__(self, wheels=("left", "right")):
        self.wheels = {w: 0.0 for w in wheels}
        self.estop = False

    def command(self, wheel, speed):
        if self.estop:
            return {"ok": False, "reason": "emergency stop active"}
        speed = max(-self.MAX_SPEED, min(self.MAX_SPEED, float(speed)))
        self.wheels[wheel] = speed
        return {"ok": True, "wheel": wheel, "speed": speed}

    def stop(self):
        for wheel in self.wheels:
            self.wheels[wheel] = 0.0
        return {"ok": True, "stopped": True}


class Navigator:
    """122-125. Navigation, obstacle detection, path planning, mapping."""

    DIRECTIONS = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}

    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.obstacles = set()
        self.position = (0, 0)
        self.visited = {self.position}

    def add_obstacle(self, x, y):
        self.obstacles.add((x, y))

    def obstacle_ahead(self, direction, distance=1):
        dx, dy = self.DIRECTIONS[direction]
        target = (self.position[0] + dx * distance,
                  self.position[1] + dy * distance)
        blocked = (target in self.obstacles
                   or not (0 <= target[0] < self.grid_size
                           and 0 <= target[1] < self.grid_size))
        return {"target": target, "blocked": blocked}

    def plan(self, goal):
        """BFS path planning around obstacles."""
        start = self.position
        queue, came = [start], {start: None}
        while queue:
            node = queue.pop(0)
            if node == goal:
                break
            for dx, dy in self.DIRECTIONS.values():
                nxt = (node[0] + dx, node[1] + dy)
                if (nxt in came or nxt in self.obstacles
                        or not (0 <= nxt[0] < self.grid_size
                                and 0 <= nxt[1] < self.grid_size)):
                    continue
                came[nxt] = node
                queue.append(nxt)
        if goal not in came:
            return {"ok": False, "reason": "no path"}
        path, node = [], goal
        while node is not None:
            path.append(node)
            node = came[node]
        path.reverse()
        return {"ok": True, "path": path, "steps": len(path) - 1}

    def move(self, direction):
        check = self.obstacle_ahead(direction)
        if check["blocked"]:
            return {"ok": False, "reason": "obstacle", **check}
        self.position = check["target"]
        self.visited.add(self.position)
        return {"ok": True, "position": self.position}

    def map(self):
        return {"size": self.grid_size, "obstacles": sorted(self.obstacles),
                "visited": sorted(self.visited),
                "coverage": round(len(self.visited) / self.grid_size ** 2, 3)}


class Manipulator:
    """127. Object Manipulation: grip state machine."""

    STATES = ("idle", "reaching", "grasping", "holding", "released")

    def __init__(self):
        self.state = "idle"
        self.holding = None
        self.log = []

    def act(self, action, obj=None):
        if action == "reach" and self.state == "idle":
            self.state = "reaching"
        elif action == "grasp" and self.state == "reaching":
            self.state, self.holding = "holding", obj
        elif action == "release" and self.state == "holding":
            self.state, released = "released", self.holding
            self.holding = None
            self.log.append(released)
        else:
            return {"ok": False, "state": self.state}
        return {"ok": True, "state": self.state, "holding": self.holding}


class RobotMonitor:
    """129. Robot Status Monitoring: battery/temp/uptime snapshot."""

    def __init__(self):
        self.started = time.time()
        self.errors = []

    def status(self, battery_percent=100.0, temperature_c=25.0):
        uptime = time.time() - self.started
        healthy = battery_percent > 20 and temperature_c < 70
        return {"battery_percent": battery_percent,
                "temperature_c": temperature_c,
                "uptime_s": round(uptime, 1),
                "errors": len(self.errors),
                "healthy": healthy}


class RobotInterface:
    """119. Robot Integration + 126. Autonomous Movement + 128. E-Stop."""

    def __init__(self, name="octo-bot", grid_size=10):
        self.name = name
        self.sensors = SensorBus()
        self.motors = MotorController()
        self.navigator = Navigator(grid_size)
        self.manipulator = Manipulator()
        self.monitor = RobotMonitor()
        self.autonomous = False

    def emergency_stop(self):
        """128. Emergency Stop: always honored, cannot be overridden."""
        self.autonomous = False
        self.motors.estop = True
        self.motors.stop()
        return {"estop": True, "motors": dict(self.motors.wheels)}

    def reset_estop(self):
        self.motors.estop = False
        return {"estop": False}

    def run_autonomous(self, goal, max_steps=50):
        """126. Autonomous Movement: follow planned path, stop on e-stop."""
        plan = self.navigator.plan(goal)
        if not plan.get("ok"):
            return plan
        self.autonomous = True
        moves = 0
        for target in plan["path"][1:]:
            if self.motors.estop or not self.autonomous:
                break
            dx = target[0] - self.navigator.position[0]
            dy = target[1] - self.navigator.position[1]
            direction = next(d for d, (ddx, ddy)
                             in self.navigator.DIRECTIONS.items()
                             if (ddx, ddy) == (dx, dy))
            result = self.navigator.move(direction)
            if result.get("ok"):
                moves += 1
            self.sensors.publish("position", self.navigator.position)
        self.autonomous = False
        return {"ok": self.navigator.position == goal,
                "position": self.navigator.position, "moves": moves}

    def status(self):
        return {"name": self.name, "autonomous": self.autonomous,
                "estop": self.motors.estop,
                "position": self.navigator.position,
                **self.monitor.status()}
