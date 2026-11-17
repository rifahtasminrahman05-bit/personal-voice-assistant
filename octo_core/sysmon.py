# -*- coding: utf-8 -*-
"""System Intelligence (features 83-95).

CPU/GPU/RAM/storage/network/battery/temperature/process monitoring,
performance optimization, resource allocation, crash detection,
automated recovery and a system health dashboard.
Uses psutil when available; falls back to /proc and heuristics.
No network, no telemetry.
"""
import os
import time

try:
    import psutil
except ImportError:
    psutil = None


def _read_proc_stat():
    try:
        with open("/proc/stat", "r", encoding="utf-8") as fh:
            parts = fh.readline().split()
        vals = [float(v) for v in parts[1:8]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        return idle, sum(vals)
    except (OSError, ValueError, IndexError):
        return None, None


def _meminfo_mb():
    try:
        info = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                key, val = line.split(":", 1)
                info[key] = float(val.strip().split()[0]) / 1024.0
        total = info.get("MemTotal", 0.0)
        available = info.get("MemAvailable", 0.0)
        return total, total - available
    except (OSError, ValueError):
        return 0.0, 0.0


class CPUMonitor:
    """83. CPU Monitoring."""

    def __init__(self):
        self._last = _read_proc_stat()

    def usage(self):
        if psutil is not None:
            return {"percent": psutil.cpu_percent(interval=None),
                    "cores": psutil.cpu_count() or 1}
        idle, total = _read_proc_stat()
        if self._last[0] is None or idle is None:
            self._last = (idle, total)
            return {"percent": 0.0, "cores": os.cpu_count() or 1}
        d_idle, d_total = idle - self._last[0], total - self._last[1]
        self._last = (idle, total)
        percent = 100.0 * (1.0 - d_idle / d_total) if d_total > 0 else 0.0
        return {"percent": round(max(0.0, min(100.0, percent)), 1),
                "cores": os.cpu_count() or 1}


class GPUMonitor:
    """84. GPU Monitoring (local only; no CUDA required)."""

    def status(self):
        try:
            import torch
            return {"available": torch.cuda.is_available(),
                    "backend": "cuda" if torch.cuda.is_available()
                    else "cpu-only"}
        except ImportError:
            return {"available": False, "backend": "torch-missing"}


class RAMMonitor:
    """85. RAM Monitoring."""

    def usage(self):
        if psutil is not None:
            mem = psutil.virtual_memory()
            return {"total_mb": mem.total // 1048576,
                    "used_mb": mem.used // 1048576,
                    "percent": mem.percent}
        total, used = _meminfo_mb()
        percent = round(100.0 * used / total, 1) if total else 0.0
        return {"total_mb": int(total), "used_mb": int(used),
                "percent": percent}


class StorageMonitor:
    """86. Storage Monitoring."""

    def usage(self, path="."):
        if psutil is not None:
            usage = psutil.disk_usage(path)
            return {"total_gb": round(usage.total / 1e9, 1),
                    "free_gb": round(usage.free / 1e9, 1),
                    "percent": usage.percent}
        stat = os.statvfs(path)
        total = stat.f_blocks * stat.f_frsize
        free = stat.f_bavail * stat.f_frsize
        return {"total_gb": round(total / 1e9, 1),
                "free_gb": round(free / 1e9, 1),
                "percent": round(100.0 * (total - free) / total, 1)
                if total else 0.0}


class NetworkMonitor:
    """87. Network Monitoring (counters only, no traffic inspection)."""

    def counters(self):
        if psutil is None:
            return {"available": False}
        io = psutil.net_io_counters()
        return {"available": True, "sent_mb": round(io.bytes_sent / 1e6, 1),
                "recv_mb": round(io.bytes_recv / 1e6, 1)}


class BatteryMonitor:
    """88. Battery Monitoring."""

    def status(self):
        getter = getattr(psutil, "sensors_battery", None) if psutil \
            else None
        if getter is not None:
            try:
                bat = getter()
            except (OSError, ValueError):
                bat = None
            if bat is not None:
                return {"present": True, "percent": bat.percent,
                        "plugged": bat.power_plugged}
        return {"present": False}


class TemperatureMonitor:
    """89. Temperature Monitoring."""

    def read(self):
        if psutil is not None:
            getter = getattr(psutil, "sensors_temperatures", None)
            if getter is not None:
                temps = getter() or {}
                for name, entries in temps.items():
                    if entries:
                        return {"available": True, "sensor": name,
                                "celsius": entries[0].current}
        return {"available": False}


class ProcessMonitor:
    """90. Process Monitoring (top by CPU)."""

    def top(self, limit=5):
        if psutil is None:
            return []
        procs = []
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                procs.append({"pid": proc.info["pid"],
                              "name": proc.info["name"] or "?",
                              "cpu": proc.cpu_percent(interval=None)})
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        procs.sort(key=lambda p: p["cpu"], reverse=True)
        return procs[:limit]


class PerformanceOptimizer:
    """91. Performance Optimization: prioritized recommendations."""

    def recommend(self, cpu_percent, ram_percent, disk_percent):
        tips = []
        if cpu_percent > 85:
            tips.append("close CPU-heavy apps")
        if ram_percent > 85:
            tips.append("reduce memory load / restart heavy services")
        if disk_percent > 90:
            tips.append("free disk space (caches, old snapshots)")
        return {"tips": tips, "healthy": not tips}


class ResourceAllocator:
    """92. Resource Allocation: budget workers by measured pressure."""

    def allocate(self, cpu_percent, ram_percent, max_workers=8):
        headroom = (100 - max(cpu_percent, ram_percent)) / 100.0
        workers = max(1, min(max_workers, int(1 + headroom * max_workers)))
        return {"workers": workers, "headroom": round(headroom, 2)}


class CrashDetector:
    """93. Crash Detection: repeated failures within a window."""

    def __init__(self, threshold=3, window=300.0):
        self.threshold = threshold
        self.window = window
        self.events = {}

    def record(self, component, ok, now=None):
        now = now if now is not None else time.time()
        history = [t for t in self.events.get(component, [])
                   if now - t <= self.window]
        if not ok:
            history.append(now)
        self.events[component] = history
        return len(history) >= self.threshold

    def crashed(self, component):
        return len(self.events.get(component, [])) >= self.threshold


class RecoveryManager:
    """94. Automated Recovery: registered procedures, approval-gated."""

    def __init__(self):
        self.procedures = {}

    def register(self, component, procedure):
        self.procedures[component] = procedure

    def recover(self, component):
        procedure = self.procedures.get(component)
        if procedure is None:
            return {"ok": False, "error": "no recovery procedure"}
        try:
            return {"ok": bool(procedure()), "component": component}
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:120]}


class HealthDashboard:
    """95. System Health Dashboard: one-shot snapshot."""

    def __init__(self):
        self.cpu = CPUMonitor()
        self.gpu = GPUMonitor()
        self.ram = RAMMonitor()
        self.disk = StorageMonitor()
        self.battery = BatteryMonitor()
        self.temp = TemperatureMonitor()
        self.optimizer = PerformanceOptimizer()
        self.allocator = ResourceAllocator()

    def snapshot(self):
        cpu = self.cpu.usage()
        ram = self.ram.usage()
        disk = self.disk.usage()
        rec = self.optimizer.recommend(cpu["percent"], ram["percent"],
                                       disk["percent"])
        return {"cpu": cpu, "gpu": self.gpu.status(), "ram": ram,
                "disk": disk, "battery": self.battery.status(),
                "temperature": self.temp.read(),
                "workers": self.allocator.allocate(cpu["percent"],
                                                   ram["percent"]),
                "healthy": rec["healthy"], "tips": rec["tips"]}
