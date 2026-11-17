# -*- coding: utf-8 -*-
"""Cybersecurity Intelligence — attack surface analysis and incident
timeline reconstruction (defensive tooling for authorized systems only).

AttackSurfaceAnalyzer: static scan of a Python/config tree for exposed
surface — dangerous calls, shell execution, bind-all hosts, debug flags,
hardcoded secrets.
IncidentTimeline: parses timestamped security logs, orders events, clusters
them into attack phases (recon -> compromise -> escalation -> exfiltration).
"""
import os
import re
from collections import Counter

DANGEROUS = [
    (r"\beval\s*\(", "eval() call", "high"),
    (r"\bexec\s*\(", "exec() call", "high"),
    (r"os\.system\s*\(", "os.system shell call", "high"),
    (r"subprocess\.\w+\([^)]*shell\s*=\s*True", "subprocess with shell=True", "high"),
    (r"pickle\.loads?\s*\(", "pickle deserialization", "medium"),
    (r"yaml\.load\s*\((?![^)]*Loader)", "yaml.load without safe Loader", "medium"),
    (r"verify\s*=\s*False", "TLS verification disabled", "high"),
    (r"0\.0\.0\.0", "bind-all network interface", "medium"),
    (r"debug\s*=\s*True", "debug mode enabled", "medium"),
    (r"md5|sha1\b", "weak hash algorithm", "low"),
    (r"(?i)(api[_-]?key|password|secret|token)\s*[=:]\s*[\"'][^\"']{8,}[\"']",
     "hardcoded credential", "high"),
    (r"password\.(md5|sha1)\b", "weak password hashing", "high"),
]

LOG_TS = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})\s+"
    r"(?:(?P<level>DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL)\s+)?(?P<msg>.*)")

PHASE_KEYWORDS = {
    "recon": ("scan", "probe", "enumerat", "port", "nmap", "recon"),
    "compromise": ("login failed", "auth fail", "brute", "inject", "exploit",
                   "unauthorized", "invalid user"),
    "escalation": ("sudo", "privilege", "root", "escalat", "admin"),
    "exfiltration": ("download", "upload", "transfer", "export", "outbound",
                     "exfil", "copied"),
    "cleanup": ("log clear", "deleted", "wipe", "purge"),
}


class AttackSurfaceAnalyzer:
    """Attack Surface Intelligence: analyze an authorized codebase's exposure."""

    def __init__(self, root):
        self.root = root

    def scan(self):
        findings = []
        files = 0
        for dirpath, dirs, files_ in os.walk(self.root):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for fname in files_:
                if not fname.endswith((".py", ".json", ".yml", ".yaml", ".cfg", ".ini", ".env")):
                    continue
                path = os.path.join(dirpath, fname)
                files += 1
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        for line_no, line in enumerate(f, 1):
                            for pattern, label, severity in DANGEROUS:
                                if re.search(pattern, line):
                                    findings.append({
                                        "file": os.path.relpath(path, self.root),
                                        "line": line_no, "finding": label,
                                        "severity": severity,
                                        "excerpt": line.strip()[:100],
                                    })
                except OSError:
                    continue
        by_sev = Counter(f["severity"] for f in findings)
        risk = 3 * by_sev.get("high", 0) + 2 * by_sev.get("medium", 0) \
            + by_sev.get("low", 0)
        return {"files_scanned": files, "findings": findings,
                "counts": dict(by_sev), "risk_score": risk,
                "verdict": ("critical" if by_sev.get("high", 0) > 5
                            else "elevated" if risk > 5 else "acceptable")}


class IncidentTimeline:
    """Incident Timeline Reconstruction: events -> ordered attack phases."""

    def __init__(self, keywords=None):
        self.phases = PHASE_KEYWORDS if keywords is None else keywords

    def parse(self, lines):
        events = []
        for line in lines:
            m = LOG_TS.search(line)
            if m:
                events.append({"ts": m.group("ts").replace("T", " "),
                               "level": m.group("level") or "INFO",
                               "msg": m.group("msg").strip()[:160]})
        events.sort(key=lambda e: e["ts"])
        return events

    def _phase_of(self, msg):
        msg = msg.lower()
        for phase, words in self.phases.items():
            if any(re.search(rf"\b{re.escape(w)}", msg) for w in words):
                return phase
        return None

    def reconstruct(self, lines):
        events = self.parse(lines)
        phases = []
        for event in events:
            phase = self._phase_of(event["msg"])
            if phase:
                phases.append({"ts": event["ts"], "phase": phase,
                               "msg": event["msg"][:120]})
        ordered = sorted(phases, key=lambda p: p["ts"])
        sequence = []
        for p in ordered:
            if not sequence or sequence[-1]["phase"] != p["phase"]:
                sequence.append({"phase": p["phase"], "start": p["ts"], "events": 1})
            else:
                sequence[-1]["events"] += 1
        return {"events": len(events), "phase_events": ordered,
                "attack_sequence": sequence,
                "summary": " -> ".join(s["phase"] for s in sequence)
                or "no attack indicators"}
