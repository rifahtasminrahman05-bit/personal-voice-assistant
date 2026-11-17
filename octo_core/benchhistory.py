# -*- coding: utf-8 -*-
"""Benchmark history + trend tracking for the performance dashboard."""
import datetime
import json
import os
import threading

HISTORY_FILE = os.path.join("data", "octo", "benchmarks.jsonl")


class BenchmarkHistory:
    """Performance Regression Dashboard backend: append reports, read trends."""

    def __init__(self, path=HISTORY_FILE):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._lock = threading.Lock()

    def record(self, report, label="run"):
        entry = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
                 "label": label, "report": report}
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def entries(self, n=20):
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [json.loads(l) for l in lines[-n:]]

    def trend(self, metric=("math", "score")):
        """Compare the last two recorded values of a nested metric path."""
        vals = []
        for e in self.entries(50):
            node = e["report"]
            for key in metric:
                if not isinstance(node, dict) or key not in node:
                    node = None
                    break
                node = node[key]
            if isinstance(node, (int, float)):
                vals.append((e["ts"], node))
        if len(vals) < 2:
            return {"metric": metric, "history": vals, "direction": "not-enough-data"}
        direction = "improving" if vals[-1][1] > vals[-2][1] else \
            "declining" if vals[-1][1] < vals[-2][1] else "flat"
        return {"metric": metric, "history": vals[-10:], "direction": direction}
