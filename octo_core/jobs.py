# -*- coding: utf-8 -*-
"""Scheduled tasks — cron-style recurring goals for the Octo agent.

Jobs persist in data/octo/jobs.json. A daemon thread checks due jobs and
runs them through the AgentOrchestrator; results append to the audit log.

Usage (programmatic or from the desktop app):
    jobs = JobScheduler(orchestrator)
    jobs.add("nightly log scan", goal="logscan data/auth.log", interval_minutes=1440)
    jobs.start()
"""
import datetime
import json
import os
import threading
import time

JOBS_FILE = os.path.join("data", "octo", "jobs.json")


class JobScheduler:
    def __init__(self, orchestrator, path=JOBS_FILE, audit=None):
        self.orchestrator = orchestrator
        self.path = path
        self.audit = audit
        self._jobs = []
        self._thread = None
        self._stop = threading.Event()
        self._results = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self._jobs = json.load(f)

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._jobs, f, ensure_ascii=False, indent=1)

    def add(self, name, goal, interval_minutes):
        job = {"name": name, "goal": goal,
               "interval_minutes": int(interval_minutes),
               "last_run": None, "runs": 0}
        self._jobs.append(job)
        self._save()
        return job

    def remove(self, name):
        before = len(self._jobs)
        self._jobs = [j for j in self._jobs if j["name"] != name]
        self._save()
        return len(self._jobs) < before

    def list(self):
        return [dict(j) for j in self._jobs]

    def due(self, job, now=None):
        if job["last_run"] is None:
            return True
        now = now or time.time()
        return now - job["last_run"] >= job["interval_minutes"] * 60

    def run_due(self):
        ran = []
        for job in self._jobs:
            if not self.due(job):
                continue
            trace, reflection = self.orchestrator.execute(job["goal"], verbose=False)
            job["last_run"] = time.time()
            job["runs"] += 1
            entry = {"job": job["name"], "ts": datetime.datetime.now()
                     .isoformat(timespec="seconds"),
                     "success_rate": reflection["success_rate"]}
            self._results.append(entry)
            if self.audit:
                self.audit.log("scheduled_job", "scheduler",
                               f"{job['name']}: {reflection['success_rate']}")
            ran.append(entry)
        if ran:
            self._save()
        return ran

    def results(self, n=20):
        return self._results[-n:]

    def start(self, poll_seconds=60):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def loop():
            while not self._stop.wait(poll_seconds):
                try:
                    self.run_due()
                except Exception as e:
                    if self.audit:
                        self.audit.log("scheduler_error", "scheduler", str(e))

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
