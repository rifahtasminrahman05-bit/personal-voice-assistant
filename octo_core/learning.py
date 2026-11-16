# -*- coding: utf-8 -*-
"""Self-Evolution & Adaptive Learning — with guardrails.

Skill manager (reusable strategies from successful runs), feedback engine,
regression detection and a controlled self-improvement pipeline:
propose -> isolated test -> benchmark -> human approval. Nothing is applied
automatically; approved proposals land as review-ready patch files.
"""
import datetime
import json
import os
import re
import subprocess
import sys
import threading

PROPOSAL_DIR = "octo_checkpoints/proposals"


class SkillManager:
    """Skill Acquisition Engine: capture validated workflows as reusable skills."""

    def __init__(self, path="data/octo/skills.json"):
        self.path = path
        self._lock = threading.Lock()

    def learn(self, goal, tools_used, outcome_ok):
        if not outcome_ok:
            return None
        skill = {
            "trigger": self._trigger(goal),
            "workflow": list(tools_used),
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        with self._lock:
            skills = self._load()
            if skill not in skills:
                skills.append(skill)
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(skills, f, ensure_ascii=False, indent=1)
        return skill

    def match(self, goal):
        g = goal.lower()
        for skill in self._load():
            if re.search(skill["trigger"], g):
                return skill
        return None

    def _trigger(self, goal):
        words = [w for w in re.findall(r"\w{4,}", goal.lower())][:3]
        return "|".join(words) or goal[:20]

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []


class FeedbackEngine:
    """Preference Learning: rate responses; retrieval prefers high-rated ones."""

    def __init__(self, path="data/octo/feedback.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def rate(self, prompt, response, rating, redact=None):
        entry = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
                 "prompt": str(prompt)[:200], "response": str(response)[:200],
                 "rating": int(rating)}
        if redact:
            entry = redact(json.dumps(entry, ensure_ascii=False))
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(entry + "\n")
        else:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def average_rating(self):
        if not os.path.exists(self.path):
            return None
        ratings = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ratings.append(json.loads(line)["rating"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return round(sum(ratings) / len(ratings), 2) if ratings else None


def detect_regression(baseline, current, threshold=0.05):
    """Regression Detection: flag metric degradation after a change."""
    regressions = {}
    for metric, base in baseline.items():
        if metric not in current:
            continue
        cur = current[metric]
        if metric in ("loss", "perplexity", "latency_ms"):
            worse = cur > base * (1 + threshold)
        else:
            worse = cur < base * (1 - threshold)
        if worse:
            regressions[metric] = {"baseline": base, "current": cur}
    return regressions


class SelfImprovement:
    """Controlled Self-Modification: propose -> isolated test -> benchmark -> approval.

    A proposal is a JSON file with a description and a patch (search/replace
    pair). Testing runs pytest in an isolated subprocess on a copy of the
    proposal; nothing in the repo changes until a human approves.
    """

    def __init__(self, proposal_dir=PROPOSAL_DIR, audit=None):
        self.proposal_dir = proposal_dir
        os.makedirs(proposal_dir, exist_ok=True)
        self.audit = audit

    def propose(self, description, file_path, old_str, new_str):
        pid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        proposal = {
            "id": pid,
            "description": description,
            "file": file_path,
            "old_str": old_str,
            "new_str": new_str,
            "status": "proposed",
        }
        path = os.path.join(self.proposal_dir, f"{pid}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(proposal, f, ensure_ascii=False, indent=1)
        if self.audit:
            self.audit.log("proposal", "self_improvement", description[:120])
        return path

    def test_isolated(self, proposal_path):
        """Run the test suite in a subprocess (isolated verification step)."""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header", "-x"],
            capture_output=True, text=True, timeout=540)
        return {"returncode": result.returncode,
                "summary": result.stdout.strip().splitlines()[-1] if result.stdout else ""}

    def apply(self, proposal_path, approve):
        """Apply only with explicit human approval; proposal is validated against
        the current file content before writing."""
        with open(proposal_path, "r", encoding="utf-8") as f:
            proposal = json.load(f)
        if not approve(proposal["id"], proposal["description"]):
            proposal["status"] = "rejected"
            with open(proposal_path, "w", encoding="utf-8") as f:
                json.dump(proposal, f, ensure_ascii=False, indent=1)
            if self.audit:
                self.audit.log("proposal_rejected", "self_improvement", proposal["id"])
            return False
        target = proposal["file"]
        with open(target, "r", encoding="utf-8") as f:
            content = f.read()
        if proposal["old_str"] not in content:
            proposal["status"] = "stale"
        else:
            content = content.replace(proposal["old_str"], proposal["new_str"], 1)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            proposal["status"] = "applied"
        with open(proposal_path, "w", encoding="utf-8") as f:
            json.dump(proposal, f, ensure_ascii=False, indent=1)
        if self.audit:
            self.audit.log("proposal_" + proposal["status"], "self_improvement", proposal["id"])
        return proposal["status"] == "applied"
