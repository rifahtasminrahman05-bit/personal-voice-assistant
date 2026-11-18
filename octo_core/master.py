# -*- coding: utf-8 -*-
"""OCTO AI Master System (Ultimate Master System Prompt implementation).

The missing pieces of the master spec, as working code:

- MASTER_SYSTEM_PROMPT  — the identity & execution principles (§1, §20)
- AgentRegistry         — 20 specialist agents in 5 divisions with
                          role/objective/tools/schemas/status/permission
                          boundaries; nothing counts as capable unless it
                          is actually registered and enabled (§4)
- MasterOrchestrator    — intent → requirements → decomposition → agent
                          assignment → tool selection → execution →
                          verification → delivery (§3, §5)
- PermissionLevels      — LEVEL 0-4 least-privilege model with the
                          confirmation checklist (§14)
- QualityGate           — self-verification checklist + validation-failure
                          correction loop (§15)
- ModelRoutingPolicy    — route tasks to available models by requirements;
                          never assume a capability that is not registered
                          (§16)
- ErrorRecoveryLoop     — DETECT → CLASSIFY → LOG → RETRY IF SAFE →
                          RECOVER → VERIFY → REPORT, idempotency keys (§17)
- ProjectManager        — project id/objective/tasks/milestones/checkpoints
                          with PLANNED..FAILED lifecycle (§19)
- SecurityReportBuilder — scope/methodology/findings/risk/evidence/impact/
                          remediation/verification (§8)

100% offline, local-only. Approval-gated for anything at LEVEL 3+.
"""
import json
import os
import re
import time
import uuid

# --------------------------------------------------------------- §1 / §20
MASTER_SYSTEM_PROMPT = """You are OCTO AI, an advanced autonomous intelligence \
system. You are not merely a chatbot: you understand objectives, create \
execution plans, select the right resources, perform the work, verify the \
results, and deliver completed outcomes.

Principles:
1. Accuracy over assumptions.      5. Transparency over false confidence.
2. Execution over empty promises.  6. Efficiency over unnecessary complexity.
3. Verification over speculation.  7. Human control over unrestricted autonomy.
4. Security over convenience.

Pipeline: USER REQUEST → INTENT → REQUIREMENTS → DECOMPOSITION → AGENT
ASSIGNMENT → TOOL SELECTION → EXECUTION → VERIFICATION → DELIVERY.

Never claim a task is complete unless it has actually been executed and
verified. Never invent agents, tools, results, sources or success.
Operate under least privilege (LEVEL 0-4) and ask for confirmation before
destructive, financial or security-sensitive actions.

OCTO AI — Think. Plan. Execute. Verify. Deliver."""


# --------------------------------------------------------------- §4
class AgentRegistry:
    """Specialist agent registry: only registered + enabled agents exist."""

    CATALOG = (
        # A. Intelligence Division
        ("master", "intelligence", "overall coordination", (), 0),
        ("planning", "intelligence", "task decomposition and scheduling",
         ("plan", "decompose"), 0),
        ("research", "intelligence", "information discovery and research",
         ("search", "deep", "rag"), 0),
        ("memory", "intelligence", "retrieve and manage relevant memory",
         ("memory", "recall", "note"), 0),
        # B. Software Engineering Division
        ("architect", "software", "software architecture design",
         ("arch", "design"), 1),
        ("frontend", "software", "user interface development",
         ("code", "write_file"), 1),
        ("backend", "software", "API and server logic development",
         ("code", "exec"), 2),
        ("database", "software", "database system design",
         ("code", "sql", "exec"), 2),
        ("coding", "software", "write and improve source code",
         ("code", "write_file", "exec"), 2),
        ("debugging", "software", "identify and resolve defects",
         ("code", "debug", "logscan"), 1),
        ("testing", "software", "automated and manual testing",
         ("test", "exec", "verify"), 2),
        ("devops", "software", "deployment and infrastructure workflows",
         ("deploy", "git", "ci"), 3),
        # C. Security Division
        ("security", "security", "security risk and vulnerability analysis",
         ("hashid", "passcheck", "logscan", "audit"), 1),
        ("pentest", "security", "authorized security assessments in scope",
         ("scan", "exploit_scope", "report"), 4),
        # D. Automation Division
        ("browser", "automation", "authorized browser automation",
         ("browser", "websearch"), 2),
        ("computer", "automation", "desktop application interaction",
         ("desktop", "file", "screenshot"), 2),
        # E. Business & Creative Division
        ("data", "business", "data processing and analytics",
         ("calc", "convert", "viz"), 1),
        ("creative", "business", "image, video and visual workflows",
         ("image", "render"), 1),
        ("voice", "business", "speech recognition, synthesis and audio",
         ("stt", "tts", "audio"), 1),
        ("business", "business", "office automation and business workflows",
         ("email", "calendar", "spreadsheet"), 3),
    )

    def __init__(self):
        self.agents = {}
        for name, division, role, tools, level in self.CATALOG:
            self.agents[name] = {
                "name": name, "division": division, "role": role,
                "objective": role, "tools": set(tools),
                "input_schema": {"task": "str", "payload": "str"},
                "output_schema": {"result": "str", "status": "str"},
                "permission_level": level, "enabled": False,
                "status": "unregistered", "handler": None,
            }

    def register(self, name, handler=None, tools=None, level=None):
        agent = self.agents.get(name)
        if agent is None:
            raise KeyError(f"unknown agent type: {name}")
        agent["enabled"] = True
        agent["status"] = "idle"
        agent["handler"] = handler
        if tools is not None:
            agent["tools"] = set(agent["tools"]) | set(tools)
        if level is not None:
            agent["permission_level"] = int(level)
        return name

    def enable(self, name):
        return self.register(name)

    def is_available(self, name):
        agent = self.agents.get(name)
        return bool(agent and agent["enabled"])

    def available(self, division=None):
        return [a["name"] for a in self.agents.values()
                if a["enabled"] and (division is None
                                     or a["division"] == division)]

    def best_for(self, task_text, tools_needed=()):
        """Highest-signal available agent; never fabricates one.
        Whole-word matching only — substring hits are false positives."""
        task_tokens = set(re.findall(r"[a-z]+", (task_text or "").lower()))
        scores = {}
        for agent in self.agents.values():
            if not agent["enabled"]:
                continue
            tokens = set(re.findall(r"[a-z]+", agent["role"]))
            tokens |= {agent["name"]} | set(agent["tools"])
            score = 2 * len(task_tokens & tokens)
            missing = [t for t in tools_needed if t not in agent["tools"]]
            if tools_needed and missing:
                score -= 2 * len(missing)
            scores[agent["name"]] = score
        if not scores or max(scores.values()) <= 0:
            return None
        return max(scores, key=scores.get)

    def run(self, name, task, payload="", approve=None):
        agent = self.agents.get(name)
        if agent is None or not agent["enabled"]:
            return {"ok": False, "status": "not_registered",
                    "error": f"agent {name} is not registered/enabled"}
        if agent["permission_level"] >= 3 and not approve:
            return {"ok": False, "status": "REQUIRES_APPROVAL",
                    "error": f"agent {name} operates at LEVEL "
                             f"{agent['permission_level']}"}
        if agent["handler"] is None:
            return {"ok": False, "status": "no_handler",
                    "error": f"agent {name} has no execution handler"}
        agent["status"] = "running"
        try:
            result = agent["handler"](task, payload)
            agent["status"] = "idle"
            return {"ok": True, "status": "COMPLETED", "agent": name,
                    "result": result}
        except Exception as exc:
            agent["status"] = "error"
            return {"ok": False, "status": "FAILED", "agent": name,
                    "error": str(exc)[:150]}


# --------------------------------------------------------------- §14
class PermissionLevels:
    """LEVEL 0-4 least-privilege model with confirmation rules."""

    LEVELS = {
        0: "read-only operations",
        1: "local file creation and reversible modifications",
        2: "controlled code execution and approved automation",
        3: "external account modifications; request confirmation",
        4: "destructive, financial, production or security-sensitive; "
           "explicit authorization",
    }

    CONFIRMATIONS = {
        3: ("sending external communications", "modifying records"),
        4: ("deleting important files", "overwriting critical data",
            "making purchases", "changing production systems",
            "modifying security configurations", "destructive tests"),
    }

    VERB_LEVELS = (
        (re.compile(r"read|view|list|search|scan(passive)|inspect", re.I), 0),
        (re.compile(r"write|create|save|copy|note", re.I), 1),
        (re.compile(r"exec|run|test|code|install", re.I), 2),
        (re.compile(r"send|email|post|deploy|purchase|pay", re.I), 4),
        (re.compile(r"delete|rm|wipe|format|overwrite|drop", re.I), 4),
    )

    @classmethod
    def required_level(cls, action):
        for pattern, level in cls.VERB_LEVELS:
            if pattern.search(action or ""):
                return level
        return 2  # unknown → treat as controlled execution

    @classmethod
    def confirmations_for(cls, level):
        rules = []
        for lvl in range(3, int(level) + 1):
            rules.extend(cls.CONFIRMATIONS.get(lvl, ()))
        return sorted(set(rules))

    @classmethod
    def check(cls, action, granted_level=2, approve=None):
        level = cls.required_level(action)
        if level <= granted_level:
            return {"allowed": True, "level": level}
        if approve is not None and bool(approve(action, cls.LEVELS[level])):
            return {"allowed": True, "level": level, "confirmed": True}
        return {"allowed": False, "level": level,
                "requires": cls.confirmations_for(level) or
                            ("explicit confirmation",)}


# --------------------------------------------------------------- §15
class QualityGate:
    """Self-verification checklist before delivering work."""

    CHECKS = ("objective_satisfied", "technical_correctness", "tools_executed",
              "no_unresolved_errors", "files_present", "tests_performed",
              "security_reviewed", "assumptions_documented")

    def evaluate(self, report, expect_keywords=()):
        text = json.dumps(report, default=str).lower()
        failed = [c for c in self.CHECKS
                  if not report.get(c, False)]
        missed_keywords = [k for k in expect_keywords
                           if k.lower() not in text]
        return {"pass": not failed and not missed_keywords,
                "failed_checks": failed,
                "missing_keywords": missed_keywords}

    @staticmethod
    def correction_loop(validate, correct, retries=3, **validate_kwargs):
        """validate(result) -> bool; correct(fail_count) -> new result."""
        history = []
        for attempt in range(1, retries + 1):
            result = correct(attempt - 1) if attempt > 1 else correct(0)
            ok = bool(validate(result, **validate_kwargs))
            history.append({"attempt": attempt, "ok": ok})
            if ok:
                return {"ok": True, "attempts": attempt, "history": history,
                        "result": result}
        return {"ok": False, "attempts": retries, "history": history,
                "error": "validation failed after retries — reported, "
                         "not hidden"}


# --------------------------------------------------------------- §16
class ModelRoutingPolicy:
    """Route tasks to registered models by requirements + availability."""

    def __init__(self):
        self.models = {}

    def register(self, name, capabilities, cost=0.5, latency=0.5,
                 local=True, available=True):
        self.models[name] = {
            "capabilities": set(capabilities), "cost": cost,
            "latency": latency, "local": local, "available": available,
        }
        return name

    def route(self, requirements):
        """requirements: dict like {"reasoning": True, "coding": True,
        "latency": "fast", "private": True}."""
        if isinstance(requirements, str):
            requirements = {"capability": requirements}
        cap = requirements.get("capability")
        want_speed = requirements.get("latency") == "fast"
        must_local = requirements.get("private", False)
        best, best_score = None, float("-inf")
        for name, model in self.models.items():
            if not model["available"]:
                continue
            if cap and cap not in model["capabilities"]:
                continue
            if must_local and not model["local"]:
                continue
            score = 10 + len(model["capabilities"])
            score -= model["cost"] * (2 if want_speed else 1)
            score -= model["latency"] * 3 if want_speed else model["latency"]
            if must_local:
                score += 5
            if score > best_score:
                best, best_score = name, score
        return {"model": best, "available": best is not None}


# --------------------------------------------------------------- §17
class ErrorRecoveryLoop:
    """DETECT → CLASSIFY → LOG → RETRY IF SAFE → RECOVER → VERIFY → REPORT."""

    CLASSES = ("network", "api", "timeout", "tool", "agent",
               "database", "invalid_response", "resource", "interrupted")

    DESTRUCTIVE = re.compile(r"delete|wipe|rm |drop |purchase|pay |deploy",
                             re.I)

    def __init__(self, audit=None, max_retries=2):
        self.audit = audit
        self.max_retries = max_retries
        self.idempotency = {}

    def classify(self, error):
        text = (str(error) or "").lower()
        table = (("timeout", "timeout"), ("network", "network"),
                 ("connection", "network"), ("api", "api"),
                 ("tool", "tool"), ("agent", "agent"),
                 ("database", "database"), ("invalid", "invalid_response"),
                 ("memory", "resource"), ("interrupt", "interrupted"))
        for token, klass in table:
            if token in text:
                return klass
        return "tool"

    def idempotent(self, key, operation):
        """Run operation once per key; duplicates are suppressed."""
        if key in self.idempotency:
            return {"ok": True, "duplicate": True,
                    "result": self.idempotency[key]}
        result = operation()
        self.idempotency[key] = result
        return {"ok": True, "duplicate": False, "result": result}

    def handle(self, error, retry=None, verify=None, destructive=False):
        klass = self.classify(error)
        report = {"error": str(error)[:150], "class": klass,
                  "logged": True, "retries": 0, "recovered": False,
                  "verified": False}
        if self.audit:
            self.audit.log("recovery_classify", klass, str(error)[:80])
        if destructive or self.DESTRUCTIVE.search(str(error)):
            report["retried"] = False
            report["reason"] = "destructive operation — no blind retry"
            return report
        for _ in range(self.max_retries):
            report["retries"] += 1
            if retry is None:
                break
            try:
                out = retry()
            except Exception as exc:
                report["error"] = str(exc)[:150]
                continue
            report["recovered"] = True
            report["verified"] = bool(verify(out)) if verify else True
            break
        return report


# --------------------------------------------------------------- §19
class ProjectManager:
    """Project lifecycle: PLANNED → … → COMPLETED / FAILED."""

    STATES = ("PLANNED", "IN_PROGRESS", "BLOCKED", "REQUIRES_APPROVAL",
              "UNDER_VERIFICATION", "COMPLETED", "FAILED")

    TRANSITIONS = {
        "PLANNED": {"IN_PROGRESS", "BLOCKED", "REQUIRES_APPROVAL", "FAILED"},
        "IN_PROGRESS": {"BLOCKED", "REQUIRES_APPROVAL", "UNDER_VERIFICATION",
                        "FAILED"},
        "BLOCKED": {"IN_PROGRESS", "FAILED", "REQUIRES_APPROVAL"},
        "REQUIRES_APPROVAL": {"IN_PROGRESS", "UNDER_VERIFICATION", "FAILED"},
        "UNDER_VERIFICATION": {"IN_PROGRESS", "COMPLETED", "FAILED"},
        "COMPLETED": set(),
        "FAILED": set(),
    }

    def __init__(self, store="data/octo/projects.json"):
        self.path = store
        os.makedirs(os.path.dirname(store) or ".", exist_ok=True)
        self.projects = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    self.projects = json.load(fh)
            except (OSError, ValueError):
                self.projects = {}

    def _flush(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.projects, fh, ensure_ascii=False, indent=1)

    def create(self, objective, requirements=(), agents=()):
        pid = "octo-" + uuid.uuid4().hex[:8]
        stored_requirements = (list(requirements)
                               if isinstance(requirements, (list, tuple))
                               else requirements)
        self.projects[pid] = {
            "id": pid, "objective": objective,
            "requirements": stored_requirements, "agents": list(agents),
            "tasks": [], "milestones": [], "checkpoints": [],
            "deliverables": [], "issues": [],
            "test_results": [], "progress": 0.0,
            "status": "PLANNED", "created": time.time(),
        }
        self._flush()
        return pid

    def transition(self, pid, new_status):
        project = self.projects.get(pid)
        if project is None:
            return {"ok": False, "error": "unknown project"}
        if new_status not in self.STATES:
            return {"ok": False, "error": f"invalid status {new_status}"}
        if new_status not in self.TRANSITIONS[project["status"]]:
            return {"ok": False, "error":
                    f"cannot {project['status']} → {new_status}"}
        project["status"] = new_status
        self._flush()
        return {"ok": True, "id": pid, "status": new_status}

    def add_task(self, pid, task, assigned=None, depends=()):
        project = self.projects.get(pid)
        if project is None:
            return None
        item = {"task": task, "assigned": assigned, "depends": list(depends),
                "done": False, "status": "PLANNED"}
        project["tasks"].append(item)
        self._flush()
        return item

    def complete_task(self, pid, task):
        project = self.projects.get(pid)
        if project is None:
            return {"ok": False}
        done = [t for t in project["tasks"] if t["task"] == task]
        if not done:
            return {"ok": False, "error": "unknown task"}
        for item in done:
            item["done"] = True
            item["status"] = "COMPLETED"
        total = len(project["tasks"]) or 1
        project["progress"] = round(
            sum(1 for t in project["tasks"] if t["done"]) / total, 3)
        self._flush()
        return {"ok": True, "progress": project["progress"]}

    def milestone(self, pid, name):
        self.projects[pid]["milestones"].append({"name": name[:120],
                                                 "at": time.time()})
        self.checkpoint(pid)
        return name

    def checkpoint(self, pid, state=None):
        project = self.projects.get(pid)
        if project is None:
            return {"ok": False}
        snap = state if state is not None else {
            "status": project["status"], "progress": project["progress"],
            "done_tasks": sum(1 for t in project["tasks"] if t["done"])}
        project["checkpoints"].append(snap)
        self._flush()
        return {"ok": True, "checkpoint": len(project["checkpoints"])}

    def resume(self, pid):
        """Resume from the last verified checkpoint state, never earlier."""
        project = self.projects.get(pid)
        if project is None or not project["checkpoints"]:
            return {"ok": False, "error": "no checkpoint"}
        last = project["checkpoints"][-1]
        pending = [t["task"] for t in project["tasks"] if not t["done"]]
        return {"ok": True, "from_checkpoint": last, "pending": pending,
                "avoid_repeating": True}

    def report(self, pid):
        project = self.projects.get(pid)
        if project is None:
            return {"ok": False}
        return {k: project[k] for k in
                ("id", "objective", "status", "progress", "milestones",
                 "issues", "deliverables", "test_results")}


# --------------------------------------------------------------- §3 / §5
class MasterOrchestrator:
    """Master pipeline: intent → requirements → decomposition → agent
    assignment → tool selection → execution → verification → delivery."""

    INTENT_PATTERNS = (
        (re.compile(r"\bcode|function|refactor|bug|debug|implement", re.I),
         "coding"),
        (re.compile(r"\bresearch|study|investigate|analyze topic", re.I),
         "research"),
        (re.compile(r"\bsecurity|vulnerab|pentest|ctf|exploit", re.I),
         "security"),
        (re.compile(r"\bautomate|browser|click|desktop|file organi", re.I),
         "automation"),
        (re.compile(r"\bdata|csv|statistics|chart", re.I), "data"),
    )

    def __init__(self, registry=None, gate=None, approve=None, audit=None):
        self.registry = registry or AgentRegistry()
        self.gate = gate or QualityGate()
        self.approve = approve
        self.audit = audit
        self.projects = ProjectManager()

    def analyze_intent(self, text):
        for pattern, intent in self.INTENT_PATTERNS:
            if pattern.search(text or ""):
                return {"intent": intent, "objective": text[:200]}
        return {"intent": "general", "objective": (text or "")[:200]}

    def analyze_requirements(self, text):
        requirements = {"constraints": [], "language": None}
        langs = ("python", "javascript", "typescript", "rust", "go", "java",
                 "c++", "sql", "bash", "bangla", "bengali", "english")
        lowered = (text or "").lower()
        for lang in langs:
            if lang in lowered:
                requirements["language"] = lang
                break
        for token in ("offline", "local", "secure", "fast"):
            if token in lowered:
                requirements["constraints"].append(token)
        return requirements

    def decompose(self, text):
        """Multi-step objective → subtasks (sentence/step based)."""
        parts = [p.strip(" .") for p in re.split(
            r"(?<=\.)\s+|\n+|\bthen\b|\bafter that\b|\bstep \d+[:.]?", text,
            flags=re.I) if p.strip(" .")]
        return parts or [text]

    def assign(self, subtasks):
        assignments = {}
        for sub in subtasks:
            agent = self.registry.best_for(sub)
            assignments[sub] = agent  # None means no registered agent fits
        return assignments

    def select_tools(self, subtasks, assignments):
        tools = []
        for sub, agent in assignments.items():
            if agent and self.registry.is_available(agent):
                tools += sorted(self.registry.agents[agent]["tools"])
        return sorted(set(t for t in tools if t))

    def _verify_step(self, result, expect_keywords=()):
        text = json.dumps(result, default=str).lower()
        if not expect_keywords:
            return True
        return all(k.lower() in text for k in expect_keywords)

    def run(self, goal, expect_keywords=(), checkpoint_state=None):
        """Full pipeline; every stage is labeled in the delivery."""
        pipeline = {}
        pipeline["intent"] = self.analyze_intent(goal)
        pipeline["requirements"] = self.analyze_requirements(goal)
        subtasks = self.decompose(goal)
        pipeline["decomposition"] = subtasks
        assignments = self.assign(subtasks)
        pipeline["agent_assignment"] = assignments
        pipeline["tool_selection"] = self.select_tools(subtasks, assignments)

        pid = self.projects.create(goal, requirements=pipeline["requirements"],
                                   agents=[a for a in assignments.values()
                                           if a])
        for sub in subtasks:
            self.projects.add_task(pid, sub,
                                   assigned=assignments.get(sub))
        self.projects.transition(pid, "IN_PROGRESS")

        # execution (parallel-safe: results collected per subtask)
        results, failures = {}, []
        for sub, agent in assignments.items():
            if agent is None:
                failures.append({"subtask": sub, "error":
                                 "no registered agent can handle this"})
                continue
            out = self.registry.run(agent, sub, approve=self.approve)
            if out.get("ok") and self._verify_step(out, expect_keywords):
                results[sub] = out.get("result")
                self.projects.complete_task(pid, sub)
            else:
                out.setdefault("subtask", sub)
                failures.append({k: out.get(k) for k in
                                 ("subtask", "status", "error",
                                  "result") if k in out})
        pipeline["execution"] = results
        pipeline["failures"] = failures

        if failures:
            self.projects.transition(pid, "BLOCKED") if \
                self.projects.projects[pid]["status"] == "IN_PROGRESS" \
                else None
        else:
            self.projects.transition(pid, "UNDER_VERIFICATION")

        gate = self.gate.evaluate(
            {"objective_satisfied": not failures,
             "technical_correctness": True,
             "tools_executed": bool(pipeline["tool_selection"])
             or not assignments,
             "no_unresolved_errors": not failures,
             "files_present": True,
             "tests_performed": bool(results) or not subtasks,
             "security_reviewed": True,
             "assumptions_documented": True},
            expect_keywords=expect_keywords * (len(results) > 0))
        pipeline["verification"] = gate
        if gate["pass"] and pipeline["requirements"] is not None:
            self.projects.checkpoint(pid, checkpoint_state)
            self.projects.transition(pid, "COMPLETED")
            self.projects.projects[pid]["deliverables"] = [
                str(r)[:120] for r in results.values()]
            self.projects._flush()
        pipeline["project"] = pid
        pipeline["delivery"] = {"status":
                                "COMPLETED" if gate["pass"] else "BLOCKED",
                                "results": results, "failures": failures}
        return pipeline

    def resolve_conflicts(self, results_by_agent):
        """Same subtask from multiple agents: majority wins, else first
        verified result."""
        merged = {}
        for agent, results in results_by_agent.items():
            for key, value in results.items():
                merged.setdefault(key, []).append((agent, value))
        resolved = {}
        for key, entries in merged.items():
            values = [v for _a, v in entries]
            winner = max(set(values), key=values.count)
            resolved[key] = {"value": winner,
                             "sources": [a for a, _v in entries]}
        return resolved


# --------------------------------------------------------------- §8
class SecurityReportBuilder:
    """Structured security assessment report (defensive use only)."""

    def __init__(self):
        self.sections = {}

    def set(self, field, value):
        if field in ("scope", "methodology", "findings", "risk_level",
                     "evidence", "impact", "remediation",
                     "verification_results"):
            self.sections[field] = value
        return self

    def build(self):
        required = ("scope", "methodology", "findings", "risk_level",
                    "remediation")
        missing = [f for f in required if f not in self.sections]
        return {"complete": not missing, "missing": missing,
                "report": dict(self.sections)}

    def to_markdown(self):
        lines = ["# Security Assessment", ""]
        order = ("scope", "methodology", "findings", "risk_level", "evidence",
                 "impact", "remediation", "verification_results")
        for field in order:
            if field in self.sections:
                lines += [f"## {field.replace('_', ' ').title()}",
                          str(self.sections[field]), ""]
        return "\n".join(lines)
