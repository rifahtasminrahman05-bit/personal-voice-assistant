# -*- coding: utf-8 -*-
"""Autonomous Agent & Execution Architecture.

AgentOrchestrator coordinates specialized agents (researcher, coder, security)
over the existing octo.agi tool set, with dynamic task decomposition,
agent-to-agent messaging, execution verification, long-horizon state and a
human-in-the-loop approval hook. All tool calls pass through the security
layer's permission checks and audit log.
"""
import datetime
import re
import threading

from octo.agi import TOOLS, plan

from .cognitive import decompose, reflect
from .infrastructure import Scheduler
from .security import AuditLogger, PermissionManager, redact_secrets


def tool_code(code):
    """Sandboxed code execution tool — used by the coding agent."""
    from .sandbox_code import run_code
    result = run_code(code, timeout=5)
    out = result["stdout"].strip() or result["stderr"].strip() or "(no output)"
    status = "ok" if result["ok"] else "failed"
    return f"[{status}] {out[:400]}"


TOOLS_EXT = dict(TOOLS, code=tool_code)


class Message:
    """Agent-to-Agent Communication: structured delegation envelope."""

    def __init__(self, sender, receiver, task, payload="", reply=None):
        self.sender = sender
        self.receiver = receiver
        self.task = task
        self.payload = payload
        self.reply = reply
        self.ts = datetime.datetime.now().isoformat(timespec="seconds")

    def as_dict(self):
        return {"from": self.sender, "to": self.receiver, "task": self.task,
                "payload": self.payload[:200], "reply": (self.reply or "")[:200]}


class Agent:
    def __init__(self, name, role, skills):
        self.name = name
        self.role = role
        self.skills = set(skills)

    def can_handle(self, task):
        text = task.lower()
        return any(re.search(skill, text) for skill in self.skills)


class ExecutionVerifier:
    """Execution Verification Engine: did the action produce a usable result?"""

    @staticmethod
    def verify(output, expect_keywords=()):
        out = str(output)
        lowered = out.lower()
        if not out.strip():
            return {"ok": False, "reason": "empty output"}
        if "failed" in lowered or "error" in lowered or "could not" in lowered:
            return {"ok": False, "reason": out[:120]}
        missing = [k for k in expect_keywords if k.lower() not in lowered]
        if missing:
            return {"ok": False, "reason": f"missing keywords: {missing}"}
        return {"ok": True, "reason": "output verified"}


class AgentOrchestrator:
    """Coordinates agents, decomposes goals, routes tasks, verifies results."""

    def __init__(self, approve=None, audit=None, max_steps=8):
        self.agents = {
            "researcher": Agent("researcher", "research & synthesis",
                                (r"research|deep|find|study|গবেষণা|খোঁজ",
                                 r"web|search|সার্চ")),
            "coder": Agent("coder", "code, math & conversion",
                           (r"calc|convert|code|function|debug|program|হিসাব",)),
            "security": Agent("security", "defensive security analysis",
                              (r"hash|password|log|cve|brute|নিরাপত্তা",)),
        }
        self.permissions = PermissionManager()
        self.audit = audit or AuditLogger()
        from .security import ApprovalGateway
        self.gateway = ApprovalGateway(approve)
        self.verifier = ExecutionVerifier()
        self.max_steps = max_steps
        self.state = {}
        self._lock = threading.Lock()

    def route(self, task):
        for agent in self.agents.values():
            if agent.can_handle(task):
                return agent.name
        return "researcher"

    def delegate(self, sender, receiver, task, payload=""):
        msg = Message(sender, receiver, task, payload)
        self.audit.log("delegate", sender, f"{receiver}: {task[:80]}")
        if not self.permissions.check(receiver, "answer"):
            msg.reply = f"denied: {receiver} lacks permission"
            return msg
        try:
            msg.reply = str(TOOLS.get("search", lambda a: "no tool")(task))[:400]
        except Exception as e:
            msg.reply = f"delegation failed: {e}"
        return msg

    def _step_allowed(self, agent, tool, arg):
        if not self.permissions.check(agent, tool):
            return False, f"permission denied: {agent}/{tool}"
        if tool in self.gateway.SENSITIVE:
            if not self.gateway.request(tool, f"{agent} wants to run {tool}({arg[:60]})"):
                return False, f"human approval denied for {tool}"
        return True, ""

    def execute(self, goal, verbose=True):
        """Long-Horizon Task Execution: decompose -> route -> tools -> verify -> reflect."""
        self.audit.log("goal", "orchestrator", goal)
        subtasks = decompose(goal)
        trace = []
        for i, sub in enumerate(subtasks[:self.max_steps]):
            agent_name = self.route(sub["task"])
            steps = plan(sub["task"])
            for tool, arg in steps:
                allowed, reason = self._step_allowed(agent_name, tool, arg)
                if not allowed:
                    trace.append((tool, reason))
                    continue
                try:
                    result = str(TOOLS_EXT[tool](arg))
                except Exception as e:
                    result = f"tool '{tool}' failed: {e}"
                verdict = self.verifier.verify(result)
                self.audit.log("tool_call", agent_name, f"{tool}: {result[:120]}")
                trace.append((tool, result))
                with self._lock:
                    self.state[f"subtask:{i}"] = {"agent": agent_name, "tool": tool,
                                                  "ok": verdict["ok"]}
                if verbose:
                    print(f"[{agent_name}] {tool}: {result[:100]}")
                if tool == "answer":
                    break
        reflection = reflect(trace)
        self.audit.log("reflection", "orchestrator", str(reflection)[:200])
        return trace, reflection

    def execute_parallel(self, goal, verbose=False):
        """Parallel Task Processing: independent subtasks run concurrently,
        each routed to its own agent. Returns merged trace + reflection."""
        self.audit.log("goal_parallel", "orchestrator", goal)
        subtasks = decompose(goal)
        scheduler = Scheduler(workers=min(4, max(1, len(subtasks))))

        def run_sub(sub):
            agent_name = self.route(sub["task"])
            trace = []
            for tool, arg in plan(sub["task"])[:3]:
                allowed, reason = self._step_allowed(agent_name, tool, arg)
                if not allowed:
                    trace.append((tool, reason))
                    continue
                try:
                    result = str(TOOLS_EXT[tool](arg))
                except Exception as e:
                    result = f"tool '{tool}' failed: {e}"
                self.audit.log("tool_call", agent_name, f"{tool}: {result[:100]}")
                trace.append((tool, result))
                if tool in ("answer", "code"):
                    break
            return agent_name, trace

        for sub in subtasks:
            scheduler.submit(run_sub, sub)
        scheduler.shutdown()

        merged = []
        for status, payload in scheduler.poll():
            if status == "ok":
                agent_name, trace = payload
                for tool, out in trace:
                    merged.append((f"{agent_name}/{tool}", out))
                    if verbose:
                        print(f"[{agent_name}] {tool}: {out[:100]}")
        reflection = reflect(merged)
        self.audit.log("reflection_parallel", "orchestrator", str(reflection)[:200])
        return merged, reflection
