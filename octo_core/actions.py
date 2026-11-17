# -*- coding: utf-8 -*-
"""Action & Tool System — safe, extensible capability layer
(Components Deep Dive).

ToolRegistry (dynamic discovery/registration), ActionEngine (permission +
safety checked execution with feedback loop and full action history),
GuardedDesktopAutomation (off by default, approval-gated), ExternalTools
(local scripts/plugins). 100% offline; every action audited.
"""
import re

from .security import ApprovalGateway, AuditLogger, PermissionManager, redact_secrets


class ToolRegistry:
    """Dynamic discovery, registration and management of available tools."""

    def __init__(self):
        self._tools = {}

    def register(self, name, fn, *, risky=False, description=""):
        self._tools[name] = {"fn": fn, "risky": risky, "description": description}
        return name

    def unregister(self, name):
        return self._tools.pop(name, None) is not None

    def get(self, name):
        return self._tools.get(name)

    def discover(self, query=""):
        """Tool discovery by keyword match over name + description."""
        q = query.lower()
        out = []
        for name, meta in sorted(self._tools.items()):
            if not q or q in name.lower() or q in meta["description"].lower():
                out.append({"name": name, "risky": meta["risky"],
                            "description": meta["description"]})
        return out

    def __contains__(self, name):
        return name in self._tools


class ActionEngine:
    """Controlled execution with permission and safety checks."""

    DANGEROUS = re.compile(
        r"\b(rm -rf|mkfs|dd if=|format c:|shutdown|reg delete|"
        r"drop table|curl .*\|\s*sh)\b", re.I)

    def __init__(self, registry=None, permissions=None, audit=None,
                 approval=None):
        self.registry = registry or ToolRegistry()
        self.permissions = permissions or PermissionManager()
        self.permissions.grant("octo-agent", set())
        self.audit = audit or AuditLogger()
        self.approval = approval
        self.history = []
        self.feedback = []

    def execute(self, tool, arg="", approve=None):
        entry = self.registry.get(tool)
        if entry is None:
            return {"ok": False, "error": f"unknown tool: {tool}"}
        if self.DANGEROUS.search(str(arg)):
            self.audit.log("action_blocked", tool, "dangerous pattern in argument")
            return {"ok": False, "error": "blocked: dangerous argument pattern"}
        if not self.permissions.check("octo-agent", tool):
            self.audit.log("action_denied", tool, "permission manager refused")
            return {"ok": False, "error": "permission denied (least privilege)"}
        if entry["risky"]:
            gate = ApprovalGateway(approve=approve or self.approval)
            if not gate.request(tool, f"risky tool '{tool}' with arg: {str(arg)[:80]}"):
                self.audit.log("action_unapproved", tool, "awaiting human approval")
                return {"ok": False, "error": "requires explicit human approval"}
        try:
            output = str(entry["fn"](arg))
        except Exception as exc:
            self.history.append({"tool": tool, "arg": str(arg)[:200],
                                 "ok": False, "error": str(exc)[:200]})
            self.audit.log("action_error", tool, str(exc)[:200])
            return {"ok": False, "error": str(exc)[:200]}
        self.history.append({"tool": tool, "arg": str(arg)[:200], "ok": True,
                             "output": output[:200]})
        self.audit.log("action_ok", tool, redact_secrets(output)[:200])
        return {"ok": True, "output": redact_secrets(output)[:400]}

    def approval_gateway(self):
        return ApprovalGateway()

    def observe_outcome(self, tool, success, note=""):
        """Feedback loop: outcomes inform future decisions."""
        self.feedback.append({"tool": tool, "success": success, "note": note[:120]})
        reliability = (sum(1 for f in self.feedback
                           if f["tool"] == tool and f["success"])
                       / max(sum(1 for f in self.feedback
                                 if f["tool"] == tool), 1))
        return {"tool": tool, "reliability": round(reliability, 2)}

    def log(self, n=20):
        return self.history[-n:]


class GuardedDesktopAutomation:
    """Guarded interaction with the local OS — disabled unless explicitly
    enabled AND each specific action is approval-gated."""

    def __init__(self, enabled=False, approval=None):
        self.enabled = enabled
        self.approval = approval
        self.executed = []

    def allow(self, approval_fn):
        self.enabled = True
        self.approval = approval_fn

    def run(self, action, arg=""):
        if not self.enabled:
            return {"ok": False, "error": "desktop automation is disabled"}
        if self.approval is None or not self.approval(f"{action} {arg}"):
            return {"ok": False, "error": "action not approved by user"}
        self.executed.append({"action": action, "arg": arg[:120]})
        return {"ok": True, "note": f"approved action '{action}' recorded "
                                    f"(real OS execution happens in the UI layer)"}


class ExternalTools:
    """Support for local scripts, APIs and plugins (local paths only)."""

    def __init__(self, allowed_root="."):
        self.allowed_root = allowed_root
        self.scripts = {}

    def register_script(self, name, path):
        import os
        real = os.path.realpath(path)
        root = os.path.realpath(self.allowed_root)
        if not real.startswith(root):
            return {"ok": False, "error": "script outside allowed root"}
        self.scripts[name] = real
        return {"ok": True, "name": name}

    def call(self, name, args=""):
        import subprocess
        path = self.scripts.get(name)
        if path is None:
            return {"ok": False, "error": f"unknown script: {name}"}
        try:
            proc = subprocess.run([path, str(args)], capture_output=True,
                                  text=True, timeout=10)
            return {"ok": proc.returncode == 0,
                    "output": proc.stdout.strip()[:400] or proc.stderr.strip()[:400]}
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "script timed out"}
        except OSError as exc:
            return {"ok": False, "error": str(exc)[:200]}
