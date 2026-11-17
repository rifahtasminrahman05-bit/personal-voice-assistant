# -*- coding: utf-8 -*-
"""Security & Governance Layer — AI safety for the Octo agent stack.

Prompt-injection defense, least-privilege tool permissions, sandboxed
expression execution, secrets redaction, output safety validation,
audit logging and a human-approval gateway.
"""
import ast
import datetime
import json
import operator
import os
import re
import threading

INJECTION_PATTERNS = (
    r"ignore (all |any )?((previous|prior|above) )?(instructions?|prompts?|rules?)",
    r"disregard (your |the |all )?(instructions?|rules?|guidelines)",
    r"reveal|show|print (me )?(your )?(system prompt|instructions|rules)",
    r"you are now|act as (if|though) you (are|were)|new persona",
    r"<\|?(im_start|system|endoftext)\|?>",
    r"পুরোনো নির্দেশ ভুলে|সিস্টেম প্রম্পট দেখাও",
)

SECRET_PATTERNS = (
    (r"(sk-[A-Za-z0-9]{20,})", "[REDACTED_KEY]"),
    (r"(ghp_[A-Za-z0-9]{36,})", "[REDACTED_TOKEN]"),
    (r"(gh_[A-Za-z0-9_]{30,})", "[REDACTED_TOKEN]"),
    (r"(?i)(api[_-]?key\s*[=:]\s*)\S{8,}", r"\1[REDACTED]"),
    (r"(?i)(token\s*[=:]\s*)\S{8,}", r"\1[REDACTED]"),
    (r"(?i)(password\s*[=:]\s*)\S{4,}", r"\1[REDACTED]"),
    (r"(-----BEGIN [A-Z ]*PRIVATE KEY-----)", "[REDACTED_PRIVATE_KEY]"),
)

SAFE_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
            ast.USub: operator.neg, ast.UAdd: operator.pos}


def detect_prompt_injection(text):
    """Return (risk_level, matched_patterns) for suspicious instructions."""
    matches = []
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.I):
            matches.append(pattern)
    risk = "high" if len(matches) >= 2 else "medium" if matches else "none"
    return risk, matches


def redact_secrets(text):
    """Secrets Protection: strip credentials from any text before logging."""
    for pattern, repl in SECRET_PATTERNS:
        text = re.sub(pattern, repl, text)
    return text


def sandbox_exec(expr, max_length=200):
    """Sandboxed Execution: evaluate whitelisted arithmetic only (AST-walk).

    No eval/exec, no names, no attribute access — a pure constant-folder.
    """
    if len(expr) > max_length:
        raise ValueError("expression too long for sandbox")
    for node in ast.walk(ast.parse(expr, mode="eval")):
        if isinstance(node, ast.Expression):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            continue
        if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPS:
            continue
        if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPS:
            continue
        if isinstance(node, (ast.operator, ast.unaryop)) and type(node) in SAFE_OPS:
            continue
        raise ValueError(f"blocked node in sandbox: {type(node).__name__}")
    return _fold(ast.parse(expr, mode="eval").body)


def _fold(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return SAFE_OPS[type(node.op)](_fold(node.left), _fold(node.right))
    if isinstance(node, ast.UnaryOp):
        return SAFE_OPS[type(node.op)](_fold(node.operand))
    raise ValueError("unreachable sandbox node")


class PermissionManager:
    """Tool Permission System: least-privilege allowlist for agent tools."""

    DEFAULT_ROLES = {
        "researcher": {"research", "deep", "websearch", "search", "answer", "now", "note", "memory"},
        "coder": {"calc", "convert", "answer", "search", "note"},
        "security": {"hashid", "passcheck", "logscan", "search", "answer"},
        "admin": None,
    }

    def __init__(self):
        self._grants = {}
        self._lock = threading.Lock()

    def grant(self, agent, tools):
        with self._lock:
            self._grants[agent] = set(tools)

    def revoke(self, agent):
        with self._lock:
            self._grants.pop(agent, None)

    def check(self, agent, tool):
        if agent in self._grants:
            return tool in self._grants[agent]
        role_tools = self.DEFAULT_ROLES.get(agent)
        return True if role_tools is None else tool in role_tools


class AuditLogger:
    """Audit Logging System: append-only JSONL trace of decisions and tool calls."""

    def __init__(self, path="data/octo/audit.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._lock = threading.Lock()

    def log(self, event, agent="system", detail=""):
        entry = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "agent": agent,
            "event": event,
            "detail": redact_secrets(str(detail))[:300],
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def tail(self, n=10):
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [json.loads(l) for l in lines[-n:]]


class ApprovalGateway:
    """Human-in-the-Loop Control + Human Approval Gateway.

    Sensitive operations require an explicit approval callback; without one
    they are denied (fail-closed).
    """

    SENSITIVE = {"write_file", "delete", "deploy", "self_modify", "exec", "network_post"}

    def __init__(self, approve=None):
        self.approve = approve

    def request(self, action, description=""):
        if action not in self.SENSITIVE:
            return True
        if self.approve is None:
            return False
        return bool(self.approve(action, description))


def validate_output(text, policies=("no_secrets", "max_length")):
    """Output Safety Validation: check generated content against policies."""
    problems = []
    if "no_secrets" in policies and redact_secrets(text) != text:
        problems.append("secrets detected in output")
    if "max_length" in policies and len(text) > 20000:
        problems.append("output exceeds 20000 chars")
    return {"safe": not problems, "problems": problems}
