# -*- coding: utf-8 -*-
"""Advanced Software Engineering Intelligence (features 31-36).

AutonomousDebugger, ArchitectureDesigner, CodeOptimizer,
DependencyRiskAnalyzer, DocumentationEngine, MigrationIntelligence.
"""
import ast
import os
import re


class AutonomousDebugger:
    """31. Autonomous Debugging Engine: traceback -> category + fix."""

    RULES = (
        ("ModuleNotFoundError", "missing dependency",
         "install the missing module or pin it in requirements"),
        ("ImportError", "bad import",
         "check the import path and package version"),
        ("KeyError", "missing dict key",
         "guard with .get() or validate input keys"),
        ("IndexError", "index out of range",
         "check sequence length before indexing"),
        ("TypeError", "type mismatch",
         "inspect argument types at the call site"),
        ("ValueError", "invalid value",
         "validate inputs before conversion"),
        ("AttributeError", "missing attribute",
         "verify the object type and add None checks"),
        ("ZeroDivisionError", "division by zero", "guard the denominator"),
        ("FileNotFoundError", "missing file",
         "check the path or create the file"),
        ("TimeoutError", "timeout",
         "increase the timeout or add retries"),
    )

    def diagnose(self, traceback_text):
        for err, category, fix in self.RULES:
            if err in traceback_text:
                line = next((l.strip() for l in traceback_text.splitlines()
                             if err in l), "")
                return {"error": err, "category": category,
                        "suggestion": fix, "evidence": line[:160]}
        return {"error": "unknown", "category": "unclassified",
                "suggestion": "collect more context around the failure",
                "evidence": ""}


class ArchitectureDesigner:
    """32. Architecture Design Intelligence: requirements -> module plan."""

    COMPONENTS = (
        (re.compile(r"\b(users?|log ?in|account|auth)\b", re.I), "auth-service"),
        (re.compile(r"\b(payment|billing|invoice)\b", re.I), "billing-service"),
        (re.compile(r"\b(chat|message|notify)\b", re.I), "messaging-service"),
        (re.compile(r"\b(report|analytics|dashboard)\b", re.I), "analytics-service"),
        (re.compile(r"\b(file|upload|storage|image)\b", re.I), "storage-service"),
        (re.compile(r"\b(api|mobile|web|client)\b", re.I), "api-gateway"),
    )

    def design(self, requirements):
        modules = [name for pattern, name in self.COMPONENTS
                   if pattern.search(requirements)]
        if modules and "api-gateway" not in modules:
            modules.insert(0, "api-gateway")
        modules.append("database")
        return {"modules": modules,
                "layers": ["interface", "services", "data"],
                "notes": "start as a monolith, split services when scale demands"}


class CodeOptimizer:
    """33. Code Optimization Engine: AST-based improvement suggestions."""

    def analyze(self, source):
        tree = ast.parse(source)
        tips = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                for sub in ast.walk(node):
                    if sub is node:
                        continue
                    if isinstance(sub, (ast.For, ast.While)):
                        tips.append({"line": node.lineno, "issue": "nested loop",
                                     "suggestion": "hoist work or use a lookup table"})
                    if isinstance(sub, ast.AugAssign) and isinstance(sub.op, ast.Add):
                        tips.append({"line": sub.lineno,
                                     "issue": "repeated += accumulation",
                                     "suggestion": "append to a list and join/extend once"})
            if (isinstance(node, ast.Compare)
                    and any(isinstance(c, (ast.List, ast.Tuple))
                            for c in node.comparators)):
                tips.append({"line": node.lineno,
                             "issue": "membership test against a literal list",
                             "suggestion": "use a set for O(1) membership"})
        unique = {(t["line"], t["issue"]): t for t in tips}
        return {"suggestions": sorted(unique.values(), key=lambda t: t["line"])}


class DependencyRiskAnalyzer:
    """34. Dependency Risk Analyzer: flag risky package declarations."""

    KNOWN = {"requests": "low", "numpy": "low", "torch": "medium",
             "flask": "low", "pytest": "low", "pytesseract": "low"}

    def assess(self, requirements):
        findings = []
        for line in requirements:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name = re.split(r"[=<>!~\[]", line)[0].strip().lower()
            risk, reasons = "low", []
            if name not in self.KNOWN and len(name) < 5:
                risk = "high"
                reasons.append("possible typosquat / unrecognized package")
            if "==" not in line:
                risk = risk if risk == "high" else "medium"
                reasons.append("unpinned version")
            findings.append({"package": name, "risk": risk, "reasons": reasons})
        return {"dependencies": findings,
                "high_risk": [f["package"] for f in findings
                              if f["risk"] == "high"]}


class DocumentationEngine:
    """35. Automated Documentation Engine: AST -> markdown API docs."""

    def document_module(self, path, module_name=None):
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        module_name = module_name or os.path.splitext(os.path.basename(path))[0]
        lines = [f"# {module_name}", ""]
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                args = [a.arg for a in node.args.args]
                doc = ast.get_docstring(node) or "(no docstring)"
                lines += [f"## {node.name}({', '.join(args)})", "", doc, ""]
            elif isinstance(node, ast.ClassDef):
                lines += [f"## class {node.name}", ""]
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and not sub.name.startswith("_"):
                        args = [a.arg for a in sub.args.args if a.arg != "self"]
                        doc = ast.get_docstring(sub) or "(no docstring)"
                        lines += [f"### {node.name}.{sub.name}({', '.join(args)})",
                                  "", doc, ""]
        return "\n".join(lines)


class MigrationIntelligence:
    """36. Software Migration Intelligence: old -> new stack plan."""

    MIGRATIONS = {
        ("python2", "python3"): ["run 2to3 for syntax", "replace print statements",
                                 "update str/unicode handling",
                                 "verify dependencies support py3"],
        ("mysql", "postgresql"): ["dump schema + data",
                                  "map types (AUTO_INCREMENT -> SERIAL)",
                                  "rewrite dialect-specific SQL",
                                  "repoint connection strings"],
        ("monolith", "microservices"): ["identify bounded contexts",
                                        "extract services behind an API",
                                        "introduce service discovery",
                                        "split data stores per service"],
    }

    def plan(self, source, target):
        steps = self.MIGRATIONS.get((source.lower(), target.lower()))
        if not steps:
            steps = ["inventory source features",
                     "map to target equivalents",
                     "migrate incrementally",
                     "verify parity with tests"]
        return {"from": source, "to": target, "steps": list(steps)}
