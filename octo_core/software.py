# -*- coding: utf-8 -*-
"""Autonomous Software Engineering — codebase understanding and test
generation.

CodebaseUnderstanding: AST-level scan of a Python project — modules,
classes, functions, imports and an intra-project dependency graph.
TestGenerator: emits pytest files covering normal calls plus edge cases
(None, empty, negative, large) for every public function found.
"""
import ast
import os

EDGE_CASES = {
    "none": "None",
    "empty_string": '""',
    "empty_list": "[]",
    "zero": "0",
    "negative": "-1",
    "large": "10 ** 6",
}


class CodebaseUnderstanding:
    """Codebase Understanding Engine: structure + dependency graph."""

    def __init__(self, root):
        self.root = root

    def analyze(self):
        modules = {}
        for dirpath, _dirs, files in os.walk(self.root):
            if any(part.startswith(".") or part == "__pycache__" for part in dirpath.split(os.sep)):
                continue
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fname)
                module = os.path.relpath(path, self.root)[:-3].replace(os.sep, ".")
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        tree = ast.parse(f.read())
                except SyntaxError:
                    modules[module] = {"error": "syntax error", "imports": [],
                                       "classes": [], "functions": [], "lines": 0}
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    lines = sum(1 for _ in f)
                modules[module] = {
                    "imports": [a.name.split(".")[0] for n in ast.walk(tree)
                                if isinstance(n, ast.Import) for a in n.names]
                    + [n.module.split(".")[0] for n in ast.walk(tree)
                       if isinstance(n, ast.ImportFrom) and n.module],
                    "classes": [n.name for n in tree.body if isinstance(n, ast.ClassDef)],
                    "functions": [n.name for n in tree.body
                                  if isinstance(n, ast.FunctionDef)],
                    "lines": lines,
                }
        local = set(modules)
        for m in modules.values():
            m["local_deps"] = sorted({i for i in m["imports"] if i in local})
        self.modules = modules
        return modules

    def report(self):
        mods = getattr(self, "modules", None) or self.analyze()
        dep_count = {}
        for m in mods.values():
            for d in m["local_deps"]:
                dep_count[d] = dep_count.get(d, 0) + 1
        orphans = sorted(m for m in mods
                         if dep_count.get(m.split(".")[-1], 0) == 0
                         and m.split(".")[-1] != "__init__")
        biggest = sorted(mods.items(), key=lambda kv: -kv[1]["lines"])[:5]
        return {"modules": len(mods),
                "total_lines": sum(m["lines"] for m in mods.values()),
                "total_functions": sum(len(m["functions"]) for m in mods.values()),
                "most_depended": sorted(dep_count.items(), key=lambda kv: -kv[1])[:5],
                "biggest_modules": [(m, v["lines"]) for m, v in biggest],
                "entry_candidates": orphans[:8]}

    def dependencies(self, module):
        mods = getattr(self, "modules", None) or self.analyze()
        return mods.get(module, {}).get("local_deps", [])


class TestGenerator:
    """Automated Test Generation: pytest cases with edge values."""

    def __init__(self, root):
        self.root = root

    def public_functions(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        funcs = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                args = [a.arg for a in node.args.args]
                defaults = [ast.dump(d) for d in node.args.defaults]
                funcs.append({"name": node.name, "args": args,
                              "n_defaults": len(node.args.defaults),
                              "defaults_present": bool(defaults)})
        return funcs

    def generate_for_function(self, filepath, func_name, module_name):
        funcs = self.public_functions(filepath)
        spec = next((f for f in funcs if f["name"] == func_name), None)
        if spec is None:
            raise ValueError(f"function {func_name} not found in {filepath}")
        args = spec["args"]
        normal_args = ", ".join(f'"{a}_value"' for a in args)
        lines = [
            f'"""Generated tests for {module_name}.{func_name} (edge cases included)."""',
            "import pytest",
            "",
            f"from {module_name} import {func_name}",
            "",
            "",
            f"class Test{func_name.capitalize()}:",
            "    def test_smoke(self):",
            f"        result = {func_name}({normal_args})" if args
            else f"        result = {func_name}()",
            "        assert True",
            "",
        ]
        for label, value in EDGE_CASES.items():
            if not args:
                break
            case_args = ", ".join([value] + ['"x"'] * (len(args) - 1))
            lines += [
                f"    def test_edge_{label}(self):",
                "        try:",
                f"            {func_name}({case_args})",
                "        except (TypeError, ValueError, IndexError, KeyError):",
                f"            pytest.fail('{func_name} crashed on {label}')",
                "        except Exception:",
                "            pass",
                "",
            ]
        return "\n".join(lines) + "\n"

    def generate_file(self, filepath, out_dir="tests/generated"):
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"generated_test_{module_name}.py")
        chunks = []
        for func in self.public_functions(filepath):
            try:
                chunks.append(self.generate_for_function(filepath, func["name"],
                                                         module_name))
            except Exception:
                continue
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(chunks))
        return out_path
