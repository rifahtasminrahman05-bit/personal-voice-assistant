# -*- coding: utf-8 -*-
"""Sandboxed code execution for the coding agent.

Runs generated Python in a fresh subprocess with a hard timeout and
(On POSIX) memory/CPU limits via preexec limits. No network restriction is
possible portably, so untrusted code should only be run with user awareness.
"""
import os
import subprocess
import sys

MAX_OUTPUT = 5000


def run_code(code, timeout=5, stdin_data=""):
    """Execute Python source in an isolated subprocess.

    Returns {"ok": bool, "stdout": str, "stderr": str, "returncode": int,
    "timed_out": bool}.
    """
    preexec = None
    kwargs = {}
    if os.name == "posix":
        def preexec():
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
            resource.setrlimit(resource.RLIMIT_CPU, (timeout + 5, timeout + 5))
            os.setsid()
        kwargs["preexec_fn"] = preexec
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            input=stdin_data, capture_output=True, text=True,
            timeout=timeout, **kwargs)
        return {"ok": proc.returncode == 0,
                "stdout": proc.stdout[:MAX_OUTPUT],
                "stderr": proc.stderr[:MAX_OUTPUT],
                "returncode": proc.returncode, "timed_out": False}
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": f"timed out after {timeout}s",
                "returncode": -1, "timed_out": True}
    except Exception as e:
        return {"ok": False, "stdout": "", "stderr": str(e)[:MAX_OUTPUT],
                "returncode": -1, "timed_out": False}


def run_tests_snippet(code, tests, timeout=10):
    """Run generated code against assertion-style tests in one sandbox.

    `tests` is Python source containing asserts over the generated module.
    """
    harness = code + "\n\n# --- verification ---\n" + tests
    return run_code(harness, timeout=timeout)
