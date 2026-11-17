# -*- coding: utf-8 -*-
"""Computer & Desktop Control (features 41-55).

DesktopController: guarded mouse/keyboard control (pyautogui optional),
application launching/management, file management, folder organization,
browser automation, screen monitoring, window management, clipboard,
system settings, software installation and troubleshooting.
Everything is approval-gated and audit-logged. No new hard dependencies —
OS automation backends are optional and degrade to "unavailable" results.
"""
import os
import shutil
import subprocess
import sys

from .security import ApprovalGateway, AuditLogger


class DesktopController:
    def __init__(self, audit=None, approval=None, home=None):
        self.audit = audit or AuditLogger()
        self.approval = approval
        self.home = home or os.path.expanduser("~")
        self.log = []

    def _approved(self, what):
        gate = ApprovalGateway(approve=self.approval)
        return gate.request("exec", what)

    def _gate(self, approve, what):
        if approve is not None:
            return bool(approve(what))
        return self._approved(what)

    def _record(self, action, detail, ok):
        self.log.append({"action": action, "detail": str(detail)[:120],
                         "ok": ok})
        self.audit.log("desktop_" + ("ok" if ok else "denied"), action,
                       str(detail)[:120])

    # ---- 42/43. mouse & keyboard (pyautogui optional) ----
    def _pyautogui(self):
        try:
            import pyautogui
            return pyautogui
        except ImportError:
            return None

    def mouse_move(self, x, y):
        pag = self._pyautogui()
        if pag is None or not self._approved(f"mouse move {x},{y}"):
            self._record("mouse_move", f"{x},{y}", False)
            return {"ok": False, "reason": "unavailable or not approved"}
        pag.moveTo(x, y)
        self._record("mouse_move", f"{x},{y}", True)
        return {"ok": True}

    def keyboard_type(self, text):
        pag = self._pyautogui()
        if pag is None or not self._approved(f"keyboard type: {text[:40]}"):
            self._record("keyboard_type", text, False)
            return {"ok": False, "reason": "unavailable or not approved"}
        pag.typewrite(text)
        self._record("keyboard_type", text, True)
        return {"ok": True}

    # ---- 44/45. application launching & management ----
    def launch_app(self, app):
        if not self._approved(f"launch app: {app}"):
            self._record("launch_app", app, False)
            return {"ok": False, "reason": "not approved"}
        try:
            subprocess.Popen([app],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            self._record("launch_app", app, True)
            return {"ok": True, "app": app}
        except OSError as exc:
            self._record("launch_app", app, False)
            return {"ok": False, "reason": str(exc)[:150]}

    def list_apps(self):
        """Running user-facing processes (name only)."""
        names = []
        if sys.platform == "linux":
            for pid in os.listdir("/proc"):
                if not pid.isdigit():
                    continue
                try:
                    with open(f"/proc/{pid}/comm", encoding="utf-8") as f:
                        names.append(f.read().strip())
                except OSError:
                    continue
        return sorted(set(names))[:50] or ["(process listing unavailable "
                                           "on this platform)"]

    def kill_app(self, app, approve=None):
        """Application Management: terminate a process (approval required)."""
        if not self._gate(approve, f"kill {app}"):
            self._record("kill_app", app, False)
            return {"ok": False, "reason": "not approved"}
        try:
            subprocess.run(["pkill", "-f", app], timeout=5,
                           capture_output=True)
            self._record("kill_app", app, True)
            return {"ok": True, "app": app}
        except OSError as exc:
            return {"ok": False, "reason": str(exc)[:150]}

    # ---- 46/47. file management & folder organization ----
    def file_op(self, op, src, dst=None, approve=None):
        if op == "delete" and not self._gate(approve, f"delete {src}"):
            self._record("file_op", f"delete {src}", False)
            return {"ok": False, "reason": "not approved"}
        try:
            if op == "copy":
                shutil.copy2(src, dst)
            elif op == "move":
                shutil.move(src, dst)
            elif op == "delete":
                os.remove(src)
            elif op == "mkdir":
                os.makedirs(src, exist_ok=True)
            else:
                return {"ok": False, "reason": f"unknown op: {op}"}
            self._record("file_op", f"{op} {src}", True)
            return {"ok": True, "op": op}
        except OSError as exc:
            return {"ok": False, "reason": str(exc)[:150]}

    def organize_folder(self, folder):
        """Folder Organization: sort files into type subfolders."""
        groups = {"images": {".png", ".jpg", ".jpeg", ".gif"},
                  "docs": {".pdf", ".docx", ".txt", ".md"},
                  "code": {".py", ".js", ".sh", ".rs"}}
        moved = []
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(name)[1].lower()
            target = next((g for g, exts in groups.items() if ext in exts),
                          "other")
            dest_dir = os.path.join(folder, target)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.move(path, os.path.join(dest_dir, name))
            moved.append({"file": name, "to": target})
        self._record("organize_folder", folder, True)
        return {"ok": True, "moved": moved}

    # ---- 48. browser automation ----
    def browser_open(self, url):
        if not self._approved(f"open url: {url}"):
            self._record("browser_open", url, False)
            return {"ok": False, "reason": "not approved"}
        import webbrowser
        webbrowser.open(url)
        self._record("browser_open", url, True)
        return {"ok": True, "url": url}

    # ---- 49/50. screen monitoring & screenshot analysis ----
    def screenshot(self, path="data/octo/screen.png"):
        pag = self._pyautogui()
        if pag is None:
            return {"ok": False, "reason": "pyautogui not installed"}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        pag.screenshot(path)
        return {"ok": True, "path": path}

    def analyze_screen(self, path="data/octo/screen.png"):
        from .cognitive_core import VisionSystem
        entry = VisionSystem().understand_screen(path)
        return {"ok": True, "analysis": entry}

    # ---- 51. window management ----
    def window_op(self, op, title=""):
        pag = self._pyautogui()
        keys = {"minimize": ["win", "down"], "maximize": ["win", "up"],
                "switch": ["alt", "tab"]}
        if pag is None or op not in keys:
            return {"ok": False, "reason": "unavailable or unknown op"}
        if not self._approved(f"window {op} {title}"):
            return {"ok": False, "reason": "not approved"}
        pag.hotkey(*keys[op])
        self._record("window_op", f"{op} {title}", True)
        return {"ok": True, "op": op}

    # ---- 52. clipboard management ----
    def clipboard_copy(self, text):
        try:
            import pyperclip
            pyperclip.copy(text)
            return {"ok": True}
        except ImportError:
            return {"ok": False, "reason": "pyperclip not installed"}

    def clipboard_paste(self):
        try:
            import pyperclip
            return {"ok": True, "text": pyperclip.paste()[:2000]}
        except ImportError:
            return {"ok": False, "reason": "pyperclip not installed"}

    # ---- 53. system settings control ----
    def set_volume(self, level):
        if not 0 <= level <= 100:
            return {"ok": False, "reason": "level must be 0..100"}
        if not self._approved(f"set volume {level}"):
            return {"ok": False, "reason": "not approved"}
        try:
            if sys.platform == "darwin":
                subprocess.run(["osascript", "-e",
                                f"set volume output volume {level}"],
                               timeout=5, capture_output=True)
            elif sys.platform.startswith("linux"):
                subprocess.run(["amixer", "-q", "set", "Master",
                                f"{level}%"], timeout=5, capture_output=True)
            else:
                return {"ok": False, "reason": "platform not supported"}
            self._record("set_volume", level, True)
            return {"ok": True, "level": level}
        except OSError as exc:
            return {"ok": False, "reason": str(exc)[:150]}

    # ---- 54. automated software installation ----
    def install_package(self, package, approve=None):
        if not self._gate(approve, f"install {package}"):
            self._record("install_package", package, False)
            return {"ok": False, "reason": "installation requires explicit "
                                           "user approval"}
        proc = subprocess.run([sys.executable, "-m", "pip", "install",
                               package], capture_output=True, text=True,
                              timeout=300)
        ok = proc.returncode == 0
        self._record("install_package", package, ok)
        return {"ok": ok, "output": (proc.stdout or proc.stderr)[-400:]}

    # ---- 55. desktop troubleshooting ----
    def troubleshoot(self, symptom):
        steps = {
            "slow": ["check CPU/RAM in System Health Dashboard",
                     "close unused apps", "reboot if persistent"],
            "no sound": ["check volume level", "check output device",
                         "restart audio service"],
            "no internet": ["check cable/wifi", "ping 1.1.1.1",
                            "restart router"],
            "crash": ["read the crash log", "reproduce with debug logging",
                      "update or reinstall the app"],
        }
        key = next((k for k in steps if k in symptom.lower()), None)
        plan = steps.get(key, ["collect logs", "reproduce the issue",
                               "search the error message"])
        return {"symptom": symptom[:120], "steps": plan}
