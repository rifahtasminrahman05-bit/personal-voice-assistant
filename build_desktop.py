# -*- coding: utf-8 -*-
"""Build a standalone Octo Desktop app with PyInstaller.

macOS:   python build_desktop.py          -> dist/OctoDesktop.app
Windows: python build_desktop.py          -> dist\\OctoDesktop\\OctoDesktop.exe
Linux:   python build_desktop.py          -> dist/OctoDesktop/OctoDesktop

Requires: pip install pyinstaller
"""
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "OctoDesktop"


def main():
    try:
        import PyInstaller
    except ImportError:
        sys.exit("PyInstaller missing: pip install pyinstaller")

    sep = ";" if platform.system() == "Windows" else ":"
    args = [
        sys.executable, "-m", "PyInstaller",
        "--name", NAME,
        "--noconfirm",
        "--windowed",
        "--add-data", os.path.join("data", "octo") + sep + "data/octo",
        "--add-data", os.path.join("core") + sep + "core",
        "--add-data", os.path.join("speech") + sep + "speech",
        "--hidden-import", "octo_core.desktop",
        "--collect-all", "octo",
        "--collect-all", "octo_core",
        os.path.join("octo_desktop.py"),
    ]
    if platform.system() == "Darwin":
        args.insert(1, "--windowed")
    print(" ".join(args))
    result = subprocess.run(args, cwd=HERE)
    if result.returncode != 0:
        sys.exit("build failed")
    print(f"\ndone: dist/{NAME}"
          + (".app" if platform.system() == "Darwin" else ""))


if __name__ == "__main__":
    main()
