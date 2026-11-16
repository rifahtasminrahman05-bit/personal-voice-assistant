#!/bin/bash
# Octo Desktop launcher for macOS (double-clickable in Finder)
cd "$(dirname "$0")"
if [ ! -x "$(command -v python3)" ]; then
    echo "Python 3 not found. Install it from https://www.python.org/downloads/"
    exit 1
fi
exec python3 octo_desktop.py
