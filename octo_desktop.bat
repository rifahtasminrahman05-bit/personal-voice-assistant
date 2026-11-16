@echo off
rem Octo Desktop launcher for Windows
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 octo_desktop.py
) else (
    python octo_desktop.py
)
if %errorlevel% neq 0 (
    echo.
    echo Octo Desktop failed to start. Check that Python 3 with Tkinter is installed.
    pause
)
