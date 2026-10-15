@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
    echo Install the desktop requirements in .venv before launching the workbench.
    echo See README.md for the installation commands.
    pause
    exit /b 1
)
start "Traffic Research Workbench" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0scripts\run_desktop.py"
