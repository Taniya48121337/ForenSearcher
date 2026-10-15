@echo off
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. All steps and messages live in scripts\_launch.py.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python scripts\_launch_evid.py evid_full
pause
