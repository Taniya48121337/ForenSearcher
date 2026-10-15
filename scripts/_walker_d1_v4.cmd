@echo off
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. Steps and messages live in scripts\walker_run.py.
rem walker D1 v4 (2026-10-12): follow-up programs per gap type (windowed align for correspondence extents, both-side frames, range diff), answers restate findings, round wk4.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python scripts\walker_run.py d1 --round wk4 %*
pause
