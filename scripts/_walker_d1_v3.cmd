@echo off
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. Steps and messages live in scripts\walker_run.py.
rem walker D1 v3 (2026-10-12): gap-driven second round (GIA dispatches a targeted follow-up when a report names a fixable gap), round wk3.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python scripts\walker_run.py d1 --round wk3 %*
pause
