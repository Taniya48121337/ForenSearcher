@echo off
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. Steps and messages live in scripts\walker_run.py.
rem walker D1 v6 (2026-10-13): wk5 plus visible-only numbers, frame-exact compare for equal-length pairs, shot-list read, citation rules; round wk6.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python scripts\walker_run.py d1 --round wk6 %*
pause
