@echo off
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. Steps and messages live in scripts\walker_run.py.
rem walker revise (2026-10-13): answers judged negative in round wk6 are rewritten once from the checker's notes, then the round is re-verified, re-exported and compared.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python scripts\walker_run.py revise --round wk6 %*
pause
