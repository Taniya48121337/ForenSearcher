@echo off
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. Steps and messages live in scripts\walker_run.py.
rem walker D1 v2 (2026-10-12): same 9 cases as wk1 after the fixes (uncovered ranges, asr, visible numbers, answer prompt); round wk2.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python scripts\walker_run.py d1 --round wk2 %*
pause
