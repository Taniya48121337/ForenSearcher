@echo off
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. Steps and messages live in scripts\walker_run.py.
rem walker D1 (2026-10-12): 9 given-media cases, program walks + Codex fills report/answer, Sonnet verifier, compare with r0e/tp2.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python scripts\walker_run.py d1 %*
pause
