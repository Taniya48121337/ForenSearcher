@echo off
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. Steps and messages live in scripts\walker_run.py.
rem walker D1 v5 (2026-10-13): merged catalog hi@0.8.0 / gia@0.9.0 (compare_media, video_structure), answer checked against case GT with one evidence-bound rewrite, round wk5.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python scripts\walker_run.py d1 --round wk5 %*
pause
