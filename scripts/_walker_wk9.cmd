@echo off
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. Steps and messages live in the python scripts.
rem walker wk9 (2026-10-14), after the wk8 review: E family reads the earliest dated page and can cite date records,
rem A family uses only frames still in view for text_image, C family reports quote the transcript, search-task report
rem rules for the teacher. Only the D2 / D3 cases run (B family and v9 are unchanged since wk8).
rem First retries the GroundLie360 unzip (falls back to 7-Zip when Python cannot open vid.zip), so the video claim
rem cases can be built.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if not exist plan-harness-1\datasets\GroundLie360\videos_index.json python scripts\datasets_prepare.py g360 --zip "E:\Downloads\vid.zip"
python scripts\walker_d2_cases.py
python scripts\walker_run.py d1 --round wk9 --d2 --skip-b %*
pause
