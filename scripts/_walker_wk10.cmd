@echo off
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. Steps and messages live in the python scripts.
rem walker wk10 (2026-10-14), after the wk9 review: the walker now reads search results from the truncated preview
rem (wk9 1KkAwCyj7nI found its original with a 100% picture match but reported "not found"), searches once more when
rem the best match covers under 60% of the material, and does not count a query that failed with a provider error.
rem Runs the find-original cases (fmnv_a_*) and, once GroundLie360 is indexed, the video claim cases (groundlie360_v*).
rem The GroundLie360 unzip now keeps the videos before the truncation point of vid.zip and builds the index.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if not exist plan-harness-1\datasets\GroundLie360\videos_index.json python scripts\datasets_prepare.py g360 --zip "E:\Downloads\vid.zip"
python scripts\walker_d2_cases.py
python scripts\walker_run.py d1 --round wk10 --d2 --skip-b --only fmnv_a_ groundlie360_v %*
pause
