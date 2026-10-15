@echo off
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. Steps and messages live in the python scripts.
rem walker wk10 with the Codex model gpt-6.1-sol as teacher (2026-10-14): gpt-6-astra is rate-limited (429).
rem Same cases and code as _walker_wk10.cmd (find-original fmnv_a_* and GroundLie360 video claims groundlie360_v*);
rem the teacher model is written into each run_config.json (walker.teacher_model). If Codex refuses the model,
rem the preflight stops before any case runs.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set FOREN_CODEX_MODEL=gpt-6.1-sol
if not exist plan-harness-1\datasets\GroundLie360\videos_index.json python scripts\datasets_prepare.py g360 --zip "E:\Downloads\vid.zip"
python scripts\walker_d2_cases.py
python scripts\walker_run.py d1 --round wk10 --d2 --skip-b --only fmnv_a_ groundlie360_v %*
pause
