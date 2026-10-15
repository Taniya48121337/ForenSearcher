@echo off
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese.
rem r0 freeze check (2026-10-15): 1) compare specs / tool layer hash / verifier hash / teacher with specs\r0_freeze.json
rem 2) one B-family smoke run (videosham_4126) with the mock slot, no LLM, no verify; check run_config fields.
rem Safe to run while the wk14/wk15 batch is running (different round name r0smoke, B family does not search).
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python scripts\walker_run.py freeze --round r0smoke > _r0_freeze_smoke_log.txt 2>&1
set FOREN_ORACLE_SLOT=mock
python scripts\walker_run.py d1 --round r0smoke --cases videosham_4126 --no-verify >> _r0_freeze_smoke_log.txt 2>&1
findstr /C:"tool_layer_hash" /C:"gia_report_review_delay_s" /C:"render_profile" /C:"image_input_limit" /C:"teacher_model" /C:"teacher_role" /C:"\"ok\"" experiments\sft_data\runs\videosham_4126_r0smoke_1\run_config.json >> _r0_freeze_smoke_log.txt
type _r0_freeze_smoke_log.txt
pause
