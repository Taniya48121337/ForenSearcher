@echo off
rem Precision A/B (10-06): Qwen3-VL-8B bf16 vs Q4 on the same 20 EVID samples (round evidb12bf vs evidb12), Sonnet scoring.
rem First run downloads qwen3-vl:8b-instruct-bf16 (about 18 GB). Then about 2-3 hours. Safe to close and double-click again.
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. All steps and messages live in scripts\_launch_evid.py.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OLLAMA_NUM_PARALLEL=1
python scripts\_launch_evid.py evid_ab_bf16
pause
