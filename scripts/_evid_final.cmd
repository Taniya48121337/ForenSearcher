@echo off
rem EVID-Bench official baseline (10-07): samples 131-222 only (002-130 were used for diagnosis).
rem Runs Qwen3-VL-8B bf16 (about 11 h), then Qwen3.8-27B Q4 (about 9 h), each scored by Sonnet.
rem Safe to close and double-click again: finished runs are skipped. Only one model: set FOREN_FINAL_MODELS=27b
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. All steps and messages live in scripts\_evid_final.py.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OLLAMA_NUM_PARALLEL=1
python scripts\_evid_final.py
pause
