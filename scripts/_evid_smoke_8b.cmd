@echo off
rem Fix plan P1 smoke on Qwen3-VL-8B (specs hi@0.7.0 / gia@0.8.0, English inputs, round evidb12), then Sonnet scoring.
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. All steps and messages live in scripts\_launch_evid.py.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set FOREN_EVID_MODEL_PROFILE=8b
python scripts\_launch_evid.py evid_smoke
pause
