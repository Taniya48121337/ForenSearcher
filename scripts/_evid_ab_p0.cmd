@echo off
rem Fix plan P0 A/B: Qwen3-VL-8B on 20 samples, round evidb6 vs evidb4, both scored by Sonnet (about 1-2 hours).
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. All steps and messages live in scripts\_launch_evid.py.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python scripts\_launch_evid.py evid_ab_p0
pause
