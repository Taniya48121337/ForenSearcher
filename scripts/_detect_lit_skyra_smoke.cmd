@echo off
rem Skyra smoke test (10-12): one chunk of video 152 with step timings, to see where the skyra stage stalls.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set LIT_SKYRA_SMOKE=1
python scripts\_detect_lit_probe.py --stage skyra
pause
