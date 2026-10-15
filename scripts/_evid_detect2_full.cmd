@echo off
rem DETECT v2.1 full run (only after the 12-sample A/B confirms): EVID 131-222 test samples, 27B + DETECT v2, round evidC27F.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OLLAMA_NUM_PARALLEL=1
set FOREN_DETECT2_FULL=1
python scripts\_evid_detect2.py
pause
