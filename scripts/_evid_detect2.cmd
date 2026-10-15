@echo off
rem DETECT v2 small A/B (10-10): 12 AI-generation samples, 27B + compare_source, vs evidF27/evidD27.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OLLAMA_NUM_PARALLEL=1
python scripts\_evid_detect2.py
pause
