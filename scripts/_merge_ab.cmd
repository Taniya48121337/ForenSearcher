@echo off
rem DETECT merged into SEARCH / INSPECT (10-12): small A/B. EVID 14 samples (evidM27a) + VideoSham 5 / VCSL 4 (abO27 vs abM27).
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OLLAMA_NUM_PARALLEL=1
python scripts\_merge_ab.py
pause
