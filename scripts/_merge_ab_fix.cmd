@echo off
rem 10-13: rerun the 14 EVID samples of the merged-catalog A/B after the evidM27a fixes (round evidM27b, EVID part only).
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OLLAMA_NUM_PARALLEL=1
set FOREN_MERGE_ROUND=evidM27b
set FOREN_MERGE_PARTS=evid
python scripts\_merge_ab.py
pause
