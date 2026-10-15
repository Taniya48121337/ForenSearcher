@echo off
rem EVID-Bench 131-222 rerun with DETECT (10-08): same setup as _evid_final.cmd plus run_cases --detect on.
rem Steps: DETECT unit tests, real-footage reference (if missing), 27B (evidD27) then 8B (evidD8), Sonnet scoring,
rem comparison table experiments\evid\scores\detect_131_222.md. Safe to close and double-click again.
rem Only one model: set FOREN_FINAL_MODELS=27b. ASCII only on purpose; messages live in scripts\_evid_detect.py.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OLLAMA_NUM_PARALLEL=1
python scripts\_evid_detect.py
pause
