@echo off
rem Source-finding experiment, one night (10-06): A6 oracle, A7 old-query replay, search-arm self-test,
rem then model arms A9, A5, A4, A8, A8G, A8T (in that order; arms already scored are skipped) on the same 20 EVID samples.
rem No deadline by default; set FOREN_NIGHT_UNTIL=09:00 to skip arms that cannot finish by then. Safe to close and rerun.
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. All steps and messages live in scripts\_evid_night.py.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OLLAMA_NUM_PARALLEL=1
rem set FOREN_NIGHT_UNTIL=09:00
python scripts\_evid_night.py
pause
