@echo off
rem 10-14: merged catalog, fix 4 (DETECT v2.4 output wording restored, report citation salvage) with --auto-tools on.
rem two EVID rounds, evidM27g then evidM27h. Close the window any time; double-click again to resume.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OLLAMA_NUM_PARALLEL=1
set FOREN_MERGE_PARTS=evid
set FOREN_MERGE_AUTO=on
set FOREN_MERGE_ROUND=evidM27g
python scripts\_merge_ab.py
set FOREN_MERGE_ROUND=evidM27h
python scripts\_merge_ab.py
pause
