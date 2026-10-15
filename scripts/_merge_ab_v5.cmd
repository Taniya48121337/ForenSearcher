@echo off
rem 10-14: merged catalog, fix 5 (tool records hold conclusions only, structure noise folded, auto compare only: --auto-tools compare).
rem two EVID rounds, evidM27i then evidM27j. Close the window any time; double-click again to resume.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OLLAMA_NUM_PARALLEL=1
set FOREN_MERGE_PARTS=evid
set FOREN_MERGE_AUTO=compare
set FOREN_MERGE_ROUND=evidM27i
python scripts\_merge_ab.py
set FOREN_MERGE_ROUND=evidM27j
python scripts\_merge_ab.py
pause
