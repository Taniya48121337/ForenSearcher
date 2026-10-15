@echo off
rem 10-13: merged catalog after fix 2 (GIA numbered-answer check, wider text_image budget): two EVID rounds, evidM27c then evidM27d.
rem Close the window any time; double-click again to resume (finished runs are skipped).
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OLLAMA_NUM_PARALLEL=1
set FOREN_MERGE_PARTS=evid
set FOREN_MERGE_ROUND=evidM27c
python scripts\_merge_ab.py
set FOREN_MERGE_ROUND=evidM27d
python scripts\_merge_ab.py
pause
