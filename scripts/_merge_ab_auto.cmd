@echo off
rem 10-14: merged catalog with --auto-tools on (auto compare_media on a content_match source, auto video_structure at ingest)
rem two EVID rounds, evidM27e then evidM27f. Close the window any time; double-click again to resume.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OLLAMA_NUM_PARALLEL=1
set FOREN_MERGE_PARTS=evid
set FOREN_MERGE_AUTO=on
set FOREN_MERGE_ROUND=evidM27e
python scripts\_merge_ab.py
set FOREN_MERGE_ROUND=evidM27f
python scripts\_merge_ab.py
pause
