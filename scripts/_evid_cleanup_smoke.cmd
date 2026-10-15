@echo off
rem Delete EVID smoke / pipeline-fix rounds (evidb0-3, evidb5, evidb7-11; 25 run folders). Permanent delete, not recycle bin.
rem Keeps evidb4, evidb6, evidb12, evidb12bf, evidq27a, scores, cases, search_snapshots.
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. Messages live in scripts\_cleanup_evid_smoke.py.
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python scripts\_cleanup_evid_smoke.py
echo.
echo Above is the preview. Close this window to cancel, or
pause
python scripts\_cleanup_evid_smoke.py --apply
pause
