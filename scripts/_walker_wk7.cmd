@echo off
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. Steps and messages live in the python scripts.
rem walker wk7 (2026-10-13): B family gets OCR on text-shaped changed regions; D2 adds A (find original) and C (claim check)
rem cases built from FMNV and GroundLie360 by scripts\walker_d2_cases.py (downloads originals with yt-dlp, cuts with PyAV).
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python scripts\walker_d2_cases.py
if errorlevel 1 echo D2 cases were not built; the B family still runs. See _walker_d2*.txt
if exist experiments\sft_data\walker_d2_cases.json (
  python scripts\walker_run.py d1 --round wk7 --d2 %*
) else (
  python scripts\walker_run.py d1 --round wk7 %*
)
pause
