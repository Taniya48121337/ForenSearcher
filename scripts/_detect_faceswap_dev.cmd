@echo off
rem DETECT face_swap_scan dev check (10-10): per-person DFDC scores, swapped vs real persons.
rem Output: outputs\detect\faceswap_dev\summary.md
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python scripts\_detect_faceswap_dev.py %*
pause
