@echo off
rem DETECT compare_source offline check (10-10): dev samples vs their GT originals.
rem Output: outputs\detect\compare_dev\summary.md
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python scripts\_detect_compare_dev.py --only 155 160 163 167 171 201 205 213 216 220 %*
pause
