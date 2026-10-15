@echo off
rem DETECT dev probe (10-10): alignment labels + AIGI frame detectors + face identity on 18 EVID dev samples.
rem Output: outputs\detect\dev_probe\summary.md
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python scripts\_detect_dev_probe.py %*
pause
