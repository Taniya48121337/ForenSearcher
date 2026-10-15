@echo off
rem DETECT dev features (10-10): per-frame backbone features + shot cuts on dev samples, for leave-one-video-out tests.
rem Output: outputs\detect\dev_features\*.npz
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python scripts\_detect_dev_features.py %*
pause
