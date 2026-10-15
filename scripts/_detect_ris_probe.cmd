@echo off
rem Reverse image search recall probe (10-12): Google Cloud Vision Web Detection on dev keyframes.
rem Put the API key in .local\google_vision_key.txt first. Output: outputs\detect\ris_probe\summary.md
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python scripts\_detect_ris_probe.py %*
pause
