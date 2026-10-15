@echo off
rem Check reverse-image-search API keys in .local\ (free checks).
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python scripts\_check_image_keys.py
pause
