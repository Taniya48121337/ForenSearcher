@echo off
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. Messages live in scripts\datasets_prepare.py.
rem 2026-10-13: unzip GroundLie360 videos (E:\Downloads\vid.zip), then download the FMNV videos that yt-dlp can reach.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python scripts\datasets_prepare.py g360 --zip "E:\Downloads\vid.zip"
python scripts\datasets_prepare.py fmnv %*
python scripts\datasets_prepare.py status
pause
