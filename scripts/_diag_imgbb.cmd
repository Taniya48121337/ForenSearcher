@echo off
rem Diagnose imgbb uploads with public test images only (no investigation material, no Serper credit).
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python scripts\_diag_imgbb.py
pause
