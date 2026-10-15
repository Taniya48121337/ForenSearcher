@echo off
rem Face-swap detector probe (10-10): SBI + DFDC winner on 6 EVID identity-swap videos, their originals, 6 real dev videos.
rem Output: outputs\detect\faceswap_probe\results.json and faces_*.jpg. ASCII only on purpose.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python scripts\_faceswap_probe.py
pause
