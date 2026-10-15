@echo off
rem Project cleanup round 2 (2026-10-14). Scans first (read-only), asks about optional groups, deletes only after you type y.
rem Permanent delete, not recycle bin. Skips evidM27g/h and any run changed in the last 3 hours. ASCII only on purpose.
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python scripts\_cleanup_disk_20261014.py %*
echo.
pause
