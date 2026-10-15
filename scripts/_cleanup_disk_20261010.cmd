@echo off
rem E drive cleanup 2026-10-10. Scans first (read-only), asks about optional groups, deletes only after you type y.
rem Permanent delete, not recycle bin. Log: _cleanup log file in the project root. ASCII only on purpose (cmd.exe mis-splits Chinese lines).
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python scripts\_cleanup_disk_20261010.py
echo.
pause
