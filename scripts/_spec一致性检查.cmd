@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
python scripts\_spec_match_report.py %* > _spec一致性检查日志.txt 2>&1
echo EXIT %errorlevel% >> _spec一致性检查日志.txt
