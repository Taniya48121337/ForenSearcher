@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set LOG=_起草新案卷日志.txt
echo 起草新案卷 开始 %date% %time% > %LOG%
python scripts\_draft_new_cases.py %* >> %LOG% 2>&1
echo EXIT %errorlevel% %date% %time% >> %LOG%
