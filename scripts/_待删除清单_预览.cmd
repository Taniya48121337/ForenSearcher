@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
python scripts\_cleanup_20260924.py > _待删除清单预览日志.txt 2>&1
echo EXIT %errorlevel% >> _待删除清单预览日志.txt
