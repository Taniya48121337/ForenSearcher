@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONUTF8=1
python scripts\_d_source_states.py > _来源D状态导出日志.txt 2>&1
echo EXIT %errorlevel% >> _来源D状态导出日志.txt
echo 完成，结果在 _来源D状态导出日志.txt
pause
