@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
echo 本地核验冒烟：只跑 true_4124903_r1 一个 run，六步全走（不花钱）
python scripts\complete_verify_local.py --runs true_4124903_r1 %*
echo.
echo exit code: %errorlevel%
pause
