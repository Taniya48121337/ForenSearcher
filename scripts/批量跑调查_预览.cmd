@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
echo 预览：按案卷列出要跑的 run（不跑、不花钱）。可在后面加 --split all / --ids true:4124903 / --samples 4 / --round r0
python -m foren_searcher.datagen.run_cases --dry-run %*
echo.
pause
