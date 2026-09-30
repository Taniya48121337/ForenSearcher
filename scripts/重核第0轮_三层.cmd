@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
echo 三层重核（09-28 起改用本地核验模型，不花钱）：只跑第 1 步
python scripts\complete_verify_local.py --to 1 %*
echo 结束（若上面有报错请截图给我）
pause >nul
