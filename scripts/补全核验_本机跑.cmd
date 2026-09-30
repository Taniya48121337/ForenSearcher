@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo 补全核验：12 个 bridge run，六步（默认本地核验模型，不花钱；先跑 本地核验冒烟_1个run.cmd 确认没问题）
python scripts\complete_verify_local.py %*
echo.
echo exit code: %errorlevel%
pause
