@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
echo 本机 Codex CLI 教师（gpt-6-astra，走 ChatGPT 订阅额度，不走 OpenRouter）批量跑调查，只能串行。
echo 先用 --limit 1 或 --ids ... 试一个；跑完加 --then-verify 可直接接本地核验与导出。
python -m foren_searcher.datagen.run_cases --teacher codex %*
echo.
echo exit code: %errorlevel%
pause
