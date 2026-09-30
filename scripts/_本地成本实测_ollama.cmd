@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
set LOG=_本地成本实测日志.txt
echo 本地成本实测：5 个案例，Ollama qwen3-vl 教师 + gemma3 核验，轮次 lcost 开始 %date% %time% > %LOG%
python -m foren_searcher.datagen.run_cases --limit 5 --round lcost --endpoint http://127.0.0.1:11434/v1 --model qwen3vl-8b-32k:latest --then-verify --judge-endpoint http://127.0.0.1:11434/v1 --judge-model gemma3-12b-16k:latest %* >> %LOG% 2>&1
echo EXIT %errorlevel% %date% %time% >> %LOG%
