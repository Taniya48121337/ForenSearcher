@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
set FOREN_JUDGE_ENDPOINT=http://127.0.0.1:11434/v1
set FOREN_JUDGE_MODEL=gemma3-12b-16k:latest
set LOG=_本地教师冒烟日志.txt
echo 本地教师冒烟（Ollama qwen3-vl 教师 + gemma3 核验，videosham_1028，轮次 l0） > %LOG%
python -m foren_searcher.datagen.run_cases --ids videosham:1028 --round l0 --endpoint http://127.0.0.1:11434/v1 --model qwen3vl-8b-32k:latest --then-verify --judge-endpoint http://127.0.0.1:11434/v1 --judge-model gemma3-12b-16k:latest %* >> %LOG% 2>&1
echo EXIT %errorlevel% >> %LOG%
