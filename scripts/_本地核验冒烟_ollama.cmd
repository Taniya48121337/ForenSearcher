@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
set FOREN_JUDGE_ENDPOINT=http://127.0.0.1:11434/v1
set FOREN_JUDGE_MODEL=gemma3-12b-16k:latest
set LOG=_本地核验冒烟日志.txt
echo 本地核验冒烟（Ollama gemma3-12b，true_4124903_r1，六步） > %LOG%
python scripts\complete_verify_local.py --runs true_4124903_r1 %* >> %LOG% 2>&1
echo EXIT %errorlevel% >> %LOG%
