@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
set LOG=_来源D离线换题日志.txt
echo 正在运行（Ollama 需已启动），时间较长，日志写到 %LOG% ……
echo 来源 D 离线换题：Ollama qwen3-vl 教师 + gemma3 核验 开始 %date% %time% > %LOG%
python -m foren_searcher.datagen.offline_gia --endpoint http://127.0.0.1:11434/v1 --model qwen3vl-8b-32k:latest --then-verify --judge-endpoint http://127.0.0.1:11434/v1 --judge-model gemma3-12b-16k:latest %* >> %LOG% 2>&1
echo EXIT %errorlevel% %date% %time% >> %LOG%
echo === pytest 来源 D >> %LOG%
python -m pytest -q -p no:cacheprovider tests\test_datagen_offline_gia.py >> %LOG% 2>&1
echo PYTEST_EXIT %errorlevel% >> %LOG%
echo 完成，结果在 %LOG%
pause
