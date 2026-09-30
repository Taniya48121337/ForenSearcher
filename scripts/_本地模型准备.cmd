@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set LOG=_本地模型准备日志.txt
python -c "import shutil;u=shutil.disk_usage('C:/');print('C 盘剩余 %.1f GB' % (u.free/2**30))" > %LOG% 2>&1
echo 下载教师模型 qwen3-vl:8b-instruct（6.1GB）... >> %LOG%
ollama pull qwen3-vl:8b-instruct >> %LOG% 2>&1
echo pull teacher exit %errorlevel% >> %LOG%
echo 下载核验模型 gemma3:12b（8.1GB）... >> %LOG%
ollama pull gemma3:12b >> %LOG% 2>&1
echo pull judge exit %errorlevel% >> %LOG%
ollama create qwen3vl-8b-32k -f scripts\ollama\teacher.Modelfile >> %LOG% 2>&1
echo create teacher exit %errorlevel% >> %LOG%
ollama create gemma3-12b-16k -f scripts\ollama\judge.Modelfile >> %LOG% 2>&1
echo create judge exit %errorlevel% >> %LOG%
ollama list >> %LOG% 2>&1
echo === 核验模型试答 >> %LOG%
python -m foren_searcher.datagen.verify.judges --check --endpoint http://127.0.0.1:11434/v1 --model gemma3-12b-16k:latest >> %LOG% 2>&1
echo === 教师模型试答 >> %LOG%
python -m foren_searcher.datagen.verify.judges --check --endpoint http://127.0.0.1:11434/v1 --model qwen3vl-8b-32k:latest >> %LOG% 2>&1
echo DONE >> %LOG%
