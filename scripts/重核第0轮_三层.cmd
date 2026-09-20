@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
echo 三层重核 v2 全量（qwen/qwen3.8-27b，推理 low，带缓存；试跑过的 run 命中缓存不再付费）
python -m foren_searcher.datagen.verify.reverify --runs experiments\sft_data\runs --out experiments\sft_data\verdicts_r0_judged.jsonl --judge openrouter --model qwen/qwen3.8-27b 2>&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath '%~dp0..\_重核三层结果.txt'"
echo 完成
pause >nul
