@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
echo 三层重核（核验模型 qwen/qwen3.8-27b，走 OpenRouter，密钥读 .local\openrouter.json）...
python -m foren_searcher.datagen.verify.reverify --runs experiments\sft_data\runs --out experiments\sft_data\verdicts_r0_judged.jsonl --judge openrouter --model qwen/qwen3.8-27b > "%~dp0..\_重核三层结果.txt" 2>&1
echo 完成，结果在 _重核三层结果.txt 与 experiments\sft_data\verdicts_r0_judged.jsonl.summary.json
pause >nul
