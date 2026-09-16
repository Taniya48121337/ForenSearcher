@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
echo 第一层（纯程序）重核 experiments\sft_data\runs 下全部已发布报告...
python -m foren_searcher.datagen.verify.reverify --runs experiments\sft_data\runs --out experiments\sft_data\verdicts_r0.jsonl > "%~dp0..\_重核结果.txt" 2>&1
echo 完成，结果在 _重核结果.txt 与 experiments\sft_data\verdicts_r0.jsonl.summary.json
pause >nul
