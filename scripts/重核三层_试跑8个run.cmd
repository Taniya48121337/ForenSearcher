@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
echo 三层重核试跑（09-28 起改用本地核验模型，不花钱）：只跑前 8 个 run
python -m foren_searcher.datagen.verify.reverify --runs experiments\sft_data\runs --out experiments\sft_data\verdicts_r0_judged_pilot.local.jsonl --judge local --limit 8 --log _重核三层试跑结果.txt %*
echo 结束（若上面有报错请截图给我）
pause >nul
