@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
echo [1/2] 导出 true_4124903_r1（hi@0.2.0 + gia@0.2.0；若已跑过 补全核验_本机跑.cmd 则自动叠加模型判定）...
set UNITS=
if exist experiments\sft_data\verdicts_units_judged.jsonl set UNITS=--units experiments\sft_data\verdicts_units_judged.jsonl
python -m foren_searcher.datagen.export --runs experiments\sft_data\runs\true_4124903_r1 --cases experiments\sft_data\cases --verdicts experiments\sft_data\verdicts_r0_judged.jsonl --hi-spec hi@0.2.0 --gia-spec gia@0.2.0 --out "experiments\sft_data\export\r0_true4124903_hi@0.2.0_gia@0.2.0_compact_16k" --renderer compact_16k %UNITS%
if errorlevel 1 goto fail
echo [2/2] 用 Qwen3.5-9B processor 核对 label mask 与 token 数 ...
python -m foren_searcher.datagen.check_inputs --export "experiments\sft_data\export\r0_true4124903_hi@0.2.0_gia@0.2.0_compact_16k" --model .models\Qwen3.5-9B --files hi_accepted.jsonl gia_accepted.jsonl hi_held.jsonl hi_rejected.jsonl
if errorlevel 1 goto fail
echo 完成：experiments\sft_data\export\r0_true4124903_hi@0.2.0_gia@0.2.0_compact_16k\manifest.json
pause
exit /b 0
:fail
echo 失败，请看上方输出。
pause
exit /b 1
