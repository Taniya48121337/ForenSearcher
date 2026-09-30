@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
set FOREN_JUDGE_ENDPOINT=http://127.0.0.1:11434/v1
set FOREN_JUDGE_MODEL=gemma3-12b-16k:latest
set LOG=_本地全量核验日志.txt
echo 本地全量核验（Ollama gemma3-12b，12 个 bridge run，六步）开始 %date% %time% > %LOG%
python scripts\complete_verify_local.py %* >> %LOG% 2>&1
echo VERIFY_EXIT %errorlevel% %date% %time% >> %LOG%
echo === 训练链路冒烟（极小随机模型，用本地全量导出） >> %LOG%
python training\sft_lora.py --config training\configs\hi_r0.json --smoke-tiny --export-dir "experiments\sft_data\export\r0_bridge12_hi@0.2.0_gia@0.3.0_compact_16k.local-gemma3-12b-16k-latest" --output-dir outputs\lora\hi_local_smoke >> %LOG% 2>&1
echo SMOKE_HI_EXIT %errorlevel% >> %LOG%
python training\sft_lora.py --config training\configs\gia_r0.json --smoke-tiny --export-dir "experiments\sft_data\export\r0_bridge12_hi@0.2.0_gia@0.3.0_compact_16k.local-gemma3-12b-16k-latest" --output-dir outputs\lora\gia_local_smoke >> %LOG% 2>&1
echo SMOKE_GIA_EXIT %errorlevel% >> %LOG%
echo DONE %date% %time% >> %LOG%
