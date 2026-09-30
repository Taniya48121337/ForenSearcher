@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
set FOREN_JUDGE_ENDPOINT=http://127.0.0.1:11434/v1
set FOREN_JUDGE_MODEL=gemma3-12b-16k:latest
set LOG=_对照重跑日志_第3-4步.txt
set OLDRUNS=true_4124903_r1 true_4124903_r2 true_17838107_r1 true_20205316753904_validation true_20361105_visibility_validation true_24171541709824_r1 true_39132300_validation_r3 videosham_0069_r1 videosham_1007_validation videosham_1028_r2 videosham_2010_r1 videosham_4126_split_intervals_r7
echo ==== 第 3-4 步：r0b 新 run 本地核验（Ollama gemma3-12b）+ 导出 + 对照表 %date% %time% > %LOG%
echo 先确认 Ollama 在跑（没跑就启动一个最小化窗口，等 10 秒）
python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:11434/v1/models',timeout=3).status_code<500 else 1)" >nul 2>&1
if errorlevel 1 (
  echo Ollama 没在跑，启动中... >> %LOG%
  start "ollama" /min ollama serve
  timeout /t 10 >nul
)
echo [3/4] 本地核验六步 + 导出（r0b，hi@0.3.0 + gia@0.4.0） >> %LOG%
python scripts\complete_verify_local.py --runs-glob r0b --hi-spec hi@0.3.0 --gia-spec gia@0.4.0 --round r0b >> %LOG% 2>&1
echo VERIFY_EXIT %errorlevel% >> %LOG%
echo [4/4] 对照表 >> %LOG%
python scripts\compare_rounds.py --before-runs %OLDRUNS% --after-round r0b --before-export "experiments\sft_data\export\r0_bridge12_hi@0.2.0_gia@0.3.0_compact_16k.local-gemma3-12b-16k-latest" --after-export "experiments\sft_data\export\r0b_11runs-*" --labels r0 r0b --out experiments\sft_data\compare_r0_vs_r0b.md >> %LOG% 2>&1
type experiments\sft_data\compare_r0_vs_r0b.md
echo.
echo 完成。日志：%LOG%
pause
