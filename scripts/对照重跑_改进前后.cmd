@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
set LOG=_对照重跑日志.txt
echo ==== 改进前后对照（09-28）：同 11 个案例，Codex 教师重跑一遍 > %LOG%
echo 改动：核查站点黑名单 + 日期截断（阶段 04）、hi@0.3.0 / gia@0.4.0（rationale 字段 + 教师指令"优先 SEARCH"）、导出权重与沾染规则 >> %LOG%
echo. >> %LOG%
set CASES=true:4124903 true:17838107 true:20205316753904 true:20361105 true:24171541709824 true:39132300 videosham:0069 videosham:1007 videosham:1028 videosham:2010 videosham:4126
set OLDRUNS=true_4124903_r1 true_4124903_r2 true_17838107_r1 true_20205316753904_validation true_20361105_visibility_validation true_24171541709824_r1 true_39132300_validation_r3 videosham_0069_r1 videosham_1007_validation videosham_1028_r2 videosham_2010_r1 videosham_4126_split_intervals_r7
set R=experiments\sft_data\runs
set OLDRUNPATHS=%R%\true_4124903_r1 %R%\true_4124903_r2 %R%\true_17838107_r1 %R%\true_20205316753904_validation %R%\true_20361105_visibility_validation %R%\true_24171541709824_r1 %R%\true_39132300_validation_r3 %R%\videosham_0069_r1 %R%\videosham_1007_validation %R%\videosham_1028_r2 %R%\videosham_2010_r1 %R%\videosham_4126_split_intervals_r7

echo [1/4] 旧 12 个 run 按新导出规则重新导出（沾染标记、权重调整；用 09-25 的 OpenRouter 判定，不调模型）
echo [1/4] 旧 run 新规则重导出 >> %LOG%
python -m foren_searcher.datagen.export --runs %OLDRUNPATHS% --cases experiments\sft_data\cases --verdicts experiments\sft_data\verdicts_r0_judged.jsonl --units experiments\sft_data\verdicts_units_judged.jsonl --hi-spec hi@0.2.0 --gia-spec gia@0.3.0 --renderer compact_16k --out "experiments\sft_data\export\r0_bridge12_hi@0.2.0_gia@0.3.0_compact_16k.rules0928" >> %LOG% 2>&1
if errorlevel 1 goto fail

echo [2/4] Codex 教师重跑 11 个案例（轮次标签 r0b；黑名单 exclude；串行，约 1–2 小时）
echo [2/4] Codex 重跑 >> %LOG%
python -m foren_searcher.datagen.run_cases --teacher codex --round r0b --ids %CASES% --blocklist exclude --hi-spec hi@0.3.0 --gia-spec gia@0.4.0 >> %LOG% 2>&1
if errorlevel 1 echo （有 run 失败，继续做对照；失败的 run 看 experiments\sft_data\batch_logs） >> %LOG%

echo [3/4] 新 run 本地核验六步 + 导出（gemma3-12b；与旧 run 的 .local-gemma 导出同核验器可比；要 OpenRouter 就改成 --judge openrouter --yes-spend）
echo [3/4] 本地核验 + 导出 >> %LOG%
python scripts\complete_verify_local.py --runs-glob r0b --hi-spec hi@0.3.0 --gia-spec gia@0.4.0 --round r0b >> %LOG% 2>&1
if errorlevel 1 echo （核验/导出失败，只做教师侧对照） >> %LOG%

echo [4/4] 对照表
echo [4/4] 对照 >> %LOG%
python scripts\compare_rounds.py --before-runs %OLDRUNS% --after-round r0b --before-export "experiments\sft_data\export\r0_bridge12_hi@0.2.0_gia@0.3.0_compact_16k.local-gemma3-12b-16k-latest" --after-export "experiments\sft_data\export\r0b_11runs-*" --labels r0 r0b --out experiments\sft_data\compare_r0_vs_r0b.md >> %LOG% 2>&1
type experiments\sft_data\compare_r0_vs_r0b.md
echo.
echo 完成。日志：%LOG%；对照表：experiments\sft_data\compare_r0_vs_r0b.md
pause
exit /b 0
:fail
echo 失败，看 %LOG%
pause
exit /b 1
