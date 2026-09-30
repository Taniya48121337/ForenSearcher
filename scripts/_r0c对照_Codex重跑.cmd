@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
rem Codex 教师固定用 gpt-6-astra（与 r0b 同一个教师；Codex 默认模型 09-30 变成了 ChatGPT 账号不支持的 gpt-6.1-sol）
set FOREN_CODEX_MODEL=gpt-6-astra
set LOG=_r0c对照日志.txt
set CASES=true:4124903 true:17838107 true:20205316753904 true:20361105 true:24171541709824 true:39132300 videosham:0069 videosham:1007 videosham:1028 videosham:2010 videosham:4126
set R0B=true_17838107_r0b_1 true_20205316753904_r0b_1 true_20361105_r0b_1 true_24171541709824_r0b_1 true_39132300_r0b_1 true_4124903_r0b_1 videosham_0069_r0b_1 videosham_1007_r0b_1 videosham_1028_r0b_1 videosham_2010_r0b_1 videosham_4126_r0b_1
echo ==== r0c 对照（09-30）：同 11 个案例，hi@0.4.0 / gia@0.5.0（教师指令"搜事件本身、找一手来源"+ 格式示例带 rationale） %date% %time% > %LOG%
echo 正在运行：Codex 重跑 11 个案例（约 1–1.5 小时）→ Sonnet 核验与导出（约 30 分钟）→ 对照表。日志 %LOG% ……
echo 中途关窗口也没关系：再双击会跳过已跑完的 run，核验命中缓存。

echo [0] 检查 Codex 能否用 %FOREN_CODEX_MODEL% >> %LOG%
codex exec --ephemeral --skip-git-repo-check --sandbox read-only -m %FOREN_CODEX_MODEL% "只回复 ok" >> %LOG% 2>&1 < nul
if errorlevel 1 (
  echo Codex 用不了 %FOREN_CODEX_MODEL%，没有开跑。看 %LOG% 末尾的报错。
  echo CODEX_PREFLIGHT_FAILED >> %LOG%
  pause
  exit /b 1
)
echo [1/4] Codex 教师重跑（轮次 r0c；黑名单 exclude；录下检索快照，供之后的机制对照回放） >> %LOG%
python -m foren_searcher.datagen.run_cases --teacher codex --retry-failed --round r0c --ids %CASES% --blocklist exclude --search-mode record --hi-spec hi@0.4.0 --gia-spec gia@0.5.0 >> %LOG% 2>&1
echo RUN_EXIT %errorlevel% %date% %time% >> %LOG%

echo [2/4] Sonnet 核验 + 导出（VideoSham 权重份额压到 0.34） >> %LOG%
python scripts\complete_verify_local.py --judge claude-cli --runs-glob r0c --round r0c --hi-spec hi@0.4.0 --gia-spec gia@0.5.0 --skip 4 >> %LOG% 2>&1 < nul
echo VERIFY_EXIT %errorlevel% %date% %time% >> %LOG%

echo [3/4] 对照表 r0b → r0c >> %LOG%
python scripts\compare_rounds.py --before-runs %R0B% --after-round r0c --before-export "experiments\sft_data\export\r0b_11runs-c7ba8bd2_hi@0.3.0_gia@0.4.0_compact_16k.claude-cli-claude-sonnet-5-5" --after-export "experiments\sft_data\export\r0c_*claude-cli*" --labels r0b r0c --out experiments\sft_data\compare_r0b_vs_r0c.md >> %LOG% 2>&1
echo COMPARE_EXIT %errorlevel% >> %LOG%

echo [4/4] 样例页 >> %LOG%
python scripts\_peek_export.py "experiments\sft_data\export\r0c_*claude-cli*" --hi 16 --compare-verdicts "experiments\sft_data\verdicts_r0b_judged.claude-cli*.jsonl" "experiments\sft_data\verdicts_r0c_judged.claude-cli*.jsonl" >> %LOG% 2>&1
echo DONE %date% %time% >> %LOG%
echo 完成。对照表 experiments\sft_data\compare_r0b_vs_r0c.md；日志 %LOG%
pause
