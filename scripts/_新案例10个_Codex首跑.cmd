@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
rem Codex 教师固定用 gpt-6-astra（与 r0b 同一个教师；Codex 默认模型 09-30 变成了 ChatGPT 账号不支持的 gpt-6.1-sol）
set FOREN_CODEX_MODEL=gpt-6-astra
set LOG=_新案例r0c日志.txt
set CASES=averitec:1707 averitec:1766 averitec:392 averimatec:423 averimatec:660 averimatec:699 qvhighlights:9167 vcsl:pair_1 vcsl:pair_5 vcsl:pair_6
echo ==== 新增案例里属于 dev_train 的 10 个：Codex 教师首跑（hi@0.4.0 / gia@0.5.0） %date% %time% > %LOG%
echo 正在运行：Codex 跑 10 个新案例（约 1–1.5 小时）→ Sonnet 核验与导出（约 30 分钟）。日志 %LOG% ……
echo 只跑训练划分的案例；dev_select / dev_confirm / dev_eval 的案例留作评估，不产训练数据。

echo [0] 检查 Codex 能否用 %FOREN_CODEX_MODEL% >> %LOG%
codex exec --ephemeral --skip-git-repo-check --sandbox read-only -m %FOREN_CODEX_MODEL% "只回复 ok" >> %LOG% 2>&1 < nul
if errorlevel 1 (
  echo Codex 用不了 %FOREN_CODEX_MODEL%，没有开跑。看 %LOG% 末尾的报错。
  echo CODEX_PREFLIGHT_FAILED >> %LOG%
  pause
  exit /b 1
)
echo [1/3] Codex 教师首跑（轮次 r0cn；黑名单 exclude；录下检索快照） >> %LOG%
python -m foren_searcher.datagen.run_cases --teacher codex --retry-failed --round r0cn --ids %CASES% --blocklist exclude --search-mode record --hi-spec hi@0.4.0 --gia-spec gia@0.5.0 >> %LOG% 2>&1
echo RUN_EXIT %errorlevel% %date% %time% >> %LOG%

echo [2/3] Sonnet 核验 + 导出 >> %LOG%
python scripts\complete_verify_local.py --judge claude-cli --runs-glob r0cn --round r0cn --hi-spec hi@0.4.0 --gia-spec gia@0.5.0 --skip 4 >> %LOG% 2>&1 < nul
echo VERIFY_EXIT %errorlevel% %date% %time% >> %LOG%

echo [3/3] 取证指标与样例页 >> %LOG%
python scripts\compare_rounds.py --before-runs experiments\sft_data\runs\_none_ --after-round r0cn --after-export "experiments\sft_data\export\r0cn_*claude-cli*" --labels 无 r0cn --out experiments\sft_data\stats_r0cn.md >> %LOG% 2>&1
python scripts\_peek_export.py "experiments\sft_data\export\r0cn_*claude-cli*" --hi 16 >> %LOG% 2>&1
echo DONE %date% %time% >> %LOG%
echo 完成。取证指标 experiments\sft_data\stats_r0cn.md；日志 %LOG%
pause
