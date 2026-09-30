@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
rem Codex 教师固定用 gpt-6-astra（与 r0b 同一个教师；Codex 默认模型 09-30 变成了 ChatGPT 账号不支持的 gpt-6.1-sol）
set FOREN_CODEX_MODEL=gpt-6-astra
set LOG=_机制对照日志.txt
set TRUE6=true:4124903 true:17838107 true:20205316753904 true:20361105 true:24171541709824 true:39132300
echo ==== 机制对照（09-30，要在 _r0c对照_Codex重跑.cmd 之后跑）：两项结构改动各开一组，和 r0c 同案例比 %date% %time% > %LOG%
echo 正在运行：TRUE 6 案例开"搜索附摘录"、VideoSham 4126 开"附图上限 16"，约 1 小时。日志 %LOG% ……
echo 检索用 r0c 录下的快照回放（未命中才联网补录），两组看到的网页尽量相同。

echo [0] 检查 Codex 能否用 %FOREN_CODEX_MODEL% >> %LOG%
codex exec --ephemeral --skip-git-repo-check --sandbox read-only -m %FOREN_CODEX_MODEL% "只回复 ok" >> %LOG% 2>&1 < nul
if errorlevel 1 (
  echo Codex 用不了 %FOREN_CODEX_MODEL%，没有开跑。看 %LOG% 末尾的报错。
  echo CODEX_PREFLIGHT_FAILED >> %LOG%
  pause
  exit /b 1
)
echo [1/4] A 组：SEARCH 打开网页后附 600 字相关摘录（search_excerpt_chars=600），TRUE 6 个案例，轮次 r0cx >> %LOG%
python -m foren_searcher.datagen.run_cases --teacher codex --retry-failed --round r0cx --ids %TRUE6% --blocklist exclude --search-mode replay_or_record --hi-spec hi@0.4.0 --gia-spec gia@0.5.0 --harness-override search_excerpt_chars=600 >> %LOG% 2>&1
echo A_EXIT %errorlevel% %date% %time% >> %LOG%

echo [2/4] B 组：每次最多附图 16 张（image_input_limit=16），videosham:4126，轮次 r0ci >> %LOG%
python -m foren_searcher.datagen.run_cases --teacher codex --retry-failed --round r0ci --ids videosham:4126 --blocklist exclude --search-mode replay_or_record --hi-spec hi@0.4.0 --gia-spec gia@0.5.0 --harness-override image_input_limit=16 >> %LOG% 2>&1
echo B_EXIT %errorlevel% %date% %time% >> %LOG%

set R=experiments\sft_data\runs
echo [3/4] A 组对照：r0c 的 TRUE 6 → r0cx >> %LOG%
python scripts\compare_rounds.py --before-runs %R%\true_4124903_r0c_1 %R%\true_17838107_r0c_1 %R%\true_20205316753904_r0c_1 %R%\true_20361105_r0c_1 %R%\true_24171541709824_r0c_1 %R%\true_39132300_r0c_1 --after-round r0cx --labels r0c 摘录 --out experiments\sft_data\compare_excerpt_r0c_vs_r0cx.md >> %LOG% 2>&1
echo [4/4] B 组对照：r0c 的 4126 → r0ci >> %LOG%
python scripts\compare_rounds.py --before-runs %R%\videosham_4126_r0c_1 --after-round r0ci --labels r0c 附图16 --out experiments\sft_data\compare_imagelimit_r0c_vs_r0ci.md >> %LOG% 2>&1
echo DONE %date% %time% >> %LOG%
echo 完成。对照表在 experiments\sft_data\compare_excerpt_*.md 和 compare_imagelimit_*.md；日志 %LOG%
pause
