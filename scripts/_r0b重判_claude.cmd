@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
set LOG=_r0b重判_claude日志.txt
echo r0b 用本机 Claude CLI（Sonnet）重判 开始 %date% %time% > %LOG%
echo 正在用 Claude CLI 重判 r0b 的 11 个 run，预计 30-60 分钟，日志写到 %LOG% ……
echo 中途关窗口也没关系：已判过的都在缓存里，重新双击会接着判。
echo === 0 案卷修补（已改过会跳过） >> %LOG%
python scripts\_patch_cases_20260929.py --apply >> %LOG% 2>&1
echo === 诊断：claude 在哪、怎么登录 >> %LOG%
where claude >> %LOG% 2>&1
python -c "from foren_searcher.datagen.verify.judges import claude_cli_candidates as c, find_claude_cli as f, claude_auth_method as a; print('候选', [str(x) for x in c() if x.is_file()]); cli=f(); print('使用', cli); print('登录方式', a(cli))" >> %LOG% 2>&1
echo === 1 核验 + 导出（跳过第 4 步召回校准） >> %LOG%
python scripts\complete_verify_local.py --judge claude-cli --runs-glob r0b --round r0b --hi-spec hi@0.3.0 --gia-spec gia@0.4.0 --skip 4 >> %LOG% 2>&1 < nul
echo VERIFY_EXIT %errorlevel% %date% %time% >> %LOG%
echo === 2 样例页与 gemma 对照 >> %LOG%
python scripts\_peek_export.py "experiments\sft_data\export\r0b_*claude-cli*" --hi 16 --compare-verdicts "experiments\sft_data\verdicts_r0b_judged.local-gemma*.jsonl" "experiments\sft_data\verdicts_r0b_judged.claude-cli*.jsonl" >> %LOG% 2>&1
echo PEEK_EXIT %errorlevel% %date% %time% >> %LOG%
echo DONE >> %LOG%
echo 完成。结果在 %LOG%；样例页在 experiments\sft_data\export\r0b_...claude-cli...\样例.md
pause
