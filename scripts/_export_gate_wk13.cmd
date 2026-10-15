@echo off
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese.
rem export quality gate trial (2026-10-15): re-export wk13 with the new datagen\export.py (B2-B9, C3, C5 rules), then run
rem scripts\export_gate.py and scripts\walker_coverage.py --export on it; also run the gate on the old wk13 export as-is,
rem and re-export r0e under the current specs (C5). Reads runs and verdicts only; writes new export folders, does not touch runs.
rem Outputs: experiments\sft_data\export\wk13gate_*\gate_report.md, by_branch.md, gate_b2_review.md; _export_gate_wk13_log.txt
rem Optional: set GATE_MODEL=.models\Qwen3.5-9B (or another local Qwen3.5 processor folder) for the exact mask / token check.
cd /d "%~dp0.."
setlocal enabledelayedexpansion
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set S=experiments\sft_data
set LOG=_export_gate_wk13_log.txt
set MODELARG=
if defined GATE_MODEL set MODELARG=--model %GATE_MODEL%
set RUNS=
for /d %%R in (%S%\runs\*_wk13_1) do set RUNS=!RUNS! %%R
set R0E=
for /d %%R in (%S%\runs\*_r0e_1) do set R0E=!R0E! %%R
echo [1/5] re-export wk13 > %LOG%
python -m foren_searcher.datagen.export --runs %RUNS% --cases %S%\cases --verdicts %S%\verdicts_wk13_judged.claude-cli-claude-sonnet-5-5.27runs-77d4cdb7.jsonl --units %S%\verdicts_units_wk13_judged.claude-cli-claude-sonnet-5-5.27runs-77d4cdb7.jsonl --hi-spec hi@0.10.0 --gia-spec gia@0.9.0 --renderer compact_16k --out %S%\export\wk13gate_hi@0.10.0_gia@0.9.0_compact_16k >> %LOG% 2>&1
echo [2/5] gate on the re-export >> %LOG%
python scripts\export_gate.py --export %S%\export\wk13gate_hi@0.10.0_gia@0.9.0_compact_16k %MODELARG% --phash >> %LOG% 2>&1
python scripts\walker_coverage.py --export %S%\export\wk13gate_hi@0.10.0_gia@0.9.0_compact_16k >> %LOG% 2>&1
echo [3/5] gate on the old wk13 export as-is >> %LOG%
python scripts\export_gate.py --export "%S%\export\wk13_27runs-77d4cdb7_hi@0.10.0_gia@0.9.0_compact_16k.claude-cli-claude-sonnet-5-5" >> %LOG% 2>&1
echo [4/5] r0e re-export under the current specs (C5) >> %LOG%
python -m foren_searcher.datagen.export --runs %R0E% --cases %S%\cases --verdicts %S%\verdicts_r0e_judged.claude-cli-claude-sonnet-5-5.21runs-3b6e1ae5.jsonl --units %S%\verdicts_units_r0e_judged.claude-cli-claude-sonnet-5-5.21runs-3b6e1ae5.jsonl --hi-spec hi@0.10.0 --gia-spec gia@0.9.0 --renderer compact_16k --origin-tag r0e --out %S%\export\r0egate_hi@0.10.0_gia@0.9.0_compact_16k >> %LOG% 2>&1
echo [5/5] gate on the r0e re-export >> %LOG%
python scripts\export_gate.py --export %S%\export\r0egate_hi@0.10.0_gia@0.9.0_compact_16k >> %LOG% 2>&1
type %LOG%
pause
