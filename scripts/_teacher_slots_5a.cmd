@echo off
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. Steps and messages live in the python scripts.
rem Conversation 5a (2026-10-15): teacher split by slot + verifier over API. Nothing here uses Codex or the Claude CLI.
rem   S0 rationale / S1 query / S2 task text / S3 next step -> qwen backend  (qwen/qwen3.5-397b-a17b on OpenRouter here; H20 vLLM on the server)
rem   S4 report / S5 answer (+ post-verify rewrite)          -> strong backend (openai/gpt-6-astra on OpenRouter)
rem   verifier: Sonnet over the OpenAI-compatible API       (anthropic/claude-sonnet-5.5 on OpenRouter, effort medium)
rem All settings come from specs\r0_freeze.json; the OpenRouter key is the one saved on this PC (system proxy).
rem Steps: 1 offline tests  2 freeze preflight  3 API verifier check  4 smoke: videosham_4126 (B) + fmnv_c_RXR0B1SxiU0 (C),
rem with verification, round r0s1, then audit (under USD 2). Re-running skips finished runs and cached verdicts.
rem Optional (only with the argument wk13, e.g. "scripts\_teacher_slots_5a.cmd wk13"): 5 re-verify the 27 wk13 runs with the
rem API verifier (round tag wk13api; about USD 20-25, measured at about USD 0.85 of verification per case in the r0s1 smoke)  6 CLI vs API agreement table.
rem Output: _teacher_slots_5a_log.txt, experiments\sft_data\teacher_slots_audit_r0s1.md, experiments\sft_data\walker_wk13_cli_vs_api.md
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set LOG=_teacher_slots_5a_log.txt
echo ==== 5a %DATE% %TIME% > %LOG%
rem a second window follows the log live (all python output goes to the log file, not to this window)
start "5a log" powershell -NoProfile -Command "Get-Content -Path '%CD%\%LOG%' -Wait -Encoding UTF8"
echo Running. Live output is in the second window (and in %LOG%). This window only shows the step names.
echo [1/6] offline tests
echo [1/6] offline tests >> %LOG%
python -m pytest -q tests\test_teacher_slots_5a_20261015.py tests\test_walker_endpoint_slot_20261014.py >> %LOG% 2>&1
echo [2/6] freeze preflight
echo [2/6] freeze preflight >> %LOG%
python scripts\walker_run.py freeze --round r0s1 >> %LOG% 2>&1
echo [3/6] API verifier check
echo [3/6] API verifier check >> %LOG%
python -m foren_searcher.datagen.api_judge --check >> %LOG% 2>&1
python -m foren_searcher.datagen.api_judge --check-schemas >> %LOG% 2>&1
echo [4/6] smoke r0s1
echo [4/6] smoke r0s1 >> %LOG%
python scripts\walker_run.py d1 --round r0s1 --cases videosham_4126 fmnv_c_RXR0B1SxiU0 --jobs 2 >> %LOG% 2>&1
python scripts\teacher_slots_audit.py --round r0s1 >> %LOG% 2>&1
if /i not "%~1"=="wk13" goto done
echo [5/6] wk13 with the API verifier
echo [5/6] wk13 with the API verifier >> %LOG%
python scripts\complete_verify_local.py --judge api --runs-glob wk13 --round wk13api --hi-spec hi@0.10.0 --gia-spec gia@0.9.0 --skip 4 6 --yes-spend >> %LOG% 2>&1
echo [6/6] CLI vs API agreement
echo [6/6] CLI vs API agreement >> %LOG%
python scripts\judge_cli_api_agreement.py --round wk13 --api-round wk13api >> %LOG% 2>&1
:done
echo ==== done %DATE% %TIME% >> %LOG%
echo Done. Full output: %LOG%   Audit: experiments\sft_data\teacher_slots_audit_r0s1.md
pause
