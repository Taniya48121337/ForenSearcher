@echo off
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. Steps and messages live in the python scripts.
rem Conversation 5b (2026-10-15): r0 cost cut. All settings come from specs\r0_freeze.json:
rem   S0-S3 -> qwen backend (qwen/qwen3.5-397b-a17b, OpenRouter here; H20 vLLM on the server)
rem   S4/S5 (+ post-verify rewrite) -> strong backend = Codex CLI (ChatGPT subscription, gpt-6-astra, effort medium, serialized)
rem   verifier -> Anthropic API direct, claude-sonnet-5-5, Message Batches + prompt caching (--judge api-batch)
rem Needs: ANTHROPIC_API_KEY set for this user (setx ANTHROPIC_API_KEY sk-ant-... then open a new window), Codex CLI logged in
rem with the ChatGPT account, the OpenRouter key saved on this PC (Qwen slots), system proxy as usual.
rem Steps: 1 offline tests  2 freeze preflight  3 Anthropic verifier check (key, model, 9 schemas)
rem        4 A/B: re-verify the two r0s1 runs (5a smoke) with api-batch as round r0s1d (about USD 0.7)
rem        5 measure: 5 wk13 cases (B videosham_4126 vcsl_pair_5, C fmnv_c_RXR0B1SxiU0 groundlie360_c_46819210, E averimatec_e9)
rem          as round r0c5 with --jobs 2: trajectories + batch verify + rewrite + re-verify + export (about USD 1.5-2.5)
rem        6 report: experiments\sft_data\r0_cost_r0c5.md (per-case cost next to 5a, A/B verdicts, 150-case budget)
rem Re-running skips finished runs, cached verdicts and finished batches (batch ids are kept in experiments\sft_data).
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set LOG=_r0_cost_5b_log.txt
echo ==== 5b %DATE% %TIME% > %LOG%
start "5b log" powershell -NoProfile -Command "Get-Content -Path '%CD%\%LOG%' -Wait -Encoding UTF8"
echo Running. Live output is in the second window (and in %LOG%). This window only shows the step names.
if not defined ANTHROPIC_API_KEY (
  echo ANTHROPIC_API_KEY is not set. Run: setx ANTHROPIC_API_KEY your-key   then open a new window and double-click again.
  echo ANTHROPIC_API_KEY is not set >> %LOG%
  goto done
)
set FOREN_JUDGE_API_ENDPOINT=https://api.anthropic.com/v1
set FOREN_JUDGE_API_MODEL=claude-sonnet-5-5
echo [1/6] offline tests
echo [1/6] offline tests >> %LOG%
python -m pytest -q tests\test_r0_cost_5b_20261015.py tests\test_teacher_slots_5a_20261015.py >> %LOG% 2>&1
echo [2/6] freeze preflight
echo [2/6] freeze preflight >> %LOG%
python scripts\walker_run.py freeze --round r0c5 >> %LOG% 2>&1
echo [3/6] Anthropic verifier check
echo [3/6] Anthropic verifier check >> %LOG%
python -m foren_searcher.datagen.api_judge --check --endpoint https://api.anthropic.com/v1 --model claude-sonnet-5-5 >> %LOG% 2>&1
if errorlevel 1 goto done
python -m foren_searcher.datagen.api_judge --check-schemas >> %LOG% 2>&1
echo [4/6] A/B: r0s1 re-verified with api-batch (round r0s1d)
echo [4/6] A/B: r0s1 re-verified with api-batch (round r0s1d) >> %LOG%
python scripts\complete_verify_local.py --judge api-batch --runs-glob r0s1 --round r0s1d --hi-spec hi@0.10.0 --gia-spec gia@0.9.0 --skip 4 6 --yes-spend >> %LOG% 2>&1
echo [5/6] 5 cases, round r0c5 (Codex strong, --jobs 2)
echo [5/6] 5 cases, round r0c5 >> %LOG%
python scripts\walker_run.py d1 --round r0c5 --cases videosham_4126 vcsl_pair_5 fmnv_c_RXR0B1SxiU0 groundlie360_c_46819210 averimatec_e9 --jobs 2 >> %LOG% 2>&1
echo [6/6] cost report
echo [6/6] cost report >> %LOG%
python scripts\r0_cost_report.py --round r0c5 --ab r0s1d --cases 150 >> %LOG% 2>&1
:done
echo ==== done %DATE% %TIME% >> %LOG%
echo Done. Full output: %LOG%   Report: experiments\sft_data\r0_cost_r0c5.md
pause
