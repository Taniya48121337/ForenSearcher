@echo off
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. Steps and messages live in the python scripts.
rem walker wk12 (2026-10-14): same as wk11 (21 search-family cases, --arms) but the language-slot teacher is Codex gpt-6-astra,
rem so the result table walker_wk12_arms.md compares directly with walker_wk11_arms.md (Sonnet teacher).
rem FOREN_ORACLE_BON=1: no extra drafts (best-of-N by ground truth never differed from the first draft in wk11), saves Codex quota.
rem If Codex hits its usage limit (429) on two runs in a row the batch stops; double-click again later, finished runs are skipped.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set FOREN_CODEX_MODEL=gpt-6-astra
set FOREN_ORACLE_BON=1
python scripts\walker_run.py d1 --round wk12 --d2 --skip-b --teacher codex --arms %*
pause
