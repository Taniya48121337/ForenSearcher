@echo off
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. Steps and messages live in the python scripts.
rem walker wk11 (2026-10-14): language-slot teacher = local Claude Code CLI (Sonnet) instead of Codex, on the 21 search-family
rem cases (D2: find-original, claim checks incl. GroundLie360 videos, earliest copy, where taken).
rem --arms: at the final answer step four versions are written on the same state and verified separately:
rem   wk11  d0 first draft, no ground truth
rem   wk11b original method (ground-truth words as hints + all verifier remarks fed back)
rem   wk11g evidence-gated hints (point only to E ids whose visible findings contain the point) + evidence-only revise
rem   wk11n best of 3 drafts picked by ground truth, no hint, no revise
rem Result table: experiments\sft_data\walker_wk11_arms.md . Re-running skips finished runs.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set FOREN_ORACLE_CLAUDE_MODEL=sonnet
python scripts\walker_run.py d1 --round wk11 --d2 --skip-b --teacher claude --arms %*
pause
