@echo off
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. Steps and messages live in the python scripts.
rem walker wk8 (2026-10-13): hi@0.9.0 adds the SEARCH date lookup; new E family (AVerImaTeC: where and when an image was
rem first published) and GroundLie360 claims with video (needs scripts\_datasets_prepare.cmd to have unzipped vid.zip first);
rem ends with a tool coverage table (scripts\walker_coverage.py).
rem 10-13 later: hi@0.10.0 (compare_media without mode, video opens fetch subtitles), L family (where a photo was taken,
rem text-to-image reference photos) and two v9 variant runs (max_evidence_cards=6, QUERY_MEMORY before the report).
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python scripts\walker_d2_cases.py
if exist experiments\sft_data\walker_d2_cases.json (
  python scripts\walker_run.py d1 --round wk8 --d2 --variants %*
) else (
  python scripts\walker_run.py d1 --round wk8 --variants %*
)
pause
