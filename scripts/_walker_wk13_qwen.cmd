@echo off
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. Steps and messages live in the python scripts.
rem walker wk13-wk15 (2026-10-14 night): the overhauled pipeline, teacher = Qwen on OpenRouter, the 30 cases
rem (9 given-media B cases + the 21 search cases of wk11/wk12). Verification stays Sonnet (Claude CLI) as the reference.
rem   wk13  qwen/qwen3.5-122b-a10b   (+ the same model verifies wk13 a second time: agreement with Sonnet)
rem   wk14  qwen/qwen3.5-397b-a17b   (largest open Qwen that fits 8x H20 in FP8)
rem   wk15  qwen/qwen3.8-27b          (newest open Qwen, dense 27B)
rem The OpenRouter key is the one saved on this PC (python -m foren_searcher.openrouter_settings); it goes through the
rem system proxy. Each round stops by itself after two runs in a row fail on 429 / 402; double-click again to resume.
rem Tables: experiments\sft_data\walker_teachers_wk13_wk14_wk15.md, walker_teachers_wk12_wk13_wk14_wk15.md, walker_wk13_judges.md
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set FOREN_ORACLE_SLOT_THINKING=1
set JOBS=3
python scripts\walker_run.py d1 --round wk13 --d2 --teacher openrouter --teacher-model qwen/qwen3.5-122b-a10b --jobs %JOBS% --second-judge qwen/qwen3.5-122b-a10b %*
python scripts\walker_run.py d1 --round wk14 --d2 --teacher openrouter --teacher-model qwen/qwen3.5-397b-a17b --jobs %JOBS% %*
python scripts\walker_run.py d1 --round wk15 --d2 --teacher openrouter --teacher-model qwen/qwen3.8-27b --jobs %JOBS% %*
python scripts\walker_teacher_compare.py --rounds wk13 wk14 wk15 --cases videosham_0069 videosham_1007 videosham_1028 videosham_2010 videosham_4126 vcsl_pair_1 vcsl_pair_5 vcsl_pair_6 qvhighlights_9167 fmnv_a_3GwQdrhp__k fmnv_a_1KkAwCyj7nI fmnv_a_4u12EqPAHqU fmnv_c_RXR0B1SxiU0 fmnv_c_TIu19O02X2c fmnv_c_eC8ctVxSglU groundlie360_c_46819210 groundlie360_c_46581202 averimatec_e2 averimatec_e4 averimatec_e9 averimatec_e12 averimatec_e17 averimatec_e28 averimatec_l208 averimatec_l623 averimatec_l739 averimatec_l101 groundlie360_v183204 groundlie360_v1914403 groundlie360_v1815915
python scripts\walker_teacher_compare.py --rounds wk12 wk13 wk14 wk15 --cases fmnv_a_3GwQdrhp__k fmnv_a_1KkAwCyj7nI fmnv_a_4u12EqPAHqU fmnv_c_RXR0B1SxiU0 fmnv_c_TIu19O02X2c fmnv_c_eC8ctVxSglU groundlie360_c_46819210 groundlie360_c_46581202 averimatec_e2 averimatec_e4 averimatec_e9 averimatec_e12 averimatec_e17 averimatec_e28 averimatec_l208 averimatec_l623 averimatec_l739 averimatec_l101 groundlie360_v183204 groundlie360_v1914403 groundlie360_v1815915
pause
