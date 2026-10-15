@echo off
rem Other benchmarks (GroundLie360, FMNV) with the EVID full-run setup: 27B + DETECT v2.4. Pilot 20 per dataset by default.
rem Larger run: set FOREN_BENCH_N=100 in this file (before the python line).
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OLLAMA_NUM_PARALLEL=1
python scripts\_bench_run.py
pause
