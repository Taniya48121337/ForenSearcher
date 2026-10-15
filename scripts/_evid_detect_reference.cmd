@echo off
rem DETECT real-footage reference only (10-08): unit tests, then scripts\detect_build_reference.py on EVID 002-130.
rem Writes configs\detect_reference_d3_clipb32.json; does not run EVID. Log: _EVID_DETECT*.txt
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set FOREN_DETECT_ONLY_REFERENCE=1
python scripts\_evid_detect.py
pause
