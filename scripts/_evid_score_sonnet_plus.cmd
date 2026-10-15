@echo off
rem Sonnet scoring for the eval_plus A/B smoke (round evidq27p).
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. All steps and messages live in scripts\_launch.py.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set FOREN_EVID_ROUND=evidq27p
python scripts\_launch_evid.py evid_score_sonnet
pause
