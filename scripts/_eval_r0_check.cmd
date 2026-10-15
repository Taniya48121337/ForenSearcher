@echo off
rem r0 eval (checklist E2/E3): read-only check of the frozen eval set, search snapshots (every blob), freeze file and
rem registered script hashes. Writes experiments\eval\r0\reports\check_<time>.json. Changes nothing.
rem Then (once, before E2): python scripts\eval_r0.py seed   -- copies the 20 EVID snapshots into experiments\eval\r0\search_snapshots
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese. Messages live in scripts\eval_r0.py.
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python scripts\eval_r0.py check > _eval_r0_check_log.txt 2>&1
type _eval_r0_check_log.txt
pause
