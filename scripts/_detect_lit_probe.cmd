@echo off
rem Literature detectors on EVID dev samples (10-12): DDA, Effort, TruFor, Skyra-RL. Output: outputs\detect\lit_probe\summary.md
rem Re-run safely: finished stages are skipped. Only some stages: _detect_lit_probe.cmd --stages frames summary
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
python scripts\_detect_lit_probe.py %*
pause
