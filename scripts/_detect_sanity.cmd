@echo off
rem Positive controls (10-12): AIGI detectors on a public real/fake image set, TruFor on its own samples + CocoGlide, Yandex on GT originals.
rem Output: outputs\detect\sanity\summary.md
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
python scripts\_detect_sanity.py %*
pause
