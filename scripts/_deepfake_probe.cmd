@echo off
rem Offline test of open-source face-swap detectors (SBI EfficientNet-B4 FF++, DeepfakeBench Xception) on EVID face-swap samples (10-10).
rem Downloads weights to .models\deepfake (pip --target there, base Python untouched). Output: outputs\detect\deepfake_probe\summary.md
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
python scripts\_deepfake_probe.py
pause
