@echo off
rem Turn on the imgbb image host for this window only (local frames -> imgbb, deleted after 10 min -> Serper Google Lens;
rem if imgbb fails the frame goes to litterbox instead, deleted after 1 h)
rem and run one Lens search the way a local frame would (billed once).
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set FOREN_LENS_IMAGE_HOST=imgbb
python scripts\_check_image_keys.py --lens-local
pause
