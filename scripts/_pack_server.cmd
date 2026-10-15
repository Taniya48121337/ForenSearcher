@echo off
rem Pack the code for the server baselines into dist\ForenSearcher_server_<date>.tar.gz
cd /d "%~dp0.."
set PYTHONUTF8=1
python scripts\_pack_server.py
pause
