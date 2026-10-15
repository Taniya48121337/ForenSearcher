@echo off
rem Update yt-dlp for YouTube downloads (needs 2026.08.19 or newer; older versions get HTTP 403 on every download).
rem ASCII only on purpose: cmd.exe mis-splits lines that contain Chinese.
cd /d "%~dp0.."
python -m pip install -U "yt-dlp[default]"
python -c "import yt_dlp.version as v; print('yt-dlp version now:', v.__version__)"
pause
