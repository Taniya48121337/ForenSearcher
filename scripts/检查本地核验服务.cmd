@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
echo 检查本地核验服务（默认 http://127.0.0.1:8000/v1；别的地址：本文件后面加 --endpoint http://...:端口/v1）
python -m foren_searcher.datagen.verify.judges --check %*
echo.
pause
