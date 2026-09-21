@echo off
chcp 65001 >nul
rem 演示站脚本放在 scripts/ 下，先回到项目根目录，并强制从本仓库的 src 加载。
cd /d "%~dp0.."
set "PYTHONPATH=%CD%\src"
set "PYTHONUTF8=1"
"%USERPROFILE%\anaconda3\python.exe" -c "import foren_searcher,sys;print('foren_searcher 加载自:',foren_searcher.__file__)"
"%USERPROFILE%\anaconda3\python.exe" -m foren_searcher.web_demo --open
if errorlevel 1 pause
