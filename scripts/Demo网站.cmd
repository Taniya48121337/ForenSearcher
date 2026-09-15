@echo off
rem 启动演示网站。放在 scripts/ 下，所以先回到项目根目录。
cd /d "%~dp0.."
set "FOREN_HARNESS_ROOT=C:\Users\Zya\Documents\Codex\2026-07-23\plan-harness-1\work\harness-1"
"%USERPROFILE%\anaconda3\python.exe" -m foren_searcher.web_demo --open
if errorlevel 1 pause
