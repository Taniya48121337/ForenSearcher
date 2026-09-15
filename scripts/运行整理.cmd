@echo off
chcp 65001 >nul
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0整理_20260915.ps1"
echo.
echo ===== 脚本结束，按任意键关闭 =====
pause >nul
