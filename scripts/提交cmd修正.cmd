@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add -A
git commit -m "重核 CLI 加 --log；启动脚本去掉 powershell 管道"
git log --oneline -3 > "%~dp0..\_git提交结果.txt"
git status --short >> "%~dp0..\_git提交结果.txt"
echo 完成
pause >nul
