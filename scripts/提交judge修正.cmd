@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add -A
git commit -m "OpenRouterJudge 首次调用前自动 preload；补测试"
git log --oneline -3 > "%~dp0..\_git提交结果.txt"
git status --short >> "%~dp0..\_git提交结果.txt"
echo 完成
pause >nul
