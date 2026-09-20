@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add -A
git commit -m "核验器：坏 JSON 引号修复与失败原文留档；引语模糊回查；记录未采信部分"
git log --oneline -3 > "%~dp0..\_git提交结果.txt"
git status --short >> "%~dp0..\_git提交结果.txt"
echo 完成
pause >nul
