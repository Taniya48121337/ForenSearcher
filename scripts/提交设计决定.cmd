@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add -A
git commit -m "设计决定：DESIGN.md 第 7-9 节定案、新增讨论记录与数据集候选表、更新 README"
git log --oneline -3 > "%~dp0..\_git提交结果.txt"
git status --short >> "%~dp0..\_git提交结果.txt"
echo 完成
pause >nul
