@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add -A
git commit -m "将 SFT 候选语料 jsonl 与 manifest 纳入 git（图片仍排除）"
git log --oneline -3 > "%~dp0..\_git提交结果.txt"
git status --short >> "%~dp0..\_git提交结果.txt"
echo 完成
pause >nul
