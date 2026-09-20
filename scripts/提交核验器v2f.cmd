@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add -A
git commit -m "核验器：来源全文上限放宽到 120K；scope_overreach 提示记 minor；失败异常写入 pending"
git log --oneline -3 > "%~dp0..\_git提交结果.txt"
git status --short >> "%~dp0..\_git提交结果.txt"
echo 完成
pause >nul
