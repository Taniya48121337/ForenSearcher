@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add -A
git commit -m "核验器修正：可见性取全回合并集、数字比对保留全部位数并支持分组写法、重核改为只读副本；第 0 轮首次结果记入说明"
git log --oneline -3 > "%~dp0..\_git提交结果.txt"
git status --short >> "%~dp0..\_git提交结果.txt"
echo 完成
pause >nul
