@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add -A
git commit -m "核验器：结构化输出 schema 防坏 JSON；某对判定失败时不报无据；汇总程序兜底；转述句不记 attribution_drift；--log；启动脚本去管道"
git log --oneline -3 > "%~dp0..\_git提交结果.txt"
git status --short >> "%~dp0..\_git提交结果.txt"
echo 完成
pause >nul
