@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add -A
git commit -m "datagen 三层核验器 verify/：程序断言、证据锚定引语回查、充分性；Verdict 与 reverify 重核 CLI；15 项测试与说明"
git log --oneline -3 > "%~dp0..\_git提交结果.txt"
git status --short >> "%~dp0..\_git提交结果.txt"
echo 完成
pause >nul
