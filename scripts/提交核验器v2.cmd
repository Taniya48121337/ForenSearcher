@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add -A
git commit -m "核验器第二层改为按部分判定+覆盖汇总；充分性只报案卷要求；坏 JSON 重试记 pending；推理力度默认 low；回答缓存；重核逐 run 落盘"
git log --oneline -3 > "%~dp0..\_git提交结果.txt"
git status --short >> "%~dp0..\_git提交结果.txt"
echo 完成
pause >nul
