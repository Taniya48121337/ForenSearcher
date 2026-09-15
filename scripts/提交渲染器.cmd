@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add -A
git commit -m "状态渲染器 state_render.py：compact_8k/compact_16k 两档，接入 MemoryPolicy、运行时约束与消息构造；configs 选档；15 项测试与说明"
git log --oneline -3 > "%~dp0..\_git提交结果.txt"
git status --short >> "%~dp0..\_git提交结果.txt"
echo 完成
pause >nul
