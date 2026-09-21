@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add -A
git commit -m "系统架构图重排版面：面板表头改单行并留出净空，方块高度按实际内容重算，连线走廊与边标签移到空白处；程序校验方块两两不重叠、标签不压框、节点内容不被裁切"
git log --oneline -2 > "%~dp0..\_git提交结果.txt"
git status --short >> "%~dp0..\_git提交结果.txt"
echo 完成
pause >nul
