@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add -A
git commit -m "系统架构页：三张图合并为一张（a 全局调查 / b 工作区内部 / c 全局库与证据链），支持拖动平移与缩放；去掉更新时间与状态类标注；生成流程页的升级标注去掉日期"
git log --oneline -2 > "%~dp0..\_git提交结果.txt"
git status --short >> "%~dp0..\_git提交结果.txt"
echo 完成
pause >nul
