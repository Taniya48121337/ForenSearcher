@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add -A
git commit -m "Demo 轨迹示例页：移除旧的样本库（按 API 取语料的网格与样例详情），改为内置 true:4124903 一个案例的完整轨迹，仍用调查脑图展示，逐步可点开看当时状态、模型输出与处置；脑图节点脚注改为进入候选/暂缓/待校准与权重"
git log --oneline -2 > "%~dp0..\_git提交结果.txt"
git status --short >> "%~dp0..\_git提交结果.txt"
echo 完成
pause >nul
