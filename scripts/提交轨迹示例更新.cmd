@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add -A
git commit -m "轨迹示例页按第 0 轮导出结果更新：校准后 W1-2 转为通过，计数改为 HI 17 收录 / 2 暂缓 / 1 待校准，抽帧步收益改为 8；新增最终答案逐句核对、事件时间线、processor token 核对与适用边界四块；脑图默认只展开轮次"
git log --oneline -2 > "%~dp0..\_git提交结果.txt"
git status --short >> "%~dp0..\_git提交结果.txt"
echo 完成
pause >nul
