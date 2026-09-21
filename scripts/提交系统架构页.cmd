@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add -A
git commit -m "Demo：训练数据改名为方法与进展；新增可交互的系统架构页（运行结构 / 工作区内部 / 记录与证据链三视图），按当前代码校正 INSPECT 八个处理器、visual_qa 与 audio_qa 已移除、GIA 动作为 QUERY_MEMORY 与 READ_RECORD，补上状态渲染器与 GIA 看原件像素；DETECT 保留为待接入的第三个工具"
git log --oneline -3 > "%~dp0..\_git提交结果.txt"
git status --short >> "%~dp0..\_git提交结果.txt"
echo 完成
pause >nul
