@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add -A
git commit -m "GIA 看原件像素：model_images 新增原始材料与其派生图分组（排在点名读取之后、报告引用之前）；global_view 的已完成报告不再夹带 HI 过程图，仅保留观察摘要；相应重写 4 处受影响断言并新增第 1 轮原件可见用例"
git log --oneline -3 > "%~dp0..\_git提交结果.txt"
git status --short >> "%~dp0..\_git提交结果.txt"
echo 完成
pause >nul
