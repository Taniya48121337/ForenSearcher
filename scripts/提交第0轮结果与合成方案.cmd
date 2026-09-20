@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add "docs/04_训练与数据/datagen_核验器说明_20260915.md" "docs/04_训练与数据/sft数据合成方案_20260921.md" "docs/03_框架实现/GIA原件像素_20260921.md" "experiments/sft_data/verdicts_r0_judged.summary49.json"
git commit -m "第 0 轮三层核验结果（49 run / 80 份报告）写入文档；SFT 数据合成方案一页纸（待签字）；GIA 原件像素改动的云端复核说明"
git log --oneline -3 > "%~dp0..\_git提交结果.txt"
git status --short >> "%~dp0..\_git提交结果.txt"
echo 完成
pause >nul
