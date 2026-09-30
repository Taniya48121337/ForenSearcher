@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add src/foren_searcher/web/training.js src/foren_searcher/web_training.py tests/test_web_training.py DESIGN.md docs/04_训练与数据/RSI融入方案_20260924.md docs/04_训练与数据/漏洞审查_20260924.md docs/04_训练与数据/sft数据构造方法v2_20260921.md docs/04_训练与数据/双模型联合自进化训练方案.md
git commit -m "demo: 数字改为第0轮核验口径（49次调查/80份报告/641组证据对照），真实样例去掉数据里不存在的rationale字段、补回答案原文漏掉的一句，核验说明更新到召回测量与R4；docs: RSI方案、漏洞审查与09-24五项决定"
git log --stat -1
pause
