@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add -A
git commit -m "datagen 地基：错误清单 errors.py、案例文件 cases.py、证据扰动器 perturb.py 及 15 项测试；数据生成审计与文献综述"
git log --oneline -3 > "%~dp0..\_git提交结果.txt"
git status --short >> "%~dp0..\_git提交结果.txt"
echo 完成
pause >nul
