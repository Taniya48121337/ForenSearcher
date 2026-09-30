@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add src/foren_searcher/web/training.js src/foren_searcher/web/training.css
git commit -m "demo: 质量核验页只保留当前的逐单元核验流水线，删除上一代七维加权评分"
git log --stat -1
pause
