@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add src/foren_searcher/web/training.js src/foren_searcher/web/training.css src/foren_searcher/web/walker
git commit -m "demo: 轨迹生成页改为脑图逐步展开，wk2 九条轨迹 103 行决策取自运行记录"
git log --stat -1
pause
