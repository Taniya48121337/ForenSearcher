@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add src/foren_searcher/web/training.js src/foren_searcher/web/training.css src/foren_searcher/web/traj
git commit -m "demo: 轨迹示例页接入 run 的真实画面（E3-E10 抽帧与来源封面 E17）"
git log --stat -1
pause
