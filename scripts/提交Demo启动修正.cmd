@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add -A
git commit -m "Demo网站.cmd 强制 PYTHONPATH=src：此前未设置，-m foren_searcher.web_demo 走的是 site-packages 里的 editable 安装（指向另一棵源码树），导致页面改动不生效；顺带去掉已剥离的 FOREN_HARNESS_ROOT，并在启动时打印实际加载路径"
git log --oneline -2 > "%~dp0..\_git提交结果.txt"
git status --short >> "%~dp0..\_git提交结果.txt"
echo 完成
pause >nul
