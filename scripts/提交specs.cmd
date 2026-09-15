@echo off
chcp 65001 >nul
cd /d "%~dp0.."
git add -A
git commit -m "datagen 版本化规格：specs.py/bootstrap.py、hi@0.1.0 与 gia@0.1.0 目录、14 项测试、规格说明文档"
git log --oneline -3 > "%~dp0..\_git提交结果.txt"
git status --short >> "%~dp0..\_git提交结果.txt"
echo 完成
pause >nul
