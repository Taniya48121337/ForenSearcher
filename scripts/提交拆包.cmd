@echo off
chcp 65001 >nul
cd /d "%~dp0.."
if exist src\foren_searcher\store.py del src\foren_searcher\store.py
for /d /r src %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
git add -A
git commit -m "代码拆包 hi/ gia/ store/，剥离 Harness-1 硬依赖，新增 configs/*.toml 与 config.py；1418 项测试通过"
git log --oneline -3 > "%~dp0..\_git提交结果.txt"
git status --short >> "%~dp0..\_git提交结果.txt"
echo 完成
pause >nul
