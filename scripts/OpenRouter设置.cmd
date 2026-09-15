@echo off
rem 设置 OpenRouter API Key。放在 scripts/ 下，所以先回到项目根目录。
setlocal
cd /d "%~dp0.."
"%USERPROFILE%\anaconda3\python.exe" -B "src\foren_searcher\openrouter_settings.py" --config ".local\openrouter.json"
if errorlevel 1 (
    echo.
    echo OpenRouter settings could not start. See the error above.
    pause
)
endlocal
