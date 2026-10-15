@echo off
rem Move the old demo front-end (case library, live workbench, old progress tabs, paper draft) out of the web folder.
rem Nothing is deleted: files go to _backup_web_20261013\removed. Login page and style.css stay (login uses them).
cd /d "%~dp0.."
set W=src\foren_searcher\web
set B=_backup_web_20261013\removed
if not exist "%B%" mkdir "%B%"
for %%F in (app.js cases.json training.js training.css training-map.js paper.js paper.css) do if exist "%W%\%%F" move /Y "%W%\%%F" "%B%\" >nul && echo moved %%F
for %%D in (traj walker) do if exist "%W%\%%D" move /Y "%W%\%%D" "%B%\%%D" >nul && echo moved %%D\
echo.
echo Done. Old files are in %B%. The new page is index.html + progress.js + progress.css + progress\
pause
