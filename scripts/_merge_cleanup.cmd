@echo off
rem 10-12 DETECT merged into SEARCH / INSPECT: move the old DETECT package, its configs, tests and scripts out of the code tree.
rem Nothing is deleted: everything goes to _backup_detect_20261012\removed\ (same relative paths). The files that were
rem changed in place have their previous versions in _backup_detect_20261012\before\.
setlocal
cd /d "%~dp0.."
set B=_backup_detect_20261012\removed
if not exist "%B%" mkdir "%B%"
call :mv src\foren_searcher\detect
call :mv tests\test_detect.py
call :mv tests\test_detect_scan_20261008.py
call :mv tests\test_detect_compare_20261010.py
call :mv tests\test_detect_find_20261011.py
call :mv configs\detect.cpu.json
call :mv configs\detect.evid.json
call :mv configs\detect.v2.json
call :mv configs\detect_reference_d3_clipb32.json
call :mv scripts\detect_build_reference.py
call :mv scripts\detect_cpu_smoke.py
echo.
echo Moved to %B%. Running the merge tests ...
if not exist outputs\pytest_tmp mkdir outputs\pytest_tmp
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python -m pytest tests\test_inspect_compare_media_20261012.py tests\test_search_source_match_20261012.py tests\test_merged_catalog_20261012.py tests\test_datagen_prompt_variants.py tests\test_diff_media.py tests\test_server_20261012.py -q -o addopts= -p no:cacheprovider --basetemp=outputs\pytest_tmp\merge_cleanup
echo.
echo Done. To undo: move the folders and files in %B% back to the same places.
pause
exit /b 0

:mv
if not exist "%~1" (
  echo skip %~1 ^(not found^)
  exit /b 0
)
for %%I in ("%B%\%~1") do if not exist "%%~dpI" mkdir "%%~dpI"
move "%~1" "%B%\%~1" >nul && echo moved %~1
exit /b 0
