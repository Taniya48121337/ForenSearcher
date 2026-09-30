@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set LOG=_本机回归测试日志.txt
echo === pytest（数据流水线相关测试） > %LOG%
python -c "import glob,sys,pytest; sys.exit(pytest.main(['-q','-p','no:cacheprovider',*sorted(glob.glob('tests/test_datagen_*.py')),'tests/test_complete_verify_paths.py']))" >> %LOG% 2>&1
echo PYTEST_EXIT %errorlevel% >> %LOG%
echo === pytest（全部离线测试） >> %LOG%
python -m pytest -q -p no:cacheprovider tests >> %LOG% 2>&1
echo PYTEST_ALL_EXIT %errorlevel% >> %LOG%
echo === spec 一致性（gia@0.3.0） >> %LOG%
python scripts\_spec_match_report.py >> %LOG% 2>&1
echo DONE >> %LOG%
