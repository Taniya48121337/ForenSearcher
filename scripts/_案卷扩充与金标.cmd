@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set LOG=_案卷扩充与金标日志.txt
echo === 1 生成 22 个新案卷 %date% %time% > %LOG%
python scripts\_build_cases_r2.py >> %LOG% 2>&1
echo BUILD_EXIT %errorlevel% >> %LOG%
echo === 2 按 family 分配划分 >> %LOG%
python scripts\_assign_splits.py --apply >> %LOG% 2>&1
echo SPLIT_EXIT %errorlevel% >> %LOG%
echo === 3 全部案卷加载与泄漏检查 >> %LOG%
python -c "from foren_searcher.datagen.cases import load_cases,check_split_leakage,split_summary;c=list(load_cases('experiments/sft_data/cases').values());print(len(c),'个案卷');print(check_split_leakage(c) or '无泄漏');print(split_summary(c))" >> %LOG% 2>&1
echo === 4 金标抽样与标注页 >> %LOG%
python scripts\_gold_sample.py >> %LOG% 2>&1
echo GOLD_EXIT %errorlevel% >> %LOG%
echo === 5 回归测试 >> %LOG%
python -c "import glob,sys,pytest; sys.exit(pytest.main(['-q','-p','no:cacheprovider',*sorted(glob.glob('tests/test_datagen_*.py')),'tests/test_complete_verify_paths.py']))" >> %LOG% 2>&1
echo PYTEST_EXIT %errorlevel% >> %LOG%
python -m pytest -q -p no:cacheprovider tests >> %LOG% 2>&1
echo PYTEST_ALL_EXIT %errorlevel% %date% %time% >> %LOG%
