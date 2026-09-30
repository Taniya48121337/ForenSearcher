@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
set LOG=_0929本机收尾日志.txt
echo 0929 本机收尾 开始 %date% %time% > %LOG%
echo 正在运行，约 5-10 分钟，日志写到 %LOG% ……
echo === 1 案卷护栏修补（699 / 39132300，已改过会跳过） >> %LOG%
python scripts\_patch_cases_20260929.py --apply >> %LOG% 2>&1
echo PATCH_EXIT %errorlevel% >> %LOG%
echo === 2 案卷 x 检索护栏对照 >> %LOG%
python scripts\_guard_case_audit.py --out experiments\sft_data\cases_draft\guard_audit_20260929.md >> %LOG% 2>&1
echo AUDIT_EXIT %errorlevel% >> %LOG%
echo === 3 verifier@0.3.1 可选证据组影响（第一层，全部本机 run） >> %LOG%
python scripts\_optional_evidence_impact.py >> %LOG% 2>&1
echo IMPACT_EXIT %errorlevel% >> %LOG%
echo === 4 金标 v2：premise_smuggling / task_not_gap_driven 任务级批次 >> %LOG%
python scripts\_gold_sample.py --plan-tasks --out experiments\sft_data\gold_set\v2_plan_tasks >> %LOG% 2>&1
echo GOLD_EXIT %errorlevel% >> %LOG%
echo === 5 回归测试（数据流水线） >> %LOG%
python -c "import glob,sys,pytest; sys.exit(pytest.main(['-q','-p','no:cacheprovider',*sorted(glob.glob('tests/test_datagen_*.py')),'tests/test_search_retrieval_guard.py','tests/test_search_recording.py','tests/test_inspect_tool_cache.py','tests/test_complete_verify_paths.py']))" >> %LOG% 2>&1
echo PYTEST_EXIT %errorlevel% >> %LOG%
echo === 6 回归测试（全部离线测试） >> %LOG%
python -m pytest -q -p no:cacheprovider tests >> %LOG% 2>&1
echo PYTEST_ALL_EXIT %errorlevel% %date% %time% >> %LOG%
echo DONE >> %LOG%
echo 完成。结果在 %LOG%，把它留着我来读。
pause
