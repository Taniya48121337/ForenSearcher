@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo 循环分卡改动的本机回归（10-12）：新测试 + 相关旧测试 + 训练链路冒烟（梯度检查点）> _循环分卡测试日志.txt
python -m pytest -q tests\test_loop_gpu_20261012.py tests\test_server_20261012.py tests\test_config.py tests\test_teacher_student_input_20261005.py tests\test_datagen_run_cases.py >> _循环分卡测试日志.txt 2>&1
echo pytest exit %errorlevel% >> _循环分卡测试日志.txt
python training\sft_lora.py --config training\configs\hi_r0.json --smoke-tiny --output-dir outputs\lora\hi_r0_smoke_gc >> _循环分卡测试日志.txt 2>&1
echo smoke exit %errorlevel% >> _循环分卡测试日志.txt
echo 完成，结果在 _循环分卡测试日志.txt
pause
