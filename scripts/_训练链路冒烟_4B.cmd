@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PYTHONPATH=src
set PYTHONUTF8=1
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo 本机 4B 训练链路冒烟（清单 E1）：
echo   先查显卡占用（walker / 评测 / vLLM / ollama 在跑就不开跑），缺 Qwen3.5-4B 权重会先问是否下载（约 9-10 GB），
echo   再训 40 步 LoRA、存盘、重新加载、对一个真实状态推理一次、输出过程序核验。
echo   日志 _训练链路冒烟_4B日志.txt；结果 outputs\lora\hi_smoke_4b\smoke_report.json
echo.
python training\smoke_4b.py %*
set RC=%errorlevel%
echo.
echo 退出码 %RC%（0 = 通过，1 = 失败，2 = 显卡被占用没开跑，3 = 缺权重）
pause
