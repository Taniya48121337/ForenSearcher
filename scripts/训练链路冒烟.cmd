@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo 同架构极小随机模型：取数据 → 训一步 → 存 LoRA → 加载 → 前向（不需要 GPU，不验证数字）
python training\sft_lora.py --config training\configs\hi_r0.json --smoke-tiny --output-dir outputs\lora\hi_r0_smoke
if errorlevel 1 goto fail
python training\sft_lora.py --config training\configs\gia_r0.json --smoke-tiny --output-dir outputs\lora\gia_r0_smoke
if errorlevel 1 goto fail
echo 冒烟通过。真实训练（5090 / 服务器）：python training\sft_lora.py --config training\configs\hi_r0.json
pause
exit /b 0
:fail
echo 失败，请看上方输出。
pause
exit /b 1
