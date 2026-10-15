#!/usr/bin/env bash
# 服务器批量跑调查：教师 vLLM（Qwen3.5-122B）在 8000 端口，核验 vLLM 在 8001 端口。
#   bash scripts/run_cases_server.sh --limit 1          # 先试 1 个案例
#   bash scripts/run_cases_server.sh                    # dev_train 全部
#   bash scripts/run_cases_server.sh --samples 4 --round r1s   # 以后学生自采
# INSPECT 操作集（09-30）：默认 auto = 基础五项 + 有权重且装了包的语音转写 / 按文字找帧 / 图内定位。
#   服务器上要与本机一致，需要 configs/server.toml 的 model_root 下有 inspect/faster-whisper-small、
#   inspect/clip-vit-base-patch32、inspect/grounding-dino-tiny，并 pip install "faster-whisper>=1.1,<2"；
#   缺哪项开跑前会打印"未开 …：原因"，run_config.inspect_operations_detail 也有记录。与旧数据对照时用 INSPECT_OPS=base。
set -euo pipefail
cd "$(dirname "$0")/.."
export FOREN_CONFIG=${FOREN_CONFIG:-configs/server.toml} PYTHONPATH=src PYTHONUNBUFFERED=1
python -m foren_searcher.datagen.run_cases --round "${ROUND:-r0}" --jobs "${JOBS:-2}" --inspect-ops "${INSPECT_OPS:-auto}" \
    --endpoint "${TEACHER_ENDPOINT:-http://127.0.0.1:8000/v1}" \
    --then-verify --judge-endpoint "${JUDGE_ENDPOINT:-http://127.0.0.1:8001/v1}" "$@"
