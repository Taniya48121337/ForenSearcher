#!/usr/bin/env bash
# 服务器批量跑调查：教师 vLLM（Qwen3.5-122B）在 8000 端口，核验 vLLM 在 8001 端口。
#   bash scripts/run_cases_server.sh --limit 1          # 先试 1 个案例
#   bash scripts/run_cases_server.sh                    # dev_train 全部
#   bash scripts/run_cases_server.sh --samples 4 --round r1s   # 以后学生自采
set -euo pipefail
cd "$(dirname "$0")/.."
export FOREN_CONFIG=${FOREN_CONFIG:-configs/server.toml} PYTHONPATH=src PYTHONUNBUFFERED=1
python -m foren_searcher.datagen.run_cases --round "${ROUND:-r0}" --jobs "${JOBS:-2}" \
    --endpoint "${TEACHER_ENDPOINT:-http://127.0.0.1:8000/v1}" \
    --then-verify --judge-endpoint "${JUDGE_ENDPOINT:-http://127.0.0.1:8001/v1}" "$@"
