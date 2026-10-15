#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"
PIDF="$FS_LOG_DIR/vllm.pid"
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  kill "$(cat "$PIDF")"; sleep 5; pkill -f "vllm serve" 2>/dev/null; ok "vLLM stopped"
else warn "no running vLLM from this package (pid file $PIDF)"; pkill -f "vllm serve" 2>/dev/null && ok "killed stray vllm serve"; fi
