#!/usr/bin/env bash
# E2 评测的单 run 墙钟上限（默认 1800 s；run_cases 没有整 run 超时）。tmux 里跑：tmux new-session -d -s e2_watchdog "bash server/e2_watchdog.sh 1800"
# 只管 E2 评测的 run 子进程：命令行是 run_cases --one、轮次名 r0base-s<N> / r0sft-s<N>、cwd 是仓库根——不碰 sft_lora.py、vLLM 或别的 run_cases。
# 超时 TERM → 20 s 后 KILL；父进程把它记成失败、继续别的 run；一个阶段跑完后 failed 的统一重排一轮，仍失败的在报告里列出（EVID 计 0 分）。
# 杀过的记在 $FS_LOG_DIR/e2_watchdog.log（默认 /workspace/logs）。训练开始前停掉（server/resume.sh train 会停）。
LIMIT=${1:-1800}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG=${FS_LOG_DIR:-/workspace/logs}/e2_watchdog.log
mkdir -p "$(dirname "$LOG")"
echo "$(date '+%F %T') watchdog start: limit ${LIMIT}s, root $ROOT" >> "$LOG"
while :; do
  for p in $(pgrep -f "datagen[.]run_cases --one .*--round r0(base|sft)-s[0-9]+ "); do
    [ "$(readlink /proc/"$p"/cwd 2>/dev/null)" = "$ROOT" ] || continue
    e=$(ps -o etimes= -p "$p" 2>/dev/null | tr -d " "); [ -z "$e" ] && continue
    if [ "$e" -gt "$LIMIT" ]; then
      name=$(tr "\0" " " < /proc/"$p"/cmdline | grep -o -- "--run-name [^ ]*" | cut -d" " -f2)
      echo "$(date '+%F %T') kill $p $name after ${e}s" >> "$LOG"
      kill -TERM "$p"; sleep 20; kill -0 "$p" 2>/dev/null && kill -KILL "$p"
    fi
  done
  sleep 30
done
