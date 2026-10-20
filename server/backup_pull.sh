#!/usr/bin/env bash
# 本机拉服务器备份（本机 Git Bash 里跑；本机没有 rsync，用 tar 走 ssh）。宿主机 10-20–10-21 维护、实例随时可能被停时用。
#   bash server/backup_pull.sh            拉一次（增量：只拉上次成功之后改过的文件）
#   bash server/backup_pull.sh --full     忽略上次的标记，全量拉一次
#   bash server/backup_pull.sh --loop     每 20 分钟拉一次（BACKUP_EVERY_S 改间隔），Ctrl-C 停
# 拉到本机 experiments/server_outbox/backup/，保留服务器上的相对路径（/workspace/logs → backup/logs，/workspace/outputs → backup/outputs）。
# 只拉文本：json / jsonl / sqlite3 / md / txt / log / tsv / html / sh / py。不拉 png / jpg / mp4 / 媒体 / .part：
#   请求附图可由状态重渲，候选视频在检索快照的 blob 里；GB 级文件不回本机。
# 训练输出（/workspace/outputs）只拉每个任务目录里最新的一个 checkpoint 与 final（adapter + 训练状态）。
# 增量标记：服务器 /workspace/.backup_last（成功收完才前移）；失败时下一次自动补。
set -uo pipefail
VAST_SSH=${VAST_SSH:-"ssh -p 41837 -o ServerAliveInterval=30 -o ConnectTimeout=20 root@69.67.154.150"}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEST="$ROOT/experiments/server_outbox/backup"
EVERY=${BACKUP_EVERY_S:-1200}
mkdir -p "$DEST"

once() {
  local full=${1:-0} stamp; stamp=$(date +%Y%m%d_%H%M%S)
  # 远端：先记新标记，再列出比旧标记新的文本文件，打包流回来
  $VAST_SSH "bash -s" > "$DEST/.incoming_$stamp.tar" 2> "$DEST/.incoming_$stamp.err" <<REMOTE
set -uo pipefail
cd /workspace
touch /workspace/.backup_new
NEWER=""
if [ "$full" != 1 ] && [ -e /workspace/.backup_last ]; then NEWER="-newer /workspace/.backup_last"; fi
{
  find repo/experiments/eval/r0/runs repo/experiments/eval/r0/reports repo/experiments/eval/r0/scores repo/server repo/server_logs \
       repo/experiments/sft_data/run_manifests repo/experiments/sft_data/batch_logs logs \
       -type f \$NEWER \( -name '*.json' -o -name '*.jsonl' -o -name '*.sqlite3' -o -name '*.sqlite3-wal' -o -name '*.md' -o -name '*.txt' \
       -o -name '*.log' -o -name '*.tsv' -o -name '*.html' -o -name '*.sh' -o -name '*.py' \) 2>/dev/null
  for f in repo/experiments/eval/r0/eval_config.json repo/experiments/eval/r0/search_snapshots/snapshot_lock.json \
           repo/experiments/eval/r0/search_snapshots/seed_manifest.json; do [ -f "\$f" ] && echo "\$f"; done
  # 训练输出：每个任务目录最新的 checkpoint-* 与 final（整目录，adapter 不大）
  if [ -d outputs ]; then
    for job in \$(find outputs -mindepth 3 -maxdepth 3 -type d 2>/dev/null); do
      last=\$(ls -d "\$job"/checkpoint-* 2>/dev/null | sort -V | tail -1)
      for d in "\$last" "\$job/final"; do [ -n "\$d" ] && [ -d "\$d" ] && find "\$d" -type f \$NEWER; done
      find "\$job" -maxdepth 1 -type f \$NEWER
    done
  fi
} | sort -u | tar -cf - -T - 2>/dev/null
REMOTE
  local rc=$? size; size=$(stat -c %s "$DEST/.incoming_$stamp.tar" 2>/dev/null || echo 0)
  if [ $rc -ne 0 ] || ! tar -tf "$DEST/.incoming_$stamp.tar" >/dev/null 2>&1; then
    echo "$(date '+%F %T') backup FAILED (ssh rc $rc, $size B): $(tail -2 "$DEST/.incoming_$stamp.err")" | tee -a "$DEST/backup.log"
    rm -f "$DEST/.incoming_$stamp.tar"; return 1
  fi
  local n; n=$(tar -tf "$DEST/.incoming_$stamp.tar" | wc -l)
  tar -xf "$DEST/.incoming_$stamp.tar" -C "$DEST" && $VAST_SSH "mv -f /workspace/.backup_new /workspace/.backup_last" 2>/dev/null
  echo "$(date '+%F %T') backup ok: $n files, $((size / 1000000)) MB$([ "$full" = 1 ] && echo ' (full)')" | tee -a "$DEST/backup.log"
  rm -f "$DEST/.incoming_$stamp.tar" "$DEST/.incoming_$stamp.err"
}

case "${1:-}" in
  --full) once 1 ;;
  --loop) while :; do once 0; sleep "$EVERY"; done ;;
  "") once 0 ;;
  *) echo "usage: $0 [--full | --loop]"; exit 2 ;;
esac
