#!/usr/bin/env bash
# 本机 ↔ Vast 单卡服务器（本机 Git Bash 里跑；本机没有 rsync / WSL，用 tar 走 ssh）。
#   bash server/sync_vast.sh archive [tag]         tag 里的代码目录 → /workspace/repo（git archive，约 27 MB；默认 sft0-frozen-2）
#   bash server/sync_vast.sh put <本机文件或目录>…   本机工作区的单个文件 / server 目录 → /workspace/repo 同路径
#   bash server/sync_vast.sh pull <远端路径>         服务器文件或目录 → 本机 experiments/server_outbox/<同名>（只拉报告、json、日志）
#   bash server/sync_vast.sh ssh '<命令>'            在服务器上跑一条命令
# 永远不推：.git、.wt_*、dist、experiments/runs、案卷媒体、gold_set、.models。key 用 scp 单独传到 /workspace/repo/.local（chmod 600）。
# 本机仓库是唯一源码：改脚本先改本机，再 put；不在服务器上直接编辑。
set -euo pipefail
VAST_SSH=${VAST_SSH:-"ssh -p 41837 -o ServerAliveInterval=30 root@69.67.154.150"}
REMOTE=${VAST_REPO:-/workspace/repo}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PATHS=(src scripts specs server training configs tests pyproject.toml claude 'experiments/sft_data/*.json' 'experiments/sft_data/cases/*.json')

case "${1:-}" in
  archive)
    cd "$ROOT"; tag=${2:-sft0-frozen-2}
    git archive --format=tar "$tag" -- "${PATHS[@]}" | $VAST_SSH "mkdir -p '$REMOTE' && tar -x -C '$REMOTE'"
    echo "archive $tag ($(git rev-parse "$tag^{commit}" | cut -c1-7)) -> $REMOTE" ;;
  put)
    shift; cd "$ROOT"
    for p in "$@"; do case "$p" in .git*|.wt_*|dist*|.models*|experiments/runs*|*gold_set*) echo "refuse: $p"; exit 1;; esac; done
    tar -cf - "$@" | $VAST_SSH "tar -x -C '$REMOTE' --no-same-owner"
    echo "put $* -> $REMOTE" ;;
  pull)
    src=${2:?remote path}; dst="$ROOT/experiments/server_outbox"; mkdir -p "$dst"
    $VAST_SSH "cd '$(dirname "$src")' && tar -cf - '$(basename "$src")'" | tar -xf - -C "$dst"
    echo "pull $src -> $dst/$(basename "$src")" ;;
  ssh) shift; $VAST_SSH "$@" ;;
  *) echo "usage: $0 archive [tag] | put <paths…> | pull <remote path> | ssh '<cmd>'"; exit 2 ;;
esac
