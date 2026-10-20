#!/usr/bin/env bash
# HI s1 迁到另一台 H200 续训的交接（10-20：这台 GPU 93% 时间 SW Thermal Slowdown，SM 均值 ~880 MHz）。负责人说"停"之后：
#   bash server/handoff_hi.sh prepare   等下一个写完的 checkpoint-<step>（不停训练），在里面写 SHA256SUMS 与 handoff.json，打印 step 与 sha
#   （中转对话把这个 checkpoint 传 HF 并按 SHA256SUMS 校验通过后）
#   bash server/handoff_hi.sh kill      停 HI（tmux sft 与 sft_lora 进程）和截止守护，写 handoff_killed
#   bash server/handoff_hi.sh status    当前最新完整 checkpoint、是否已准备、训练进程
# 新机续训：同一 output_dir 放上这个 checkpoint（与 prescan_cache.json），bash server/resume.sh train；runtime 可加 mixed_precision（不进续训身份）。
set -uo pipefail
OUT=${HI_OUT:-/workspace/outputs/lora/sft0_27b/hi_s1}
LOG=${FS_LOG_DIR:-/workspace/logs}/handoff_hi.log
latest() { for d in "$OUT"/checkpoint-*; do [ -f "$d/COMPLETE" ] && echo "$d"; done | sort -V | tail -1; }

case "${1:-status}" in
  prepare)
    base=$(latest); echo "$(date -u +%FT%TZ) prepare: waiting for a checkpoint newer than ${base:-none}" | tee -a "$LOG"
    while :; do
      now=$(latest)
      [ -n "$now" ] && [ "$now" != "$base" ] && break
      pgrep -f "training/sft_lora[.]py" >/dev/null || { echo "no training process: using latest ${now:-none}" | tee -a "$LOG"; break; }
      sleep 15
    done
    ck=$(latest); [ -n "$ck" ] || { echo "no complete checkpoint" | tee -a "$LOG"; exit 1; }
    (cd "$ck" && find . -type f ! -name SHA256SUMS ! -name handoff.json -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
    step=${ck##*-}; sums=$(sha256sum "$ck/SHA256SUMS" | cut -d" " -f1)
    printf '{"step": %s, "checkpoint": "%s", "files": %s, "sha256sums_sha256": "%s", "prepared_at": "%s", "output_dir": "%s"}\n' \
      "$step" "$ck" "$(wc -l < "$ck/SHA256SUMS")" "$sums" "$(date -u +%FT%TZ)" "$OUT" > "$ck/handoff.json"
    echo "$(date -u +%FT%TZ) prepared $ck (step $step, SHA256SUMS sha256 $sums)" | tee -a "$LOG"
    cat "$ck/SHA256SUMS" | tee -a "$LOG" ;;
  kill)
    ck=$(latest); [ -f "$ck/handoff.json" ] || { echo "latest checkpoint $ck has no handoff.json: run prepare first"; exit 1; }
    tmux kill-session -t sft 2>/dev/null; pkill -TERM -f "training/sft_lora[.]py"; sleep 30; pkill -KILL -f "training/sft_lora[.]py" 2>/dev/null
    tmux kill-session -t cutoff_guard 2>/dev/null
    echo "$(date -u +%FT%TZ) HI stopped for handoff; resume point $(cat "$ck/handoff.json")" | tee -a "$LOG"
    touch "${FS_LOG_DIR:-/workspace/logs}/handoff_killed" ;;
  status)
    echo "latest complete checkpoint: $(latest)"; [ -f "$(latest)/handoff.json" ] && cat "$(latest)/handoff.json"
    pgrep -fa "training/sft_lora[.]py" | cut -c1-120 || echo "no training process" ;;
  *) echo "usage: $0 prepare | kill | status"; exit 2 ;;
esac
