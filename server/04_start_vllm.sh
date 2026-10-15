#!/usr/bin/env bash
# Step 4: start the model server (vLLM, OpenAI-compatible) in the background and wait until it answers.
#   GPUs $FS_VLLM_GPUS, one Qwen3.8-27B bf16 replica per GPU (data parallel), port $FS_VLLM_PORT.
#   Stop it with: bash server/04_stop_vllm.sh      Log: server_logs/vllm.log
set -uo pipefail
source "$(dirname "$0")/env.sh"
MODEL="$FS_MODEL_DIR/$(basename "$FS_HF_MODEL")"
PIDF="$FS_LOG_DIR/vllm.pid"
say "Step 4: vLLM  model=$MODEL  served as '$FS_SERVED_NAME'  GPUs=$FS_VLLM_GPUS  port=$FS_VLLM_PORT"

if curl -s -m 5 "$FS_ENDPOINT/models" | grep -q "$FS_SERVED_NAME"; then ok "already running at $FS_ENDPOINT"; exit 0; fi
[ -f "$MODEL/config.json" ] || { fail "model not found at $MODEL (run step 2)"; exit 1; }
DP=$(echo "$FS_VLLM_GPUS" | tr ',' '\n' | grep -c .)

CUDA_VISIBLE_DEVICES="$FS_VLLM_GPUS" NO_PROXY="*" nohup "$FS_ENV_VLLM/bin/vllm" serve "$MODEL" \
  --served-model-name "$FS_SERVED_NAME" --host 127.0.0.1 --port "$FS_VLLM_PORT" \
  --data-parallel-size "$DP" --tensor-parallel-size 1 \
  --max-model-len "$FS_MAX_MODEL_LEN" --gpu-memory-utilization 0.90 --max-num-seqs 32 \
  --limit-mm-per-prompt '{"image": 32, "video": 0}' --reasoning-parser qwen3 --trust-remote-code \
  > "$FS_LOG_DIR/vllm.log" 2>&1 &
echo $! > "$PIDF"
say "started (pid $(cat "$PIDF")); loading $DP replicas usually takes 3-10 minutes. Progress: tail -f $FS_LOG_DIR/vllm.log"

for i in $(seq 1 120); do
  sleep 10
  if ! kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    fail "vLLM exited. Last log lines:"; tail -40 "$FS_LOG_DIR/vllm.log"
    echo; warn "common fixes: older vLLM -> pip install -U 'vllm>=0.17' in .venv-vllm;"
    warn "  out of memory -> lower FS_MAX_MODEL_LEN to 36864 or use fewer replicas;"
    warn "  '--data-parallel-size' unsupported -> set FS_VLLM_GPUS=0 (one replica) and rerun"
    exit 1
  fi
  if curl -s -m 5 "$FS_ENDPOINT/models" | grep -q "$FS_SERVED_NAME"; then ok "server is up after $((i * 10)) s"; break; fi
  [ $((i % 6)) -eq 0 ] && say "still loading ($((i * 10)) s) ... $(grep -Eo 'Loading.*|Capturing.*|Starting.*' "$FS_LOG_DIR/vllm.log" | tail -1 | cut -c1-80)"
done
curl -s -m 5 "$FS_ENDPOINT/models" | grep -q "$FS_SERVED_NAME" || { fail "not up after 20 minutes; see $FS_LOG_DIR/vllm.log"; exit 1; }

say "quick answer test (thinking off)"
"$FS_ENV_AGENT/bin/python" - <<'PY'
import json, os, time, httpx
url = os.environ["FS_ENDPOINT"] + "/chat/completions"
body = {"model": os.environ["FS_SERVED_NAME"], "temperature": 0, "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": 'Return exactly {"ok": true} and nothing else.'}]}
t = time.time()
r = httpx.post(url, json=body, timeout=300, trust_env=False)
r.raise_for_status()
m = r.json()["choices"][0]["message"]
think = m.get("reasoning_content") or m.get("reasoning") or ""
print(f"[ OK ] answer {m.get('content')!r} in {time.time() - t:.1f} s" + (f"  [WARN] thinking text present ({len(think)} chars)" if think or "<think>" in (m.get("content") or "") else "; no thinking text"))
PY
ok "Step 4 done. Next: bash server/05_preflight.sh"
