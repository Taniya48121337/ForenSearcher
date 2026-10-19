#!/usr/bin/env bash
# Register a trained LoRA with the student server and load it without a restart.
#   bash server/12_add_lora.sh hi_r1_full_s1 outputs/lora/r1/full/hi_s1/final
#   bash server/12_add_lora.sh --list
# Then point a run at it: HI_MODEL=hi_r1_full_s1 GIA_MODEL=gia_r1_full_s1 bash server/14_run_cases.sh ...
# The registry (server_logs/loras.tsv) is also what 11_loop_layout.sh passes as --lora-modules when the student (re)starts,
# so a LoRA added here survives a student restart.
set -uo pipefail
source "$(dirname "$0")/loop_env.sh"

if [ "${1:-}" = "--list" ]; then [ -s "$FS_LORA_REGISTRY" ] && cat "$FS_LORA_REGISTRY" || echo "(no LoRAs registered)"; exit 0; fi
[ $# -eq 2 ] || { echo "usage: $0 <name> <adapter dir with adapter_config.json> | --list"; exit 2; }
NAME=$1; DIR=$(cd "$2" 2>/dev/null && pwd) || { fail "no such directory: $2"; exit 1; }
[ -f "$DIR/adapter_config.json" ] || { fail "$DIR has no adapter_config.json (use the .../final directory written by sft_lora.py)"; exit 1; }
[[ "$NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || { fail "name may only use letters, digits, _ . -"; exit 1; }
[ "$NAME" = "$FS_STUDENT_NAME" ] && { fail "'$FS_STUDENT_NAME' is the base model's name"; exit 1; }
RANK=$("$FS_ENV_AGENT/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["r"])' "$DIR/adapter_config.json" 2>/dev/null)
[ -n "$RANK" ] && [ "$RANK" -gt "$FS_LORA_RANK" ] && { fail "LoRA rank $RANK > FS_LORA_RANK=$FS_LORA_RANK (raise it in loop_env.sh and restart the student)"; exit 1; }

touch "$FS_LORA_REGISTRY"
grep -v -P "^\Q$NAME\E\t" "$FS_LORA_REGISTRY" > "$FS_LORA_REGISTRY.tmp"; printf '%s\t%s\n' "$NAME" "$DIR" >> "$FS_LORA_REGISTRY.tmp"
mv "$FS_LORA_REGISTRY.tmp" "$FS_LORA_REGISTRY"
ok "registered $NAME -> $DIR ($(wc -l < "$FS_LORA_REGISTRY") LoRAs; FS_MAX_CPU_LORAS=$FS_MAX_CPU_LORAS)"
[ "$(wc -l < "$FS_LORA_REGISTRY")" -gt "$FS_MAX_CPU_LORAS" ] && warn "more LoRAs registered than FS_MAX_CPU_LORAS: vLLM will swap them in and out (slower, still correct)"

if ! served_on "$FS_STUDENT_ENDPOINT" "$FS_STUDENT_NAME"; then
  say "student not running: it will load $NAME when started (bash server/11_loop_layout.sh $([ "${FS_PROFILE:-}" = vast1 ] && echo serve || echo A))"; exit 0
fi
if served_on "$FS_STUDENT_ENDPOINT" "$NAME"; then
  curl -s -m 30 -X POST "$FS_STUDENT_ENDPOINT/unload_lora_adapter" -H 'Content-Type: application/json' -d "{\"lora_name\": \"$NAME\"}" >/dev/null
fi
RESP=$(curl -s -m 300 -w '\n%{http_code}' -X POST "$FS_STUDENT_ENDPOINT/load_lora_adapter" -H 'Content-Type: application/json' \
       -d "{\"lora_name\": \"$NAME\", \"lora_path\": \"$DIR\"}")
CODE=$(echo "$RESP" | tail -1)
if [ "$CODE" = 200 ] && served_on "$FS_STUDENT_ENDPOINT" "$NAME"; then
  ok "$NAME loaded on the running student ($FS_STUDENT_ENDPOINT)"
else
  warn "runtime loading failed (HTTP $CODE: $(echo "$RESP" | head -1 | cut -c1-200))"
  warn "the registry is updated; restart the student to load it: bash server/11_loop_layout.sh restart-student"
  exit 1
fi
