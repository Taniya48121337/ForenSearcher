#!/usr/bin/env bash
# sft0 合集门 + 训练行门（10-19 重合用；在 .wt_g2 下跑）：bash claude/_g2_work/run_gates_sft0.sh dist/sft0-<id> <标签>
set -u
D=$1; TAG=$2
M=E:/Codex/workspaces/ChatGPT/ForenSearcher
U=experiments/sft_data/export/_sft0_union
G=experiments/sft_data/export/_sft0_trainset
UN=$(python -c "import json;print(' '.join(json.load(open('$M/experiments/sft_data/unanswerable_r1.json',encoding='utf-8'))['unanswerable_case_ids']))")
ARGS="--phash --model .models/Qwen3.8-27B-processor --evid-channels experiments/sft_data/evid_channels_131_222.json --train-channels experiments/sft_data/sft0_channels.json"
# 合集（四个导出的并集）：并集目录不变，只重跑门
if [ "${SKIP_UNION:-0}" != 1 ]; then
PYTHONIOENCODING=utf-8 python scripts/export_gate.py --export $U $ARGS --unanswerable $UN > claude/_g2_work/gate_${TAG}_union.txt 2>&1
echo "UNION $?"
fi
# 训练行：accepted = sft0，held / rejected / verdicts 照并集，图联到并集的 images（并集 images 下是按 run 的联接）
# 先只拆联接（rm -rf 会跟进联接删到导出里的真图）：images 是联接就 rmdir；是目录就逐个 rmdir 里面的联接，碰到非联接就停
python - "$G" <<'PY' || exit 1
import os, sys
g = os.path.join(sys.argv[1], "images")
if os.path.isjunction(g) or os.path.islink(g):
    os.rmdir(g) if os.path.isjunction(g) else os.unlink(g)
elif os.path.isdir(g):
    for x in os.listdir(g):
        q = os.path.join(g, x)
        if not os.path.isjunction(q):
            sys.exit(f"{q} 不是联接，不删")
        os.rmdir(q)
    os.rmdir(g)
PY
rm -rf $G; mkdir -p $G
cp $D/hi.jsonl $G/hi_accepted.jsonl; cp $D/gia.jsonl $G/gia_accepted.jsonl
for f in hi_held hi_rejected gia_held gia_rejected verdicts_reports verdicts_units; do cp $U/$f.jsonl $G/; done
cp $U/manifest.json $G/
# 图：按 run 建联接，指向并集 images 下各 run 联接的目标（导出里的真图）
python - "$U" "$G" <<'PY' || exit 1
import os, subprocess, sys
u, g = (os.path.normpath(os.path.abspath(os.path.join(a, "images"))) for a in sys.argv[1:3])
os.makedirs(g)
for x in os.listdir(u):
    target = os.path.realpath(os.path.join(u, x))
    subprocess.run(["cmd", "/c", "mklink", "/J", os.path.join(g, x), target], check=True, capture_output=True)
print("image run links", len(os.listdir(g)))
PY
PYTHONIOENCODING=utf-8 python scripts/export_gate.py --export $G $ARGS --unanswerable $UN > claude/_g2_work/gate_${TAG}_trainset.txt 2>&1
echo "TRAINSET $?"
grep -E "^[BC][0-9]+ " claude/_g2_work/gate_${TAG}_union.txt | sed 's/^/union    /'
grep -E "^[BC][0-9]+ " claude/_g2_work/gate_${TAG}_trainset.txt | sed 's/^/trainset /'
