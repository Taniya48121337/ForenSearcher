"""训练端（/venv/main，transformers 5.3.0）与推理端（/workspace/venvs/vllm，transformers 5.19.0）对同一批 sft0 行的
聊天模板文本与 token id 是否逐个一致（单卡 Vast，2026-10-19）。两个环境各跑一次 dump，再 compare。只用 CPU。

    CUDA_VISIBLE_DEVICES= /venv/main/bin/python server/23_tokenizer_parity.py dump --pack <sft0 目录> --model <底座目录> --out a.json
    CUDA_VISIBLE_DEVICES= /workspace/venvs/vllm/bin/python server/23_tokenizer_parity.py dump --pack … --model … --out b.json
    python server/23_tokenizer_parity.py compare a.json b.json

取样：HI / GIA 各取图最多的 4 行 + 最长的 3 行 + 随机 3 行（种子 0），去重。比的是 prefix（推理时送进模型的部分，add_generation_prompt=True）
的文本 sha256、input_ids sha256 与长度、pixel_values / image_grid_thw 的形状与 sha256。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path


def pick(rows: list[dict], k_img: int = 4, k_long: int = 3, k_rand: int = 3) -> list[int]:
    idx = list(range(len(rows)))
    by_img = sorted(idx, key=lambda i: -len(rows[i].get("images") or []))[:k_img]
    by_len = sorted(idx, key=lambda i: -len(json.dumps(rows[i]["messages"], ensure_ascii=False)))[:k_long]
    rnd = random.Random(0).sample(idx, k_rand)
    seen, out = set(), []
    for i in by_img + by_len + rnd:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump(args: argparse.Namespace) -> int:
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    pack = Path(args.pack)
    out = {"transformers": transformers.__version__, "torch": torch.__version__, "rows": []}
    for name in ("hi.jsonl", "gia.jsonl"):
        rows = [json.loads(line) for line in (pack / name).read_text(encoding="utf-8").splitlines() if line.strip()]
        for i in pick(rows):
            row = rows[i]
            msgs = row["messages"][:-1]
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            images = []
            for rel in row.get("images") or []:
                with Image.open(pack / rel) as im:
                    images.append(im.convert("RGB"))
            enc = processor(text=[text], images=images or None, return_tensors="pt", padding=False)
            ids = enc["input_ids"][0].tolist()
            item = {"file": name, "row": i, "run": row["metadata"].get("run"), "request": row["metadata"].get("request"),
                    "images": len(images), "text_sha": sha(text.encode("utf-8")), "n_ids": len(ids),
                    "ids_sha": sha(json.dumps(ids).encode()), "text_head": text[:200], "text_tail": text[-200:]}
            for key in ("pixel_values", "image_grid_thw"):
                if key in enc:
                    t = enc[key].contiguous()
                    item[key] = {"shape": list(t.shape), "sha": sha(t.float().numpy().tobytes())}
            out["rows"].append(item)
            print(f"{name}#{i} images={len(images)} ids={len(ids)}", file=sys.stderr)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def compare(args: argparse.Namespace) -> int:
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (args.a, args.b))
    print(f"A transformers {a['transformers']} / B transformers {b['transformers']}")
    bad = 0
    for x, y in zip(a["rows"], b["rows"]):
        assert (x["file"], x["row"]) == (y["file"], y["row"])
        diffs = [k for k in ("text_sha", "n_ids", "ids_sha") if x[k] != y[k]]
        for key in ("pixel_values", "image_grid_thw"):
            if (x.get(key) or {}) != (y.get(key) or {}):
                diffs.append(key)
        bad += bool(diffs)
        print(f"{x['file']}#{x['row']:<5} images={x['images']:<2} ids {x['n_ids']}/{y['n_ids']}  {'SAME' if not diffs else 'DIFF ' + ','.join(diffs)}")
    print(f"rows {len(a['rows'])}, differing {bad}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--pack", required=True)
    d.add_argument("--model", required=True)
    d.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    return dump(args) if args.cmd == "dump" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
