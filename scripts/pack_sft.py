"""把多轮导出的 accepted 行合成一份训练集（2026-10-14 晚，服务器放量用）。training/sft_lora.py 读一个 export 目录的
<role>_accepted.jsonl，这个脚本产出的目录可以直接当它的 export_dir。

    python scripts/pack_sft.py --exports experiments/sft_data/export/ws1_* experiments/sft_data/export/ws2_* --out experiments/sft_data/sft_pack/p1

做的事：
  - 只收 accepted；teacher_model 可按 --teachers 过滤（例如只收服务器 Qwen 跑的）；--exclude-arms 去掉 wk11b/g/n 这类对照拷贝
  - 去重：同一 run 同一 request 只留一次（同一 run 被几轮导出时）；输出文本完全相同的终端行（报告 / 答案）只留一条
  - 每个案例家族封顶（HI ≤ --cap-hi、GIA plan ≤ --cap-plan、answer ≤ --cap-answer），与 export 的家族封顶同口径，跨轮再封一次
  - 图片复制进新目录，images 路径改写；行内 metadata.pack 记来源目录
  - manifest.json：各角色 / 单元 / 动作 / 族 / 教师 / 数据集的行数与权重，输入 token 估计分位数
不改任何权重规则（权重在 export 里定），也不碰 held / rejected。
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARM_SUFFIXES = ("b", "g", "n", "qv")


def _is_arm(run: str) -> bool:
    tag = run.rsplit("_", 2)[-2] if run.count("_") >= 2 else ""
    return tag.startswith("wk") and tag[-1:] in ("b", "g", "n") and tag[2:-1].isdigit()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exports", nargs="+", required=True, help="export 目录（可用通配符）")
    parser.add_argument("--out", required=True)
    parser.add_argument("--teachers", nargs="*", default=None, help="只收这些 teacher_model（子串匹配）")
    parser.add_argument("--exclude-arms", action="store_true", default=True)
    parser.add_argument("--cap-hi", type=int, default=40)
    parser.add_argument("--cap-plan", type=int, default=5)
    parser.add_argument("--cap-answer", type=int, default=3)
    args = parser.parse_args()
    dirs: list[Path] = []
    for pattern in args.exports:
        dirs += [Path(p) for p in sorted(glob.glob(pattern)) if (Path(p) / "manifest.json").is_file()]
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    seen_req: set[tuple[str, str, str]] = set()
    seen_text: set[str] = set()
    family_count: Counter[tuple[str, str]] = Counter()
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dropped: Counter[str] = Counter()
    for d in dirs:
        for role in ("hi", "gia"):
            path = d / f"{role}_accepted.jsonl"
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                m = row.get("metadata") or {}
                run = str(m.get("run") or "")
                if args.exclude_arms and _is_arm(run):
                    dropped["arm_copy"] += 1
                    continue
                if args.teachers and not any(t in str(m.get("teacher_model") or "") for t in args.teachers):
                    dropped["teacher_filter"] += 1
                    continue
                key = (role, run, str(m.get("request")))
                if key in seen_req:
                    dropped["same_request"] += 1
                    continue
                target = json.dumps(row["messages"][-1], ensure_ascii=False, sort_keys=True)
                if m.get("unit") in ("report", "answer"):
                    digest = hashlib.sha1(target.encode("utf-8")).hexdigest()
                    if digest in seen_text:
                        dropped["same_terminal_text"] += 1
                        continue
                    seen_text.add(digest)
                cap_key = "hi" if role == "hi" else ("gia_" + str(m.get("unit")))
                cap = {"hi": args.cap_hi, "gia_plan": args.cap_plan, "gia_answer": args.cap_answer}.get(cap_key, 10**9)
                fam = (cap_key, str(m.get("case_family") or m.get("case_id")))
                if family_count[fam] >= cap:
                    dropped["family_cap"] += 1
                    continue
                family_count[fam] += 1
                seen_req.add(key)
                new_images = []
                for rel in row.get("images") or []:
                    src = d / rel
                    dst = out / "images" / run / str(m.get("request")) / Path(rel).name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if src.is_file() and not dst.exists():
                        shutil.copy2(src, dst)
                    new_images.append(str(dst.relative_to(out)).replace("\\", "/"))
                row["images"] = new_images
                m["pack"] = {"from": d.name}
                row["metadata"] = m
                rows[role].append(row)
    for role in ("hi", "gia"):
        with (out / f"{role}_accepted.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows[role]:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def dist(items: list[dict[str, Any]], key: str) -> dict[str, int]:
        return dict(Counter(str((r["metadata"] or {}).get(key)) for r in items).most_common())

    def q(values: list[int], p: float) -> int:
        return sorted(values)[int(p * (len(values) - 1))] if values else 0

    manifest = {
        "exports": [d.name for d in dirs], "dropped": dict(dropped),
        "rows": {role: len(rows[role]) for role in rows},
        "weight_sum": {role: round(sum(float(r["metadata"].get("sample_weight") or 0) for r in rows[role]), 2) for role in rows},
        "by_unit": {role: dist(rows[role], "unit") for role in rows},
        "by_action": {role: dist(rows[role], "action") for role in rows},
        "by_dataset": {role: dist(rows[role], "dataset") for role in rows},
        "by_teacher": {role: dist(rows[role], "teacher_model") for role in rows},
        "families": len({r["metadata"].get("case_family") for role in rows for r in rows[role]}),
        "input_tokens_est": {role: {"p50": q([int(r["metadata"].get("input_tokens_est") or 0) for r in rows[role]], 0.5),
                                    "p95": q([int(r["metadata"].get("input_tokens_est") or 0) for r in rows[role]], 0.95),
                                    "max": q([int(r["metadata"].get("input_tokens_est") or 0) for r in rows[role]], 1.0)}
                             for role in rows},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
