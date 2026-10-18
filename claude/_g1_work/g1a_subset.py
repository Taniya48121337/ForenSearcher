"""g1a：从 _gold_sample.py --oversample-flagged 抽出的 25 条核验器标错 Finding（+ 6 答案）里按族取 14 条，重编号后重出 items / strata / 标注页。

_gold_sample 的 --oversample-flagged 会把标错的 Finding 全取（不受 --findings 限制），所以先抽到草稿目录、再用这里按族收窄：
B 族标错的只有 1 条（全取），A 4 条，C 9 条（GroundLie360 4、FMNV-C 5），每个 run 至多 1 条；答案 6 条（负 3 正 3）原样保留。
挑选按草稿里的顺序（_gold_sample 的 seed 排序），不看内容。

    python claude/_g1_work/g1a_subset.py <草稿目录> experiments/sft_data/gold_set/g1a
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("gs", ROOT / "scripts" / "_gold_sample.py")
gs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gs)  # type: ignore[union-attr]

QUOTA = {"B": 1, "A": 4, "C_groundlie360": 4, "C_fmnv": 5}


def bucket(run: str) -> str:
    if run.startswith(("videosham_", "vcsl_")):
        return "B"
    if run.startswith("fmnv_a_"):
        return "A"
    return "C_groundlie360" if run.startswith("groundlie360_") else "C_fmnv"


def main() -> int:
    draft, out = Path(sys.argv[1]), Path(sys.argv[2])
    items = json.loads((draft / "items.json").read_text(encoding="utf-8"))
    strata = json.loads((draft / "strata.json").read_text(encoding="utf-8"))
    by_id = {s["id"]: s for s in strata["strata"]}
    taken, runs, got = [], set(), {k: 0 for k in QUOTA}
    for it in items["items"]:
        sid = it["id"]
        if sid.startswith("A"):
            continue
        st = by_id[sid]
        b = bucket(st["run"])
        if got[b] < QUOTA[b] and st["run"] not in runs:
            taken.append(sid)
            runs.add(st["run"])
            got[b] += 1
    keep = taken + [it["id"] for it in items["items"] if it["id"].startswith("A")]
    rename = {old: (f"F{i:03d}" if old.startswith("F") else old) for i, old in enumerate(taken, 1)}
    rename |= {old: old for old in keep if old.startswith("A")}
    new_items, new_strata = [], []
    for it in items["items"]:
        if it["id"] in rename:
            new_items.append({**it, "id": rename[it["id"]]})
    for st in strata["strata"]:
        if st["id"] in rename:
            new_strata.append({**st, "id": rename[st["id"]], "draft_id": st["id"], "family_bucket": bucket(st["run"])})
    meta = {**items["meta"], "batch": "g1a", "subset_of_draft": str(draft.name), "quota": QUOTA, "picked": got,
            "counts": {"finding": len(taken), "answer": len(keep) - len(taken), "plan": 0}}
    if out.exists():
        raise SystemExit(f"{out} 已存在")
    (out / "media").mkdir(parents=True)
    blob = json.dumps(new_items, ensure_ascii=False)
    for img in (draft / "media").glob("*"):
        if f"media/{img.name}" in blob:
            shutil.copy2(img, out / "media" / img.name)
    gs.write_batch(out, meta, new_items, new_strata)
    print(json.dumps({"picked": got, "rename": rename}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
