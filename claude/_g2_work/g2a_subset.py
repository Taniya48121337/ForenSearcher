"""g2a：从 _gold_sample.py 抽出的 g2 全量草稿（191 Finding + 60 答案）里按"补量格的新动作"取 Finding 10 + 答案 5，重编号重出标注页。

    python claude/_g2_work/g2a_subset.py claude/_g2_work/g2a_draft experiments/sft_data/gold_set/g2a

Finding 名额（g2 补的是 A 族找原片与 B 族 VideoSham）：按 Finding 所引证据的 acquired_by 归类——
  A_compare（A 族、引 compare_media 记录）3、A_frames（A 族、引剪点旁的 get_frames 帧，且不在上一类）2、
  A_dates（A 族、引原片的日期查询记录 SEARCH/dates）2、B_ocr（B 族、引放大区域的 OCR 记录——M3c，文字篡改）3。
  类内核验器标 major/critical 的先取（过采样），再按草稿顺序；每个 run 至多 1 条。B_ocr 不足 3 条时差额由 B_compare（B 族、引 compare_media 记录）补。
答案名额：A 3、B 2；类内 negative / positive 尽量各半。
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

QUOTA = [("A_compare", 3), ("A_frames", 2), ("A_dates", 2), ("B_ocr", 3), ("B_compare", 0)]   # B_ocr 不够的名额由 B_compare 补
ANS_QUOTA = {"A": 3, "B": 2}


def fam(run: str) -> str:
    return "A" if run.startswith("fmnv_a_") else "B" if run.startswith(("videosham_", "vcsl_")) else "C"


def kinds(item: dict) -> set[str]:
    acq = {str(e.get("acquired_by") or "") for e in item.get("evidence") or []}
    f = fam(item["run"])
    out = set()
    if f == "A":
        if any(a.startswith("INSPECT/compare_media") for a in acq):
            out.add("A_compare")
        if any(a.startswith("INSPECT/get_frames") for a in acq):
            out.add("A_frames")
        if any(a.startswith("SEARCH/dates") or a.endswith("/dates") for a in acq):
            out.add("A_dates")
    if f == "B" and any(a.startswith("INSPECT/ocr") for a in acq):
        out.add("B_ocr")
    if f == "B" and any(a.startswith("INSPECT/compare_media") for a in acq):
        out.add("B_compare")
    return out


def main() -> int:
    draft, out = Path(sys.argv[1]), Path(sys.argv[2])
    items = json.loads((draft / "items.json").read_text(encoding="utf-8"))
    strata = {s["id"]: s for s in json.loads((draft / "strata.json").read_text(encoding="utf-8"))["strata"]}
    finds = [x for x in items["items"] if x["id"].startswith("F")]
    taken, runs, picked = [], set(), {}
    short = 0
    for cat, n in QUOTA:
        n = n + short if cat == "B_compare" else n
        pool = [x for x in finds if cat in kinds(x) and x["id"] not in {t for t, _ in taken}]
        pool.sort(key=lambda x: (not strata[x["id"]].get("verifier_flagged"), finds.index(x)))
        got = 0
        for x in pool:
            if got >= n:
                break
            if x["run"] in runs:
                continue
            taken.append((x["id"], cat))
            runs.add(x["run"])
            got += 1
        picked[cat] = got
        short = n - got if cat == "B_ocr" else short
    answers = [x for x in items["items"] if x["id"].startswith("A")]
    ans_taken = []
    for f, n in ANS_QUOTA.items():
        pool = [x for x in answers if fam(x["run"]) == f]
        neg = [x for x in pool if strata[x["id"]].get("verifier_disposition") == "negative"]
        pos = [x for x in pool if strata[x["id"]].get("verifier_disposition") != "negative"]
        want_neg = n // 2 if f == "B" else (n + 1) // 2
        sel = neg[:want_neg] + pos[: n - min(want_neg, len(neg))]
        ans_taken += [x["id"] for x in sel[:n]]
    rename = {old: f"F{i:03d}" for i, (old, _c) in enumerate(taken, 1)} | {old: f"A{i:03d}" for i, old in enumerate(ans_taken, 1)}
    cat_of = dict(taken)
    new_items = [{**x, "id": rename[x["id"]]} for x in items["items"] if x["id"] in rename]
    new_items.sort(key=lambda x: x["id"])
    new_strata = [{**strata[old], "id": new, "draft_id": old, "fill_cell": cat_of.get(old), "family": fam(strata[old]["run"])}
                  for old, new in rename.items()]
    meta = {**items["meta"], "batch": "g2a", "subset_of_draft": draft.name, "quota": dict(QUOTA), "picked": picked,
            "answer_quota": ANS_QUOTA, "counts": {"finding": len(taken), "answer": len(ans_taken), "plan": 0}}
    if out.exists():
        raise SystemExit(f"{out} 已存在")
    (out / "media").mkdir(parents=True)
    blob = json.dumps(new_items, ensure_ascii=False)
    for img in (draft / "media").glob("*"):
        if f"media/{img.name}" in blob:
            shutil.copy2(img, out / "media" / img.name)
    gs.write_batch(out, meta, new_items, new_strata)
    print(json.dumps({"picked": picked, "answers": [(rename[a], strata[a]["run"], strata[a].get("verifier_disposition")) for a in ans_taken],
                      "findings": [(rename[f], c, strata[f]["run"], strata[f].get("verifier_flagged")) for f, c in taken]}, ensure_ascii=False, indent=0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
