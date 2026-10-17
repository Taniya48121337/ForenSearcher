"""r1 全量案卷清单：r0g 的非 E/L 案例（71，含 2 个 v9 变体）+ 按新"可回答"定义重建的 E / L。

E / L 规则（负责人 10-17）：探测可回答的全取；不可回答的每族保留 ≤ 1/3 行（不可回答数 ≤ 可回答数 // 2）做"如实 INCOMPLETE"
教学，程序标 unanswerable、排除出 B6 分母。保留哪几个不可回答：先 r0g 里已有的（同案例 r0 → r1 对照），再按编号顺序。
未判定（两次检索都报错）的不进清单。

    python claude/_r1a_work/build_r1_cases.py [--probe experiments/sft_data/r1_answerable_probe.json]
→ experiments/sft_data/walker_r1_cases.json（d1 的 --case-list）、experiments/sft_data/unanswerable_r1.json（门 --unanswerable 读其中
  unanswerable_case_ids）
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SFT = ROOT / "experiments" / "sft_data"


def num(name: str) -> int:
    return int(re.sub(r"\D", "", name.rsplit("_", 1)[-1]) or 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=str(SFT / "r1_answerable_probe.json"))
    args = ap.parse_args()
    probe = json.loads(Path(args.probe).read_text(encoding="utf-8"))
    rows = {r["case"]: r for r in probe["rows"]}
    r0g = json.loads((SFT / "walker_r0g_cases.json").read_text(encoding="utf-8"))
    base = [c for c in r0g if not c.startswith("averimatec_")]
    r0g_el = {c for c in r0g if c.startswith("averimatec_")}
    chosen: dict[str, list[str]] = {}
    kept_un: list[str] = []
    detail = []
    for fam in ("E", "L"):
        fam_rows = [r for r in rows.values() if r["family"] == fam]
        ans = sorted((r["case"] for r in fam_rows if r.get("answerable") is True), key=num)
        un = sorted((r["case"] for r in fam_rows if r.get("answerable") is False), key=lambda c: (c not in r0g_el, num(c)))
        cap = len(ans) // 2                     # 不可回答 ≤ 1/3 行  ⇔  un ≤ ans / 2
        keep = un[:cap]
        chosen[fam] = sorted(ans + keep, key=num)
        kept_un += keep
        for r in sorted(fam_rows, key=lambda r: num(r["case"])):
            detail.append({"case": r["case"], "case_id": r.get("case_id"), "family": fam, "answerable": r.get("answerable"),
                           "in_r1": r["case"] in chosen[fam], "in_r0g": r["case"] in r0g_el, "reason": r.get("reason"),
                           "case_file": r.get("case_file")})
    cases = base + chosen["E"] + chosen["L"]
    (SFT / "walker_r1_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    case_id = {r["case"]: r.get("case_id") for r in rows.values()}
    payload = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "method": "r1 新定义（负责人 10-17）：GT 可回答且引擎前 10 召回（以图搜 + 文字搜）里有载图页才算可回答；探测见 "
                         "scripts/check_unanswerable.py → r1_answerable_probe.json。不可回答的每族保留 ≤ 1/3 行，排除出 B6 分母。",
               "probe": Path(args.probe).name, "probe_generated": probe.get("generated"),
               "summary": {fam: {"answerable": sum(1 for d in detail if d["family"] == fam and d["answerable"] is True),
                                 "unanswerable": sum(1 for d in detail if d["family"] == fam and d["answerable"] is False),
                                 "undetermined": sum(1 for d in detail if d["family"] == fam and d["answerable"] is None),
                                 "in_r1": len(chosen[fam]),
                                 "in_r1_unanswerable": sum(1 for c in kept_un if c in chosen[fam])} for fam in ("E", "L")},
               "unanswerable_case_ids": sorted(case_id[c] for c in kept_un),
               "rows": detail}
    (SFT / "unanswerable_r1.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "base": len(base), **payload["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
