"""负责人 10-18 晚定案 ③ 的模拟：分支维度降权、不删行（只读，不改 export）。

python claude/_longtail_work/cap_downweight.py > claude/_longtail_work/cap_downweight.json

口径：
- 动作行（unit=step）sample_weight 上限 3 → 2；报告行不动（COMPLETED 无错误仍 3.0）。
- "等效行数" = sample_weight 之和（权重 1 = 一行）。
- 每案例等效行数 > M：该案例全部行按同一系数缩到 M；之后每分支等效行数 > N：该分支全部行按同一系数缩到 N（报告行不受 N 限制，同 cap_sim）。
- 候选组合 = 四节模拟表的 (N, M)：(150, 20) 《可训性评估》建议、(150, 15)、(100, 12)、(60, 10)；都满足"每分支 ≤ 150、每案例 ≤ 20"。
- 长尾 = 该集合里 < 15 行或 < 5 个案例的分支（按行数判，降权不改行数）；长尾占比 = 长尾分支的权重和 / 总权重。
- 选法（定案 ③）：长尾占比最高且 E 族权重占比 < 50% 的一组；同时列 C 族（sft0 里最大的族）与 READ_RECORD 占比。
集合：r1（四节基准）与 sft0 = r1 + g1（定案 ②：r0 包不进；g2 生成完后重算）。
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_overlap import SOURCES, family_of  # noqa: E402

CANDIDATES = [("不处理", None, None), ("仅权重上限 2", None, None), ("D_doc", 150, 20), ("A", 150, 15), ("B", 100, 12), ("C", 60, 10)]
STEP_CAP = 2.0


def load_hi(path: Path) -> list[dict]:
    with (path / "hi_accepted.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line)["metadata"] for line in f if line.strip()]


def weights(rows: list[dict], label: str, n: int | None, m: int | None) -> list[float]:
    w = [float(r["sample_weight"] or 0) for r in rows]
    if label == "不处理":
        return w
    w = [min(x, STEP_CAP) if r["unit"] == "step" else x for r, x in zip(rows, w)]
    if m is not None:
        by_case: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(rows):
            by_case[r["case_id"]].append(i)
        for idx in by_case.values():
            tot = sum(w[i] for i in idx)
            if tot > m:
                for i in idx:
                    w[i] *= m / tot
    if n is not None:
        by_branch: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(rows):
            if not r["branch"].startswith("report_"):
                by_branch[r["branch"]].append(i)
        for idx in by_branch.values():
            tot = sum(w[i] for i in idx)
            if tot > n:
                for i in idx:
                    w[i] *= n / tot
    return w


def stats(rows: list[dict], w: list[float]) -> dict:
    total = sum(w)
    rows_b = Counter(r["branch"] for r in rows)
    cases_b = defaultdict(set)
    for r in rows:
        cases_b[r["branch"]].add(r["case_id"])
    tail = {b for b in rows_b if rows_b[b] < 15 or len(cases_b[b]) < 5}
    wb, wf = Counter(), Counter()
    rr = 0.0
    for r, x in zip(rows, w):
        wb[r["branch"]] += x
        wf[family_of(r["case_id"])] += x
        if r.get("action") == "READ_RECORD":
            rr += x
    per_case = Counter()
    for r, x in zip(rows, w):
        per_case[r["case_id"]] += x
    return {"weight": round(total, 1), "tail_branches": sorted(tail), "tail_share": round(sum(wb[b] for b in tail) / total, 4),
            "family_share": {k: round(v / total, 3) for k, v in wf.most_common()}, "read_record_share": round(rr / total, 3),
            "max_case_weight": round(max(per_case.values()), 1),
            "branches": {b: {"rows": rows_b[b], "weight": round(wb[b], 1), "share": round(wb[b] / total, 4)}
                         for b, _ in wb.most_common()}}


def main() -> None:
    r1 = load_hi(SOURCES["r1"])
    sft0 = r1 + load_hi(SOURCES["g1"])
    out = {}
    for name, rows in (("r1", r1), ("sft0_r1_g1", sft0)):
        res = {label: {"N": n, "M": m, **stats(rows, weights(rows, label, n, m))} for label, n, m in CANDIDATES}
        ok = [k for k, v in res.items() if v["N"] is not None and v["family_share"].get("E", 0) < 0.5]
        res["_pick"] = max(ok, key=lambda k: res[k]["tail_share"]) if ok else None
        out[name] = res
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
