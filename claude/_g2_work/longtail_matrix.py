"""教学矩阵（长尾报告二节口径）+ 坏习惯列：按 metadata.branch（HI + GIA），sft0 成员逐包列行 / 案例，四档按 sft0 合计判。只读。

    python claude/_g2_work/longtail_matrix.py --pack r1 <导出> <runs> qwen --pack g1 <导出> <runs> codex [--pack g2 ...] --out claude/_g2_work/longtail_matrix

档（《长尾分支补量方案》二节）：可学 = 行 ≥ 15 且 ≥ 5 个不同 case_id；薄 = 1–14 行（或 ≥ 15 行但 < 5 案例）；零·期望 ≥ 1 = 0 行且覆盖表
MATRIX 最少次数 > 0、不在允许为零 / 按设计为零；零·允许为零 = 其余 0 行。覆盖表类别取 scripts/walker_coverage.py，再按定案 ④ 把
get_frames pages、image→video 记"允许为零"（理由：数据里没有扫描件 / walker 不可达）；image→text 保留"期望≥1（口径问题，五节 ⑥）"。
合并按定案 ②：同一 case_id 只留最高版本包的行（包按命令行顺序，后者为高）。
坏习惯列（程序口径同 scripts/teaching_matrix.py）：模板化理由（rationale 把 E/R/F 编号与数字换占位符后在 sft0 里出现 ≥ 3 次的行）、
比对前说匹配（全部 / 断言式）、GIA 答案含可见输入里没有的实体（只对 GIA FINISH）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import teaching_matrix as tm  # noqa: E402
import walker_coverage as wc  # noqa: E402

DECIDED_ALLOWED = {"INSPECT get_frames pages": "允许为零（定案 ④：数据里没有扫描件）",
                   "SEARCH image→video": "允许为零（定案 ④：walker 不可达）"}


def coverage() -> dict[str, str]:
    out = {}
    for name, _why, _fam, minimum in wc.MATRIX:
        if name in wc.BY_DESIGN_ZERO:
            out[name] = "按设计为零"
        elif name in wc.ALLOWED_ZERO:
            out[name] = "允许为零"
        else:
            out[name] = "期望≥1" if minimum > 0 else "条件触发（最少 0）"
    out.update(DECIDED_ALLOWED)
    out["SEARCH image→text"] = "条件触发（最少 0；10-18 定：E/L 改走 image→image）"
    out["report_completed"] = out["report_incomplete"] = "表外（报告）"
    out.setdefault("SEARCH image→image", "表外（r1 新）")
    return out


def tier(rows: int, cases: int, cov: str) -> str:
    if rows >= 15 and cases >= 5:
        return "可学"
    if rows >= 1:
        return "薄·条件触发" if cov.startswith("条件触发") else "薄"
    return "零·期望≥1" if cov.startswith("期望≥1") else "零·允许为零"


def skeleton(r: str) -> str:
    return re.sub(r"\d+(?:\.\d+)?", "N", tm.EID_RE.sub("#", r))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", nargs=4, action="append", metavar=("NAME", "EXPORT", "RUNS", "TEACHER"), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cov = coverage()
    loaded = []
    for name, export, runs, teacher in args.pack:
        rc = tm.RunCache(Path(runs))
        rows = [r for role in ("hi", "gia") for r in tm.jsonl(Path(export) / f"{role}_accepted.jsonl")]
        loaded.append((name, teacher, rc, rows))
    # 定案 ②：同一 case_id 只留最高版本包
    owner = {}
    for i, (name, _t, _rc, rows) in enumerate(loaded):
        for r in rows:
            owner[r["metadata"]["case_id"]] = i
    kept = [(name, teacher, rc, r) for i, (name, teacher, rc, rows) in enumerate(loaded) for r in rows if owner[r["metadata"]["case_id"]] == i]
    dropped = {name: sum(1 for r in rows if owner[r["metadata"]["case_id"]] != i) for i, (name, _t, _rc, rows) in enumerate(loaded)}
    sk = Counter(skeleton(tm.norm_rationale(tm.target(r).get("rationale"))) for *_x, r in kept if tm.target(r).get("rationale"))
    stats = defaultdict(lambda: {"rows": 0, "cases": set(), "packs": defaultdict(lambda: [0, set()]), "teacher": Counter(),
                                 "rat": 0, "tmpl": 0, "match": 0, "match_assert": 0, "gia_ans": 0, "gia_miss": 0, "families": Counter()})
    for name, teacher, rc, r in kept:
        md = r["metadata"]
        b = md.get("branch") or "(none)"
        if md.get("role") == "gia":
            b = {"answer": "GIA FINISH", "plan": "GIA INVESTIGATE round2" if str(md.get("walker_rule", "")).startswith("G2") else "GIA INVESTIGATE round1"}.get(md.get("unit"), b)
        s = stats[b]
        s["rows"] += 1
        s["cases"].add(md["case_id"])
        s["packs"][name][0] += 1
        s["packs"][name][1].add(md["case_id"])
        s["teacher"][teacher] += 1
        s["families"][tm.family(md["case_id"])] += 1
        tgt = tm.target(r)
        rat = tm.norm_rationale(tgt.get("rationale"))
        if rat:
            s["rat"] += 1
            if sk[skeleton(rat)] >= 3:
                s["tmpl"] += 1
            if md.get("role") == "hi" and md.get("unit") == "step" and tm.MATCH_RE.search(rat):
                cards = rc.cards(str(md["run"]), str(md["request"]))
                had = any(str(c.get("acquired_by") or "").startswith("INSPECT/compare_media") for c in cards.values()) or \
                    any(op == "INSPECT:compare_media" for op in rc.history_ops(str(md["run"]), str(md["request"])))
                if not had:
                    s["match"] += 1
                    is_cmp = ((tgt.get("arguments") or {}).get("request") or {}).get("operation") == "compare_media"
                    if not is_cmp and not tm.HEDGE_RE.search(rat):
                        s["match_assert"] += 1
        if md.get("role") == "gia" and md.get("unit") == "answer":
            s["gia_ans"] += 1
            if tm.entity_misses(str(tgt.get("answer") or ""), tm.visible_text(r)):
                s["gia_miss"] += 1
    names = list(dict.fromkeys([*sorted(stats, key=lambda b: -stats[b]["rows"]), *cov]))
    packs = [p[0] for p in loaded]
    table = []
    for b in names:
        s = stats.get(b)
        rows, cases = (s["rows"], len(s["cases"])) if s else (0, 0)
        c = cov.get(b, "表外")
        t = tier(rows, cases, c)
        if c == "按设计为零" and rows == 0:
            t = "零·允许为零"
        table.append({"branch": b, "coverage": c, "tier": t, "rows": rows, "cases": cases,
                      "packs": {p: ([s["packs"][p][0], len(s["packs"][p][1])] if s and p in s["packs"] else [0, 0]) for p in packs},
                      "teacher": dict(s["teacher"]) if s else {}, "families": dict(s["families"]) if s else {},
                      "rationale_rows": s["rat"] if s else 0, "templated": s["tmpl"] if s else 0,
                      "match_before_compare": s["match"] if s else 0, "match_assertive": s["match_assert"] if s else 0,
                      "gia_answers": s["gia_ans"] if s else 0, "gia_unseen_entity": s["gia_miss"] if s else 0})
    order = {"可学": 0, "薄": 1, "薄·条件触发": 2, "零·期望≥1": 3, "零·允许为零": 4}
    table.sort(key=lambda x: (order[x["tier"]], -x["rows"]))
    tot = {k: sum(x[k] for x in table) for k in ("rows", "rationale_rows", "templated", "match_before_compare", "match_assertive", "gia_answers", "gia_unseen_entity")}
    out = {"packs": packs, "dropped_by_dedup": dropped, "tiers": dict(Counter(x["tier"] for x in table)), "totals": tot, "table": table}
    Path(args.out).with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    head = ("| 分支（metadata.branch） | 覆盖表 | " + " | ".join(f"{p} 行/案例" for p in packs) +
            " | sft0 行/案例 | 族 | 档 | 模板化理由 | 比对前说匹配（全部 / 断言式） | GIA 未见实体 |")
    lines = [f"sft0 = {' + '.join(packs)}（定案 ② 去重丢行：{dropped}）；档：{out['tiers']}", "", head, "|" + "---|" * (8 + len(packs))]
    for x in table:
        fam = " ".join(f"{k}{v}" for k, v in sorted(x["families"].items(), key=lambda kv: -kv[1]))
        pk = " | ".join(f"{a}/{b}" for a, b in x["packs"].values())
        tmpl = f"{x['templated']}/{x['rationale_rows']}" if x["rationale_rows"] else "—"
        mb = f"{x['match_before_compare']} / {x['match_assertive']}" if x["match_before_compare"] else "0"
        ge = f"{x['gia_unseen_entity']}/{x['gia_answers']}" if x["gia_answers"] else "—"
        lines.append(f"| {x['branch']} | {x['coverage']} | {pk} | {x['rows']}/{x['cases']} | {fam} | **{x['tier']}** | {tmpl} | {mb} | {ge} |")
    hi_steps = sum(x["rationale_rows"] for x in table if not x["branch"].startswith(("GIA", "report_")))
    lines += ["", f"合计：模板化理由 {tot['templated']}/{tot['rationale_rows']} = {tot['templated'] / max(1, tot['rationale_rows']):.1%}；"
              f"比对前说匹配 {tot['match_before_compare']}（{tot['match_before_compare'] / max(1, hi_steps):.1%} 的 HI 动作理由行）/ 断言式 "
              f"{tot['match_assertive']}（{tot['match_assertive'] / max(1, hi_steps):.1%}）；GIA 答案未见实体 {tot['gia_unseen_entity']}/{tot['gia_answers']} = "
              f"{tot['gia_unseen_entity'] / max(1, tot['gia_answers']):.1%}"]
    Path(args.out).with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
