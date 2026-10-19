"""sft0 合包清单：哪些行合包时跳过（行不删、包不动、export 规则不动；merge_packs.py --hold 读这份清单）（2026-10-18）。

    python scripts/sft0_holds.py premature_match --pack r1 <包目录> <runs 目录> --pack g1 ... --pack g2 ... \
        --out experiments/sft_data/sft0_holds/hold_premature_match.json

premature_match（负责人 10-18 定）："比对前就说匹配"的**断言式**动作行——HI 动作行的 rationale 含 match / same（词边界），
该步之前没有 compare_media 记录（证据卡片里没有 INSPECT/compare_media、历史里没有 compare_media 调用），且不是 compare_media 那一步
本身、也不是 whether / if / check / verify 一类带保留的问句（口径同 scripts/teaching_matrix.py 的"断言式"）。
带保留措辞的行照常进；不改写教师文本。合包时这些行记 bucket_reason = rationale_premature_match。

gold（负责人 10-19 定）：金标里人工判 unacceptable、而核验器放进了 accepted 的单元，按金标条目 id 从 strata.json 取 run + request，
记给定的 bucket_reason（例：g1a A006 → omitted_required_fact）。答案 / 计划单元是 gia 角色，Finding 所在报告是 hi 角色。

    python scripts/sft0_holds.py gold --pack g1 --item <金标目录>:A006 --reason omitted_required_fact \
        --out experiments/sft_data/sft0_holds/hold_gold_unacceptable.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import teaching_matrix as tm  # noqa: E402


def premature_match(packs: list[list[str]]) -> list[dict]:
    out = []
    for name, pack_dir, runs in packs:
        rc = tm.RunCache(Path(runs))
        for row in tm.jsonl(Path(pack_dir) / "data" / "hi_accepted.jsonl"):
            md = row.get("metadata") or {}
            if md.get("unit") != "step":
                continue
            tgt = tm.target(row)
            rat = tm.norm_rationale(tgt.get("rationale"))
            if not rat or not tm.MATCH_RE.search(rat):
                continue
            run, req = str(md.get("run")), str(md.get("request"))
            cards = rc.cards(run, req)
            had = any(str(c.get("acquired_by") or "").startswith("INSPECT/compare_media") for c in cards.values()) or \
                any(op == "INSPECT:compare_media" for op in rc.history_ops(run, req))
            is_cmp = ((tgt.get("arguments") or {}).get("request") or {}).get("operation") == "compare_media"
            if had or is_cmp or tm.HEDGE_RE.search(rat):
                continue
            out.append({"pack": name, "role": "hi", "run": run, "request": req, "case_id": md.get("case_id"),
                        "walker_rule": md.get("walker_rule"), "branch": md.get("branch"), "rationale": rat})
    return out


def plan_errors(packs: list[list[str]], units: list[str], err_type: str, severity: str) -> list[dict]:
    """10-19 负责人定：GIA 计划行（第 2 轮派工）里核验器标了 err_type（且严重度 = severity）的，按 run + request hold。
    packs：NAME PACK_DIR（读包 data/gia_accepted.jsonl 的 plan 行）；units：各批的 verdicts_units_*.jsonl。"""

    flagged: dict[tuple[str, str], list[dict]] = {}
    for f in units:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("unit") != "plan":
                continue
            errs = [e for e in d.get("errors") or [] if e.get("type") == err_type and e.get("severity") == severity]
            if errs:
                flagged[(d["subject"]["run"], d["subject"].get("request"))] = errs
    out = []
    for name, pack_dir, *_ in packs:
        for row in tm.jsonl(Path(pack_dir) / "data" / "gia_accepted.jsonl"):
            md = row.get("metadata") or {}
            key = (str(md.get("run")), str(md.get("request")))
            if md.get("unit") == "plan" and key in flagged:
                out.append({"pack": name, "role": "gia", "run": key[0], "request": key[1], "case_id": md.get("case_id"),
                            "walker_rule": md.get("walker_rule"), "errors": [{"detail": e.get("detail"), "span": e.get("span")} for e in flagged[key]]})
    return out


def single_rows(specs: list[str], packs: list[list[str]]) -> list[dict]:
    dirs = {p[0]: Path(p[1]) for p in packs if len(p) >= 2}
    out = []
    for spec in specs:
        pack, role, run, req = spec.split(":", 3)
        row = next((r for r in tm.jsonl(dirs[pack] / "data" / f"{role}_accepted.jsonl")
                    if str(r["metadata"].get("run")) == run and str(r["metadata"].get("request")) == req), None)
        if row is None:
            raise SystemExit(f"{spec}：包里没有这一行")
        md = row["metadata"]
        out.append({"pack": pack, "role": role, "run": run, "request": req, "case_id": md.get("case_id"), "unit": md.get("unit"),
                    "status": tm.target(row).get("status") or tm.target(row).get("decision")})
    return out


def gold_items(items: list[str], pack: str) -> list[dict]:
    out = []
    for spec in items:
        batch, _, item_id = spec.rpartition(":")
        strata = json.loads((Path(batch) / "strata.json").read_text(encoding="utf-8"))["strata"]
        st = next(x for x in strata if x["id"] == item_id)
        role = "gia" if item_id.startswith(("A", "P")) else "hi"
        out.append({"pack": pack, "role": role, "run": st["run"], "request": st.get("request"), "case_id": st.get("case_id"),
                    "gold_item": f"{Path(batch).name}:{item_id}", "verifier_disposition": st.get("verifier_disposition")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["premature_match", "gold", "plan_error", "row"])
    ap.add_argument("--row", nargs="*", default=[], help="row：PACK:ROLE:RUN:REQUEST")
    ap.add_argument("--rule", default=None, help="row：口径说明")
    ap.add_argument("--units", nargs="*", default=[], help="plan_error：verdicts_units_*.jsonl")
    ap.add_argument("--type", default="task_not_gap_driven", help="plan_error：错误类型")
    ap.add_argument("--severity", default="major", help="plan_error：严重度")
    ap.add_argument("--pack", nargs="+", action="append", metavar="NAME [PACK_DIR RUNS]", required=True,
                    help="premature_match：NAME PACK_DIR RUNS（可多次）；gold：只要 NAME")
    ap.add_argument("--item", nargs="*", default=[], help="gold：<金标目录>:<条目 id>")
    ap.add_argument("--reason", default=None, help="gold：bucket_reason")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.kind == "row":
        if not args.reason or not args.row or not args.rule:
            ap.error("row 需要 --row、--reason 与 --rule")
        rows, reason, rule = single_rows(args.row, args.pack), args.reason, args.rule
    elif args.kind == "plan_error":
        rows, reason = plan_errors(args.pack, args.units, args.type, args.severity), f"plan_{args.type}"
        rule = f"GIA 第 2 轮派工行里核验器标 {args.type}（{args.severity}）的（负责人 10-19 定 hold）"
    elif args.kind == "gold":
        if not args.reason or not args.item:
            ap.error("gold 需要 --item 与 --reason")
        rows, reason = gold_items(args.item, args.pack[0][0]), args.reason
        rule = "金标人工判 unacceptable、核验器放进 accepted 的单元（负责人 10-19 定 hold）"
    else:
        if any(len(p) != 3 for p in args.pack):
            ap.error("premature_match 的 --pack 要 NAME PACK_DIR RUNS")
        rows, reason = premature_match(args.pack), "rationale_premature_match"
        rule = __doc__.split("premature_match（", 1)[1].split("\n\n")[0]
    out = {"kind": args.kind, "bucket_reason": reason, "made_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "made_by": "scripts/sft0_holds.py", "rule": rule,
           "packs": [p[0] for p in args.pack], "n": len(rows),
           "by_pack": {p[0]: sum(1 for r in rows if r["pack"] == p[0]) for p in args.pack}, "rows": rows}
    path = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n", "by_pack")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
