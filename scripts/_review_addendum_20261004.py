"""复核页补充（10-04 晚）：把核验器候选改动（verifier@0.3.6c）要靠人拍板的地方加进 v1p2_hard 复核页。

1. 口径条目（回指句 / 综合句）：H020、H024、H033、H048、H050、H051。
   - 页面原来只给本句所引证据。报告条目连被回指的前文 Finding 都看不到，人没法判断"回指"成不成立。
   - 现在附上同一报告 / 答案里被依据的句子（及其来源引语），并加一个"宽口径"问题：
     允许依据这些已核实的句子时，这句是否成立。
   - 主判定仍是严格口径（只看本句所引），两个口径各记一份，评估时分别算精确率。
2. strata.json 的 verifier_flags 加一列 "0.3.6c"（程序规则 + 推理核对 + 语义覆盖），同一批人工标签直接评三版。
3. 新增 F 组"语义覆盖抽查"：核验器判"已建立 / 未建立"的要求各抽几条，人判"这些已核验片段是否足以建立该要求"。
   另附一条案卷标签口径题（17838107）。

    python scripts/_review_addendum_20261004.py --batch experiments/sft_data/gold_set/v1p2_hard \
        --r0c-reports R.v036c.jsonl --r0c-units U.v036c.jsonl --r0cn-reports R2.v036c.jsonl --r0cn-units U2.v036c.jsonl --runs RUNS
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "experiments" / "sft_data"
POLICY_ITEMS = ["H020", "H024", "H033", "H048", "H050", "H051"]
_FIRM = ("major", "critical")


def load(p: str | Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def report_statements(runs: Path, run: str, ws: str) -> dict[str, str]:
    from foren_searcher.datagen.verify.calibrate_precision import open_run
    ctx = open_run(runs / run)
    try:
        for rec in ctx.store.workspace_report_records():
            if rec.task_id == ws:
                return {str(f.get("finding_id")): str(f.get("statement") or "") for f in rec.to_dict().get("findings") or []}
    finally:
        ctx.close()
    return {}


def _bigrams(t: str) -> set[str]:
    t = re.sub(r"[\s，。、；：:,.;（）()“”\"'【】\[\]]", "", t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def quotes_of(row: dict[str, Any], sid: str, n: int = 2) -> list[dict[str, str]]:
    out = []
    for j in row.get("judgments") or []:
        if j.get("finding_id") != sid or j.get("verdict") != "supported":
            continue
        for p in j.get("supported") or []:
            if p.get("verified") is False:
                continue
            out.append({"evidence_id": j.get("evidence_id"), "source": str(j.get("source") or "")[:90], "quote": str(p.get("quote") or "")[:260]})
    return out[:n]


def firm_ids(row: dict[str, Any]) -> set[str]:
    return {str((e.get("extra") or {}).get("finding")) for e in row.get("errors") or [] if e.get("severity") in _FIRM}


def flagged(row: dict[str, Any] | None, sid: str) -> tuple[bool, list[str]]:
    if not row:
        return False, []
    types = [e["type"] for e in row.get("errors") or [] if e.get("severity") in _FIRM and (e.get("extra") or {}).get("finding") == sid]
    return bool(types), types


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", default=str(SFT / "gold_set" / "v1p2_hard"))
    p.add_argument("--r0c-reports", required=True)
    p.add_argument("--r0c-units", required=True)
    p.add_argument("--r0cn-reports", required=True)
    p.add_argument("--r0cn-units", required=True)
    p.add_argument("--runs", default=str(SFT / "runs"))
    p.add_argument("--cases", default=str(SFT / "cases"))
    a = p.parse_args()
    batch, runs = Path(a.batch), Path(a.runs)
    spec = json.loads((batch / "review_spec.json").read_text(encoding="utf-8"))
    strata = json.loads((batch / "strata.json").read_text(encoding="utf-8"))
    rep = {(r["subject"]["run"], r["subject"].get("workspace_id")): r for r in load(a.r0c_reports) + load(a.r0cn_reports) if r.get("unit") == "report"}
    ans = {(r["subject"]["run"], r["subject"].get("request")): r for r in load(a.r0c_units) + load(a.r0cn_units) if r.get("unit") == "answer"}

    # ---- 2. 0.3.6c 一列
    for s in strata["strata"]:
        row = rep.get((s["run"], s["workspace"])) if s.get("workspace") else ans.get((s["run"], s.get("request")))
        f, types = flagged(row, s["finding"])
        s.setdefault("verifier_flags", {})["0.3.6c"] = {"flagged": f, "errors": types}
    by_id = {s["id"]: s for s in strata["strata"]}

    # ---- 1. 口径条目
    for iid in POLICY_ITEMS:
        s = by_id[iid]
        if s.get("workspace"):
            row = rep[(s["run"], s["workspace"])]
            statements = report_statements(runs, s["run"], s["workspace"])
            where = f"同一报告（{s['workspace']}）"
        else:
            row = ans[(s["run"], s["request"])]
            statements = {x["id"]: x["text"] for x in (row.get("extra") or {}).get("anchored", {}).get("sentences") or []}
            where = "同一最终答案"
        sid = s["finding"]
        cand = [e for e in row.get("errors") or [] if (e.get("extra") or {}).get("finding") == sid and (e.get("extra") or {}).get("program_rule")]
        used: list[str] = []
        for e in cand:
            x = e["extra"]
            used += [i for i in (x.get("premises") or x.get("refers_to") or []) if i not in used]
        if not used:              # 候选没动它（如 H024）：按与本句的字面重合挑同单元里干净、有来源支持的句子
            bad = firm_ids(row)
            g = _bigrams(statements.get(sid, ""))
            pool = [(len(g & _bigrams(t)), k) for k, t in statements.items() if k != sid and k not in bad and quotes_of(row, k, 1)]
            used = [k for sc, k in sorted(pool, key=lambda x: -x[0])[:6] if sc]
        premises = [{"id": k, "text": statements.get(k, ""), "quotes": quotes_of(row, k)} for k in used if k in statements]
        fl, types = flagged(row, sid)
        def why(e: dict[str, Any]) -> str:
            x = e["extra"]
            if x["program_rule"] == "back_reference":
                return "回指规则：句中“上述 / 已确认的…”指向 " + "、".join(x.get("refers_to") or [])
            return "推理核对：" + str((x.get("inference") or {}).get("note") or "")
        note = "；".join(why(e) for e in cand)
        spec["items"][iid]["context"] = {
            "where": where, "premises": premises,
            "question": "宽口径：允许依据上面这些同一报告 / 答案里已核实的句子（不只是本句所引证据）时，这句话成立吗？",
            "candidate": ("候选 0.3.6c 不再判错（" + note + "）") if cand and not fl else
                         ("候选 0.3.6c 仍判错：" + "、".join(types)) if fl else "候选 0.3.6c 没有判错"}

    # ---- 3. F 组：语义覆盖抽查
    cases = {c.stem: json.loads(c.read_text(encoding="utf-8")) for c in Path(a.cases).glob("*.json")}
    extra: list[dict[str, Any]] = []

    def frag_list(rows: list[dict[str, Any]], by: list[str]) -> list[dict[str, str]]:
        out = []
        for r in rows:
            ws = r["subject"].get("workspace_id")
            tainted = set(r.get("tainted_findings") or [])
            for j in r.get("judgments") or []:
                fid = f"{ws}/{j.get('finding_id')}"
                if j.get("finding_id") in tainted or j.get("verdict") != "supported" or (by and fid not in by):
                    continue
                for q in j.get("supported") or []:
                    if q.get("verified") is True or (j.get("is_image") and q.get("verified") is None):
                        out.append({"finding": fid, "part": str(q.get("part") or "")[:160], "quote": str(q.get("quote") or "")[:320],
                                    "source": str(j.get("source") or "")[:100], "image": bool(j.get("is_image"))})
        return out[:16]

    picks = [  # (run, unit, req)：核验器判"已建立"的 6 条 + 判"未建立"的 3 条对照
        ("true_20361105_r0c_1", "W1-1", "REQ1"), ("true_17838107_r0c_1", "W1-2", "REQ3"), ("averimatec_423_r0cn_1", "W1-2", "REQ1"),
        ("averimatec_660_r0cn_1", "answer", "REQ2"), ("averimatec_699_r0cn_1", "answer", "REQ1"), ("averimatec_699_r0cn_1", "answer", "REQ2"),
        ("true_24171541709824_r0c_1", "W2-1", "REQ3"), ("averitec_1766_r0cn_1", "answer", "REQ2"), ("averimatec_423_r0cn_1", "W1-1", "REQ2"),
    ]
    for n, (run, unit, req) in enumerate(picks, 1):
        case = cases[run.rsplit("_r0c", 1)[0]]
        rtext = next(r["text"] for r in case["requirements"] if r["id"] == req)
        if unit == "answer":
            row = next(r for (rn, _), r in ans.items() if rn == run)
            sem = ((row.get("extra") or {}).get("sufficiency") or {}).get("semantic") or {}
            info = sem.get(req) or {}
            rows = [r for (rn, _), r in rep.items() if rn == run]
            scope = "整个调查（全部干净 Finding）"
        else:
            row = rep[(run, unit)]
            sc = (row.get("extra") or {}).get("semantic_coverage") or {}
            info = {"covered": req in (sc.get("covered") or []), "by": (sc.get("by") or {}).get(req) or [], "note": (sc.get("notes") or {}).get(req, "")}
            rows = [row]
            scope = f"报告 {unit}"
        frags = frag_list(rows, list(info.get("by") or []))
        if len(frags) < 3:
            frags = frag_list(rows, [])
        extra.append({"id": f"S{n:02d}", "kind": "semcov", "run": run, "unit": unit, "requirement": req, "requirement_text": rtext,
                      "scope": scope, "task": case["task_text"], "fragments": frags,
                      "candidate": {"covered": bool(info.get("covered")), "by": list(info.get("by") or []), "note": str(info.get("note") or "")},
                      "question": f"只看下面这些已核验的片段（来源原文 / 图片描述），{req}“{rtext}”是否已经建立？不看是不是案卷预设的网站。"})
    a17 = next(r for (rn, _), r in ans.items() if rn == "true_17838107_r0c_1")
    lab = next(c for c in a17.get("extra", {}).get("label_checks", [])) if a17.get("extra", {}).get("label_checks") else {}
    extra.append({"id": "L01", "kind": "label", "run": "true_17838107_r0c_1", "task": cases["true_17838107"]["task_text"],
                  "label": "基本不实", "answer_quote": str(lab.get("quote") or ""),
                  "question": "声明是“食用热十字面包可能影响呼气酒精检测读数”。答案判“有证据支持，但应限定为刚吃过后某些设备可能出现短暂的非零读数”。"
                              "案卷标签“基本不实”。这份答案的结论应不应该算与标签一致？",
                  "options": {"accept": "算一致：答案的限定（刚吃完、短暂、某些设备）与标签要表达的意思相同",
                              "reject": "不算一致：答案的主结论“有证据支持”与“基本不实”方向相反", "cannot_tell": "无法判断"}})
    spec["extra_items"] = extra
    spec["groups"]["F"] = f"语义覆盖与标签口径（{len(extra)}）"
    (batch / "review_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    strata["meta"]["verifier_versions"] = ["0.3.3", "0.3.4", "0.3.6c"]
    (batch / "strata.json").write_text(json.dumps(strata, ensure_ascii=False, indent=1), encoding="utf-8")
    flips = [(s["id"], s["verifier_flags"]["0.3.4"]["flagged"], s["verifier_flags"]["0.3.6c"]["flagged"]) for s in strata["strata"]
             if s["verifier_flags"].get("0.3.4", {}).get("flagged") != s["verifier_flags"]["0.3.6c"]["flagged"]]
    print("0.3.4 → 0.3.6c 判错变化：", flips)
    print("0.3.6c 判错条目：", [s["id"] for s in strata["strata"] if s["verifier_flags"]["0.3.6c"]["flagged"]])
    for iid in POLICY_ITEMS:
        c = spec["items"][iid]["context"]
        print(iid, [x["id"] for x in c["premises"]], c["candidate"][:80])
    for x in extra:
        print(x["id"], x.get("requirement"), x.get("candidate"), len(x.get("fragments") or []))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
