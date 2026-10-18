"""r0 训练集定稿：S0–S3 教师并排（r1 的 Qwen 版 vs g1 / g2 的 Codex 版），只报数（2026-10-19）。

    python scripts/teacher_compare.py --pack r1 <导出> <runs> qwen --pack g1 <导出> <runs> codex --pack g2 ... \
        --n 20 --out claude/_g2_work/teacher_compare

抽样：accepted 的 HI 动作行（S0 理由 / S1 查询 / S3 选页），层 = (族, 动作列)，只用两边都有的层；两边各抽 n 行、每层行数相同
（按 sha1 排序取，可复现）。量（每行）：
- 查询词长度：SEARCH 文字查询的词数（S1）；
- 是否照抄声明：查询词里落在声明原文里的词占比 ≥ 0.8 记"照抄"（声明取该步 model_view.task 里引号内 / 冒号后的部分）；
- 选页理由长度：READ_RECORD / SEARCH source（打开页、查日期、取主图）的 rationale 词数；
- 理由与动作参数的重合度：动作参数里的词（查询词、record_id、targets、source）出现在 rationale 里的占比。
同时给全体（不抽样）的均值，避免 20 行的偶然。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import teaching_matrix as tm  # noqa: E402

WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")
STOP = {"the", "a", "an", "of", "in", "on", "to", "and", "or", "for", "with", "by", "at", "is", "are", "was", "were", "from", "as", "that"}


def words(s: str) -> list[str]:
    return [w.lower() for w in WORD.findall(s or "")]


def claim_of(task: str) -> str:
    m = re.findall(r"[\"“](.+?)[\"”]", task or "")
    if m:
        return max(m, key=len)
    return (task or "").rsplit(":", 1)[-1]


def metrics(row: dict[str, Any], rc: tm.RunCache) -> dict[str, Any]:
    md = row["metadata"]
    tgt = tm.target(row)
    args = tgt.get("arguments") or {}
    rat = str(tgt.get("rationale") or "")
    out: dict[str, Any] = {"rationale_words": len(words(rat))}
    action = tgt.get("action")
    arg_tokens: list[str] = []
    if action == "SEARCH" and isinstance(args.get("query"), str) and args.get("input_type") == "text":
        q = words(args["query"])
        out["query_words"] = len(q)
        payload = rc.load(str(md["run"]), str(md["request"]), "request.json").get("payload") or {}
        claim = set(words(claim_of(str((payload.get("model_view") or {}).get("task") or ""))))
        content = [w for w in q if w not in STOP]
        share = sum(1 for w in content if w in claim) / max(1, len(content))
        out["query_claim_share"] = round(share, 3)
        out["copies_claim"] = share >= 0.8
        arg_tokens = content
    if action == "READ_RECORD" or (action == "SEARCH" and "source" in args):
        out["page_rationale_words"] = len(words(rat))
    for key in ("record_id", "source", "query"):
        v = args.get(key)
        if isinstance(v, str) and key != "query":
            arg_tokens.append(v.lower())
    for t in args.get("targets") or []:
        arg_tokens.append(str(t).lower())
    if arg_tokens:
        rl = rat.lower()
        out["rationale_arg_overlap"] = round(sum(1 for t in arg_tokens if t in rl) / len(arg_tokens), 3)
    return out


def agg(items: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float | None:
        vals = [float(x[key]) for x in items if key in x]
        return round(statistics.mean(vals), 3) if vals else None

    def n(key: str) -> int:
        return sum(1 for x in items if key in x)
    copies = [x["copies_claim"] for x in items if "copies_claim" in x]
    return {"rows": len(items), "query_rows": n("query_words"), "query_words_mean": mean("query_words"),
            "query_claim_share_mean": mean("query_claim_share"),
            "copies_claim": f"{sum(copies)}/{len(copies)}" if copies else None,
            "page_rationale_rows": n("page_rationale_words"), "page_rationale_words_mean": mean("page_rationale_words"),
            "rationale_words_mean": mean("rationale_words"), "rationale_arg_overlap_mean": mean("rationale_arg_overlap")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", nargs=4, action="append", metavar=("NAME", "EXPORT", "RUNS", "TEACHER"), required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    pools: dict[str, dict[tuple[str, str], list[tuple[dict[str, Any], tm.RunCache, str]]]] = defaultdict(lambda: defaultdict(list))
    for name, export, runs, teacher in args.pack:
        rc = tm.RunCache(ROOT / runs)
        for row in tm.jsonl(ROOT / export / "hi_accepted.jsonl"):
            md = row.get("metadata") or {}
            if md.get("unit") != "step" or md.get("rationale_source") not in ("s0_teacher", "s1_teacher", "s3_teacher"):
                continue
            cols, _ = tm.row_columns(row, rc)
            if not cols:
                continue
            pools[teacher][(tm.family(md.get("case_id")), cols[0])].append((row, rc, name))
    teachers = sorted(pools)
    if len(teachers) != 2:
        raise SystemExit(f"需要两种教师，得到 {teachers}")
    common = sorted(set(pools[teachers[0]]) & set(pools[teachers[1]]))
    # 每层行数：按层轮流取，直到 n
    alloc: dict[tuple[str, str], int] = defaultdict(int)
    i = 0
    while sum(alloc.values()) < args.n and common:
        s = common[i % len(common)]
        if alloc[s] < min(len(pools[t][s]) for t in teachers):
            alloc[s] += 1
        i += 1
        if i > 10000:
            break
    sample: dict[str, list[dict[str, Any]]] = {}
    side = {}
    for t in teachers:
        picked = []
        for s, k in alloc.items():
            ranked = sorted(pools[t][s], key=lambda x: hashlib.sha1(f"tc-{x[0]['metadata']['run']}-{x[0]['metadata']['request']}".encode()).hexdigest())
            for row, rc, pack in ranked[:k]:
                m = metrics(row, rc)
                tgt = tm.target(row)
                picked.append({"stratum": list(s), "pack": pack, "run": row["metadata"]["run"], "request": row["metadata"]["request"],
                               "action": tgt.get("action"), "arguments": tgt.get("arguments"), "rationale": tgt.get("rationale"), **m})
        sample[t] = picked
        side[t] = agg(picked)
    full = {t: agg([metrics(r, rc) for rows in pools[t].values() for r, rc, _ in rows]) for t in teachers}
    out = {"teachers": teachers, "strata": {f"{k[0]}|{k[1]}": v for k, v in alloc.items()}, "sample_metrics": side,
           "population_metrics": full, "sample": sample}
    prefix = ROOT / args.out
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# S0–S3 教师并排（只报数）", "", f"层（族 | 动作列 → 每边行数）：{json.dumps(out['strata'], ensure_ascii=False)}", "",
             "| 指标 | " + " | ".join(f"{t}（抽样）" for t in teachers) + " | " + " | ".join(f"{t}（全体）" for t in teachers) + " |",
             "|---|" + "---|" * (2 * len(teachers))]
    for key in ("rows", "query_rows", "query_words_mean", "query_claim_share_mean", "copies_claim", "page_rationale_rows",
                "page_rationale_words_mean", "rationale_words_mean", "rationale_arg_overlap_mean"):
        lines.append(f"| {key} | " + " | ".join(str(side[t].get(key)) for t in teachers) + " | "
                     + " | ".join(str(full[t].get(key)) for t in teachers) + " |")
    lines += ["", "## 并排（同层同序）", ""]
    for j in range(max(len(v) for v in sample.values())):
        for t in teachers:
            if j < len(sample[t]):
                x = sample[t][j]
                lines.append(f"- **{t}** `{x['stratum'][0]}|{x['stratum'][1]}` {x['run']} {x['request']}：`{json.dumps(x['arguments'], ensure_ascii=False)[:160]}` — {str(x['rationale'])[:220]}")
        lines.append("")
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"sample": side, "population": full}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
