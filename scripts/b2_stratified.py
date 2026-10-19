"""B2 人工抽查表（按包分层）+ 程序比对填判定（2026-10-19，负责人定；办法同 claude/_r1a_work/b2_check*.py）。只读。

    python scripts/b2_stratified.py --union experiments/sft_data/export/_sft0_union \
        --pack r1 <r1 runs> --pack g1 <g1 runs> --pack g2 <g2 runs> --pack g3 <g3 runs> \
        [--sft0 dist/sft0-…] --per-pack 5 --out <合集目录>/gate_b2_review_stratified.md

抽样：先放仍以 hidden_context 扣下的行（全放），再按包补放行的行（accepted 且带 hidden_context_released），每包凑到 --per-pack 条；
包内按 sha1(run/request) 排序取前几条（固定、可复现，且不全挤在同一 run）。包由 run 名定：_r1g1_ / _r1g2_ / _r1g3_ → g1 / g2 / g3，其余 r1。

判定（程序）：
  放行的词：出处是 E → 该 E 的全文（存储的 label + content，同 export.build_run_evidence）里有这个词（export._token_in 同一比法），
           且这个 E 的编号在学生这一步的输入里出现过；出处是 observation@N → 本 run 第 N 个事件是观察、且全文里有这个词。
           都成立 = 对；否则 = 错（误放）。
  仍扣的词：学生这一步够得着的 E（输入里出现过编号的）全文或本 run 任一观察里有这个词 = 错（误扣，列给负责人）；没有 = 对。
另把全部带 hidden_context 的行（不只抽样的 20 条）都按同一办法比一遍，出合计。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foren_searcher.datagen.export import _token_in, build_run_evidence, release_by_evidence, student_text  # noqa: E402
from foren_searcher.datagen.runlog import load_run  # noqa: E402

_EID = re.compile(r"\bE\d+\b")


def pack_of(run: str) -> str:
    for p in ("g1", "g2", "g3"):
        if f"_r1{p}_" in run:
            return p
    return "r1"


def target_text(row: dict) -> str:
    return next((m["content"] if isinstance(m["content"], str) else json.dumps(m["content"], ensure_ascii=False)
                 for m in reversed(row["messages"]) if m["role"] == "assistant"), "")


class Corpora:
    def __init__(self, runs_dirs: dict[str, Path]):
        self.runs_dirs, self.cache = runs_dirs, {}

    def get(self, run: str):
        if run not in self.cache:
            log = load_run(self.runs_dirs[pack_of(run)] / run)
            try:
                corpus = build_run_evidence(log)
                obs = {eid: (actor, text) for eid, actor, text in corpus.observations}
                self.cache[run] = (corpus, obs, {d.request_id: d for d in log.decisions})
            finally:
                log.close()
        return self.cache[run]


def check(row: dict, corpora: Corpora) -> dict:
    md = row["metadata"]
    corpus, obs, decisions = corpora.get(str(md["run"]))
    records = corpus.records
    dec = decisions.get(str(md["request"]))
    visible = set(_EID.findall(student_text(row["messages"])))
    released = md.get("hidden_context_released") or {}
    words = {}
    for w in md.get("hidden_context") or []:
        src = released.get(w)
        if src and src.startswith("E"):
            ok = src in records and _token_in(w, records[src]) and src in visible
            words[w] = {"source": src, "in_source": src in records and _token_in(w, records[src]), "source_visible": src in visible,
                        "verdict": "对" if ok else "错（误放）"}
        elif src and src.startswith("observation@"):
            n = int(src.split("@", 1)[1])
            ok = n in obs and _token_in(w, obs[n][1])
            words[w] = {"source": src, "in_source": ok, "verdict": "对" if ok else "错（误放）"}
        elif w in set(((md.get("tool_result_truncation") or {}).get("items")) or []):
            # export 把截断依赖的词交给 B4（不走放行）：这行按 B4 扣，不算 B2 误扣
            words[w] = {"source": None, "verdict": "对（截断依赖，记 B4）"}
        else:
            # 误扣 = 按 export 自己的放行口径（release_by_evidence）现在比也该放；宽口径命中（数字在没被本步引用的可见 E 里、
            # 或只在本步之后 / 别的工作区的观察里）照口径扣，记"对（口径内扣）"并列出命中处
            strict = release_by_evidence([w], dec, row["messages"], corpus) if dec is not None else {}
            hits = [e for e in sorted(visible) if e in records and _token_in(w, records[e])]
            before = dec.raw_event_id if dec is not None and dec.raw_event_id is not None else 10 ** 12
            ohits = [n for n, (a, t) in obs.items() if _token_in(w, t)]
            ohits_ok = [n for n in ohits if dec is not None and dec.role == "HI" and n < before and obs[n][0] == dec.actor_id]
            words[w] = {"source": None, "export_rule_release": strict.get(w), "visible_E_hits": hits[:5], "observation_hits": ohits[:5],
                        "observation_hits_before_same_actor": ohits_ok[:5],
                        "verdict": "错（误扣）" if strict else ("对（口径内扣）" if hits or ohits else "对")}
    bad = [v["verdict"] for v in words.values() if v["verdict"].startswith("错")]
    soft = [v["verdict"] for v in words.values() if v["verdict"] != "对"]
    return {"words": words, "verdict": bad[0] if bad else (soft[0] if soft else "对")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--union", required=True)
    ap.add_argument("--pack", nargs=2, action="append", metavar=("NAME", "RUNS"), required=True)
    ap.add_argument("--sft0", default=None, help="合好的 sft0 目录：标出抽到的行是否在训练集里")
    ap.add_argument("--per-pack", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    union = Path(args.union)
    corpora = Corpora({name: Path(runs) for name, runs in args.pack})
    flagged = []
    for role in ("hi", "gia"):
        for b in ("accepted", "held", "rejected"):
            f = union / f"{role}_{b}.jsonl"
            if f.is_file():
                for line in f.read_text(encoding="utf-8").splitlines():
                    r = json.loads(line)
                    if r["metadata"].get("hidden_context"):
                        flagged.append(r)
    in_sft0 = set()
    if args.sft0:
        for role in ("hi", "gia"):
            for line in (Path(args.sft0) / f"{role}.jsonl").read_text(encoding="utf-8").splitlines():
                m = json.loads(line)["metadata"]
                in_sft0.add((str(m["run"]), str(m["request"])))

    def key(r):
        m = r["metadata"]
        return hashlib.sha1(f"{m['run']}/{m['request']}".encode()).hexdigest()

    held = [r for r in flagged if str(r["metadata"].get("bucket_reason") or "").startswith("hidden_context")]
    sample = list(held)
    packs = [p for p, _ in args.pack]
    for p in packs:
        pool = sorted((r for r in flagged if pack_of(str(r["metadata"]["run"])) == p and r["metadata"].get("bucket") == "accepted"
                       and r["metadata"].get("hidden_context_released") and r not in sample), key=key)
        have = sum(1 for r in sample if pack_of(str(r["metadata"]["run"])) == p)
        seen_runs = set()
        picked = []
        for r in pool:                       # 先每个 run 一条，再补
            if r["metadata"]["run"] not in seen_runs:
                picked.append(r)
                seen_runs.add(r["metadata"]["run"])
        picked += [r for r in pool if r not in picked]
        sample += picked[: max(0, args.per_pack - have)]

    results = [(r, check(r, corpora)) for r in flagged]
    by_id = {id(r): res for r, res in results}
    lines = ["# B2 人工抽查（按包分层）：hidden_context", "",
             f"合集 {union.name}：带 hidden_context 词的行 {len(flagged)}（按包 {dict(Counter(pack_of(str(r['metadata']['run'])) for r in flagged))}）；"
             f"仍以 hidden_context 扣下 {len(held)} 行。",
             f"抽样：仍扣的全放（{len(held)}），再按包各补 accepted 放行行到 {args.per_pack} 条 → 共 {len(sample)} 条"
             f"（按包 {dict(Counter(pack_of(str(r['metadata']['run'])) for r in sample))}）。",
             "判定列由程序填（办法见脚本 docstring：放行词是否真在放行出处的 E 正文 / 观察里、该 E 学生看得到；仍扣词在学生够得着的证据里有没有）。", "",
             "| # | 包 | run | request | 单元 | 桶 | 词 | 放行出处 | 程序比对 | 判定 | 在 sft0 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(sample, 1):
        m, res = r["metadata"], by_id[id(r)]
        detail = "; ".join(f"{w}: " + (f"{v['source']} 正文有={v['in_source']}" + (f" 学生可见={v['source_visible']}" if "source_visible" in v else "")
                                       if v["source"] else f"可见 E 命中 {v['visible_E_hits'] or '无'} / 观察命中 {v['observation_hits'] or '无'}")
                           for w, v in res["words"].items())
        lines.append(f"| {i} | {pack_of(str(m['run']))} | {m['run']} | {m['request']} | {m.get('unit')} | {m.get('bucket')} | "
                     f"{', '.join(m['hidden_context'])} | {json.dumps(m.get('hidden_context_released') or {}, ensure_ascii=False)} | {detail} | "
                     f"**{res['verdict']}** | {'是' if (str(m['run']), str(m['request'])) in in_sft0 else '否'} |")
    s_bad = [(r, by_id[id(r)]) for r in sample if by_id[id(r)]["verdict"].startswith("错")]
    all_bad = [(r, res) for r, res in results if res["verdict"].startswith("错")]
    soft = [(r, res) for r, res in results if res["verdict"].startswith("对（")]
    tally = Counter((pack_of(str(r["metadata"]["run"])), res["verdict"]) for r, res in results)
    lines += ["", f"抽样判定：{dict(Counter(by_id[id(r)]['verdict'] for r in sample))}；清单 B2 误扣 ≤ 2。", "",
              f"全部 {len(flagged)} 行同法比对（按 包 × 判定）：{ {f'{p}/{v}': n for (p, v), n in sorted(tally.items())} }", ""]
    for title, items in (("抽样里判错的", s_bad), ("全部行里判错的（含抽样外）", all_bad),
                         ("照口径扣（截断依赖记 B4；或宽口径能找到、export 口径不放：数字没被本步引用 / 只在之后或别的工作区的观察里），列出备查", soft)):
        lines += [f"## {title}：{len(items)} 条", ""]
        for r, res in items:
            m = r["metadata"]
            lines.append(f"- {pack_of(str(m['run']))} {m['run']} {m['request']} {m.get('unit')} {m.get('bucket')}/{m.get('bucket_reason')}："
                         f"{res['verdict']}；{json.dumps(res['words'], ensure_ascii=False)}；在 sft0={'是' if (str(m['run']), str(m['request'])) in in_sft0 else '否'}")
        lines.append("")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    js = {"flagged": len(flagged), "held_hidden": len(held), "sample": len(sample),
          "sample_verdicts": dict(Counter(by_id[id(r)]["verdict"] for r in sample)),
          "all_verdicts_by_pack": {f"{p}/{v}": n for (p, v), n in sorted(tally.items())},
          "wrong": [{"pack": pack_of(str(r["metadata"]["run"])), "run": r["metadata"]["run"], "request": r["metadata"]["request"],
                     "unit": r["metadata"].get("unit"), "bucket": r["metadata"].get("bucket"), "verdict": res["verdict"], "words": res["words"],
                     "in_sft0": (str(r["metadata"]["run"]), str(r["metadata"]["request"])) in in_sft0} for r, res in all_bad]}
    Path(args.out).with_suffix(".json").write_text(json.dumps(js, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: js[k] for k in ("flagged", "held_hidden", "sample", "sample_verdicts", "all_verdicts_by_pack")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
