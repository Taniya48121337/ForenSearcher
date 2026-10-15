"""找原片实验 A7：旧流水线的查询原样灌进现在的视频搜索（10-06）。不跑模型，十来分钟。

旧 plan-harness-1 流水线（D:\\CodexEvidRuns\\evid_full_benchmark_source_preserving_v10_20260803）每个样本约 16–18 条
YouTube 查询（search_results.json 的 queries），每条 10 个结果，再加缩略图 CLIP 重排。这里把同样的查询走 run 里同一个
yt-dlp 视频搜索（每条 10 个），看原片进不进并集，与旧日志对比：

    旧·文字搜索命中   searches[*].candidates 的并集里有 GT（与这里同口径）
    旧·任一候选命中   candidates（还含旧流水线的"可见候选"通道）里有 GT
    旧·首选命中       run_summary.top_candidate 就是 GT
    新·命中 / 最好排名  现在的后端重放这些查询

一致 → 两套后端没变，差距在查询（量与写法）；新明显低 → 后端变了（看"零结果"与"桥接 0 条但裸 yt-dlp 有结果"两列）。

    python scripts/_evid_replay_old_queries.py            # → experiments/evid/scores/A7_replay.{json,md}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _evid_source_common import (AB_IDS, direct_search, gather_limited, load_sources, now,  # noqa: E402
                                 provider_search, rank_of, write_outputs)
from foren_searcher.search.providers.ytdlp import YtDlpSearchProvider  # noqa: E402

OLD_ROOT = Path(r"D:\CodexEvidRuns\evid_full_benchmark_source_preserving_v10_20260803")


def old_log(root: Path, sample: str) -> dict | None:
    base = root / "samples" / sample / "run" / "candidate_pipeline"
    path = base / "search_results.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    summary_path = base / "run_summary.json"
    top = None
    if summary_path.is_file():
        try:
            top = (json.loads(summary_path.read_text(encoding="utf-8")).get("top_candidate") or {}).get("source_id")
        except ValueError:
            top = None
    text_ids = [c.get("source_id") for s in data.get("searches") or [] for c in s.get("candidates") or []]
    return {"queries": [q for q in data.get("queries") or [] if isinstance(q, str) and q.strip()],
            "text_ids": [x for x in text_ids if x], "any_ids": [c.get("source_id") for c in data.get("candidates") or []],
            "failures": len(data.get("search_failures") or []), "top": top,
            "rerank": (data.get("thumbnail_rerank") or {}).get("ranking") or []}


async def run(ids: list[str], root: Path, limit: int, max_queries: int) -> dict:
    sources = load_sources(ids)
    provider = YtDlpSearchProvider()
    rows = []
    for sample in ids:
        log = old_log(root, sample)
        gt = sources[sample]
        row = {"sample": sample, "sources": gt, "old_log": log is not None}
        if log is None:
            rows.append(row)
            continue
        queries = log["queries"][:max_queries]
        results = await gather_limited([provider_search(provider, q, limit) for q in queries], 4)
        per_query = []
        union: list[str] = []
        for query, (found, error) in zip(queries, results):
            direct = None
            if not found:
                direct_ids, direct_error = await asyncio.to_thread(direct_search, query, limit)
                direct = {"count": len(direct_ids), "rank": rank_of(direct_ids, gt), "error": direct_error}
            per_query.append({"query": query, "count": len(found), "rank": rank_of(found, gt), "error": error,
                              "direct": direct})
            union += [x for x in found if x not in union]
        ranks = [q["rank"] for q in per_query if q["rank"]]
        row.update({
            "old_queries": len(log["queries"]), "replayed": len(queries), "old_failures": log["failures"],
            "old_text_hit": bool(set(log["text_ids"]) & set(gt)), "old_any_hit": bool(set(log["any_ids"]) & set(gt)),
            "old_top1": log["top"] in gt if log["top"] else False,
            "old_rerank_rank": rank_of(log["rerank"], gt),
            "new_hit": bool(set(union) & set(gt)), "new_best_rank": min(ranks) if ranks else None,
            "new_hit_queries": sum(1 for q in per_query if q["rank"]),
            "new_union": len(union), "new_empty": sum(1 for q in per_query if not q["count"] and not q["error"]),
            "new_errors": sum(1 for q in per_query if q["error"]),
            "bridge_empty_direct_found": sum(1 for q in per_query if not q["count"] and q["direct"] and q["direct"]["count"]),
            "queries": per_query,
        })
        rows.append(row)
        print(f"  {sample}：旧·文字 {'中' if row['old_text_hit'] else '—'} / 旧·任一 {'中' if row['old_any_hit'] else '—'}；"
              f"新 {'中（最好第 ' + str(row['new_best_rank']) + '）' if row['new_hit'] else '—'}；"
              f"{len(queries)} 条查询，零结果 {row['new_empty']}，报错 {row['new_errors']}", flush=True)
    with_log = [r for r in rows if r["old_log"]]
    count = lambda key: sum(1 for r in with_log if r.get(key))  # noqa: E731
    summary = {
        "samples": len(rows), "with_old_log": len(with_log),
        "old_text_hit": count("old_text_hit"), "old_any_hit": count("old_any_hit"), "old_top1": count("old_top1"),
        "new_hit": count("new_hit"),
        "agree_with_old_text": sum(1 for r in with_log if r.get("new_hit") == r.get("old_text_hit")),
        "new_hit_old_miss": sum(1 for r in with_log if r.get("new_hit") and not r.get("old_text_hit")),
        "old_hit_new_miss": sum(1 for r in with_log if r.get("old_text_hit") and not r.get("new_hit")),
        "queries": sum(r.get("replayed") or 0 for r in with_log),
        "new_empty": sum(r.get("new_empty") or 0 for r in with_log),
        "old_failures": sum(r.get("old_failures") or 0 for r in with_log),
        "new_errors": sum(r.get("new_errors") or 0 for r in with_log),
        "bridge_empty_direct_found": sum(r.get("bridge_empty_direct_found") or 0 for r in with_log),
    }
    return {"generated": now(), "old_root": str(root), "limit": limit, "summary": summary, "rows": rows}


def markdown(payload: dict) -> list[str]:
    s = payload["summary"]
    lines = ["# 找原片 A7：旧流水线查询在现在的后端上重放", "",
             f"生成于 {payload['generated']}；旧日志 {payload['old_root']}；每条查询前 {payload['limit']} 个。", "",
             f"- 有旧日志的样本 {s['with_old_log']}/{s['samples']}；重放查询 {s['queries']} 条",
             f"- 旧·文字搜索命中 **{s['old_text_hit']}**，旧·任一候选命中 {s['old_any_hit']}，旧·首选就是原片 {s['old_top1']}",
             f"- **新·命中 {s['new_hit']}**；与旧·文字一致 {s['agree_with_old_text']}（新中旧没中 {s['new_hit_old_miss']}，"
             f"旧中新没中 {s['old_hit_new_miss']}）",
             f"- 零结果：新 {s['new_empty']} 条（旧日志记的失败 {s['old_failures']} 条）；新报错 {s['new_errors']} 条；"
             f"桥接 0 条但裸 yt-dlp 有结果 {s['bridge_empty_direct_found']} 条",
             "", "| 样本 | 查询数 | 旧·文字 | 旧·任一 | 旧·首选 | 旧 CLIP 重排名次 | 新·命中（最好排名 / 命中查询数） | 新零结果 | 新报错 |",
             "|---|---|---|---|---|---|---|---|---|"]
    yes = lambda v: "中" if v else "—"  # noqa: E731
    for r in payload["rows"]:
        if not r["old_log"]:
            lines.append(f"| {r['sample']} | 没有旧日志 | | | | | | | |")
            continue
        new = f"中（{r['new_best_rank']} / {r['new_hit_queries']}）" if r["new_hit"] else "—"
        lines.append(f"| {r['sample']} | {r['replayed']} | {yes(r['old_text_hit'])} | {yes(r['old_any_hit'])} | "
                     f"{yes(r['old_top1'])} | {r['old_rerank_rank'] or '—'} | {new} | {r['new_empty']} | {r['new_errors']} |")
    hits = [(r["sample"], q) for r in payload["rows"] if r.get("queries") for q in r["queries"] if q["rank"]]
    if hits:
        lines += ["", "## 命中原片的查询", "", "| 样本 | 排名 | 查询 |", "|---|---|---|"]
        lines += [f"| {sample} | {q['rank']} | {q['query'].replace('|', '/')[:120]} |" for sample, q in hits]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="找原片 A7：旧流水线查询在现在的后端上重放")
    parser.add_argument("--ids", nargs="+", default=list(AB_IDS))
    parser.add_argument("--old-root", default=str(OLD_ROOT))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=20)
    parser.add_argument("--stem", default="A7_replay")
    args = parser.parse_args(argv)
    root = Path(args.old_root)
    if not root.is_dir():
        print(f"找不到旧流水线目录 {root}；A7 跳过。", flush=True)
        return 2
    payload = asyncio.run(run(args.ids, root, args.limit, args.max_queries))
    path = write_outputs(args.stem, payload, markdown(payload))
    s = payload["summary"]
    print(f"A7 旧·文字命中 {s['old_text_hit']} / 新命中 {s['new_hit']}（{s['with_old_log']} 个有旧日志）；"
          f"一致 {s['agree_with_old_text']}；新零结果 {s['new_empty']} 条；写入 {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
