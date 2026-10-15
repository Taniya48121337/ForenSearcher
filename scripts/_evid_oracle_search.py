"""找原片实验 A6：后端召回上界（10-06）。不跑模型，几分钟。

拿 GT 原片的真实标题（YouTube 公开 oEmbed）当查询，走 run 里同一个 yt-dlp 视频搜索，看原片能不能进前 5 / 前 10。
两条查询：完整标题、标题前 6 个词。原片已删除 / 私享的单独记（那种谁也搜不到）。
GT 标题只在这里用——任何正式臂都不用（实验计划"不做的"第 2 条）。

    python scripts/_evid_oracle_search.py                 # 默认 20 个对照样本 → experiments/evid/scores/A6_oracle.{json,md}

判读（实验计划）：A6 < 30% → 后端召回不足，查询策略改了也没用，先补以图搜和换源；A6 高 → 问题在查询。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _evid_source_common import (AB_IDS, direct_search, gather_limited, load_sources, now, oembed,  # noqa: E402
                                 provider_search, rank_of, write_outputs)
from foren_searcher.search.arms import query_words  # noqa: E402
from foren_searcher.search.providers.ytdlp import YtDlpSearchProvider  # noqa: E402


def short_title(title: str, n: int = 6) -> str:
    out, used = [], 0
    for token in title.split():
        used += query_words(token)
        if used > n:
            break
        out.append(token)
    return " ".join(out) or title


async def run(ids: list[str], limit: int) -> dict:
    sources = load_sources(ids)
    provider = YtDlpSearchProvider()
    meta = {vid: await asyncio.to_thread(oembed, vid) for sample in ids for vid in sources[sample]}
    jobs, keys = [], []
    for sample in ids:
        for vid in sources[sample]:
            info = meta[vid]
            if not info.get("available"):
                continue
            for kind, query in (("title", info["title"]), ("title6", short_title(info["title"]))):
                keys.append((sample, vid, kind, query))
                jobs.append(provider_search(provider, query, limit))
    results = await gather_limited(jobs, 4)
    rows = {sample: {"sample": sample, "sources": sources[sample], "available": [], "queries": []} for sample in ids}
    for (sample, vid, kind, query), (found, error) in zip(keys, results):
        direct = None
        if not found:
            direct_ids, direct_error = await asyncio.to_thread(direct_search, query, limit)
            direct = {"count": len(direct_ids), "rank": rank_of(direct_ids, [vid]), "error": direct_error}
        rows[sample]["queries"].append({"source": vid, "kind": kind, "query": query, "count": len(found),
                                        "rank": rank_of(found, [vid]), "error": error, "direct": direct})
    for sample in ids:
        row = rows[sample]
        row["available"] = [vid for vid in sources[sample] if meta[vid].get("available")]
        row["unavailable"] = {vid: meta[vid] for vid in sources[sample] if not meta[vid].get("available")}
        ranks = [q["rank"] for q in row["queries"] if q["rank"]]
        row["best_rank"] = min(ranks) if ranks else None
        row["best_rank_title"] = min((q["rank"] for q in row["queries"] if q["rank"] and q["kind"] == "title"), default=None)
    listed = list(rows.values())
    n = len(listed)
    available = [r for r in listed if r["available"]]
    hit = lambda rs, k, key="best_rank": sum(1 for r in rs if r[key] and r[key] <= k)  # noqa: E731
    summary = {
        "samples": n, "samples_with_available_source": len(available),
        "hit_at_5": hit(listed, 5), "hit_at_10": hit(listed, 10),
        "hit_at_5_title_only": hit(listed, 5, "best_rank_title"), "hit_at_10_title_only": hit(listed, 10, "best_rank_title"),
        "recall_at_10": round(hit(listed, 10) / n, 3) if n else None,
        "recall_at_10_available": round(hit(available, 10) / len(available), 3) if available else None,
        "provider_errors": sum(1 for r in listed for q in r["queries"] if q["error"]),
        "empty_via_bridge_but_direct_found": sum(1 for r in listed for q in r["queries"]
                                                 if not q["count"] and q["direct"] and q["direct"]["count"]),
    }
    return {"generated": now(), "limit": limit, "summary": summary, "rows": listed}


def markdown(payload: dict) -> list[str]:
    s = payload["summary"]
    lines = ["# 找原片 A6：后端召回上界（GT 标题当查询）", "", f"生成于 {payload['generated']}；yt-dlp 视频搜索，每条查询前 {payload['limit']} 个。", "",
             f"- 样本 {s['samples']} 个，原片仍可访问的 {s['samples_with_available_source']} 个",
             f"- **前 10 命中 {s['hit_at_10']}/{s['samples']}（{s['recall_at_10']}）**，前 5 命中 {s['hit_at_5']}；"
             f"只用完整标题：前 10 {s['hit_at_10_title_only']}，前 5 {s['hit_at_5_title_only']}",
             f"- 原片可访问的样本里前 10 命中率 {s['recall_at_10_available']}；搜索报错 {s['provider_errors']} 次；"
             f"桥接 0 条但裸 yt-dlp 有结果 {s['empty_via_bridge_but_direct_found']} 次",
             "", "| 样本 | 原片 | 可访问 | 完整标题排名 | 前 6 词排名 | 标题 |", "|---|---|---|---|---|---|"]
    for row in payload["rows"]:
        by = {(q["source"], q["kind"]): q for q in row["queries"]}
        for vid in row["sources"]:
            title = next((q["query"] for q in row["queries"] if q["source"] == vid and q["kind"] == "title"), "")
            ok = "是" if vid in row["available"] else f"否（{row['unavailable'].get(vid, {}).get('status', '?')}）"
            rank = lambda kind: (by.get((vid, kind)) or {}).get("rank") or ("—" if (vid, kind) in by else "")  # noqa: E731
            lines.append(f"| {row['sample']} | {vid} | {ok} | {rank('title')} | {rank('title6')} | {title.replace('|', '/')[:80]} |")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="找原片 A6：GT 标题当查询的后端召回上界")
    parser.add_argument("--ids", nargs="+", default=list(AB_IDS))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--stem", default="A6_oracle")
    args = parser.parse_args(argv)
    payload = asyncio.run(run(args.ids, args.limit))
    path = write_outputs(args.stem, payload, markdown(payload))
    s = payload["summary"]
    print(f"A6 前 10 命中 {s['hit_at_10']}/{s['samples']}（原片可访问 {s['samples_with_available_source']} 个），"
          f"前 5 命中 {s['hit_at_5']}；搜索报错 {s['provider_errors']}；写入 {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
