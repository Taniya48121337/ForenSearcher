"""分支可达探测：INSPECT get_frames pages（C 族 C6_pdf_pages，扫描件 PDF 渲染页面再看）。

触发条件（walker._pdf_pages_step，只在 C 族 _decide_claim 里）：读过的记录是 PDF，且页面打开器标了"没有文字层"
（openers/document.py：pages_with_text=0 / text extraction failed）——walker 渲染前两页 get_frames pages=[1,2]。
两种探法（都走 walker 同一条 search 层与页面打开器，ddgs 文字搜，免费，不发图，Serper 0 点）：
  默认：三个池的候选，C 族首条查询（替身 = 声明前 12 词，= C4_search_fallback）前 N 条结果里有没有文字层为空的 PDF；
        AVeriTeC 另看 GT 来源页。与 _branch_probe_read_offset / _branch_probe_linked_video 共用扫描缓存 probe_c_scan.json。
  --gt-pdf：定向找——AVeriTeC 训练集里 GT 来源网址是 .pdf 的条目，先打开 GT PDF 看有没有文字层，没有文字层的再用声明
        搜一次，看这份（或任何）无文字层 PDF 进不进前 N。输出 probe_pdf_pages_gt.json。

    python scripts/_branch_probe_pdf_pages.py --n 10
    python scripts/_branch_probe_pdf_pages.py --gt-pdf --max-pdfs 60
输出 claude/_longtail_work/probe_pdf_pages.json（/ probe_pdf_pages_gt.json）
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

from _branch_probe_common import (AVERITEC, LIMIT, OUT_DIR, PAGE_CONCURRENCY, c_probe_main, claim_query, close_search, make_case,
                                  now, open_search, ordered, page_props, serper_balance, used_ids, write_json)


async def gt_pdf_probe(max_pdfs: int, top: int) -> dict:
    from foren_searcher.search.contracts import SearchQueryRequest

    rows = json.loads(AVERITEC.read_text(encoding="utf-8"))
    used = used_ids()
    picks = []
    for i in ordered([str(i) for i in range(len(rows))], "probe-gtpdf"):
        if i in used:
            continue
        urls = [a.get("source_url") for q in rows[int(i)].get("questions") or [] for a in q.get("answers") or []]
        pdfs = [u for u in dict.fromkeys(urls) if u and u.lower().split("?", 1)[0].endswith(".pdf")]
        if pdfs:
            picks.append({"pool": "averitec", "id": i, "claim": rows[int(i)].get("claim"), "gt_urls": pdfs[:3],
                          "fact_check_article": rows[int(i)].get("fact_checking_article")})
    total_with_pdf = len(picks)
    picks = picks[:max_pdfs]
    out = []
    for k, cand in enumerate(picks, 1):
        started = time.monotonic()
        case = make_case("C", cand)
        row = {**cand, "case_id": case.case_id}
        with tempfile.TemporaryDirectory() as tmp:
            search, store, _desc = await open_search(case, Path(tmp), "off")
            sem = asyncio.Semaphore(PAGE_CONCURRENCY)
            row["gt_pages"] = [p if isinstance(p, dict) else {"error": type(p).__name__} for p in
                               await asyncio.gather(*(page_props(search.opener, u, None, None, sem) for u in cand["gt_urls"]),
                                                    return_exceptions=True)]
            if any(p.get("pdf_no_text") for p in row["gt_pages"]):
                row["query"] = claim_query(cand["claim"])
                try:
                    batch = await asyncio.wait_for(search.provider.search(SearchQueryRequest("text", row["query"], "text"),
                                                                          offset=0, limit=LIMIT, image_path=None), 240)
                    cands = list(batch.candidates)[:LIMIT]
                    row["pages"] = [p if isinstance(p, dict) else {"error": type(p).__name__} for p in await asyncio.gather(
                        *(page_props(search.opener, c.source_url, c.search_title, j, sem) for j, c in enumerate(cands, 1)),
                        return_exceptions=True)]
                except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
                    row["search_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    row["pages"] = []
            await close_search(search, store)
        row["seconds"] = round(time.monotonic() - started, 1)
        nt = [p for p in row["gt_pages"] if p.get("pdf_no_text")]
        hit = [p for p in row.get("pages") or [] if p.get("pdf_no_text") and (p.get("rank") or 99) <= top]
        row["gt_pdf_no_text"], row["no_text_pdf_in_top"] = bool(nt), bool(hit)
        out.append(row)
        print(f"[gt-pdf {k}/{len(picks)}] averitec {cand['id']}: GT PDF 打开 {sum(1 for p in row['gt_pages'] if 'error' not in p)}/"
              f"{len(row['gt_pages'])}，无文字层 {len(nt)}；检索前 {top} 有无文字层 PDF：{bool(hit)}；{row['seconds']:.0f}s", flush=True)
        write_json(OUT_DIR / "probe_pdf_pages_gt.json", {"generated": now(), "rows": out})
    return {"averitec_rows_with_gt_pdf": total_with_pdf, "probed": len(out),
            "gt_pdf_opened": sum(1 for r in out if any("error" not in p for p in r["gt_pages"])),
            "gt_pdf_no_text": sum(1 for r in out if r["gt_pdf_no_text"]),
            "reachable": [r["case_id"] for r in out if r["no_text_pdf_in_top"]], "rows": out}


def main() -> int:
    if "--gt-pdf" in sys.argv:
        import argparse

        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("--gt-pdf", action="store_true")
        parser.add_argument("--max-pdfs", type=int, default=60)
        parser.add_argument("--top", type=int, default=5)
        args = parser.parse_args()
        before = serper_balance()
        result = asyncio.run(gt_pdf_probe(args.max_pdfs, args.top))
        result.update({"generated": now(), "top": args.top, "serper_balance_before": before, "serper_balance_after": serper_balance()})
        write_json(OUT_DIR / "probe_pdf_pages_gt.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False))
        return 0
    return c_probe_main("pdf_no_text", "probe_pdf_pages.json", __doc__, 5,
                        "walker 读页由 S3 槽在前几条里选（C 族最多 5 次阅读），top 默认 5")


if __name__ == "__main__":
    raise SystemExit(main())
