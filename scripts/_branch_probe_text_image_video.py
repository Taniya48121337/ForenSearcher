"""分支可达探测：SEARCH text_image→video（A 族 A3_search_original_text_image，找原片的第 2 条查询）。

触发条件（walker._decide_find_original）：第 1 条文字查询的结果里没有画面对上的候选（content_match 不是 source / partial），
还有轮次、工具目录允许以图搜、视图里有材料帧——第 2 条查询就是"文字 + 中段一帧"、output_type=video。
探测两件事：
  可达 = 第 1 条文字查询（转写替身）前 10 里没有原片（→ walker 会发第 2 条 text_image）；
  有用 = text_image（Yandex 以图搜 + 同一文字的视频搜，合并）前 10 里有原片。
以图搜只发 FMNV 公开视频的帧（FMNV.zip 里预抽的中间帧），只走 Yandex；**必须给 --allow-yandex 才发帧**，
不给就只跑文字那一半。Lens 关、图床关、Serper 0 点（探测前后读余额核对）。

    python scripts/_branch_probe_text_image_video.py --n 20 --allow-yandex
输出 claude/_longtail_work/probe_text_image_video.json
"""

from __future__ import annotations

import argparse
import asyncio
import json

from _branch_probe_common import OUT_DIR, a_probe, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--allow-yandex", action="store_true", help="把 FMNV 帧发给 Yandex 以图搜（非 EVID 已过滤）")
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    payload = asyncio.run(a_probe(args.n, ["text_transcript", "text_image"], args.allow_yandex, "probe_text_image_video.json",
                                  args.redo, args.only))
    rows = payload["rows"]
    fired = [r for r in rows if "text_transcript" in (r.get("original_in_top10") or {})
             and not r["original_in_top10"]["text_transcript"]]
    useful = [r for r in fired if (r.get("original_in_top10") or {}).get("text_image")]
    payload["text_image_fires"] = {"first_text_query_missed": len(fired), "of_searched": sum(
        1 for r in rows if "text_transcript" in (r.get("original_in_top10") or {})),
        "text_image_then_finds_original": len(useful), "cases": [r["video_id"] for r in useful]}
    write_json(OUT_DIR / "probe_text_image_video.json", payload)
    print(json.dumps({k: payload[k] for k in ("candidates", "summary", "text_image_fires", "serper_balance_before",
                                              "serper_balance_after")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
