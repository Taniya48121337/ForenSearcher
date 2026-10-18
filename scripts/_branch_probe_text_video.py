"""分支可达探测：SEARCH text→video（A 族 A3_search_original_text，找原片的第 1 条文字查询）。

触发条件（walker._decide_find_original）：A 族案卷做完 A1 OCR / A2 ASR 后，第 1 条查询一定是纯文字、output_type=video
（S1 槽写查询词）。所以分支本身每个 A 案例必触发；行能不能进 accepted 取决于报告核验（g1：A 族报告 30 份 positive 7）。
这里探"有没有用"：原片（FMNV 真新闻的 YouTube 编号）能不能进视频搜前 10。两条替身查询：
  text_transcript —— 转写前 12 个内容词（学生 ASR 后可见；S1 实际会结合 OCR，偏保守）
  text_title      —— FMNV 标题（学生看不到；当上界）
走 walker 同一条 search 层（yt-dlp 视频搜 + ddgs 兜底，免费）；不发图、Serper 0 点（探测前后读余额核对）。

    python scripts/_branch_probe_text_video.py --n 20
输出 claude/_longtail_work/probe_text_video.json
"""

from __future__ import annotations

import argparse
import asyncio
import json

from _branch_probe_common import a_probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=20, help="FMNV 真新闻（A 那一半、未用、非 EVID）候选个数")
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--only", nargs="*", help="只探这些视频编号")
    args = parser.parse_args()
    payload = asyncio.run(a_probe(args.n, ["text_transcript", "text_title"], False, "probe_text_video.json", args.redo, args.only))
    print(json.dumps({k: payload[k] for k in ("candidates", "summary", "serper_balance_before", "serper_balance_after")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
