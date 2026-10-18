"""g3 阶段 2（P4）：C 族续读（READ_RECORD offset>0 / C5_read_more）与页内视频（C8）候选——复用 _branch_probe_common 的
C 族页面扫描（walker 同一条 search 层与页面打开器、Lens 关、不发图），只换候选池去重与输出位置。不建卷、不跑 walker。

    python scripts\\g3\\g3_scan_c.py --n 55            # 每个池 55 个候选（FMNV-C、GroundLie360、AVeriTeC 训练集）

挑法（长尾报告三节 a 的口径）：
- 续读：检索前 5 条里有正文 > 6000 字（read_page_chars，有 next_offset）的页，或 GT 来源页是长页（AVeriTeC）；
- C8：检索前 5 条里有正文带 YouTube / youtu.be / Vimeo / Dailymotion 链接（= walker._VIDEO_LINK）的页。
输出 claude/_g3_work/probe_c_scan.json（逐候选缓存）与 p4_c_candidates.json（两类各自的候选与命中页）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "g3"))
sys.path.insert(0, str(ROOT / "scripts"))
import g3_probe_a  # noqa: E402,F401  （导入即把 pc 的输出目录 / 已用编号 / EVID 换成 g3 口径）
import _branch_probe_common as pc  # noqa: E402

pc.AVERITEC = g3_probe_a.MAIN / "experiments" / "sft_data" / "data" / "annotations" / "averitec_train.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=55)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    pc.no_lens_env()
    before = pc.serper_balance()
    rows = asyncio.run(pc.c_scan(args.n, ["fmnv_c", "groundlie360", "averitec"]))
    after = pc.serper_balance()
    long_s = pc.c_branch_summary(rows, "long", args.top)
    video_s = pc.c_branch_summary(rows, "video_links", args.top)
    gt_long = [{"case_id": r["case_id"], "claim": (r.get("claim") or "")[:120],
                "gt_pages": [{"url": p["url"], "chars": p.get("chars")} for p in r.get("gt_pages") or [] if p.get("long")]}
               for r in rows if any(p.get("long") for p in r.get("gt_pages") or [])]
    payload = {"generated": pc.now(), "candidates": len(rows), "top": args.top, "serper_balance_before": before,
               "serper_balance_after": after,
               "read_more": {"by_pool": long_s["by_pool"], "top_long": long_s["reachable"], "gt_long": gt_long},
               "linked_video": {"by_pool": video_s["by_pool"], "reachable": video_s["reachable"]}}
    pc.write_json(pc.OUT_DIR / "p4_c_candidates.json", payload)
    print(json.dumps({"candidates": len(rows), "read_more_top": len(long_s["reachable"]), "read_more_gt": len(gt_long),
                      "linked_video": len(video_s["reachable"]), "serper": [before, after]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
