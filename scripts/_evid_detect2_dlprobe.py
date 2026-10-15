"""DETECT v2.2 原片下载自检（10-11，不用模型）：_evid_detect2.py 的 [0b] 步。

全量 evidC27F 里原片进了候选池、却没下载下来的样本：用现在的 INSPECT yt-dlp 下载（10-11 起非 403 的失败也换客户端）
逐个试下载 GT 原片，打印成功 / 失败和完整原因。下载完即删除。结果另存 outputs/detect/dlprobe_20261011.md。
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SCORES = ROOT / "experiments" / "evid" / "scores" / "evidC27F.sonnet.subset.jsonl"
OUT = ROOT / "outputs" / "detect" / "dlprobe_20261011.md"


def main() -> int:
    from foren_searcher.inspect.acquisition import ytdlp_download
    from foren_searcher.inspect.policy import InspectPolicy

    rows = [json.loads(line) for line in SCORES.read_text(encoding="utf-8").splitlines() if line.strip()]
    jobs = []
    for row in rows:
        funnel = row.get("funnel") or {}
        if funnel.get("source_in_pool") and not funnel.get("source_downloaded"):
            for vid in (funnel.get("source_hit_ids") or funnel.get("sources") or [])[:1]:
                jobs.append((row["sample_id"], row.get("type"), vid))
    work = ROOT / "outputs" / "detect" / "dlprobe_tmp"
    lines = ["# 原片下载自检（10-11）", "", f"{len(jobs)} 个样本：evidC27F 里原片进池但没下载。", "",
             "| 样本 | 类型 | 原片 | 结果 | 用时 s | 说明 |", "|---|---|---|---|---|---|"]
    ok = 0
    for sample, kind, vid in jobs:
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        url = f"https://www.youtube.com/watch?v={vid}"
        try:
            got = ytdlp_download(url, work / "v.media", InspectPolicy())
            ok += 1
            note = f"{got.details.get('method')} 客户端 {got.details.get('ytdlp_client')}, {got.details.get('byte_size')} 字节"
            if got.details.get("ytdlp_retries"):
                note += "；重试 " + "; ".join(f"{r['client']}: {r['error'][:80] or 'ok'}" for r in got.details["ytdlp_retries"])
            result = "成功"
        except Exception as exc:  # noqa: BLE001
            result = "失败"
            note = f"{type(exc).__name__}: {exc}"[:500].replace("|", "/").replace("\n", " ")
        line = f"| {sample} | {kind} | {vid} | {result} | {time.monotonic() - started:.0f} | {note} |"
        print(line, flush=True)
        lines.append(line)
    shutil.rmtree(work, ignore_errors=True)
    lines += ["", f"下载成功 {ok}/{len(jobs)}。"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"下载成功 {ok}/{len(jobs)}；表 {OUT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # noqa: BLE001 - 自检失败不挡后面的 A/B
        traceback.print_exc()
        raise SystemExit(0)
