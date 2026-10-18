"""g3 阶段 1：GroundLie360 配对的入选（负责人 10-18 定的收紧规则）。读 claude/_g3_work/probe_a_g360.json，不跑 walker。

- 只有 temporal_edit 的配对有资格做 A 族：原片必须在探测的 YouTube 前 10 里，且用 compare_media 确认是同一段素材，GT URL 才成立。
  做法：该配对前 10 里的 YouTube 结果按缩略图指纹距离取前 --top 个（默认 3，疑似的排前），yt-dlp 下 ≤ 480p 视频轨到
  claude/_g3_work/g360_confirm/（只为比对，比完删），与本地原片跑 compare_media 的同一个画面比对引擎
  （inspect.source_compare.SourceComparer，= merged.compare_media 的 visual 部分，不带人脸）；source_match 为真才算确认。
  比对结果（summary_lines、对应段、覆盖率）逐条记录。探测找到的链接不直接当 GT。
- false_title 等其余配对不做 A 族：缩略图指纹判 same / similar 的 YouTube 结果列成清单留给 C 族（本批不用），不下载。

    python scripts\\g3\\g3_confirm_g360.py [--top 3]

输出 claude/_g3_work/g360_confirm.json（temporal_edit 确认结果 + C 族清单）。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from threading import Event
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
WORK = ROOT / "claude" / "_g3_work"
PROBE = WORK / "probe_a_g360.json"
DL = WORK / "g360_confirm"
OUT = WORK / "g360_confirm.json"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def download(yt: str) -> tuple[Path | None, str | None]:
    from yt_dlp import YoutubeDL

    target = DL / yt
    target.mkdir(parents=True, exist_ok=True)
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True, "outtmpl": str(target / "%(id)s.%(ext)s"),
            "format": "bv*[height<=480][ext=mp4]/bv*[height<=480]/b[height<=480]/worst", "max_filesize": 300 * 1024 * 1024,
            "match_filter": lambda info, *_: None if (info.get("duration") or 0) <= 1200 else "longer than 20 min"}
    try:
        with YoutubeDL(opts) as ydl:
            ydl.extract_info(f"https://www.youtube.com/watch?v={yt}", download=True)
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {str(exc)[:200]}"
    files = [p for p in target.iterdir() if p.suffix in (".mp4", ".webm", ".mkv")]
    return (files[0], None) if files else (None, "no file (filtered or skipped)")


def compare(original: str, candidate: Path) -> dict[str, Any]:
    from foren_searcher.inspect.source_compare import CompareConfig, SourceComparer

    comparer = SourceComparer(None, CompareConfig(time_budget_s=600.0))
    try:
        report, _images = comparer.compare(Path(original), candidate, Event())
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    keep = ("source_match", "match", "segments_not_in_b", "summary_lines")
    out = {k: report.get(k) for k in keep if k in report}
    out["engine"] = getattr(comparer, "last_engine", "near_duplicate")
    out["correspondences"] = [{k: r.get(k) for k in ("kind", "a_range_s", "b_range_s", "mapping")} for r in (report.get("correspondences") or [])[:6]]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()
    rows = [r for r in json.loads(PROBE.read_text(encoding="utf-8"))["rows"] if "skipped" not in r]
    old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.is_file() else {}
    done = {(c["post"], c["yt"]): c for c in old.get("temporal_edit_checks") or []}
    checks, c_list = [], []
    for row in rows:
        cands = row.get("youtube_candidates") or []
        if not row.get("temporal_edit"):
            for c in row.get("likely_match") or []:
                c_list.append({"post": row["post"], "orig": row["orig"], "label": row.get("label"), "second": row.get("second"),
                               "platform": row.get("platform"), "title": row.get("title"), "yt": c["yt"], "url": c["url"],
                               "yt_title": c.get("title"), "thumb_verdict": c["verdict"], "phash": c["phash"], "dhash": c["dhash"],
                               "rank": c.get("rank")})
            continue
        for c in cands[: args.top]:
            key = (row["post"], c["yt"])
            if key in done and "error" not in (done[key].get("compare") or {}) and not done[key].get("download_error"):
                checks.append(done[key])
                continue
            item = {"post": row["post"], "orig": row["orig"], "title": row.get("title"), "yt": c["yt"], "url": c["url"],
                    "yt_title": c.get("title"), "rank": c.get("rank"), "thumb_verdict": c["verdict"], "checked_at": now()}
            path, err = download(c["yt"])
            if path is None:
                item["download_error"] = err
            else:
                item["compare"] = compare(row["orig_paths"][0], path)
                shutil.rmtree(path.parent, ignore_errors=True)
            item["confirmed"] = bool((item.get("compare") or {}).get("source_match"))
            print(f"[g360 temporal_edit] {row['post']} vs {c['yt']}：{'确认' if item['confirmed'] else '未确认'}"
                  f"（{item.get('download_error') or (item.get('compare') or {}).get('summary_lines', [''])[:1]}）", flush=True)
            checks.append(item)
    # 第二步：原片确认过的 (帖子, YouTube) 再比两次——帖子（= A 族材料）vs YouTube 结果、帖子 vs 本地原片：
    # 学生拿到的是帖子，walker 要用帖子去对 YouTube 结果；帖子的画面不在那个结果里，GT URL 就不成立
    by_post = {r["post"]: r for r in rows}
    post_checks = {(p["post"], p["yt"]): p for p in old.get("post_checks") or []}
    for item in [c for c in checks if c["confirmed"]]:
        key = (item["post"], item["yt"])
        if key in post_checks and "error" not in (post_checks[key].get("post_vs_youtube") or {}) and not post_checks[key].get("download_error"):
            continue
        row = by_post[item["post"]]
        entry = {"post": item["post"], "yt": item["yt"], "checked_at": now(),
                 "post_vs_original": compare(row["post_path"], Path(row["orig_paths"][0]))}
        path, err = download(item["yt"])
        if path is None:
            entry["download_error"] = err
        else:
            entry["post_vs_youtube"] = compare(row["post_path"], path)
            shutil.rmtree(path.parent, ignore_errors=True)
        entry["post_in_youtube"] = bool((entry.get("post_vs_youtube") or {}).get("source_match"))
        print(f"[g360 帖子] {item['post']} vs {item['yt']}：{'帖子画面在结果里' if entry['post_in_youtube'] else '对不上'}"
              f"（{(entry.get('post_vs_youtube') or {}).get('summary_lines', [entry.get('download_error')])[:1]}）", flush=True)
        post_checks[key] = entry
    accepted = sorted({p["post"] for p in post_checks.values() if p["post_in_youtube"]})
    te_rows = [r for r in rows if r.get("temporal_edit")]
    summary = {"temporal_edit_pairs": len(te_rows),
               "temporal_edit_with_youtube_results": sum(1 for r in te_rows if r.get("youtube_candidates")),
               "downloads_tried": len(checks), "download_errors": sum(1 for c in checks if c.get("download_error")),
               "confirmed_pairs": sorted({c["post"] for c in checks if c["confirmed"]}),
               "post_in_youtube_pairs": accepted,
               "false_title_etc_pairs": len(rows) - len(te_rows),
               "c_list_pairs": len({c["post"] for c in c_list})}
    OUT.write_text(json.dumps({"generated": now(), "rule": __doc__.split("\n\n")[1], "summary": summary,
                               "temporal_edit_checks": checks, "post_checks": list(post_checks.values()),
                               "c_family_list": c_list}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
