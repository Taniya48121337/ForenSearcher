"""g3 阶段 2（P3）：QVHighlights / VCSL 媒体本机各试一次（负责人 10-18：预算 30 分钟；不行就把下载清单交服务器 S2）。

- QVHighlights：同 walker_d2_cases.build_qvh（r0g 的做法）——从官方视频归档（nlp.cs.unc.edu，不经 YouTube）流式读，取训练集里
  只有一个相关窗口（10–60 s）的片段；源 YouTube 编号不在主目录 / .wt_g1 / .wt_g2 / 本工作树案卷、不在 EVID；频道（yt-dlp 元数据）
  不在 EVID 131–222 频道表。读到 --n 条或 --minutes 分钟到就停。媒体写 experiments/sft_data/data/qvhighlights/<qid>/（不入库），
  案卷 qvhighlights_<qid>.json（模板同 build_qvh，notes 记上传者 / 频道）。
- VCSL：训练集对（两侧 YouTube、各 15–120 s、标注 1–3 段），同 build_vcsl 的筛法；每个候选先 yt-dlp 原样下一次，失败再带
  浏览器 cookies（--cookies-browser，负责人 10-18 准许试一次）下一次；本机没有配置代理（环境变量 / 系统代理都没有），代理那一次
  无从试。只记下载结果，下不来的写进 S2 下载清单（编号、URL、所需字段）。

    python scripts\\g3\\g3_media_p3.py qvh --n 15 --minutes 20
    python scripts\\g3\\g3_media_p3.py vcsl --n 6 --cookies-browser chrome

输出 claude/_g3_work/p3_media_qvh.json、p3_media_vcsl.json、p3_s2_download_list.json。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts" / "g3"))
sys.path.insert(0, str(ROOT / "scripts"))
import g3_cases as g3c  # noqa: E402  （导入即把 d2._evid_text 换成读主目录 EVID 的版本）
import walker_d2_cases as d2  # noqa: E402

SFT, CASES, WORK = g3c.SFT, g3c.CASES, g3c.WORK
DATA = SFT / "data"
ANN = MAIN / "experiments" / "sft_data" / "data" / "annotations" / "qvhighlights_train.jsonl"
CASE_DIRS = (MAIN / "experiments" / "sft_data" / "cases", MAIN / ".wt_g1" / "experiments" / "sft_data" / "cases",
             MAIN / ".wt_g2" / "experiments" / "sft_data" / "cases", CASES)
log, now, load_json, save_json = g3c.log, g3c.now, g3c.load_json, g3c.save_json


def families(prefix: str) -> set[str]:
    out = set()
    for folder in CASE_DIRS:
        for path in folder.glob(f"{prefix}*.json"):
            out.add(str(load_json(path, {}).get("family_key")).partition(":")[2])
    return out


def s2_list(kind: str, items: list[dict[str, Any]]) -> None:
    path = WORK / "p3_s2_download_list.json"
    table = load_json(path, {})
    table[kind] = items
    save_json(path, table)


def qvh(n: int, minutes: float) -> None:
    import httpx

    sys.path.insert(0, str(SFT))
    from fetch_qvhighlights import StreamReader  # type: ignore

    rows: dict[str, list[dict[str, Any]]] = {}
    for line in ANN.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows.setdefault(row["vid"], []).append(row)
    seen, evid = families("qvhighlights_"), d2._evid_text()
    deadline = time.monotonic() + minutes * 60
    got: list[dict[str, Any]] = []
    stats = {"members_read": 0, "bytes_read_mb": 0.0, "skipped_used": 0, "skipped_window": 0, "skipped_evid_channel": 0}
    error = None
    try:
        with httpx.Client(timeout=90, follow_redirects=True) as client, client.stream("GET", d2.QVH_ARCHIVE) as response:
            response.raise_for_status()
            with tarfile.open(fileobj=StreamReader(response.iter_bytes(65536)), mode="r|gz") as archive:
                for member in archive:
                    stats["members_read"] += 1
                    stats["bytes_read_mb"] = round(member.offset / 1e6, 1)
                    if time.monotonic() > deadline or len(got) >= n:
                        break
                    vid = Path(member.name).stem
                    source = vid.rsplit("_", 2)[0]
                    if not member.isfile() or vid not in rows:
                        continue
                    if source in seen or source in evid:
                        stats["skipped_used"] += 1
                        continue
                    pick = next((r for r in rows[vid] if len(r.get("relevant_windows") or []) == 1
                                 and 10 <= r["relevant_windows"][0][1] - r["relevant_windows"][0][0] <= 60), None)
                    if pick is None:
                        stats["skipped_window"] += 1
                        continue
                    cid = f"qvhighlights:{pick['qid']}"
                    ok, channel = g3c.g1c.channel_check(cid, f"https://www.youtube.com/watch?v={source}")
                    seen.add(source)
                    if not ok:
                        stats["skipped_evid_channel"] += 1
                        continue
                    folder = DATA / "qvhighlights" / str(pick["qid"])
                    (folder / "inputs").mkdir(parents=True, exist_ok=True)
                    data = archive.extractfile(member)
                    (folder / "inputs" / "A.mp4").write_bytes(data.read() if data else b"")
                    (folder / "reference.json").write_text(json.dumps(pick, ensure_ascii=False), encoding="utf-8")
                    start, end = pick["relevant_windows"][0]
                    query = " ".join(str(pick["query"]).split())
                    uploader = f"{channel[0]}（{channel[1]}）" if channel and len(channel) > 1 else "取不到（yt-dlp 元数据）"
                    case = {
                        "case_id": cid, "dataset": "qvhighlights", "source_id": str(pick["qid"]), "split": "dev_train",
                        "task_kind": "temporal_locate", "task_text": d2.QVH_ZH.format(query=query),
                        "task_text_en": d2.QVH_EN.format(query=query), "family_key": f"qvhighlights:{source}",
                        "media": [{"ref": "I0", "path": f"../data/qvhighlights/{pick['qid']}/inputs/A.mp4", "media_type": "video/mp4"}],
                        "claims": [], "source_policy": {"mode": "provided_media", "excluded_keys": [], "date_cutoff": None},
                        "requirements": [
                            {"id": "REQ1", "text": "符合描述的片段在视频中的起止时间（秒）", "must_cite": True,
                             "text_en": "Start and end time (seconds) of the segment in the video that matches the description"},
                            {"id": "REQ2", "text": "支持定位的画面依据：片段内以及边界两侧的帧所显示的内容", "must_cite": True,
                             "text_en": "Visual grounds for the localization: what the frames inside the segment and on both sides of its boundaries show"},
                            {"id": "REQ3", "text": "定位精度与边界的不确定范围（未检查的区间不当作不符合）", "must_cite": False,
                             "text_en": "Localization precision and the uncertain range of the boundaries (unexamined intervals are not treated as non-matching)"},
                        ],
                        "required_evidence": [{"key": "frames", "description": "给定视频的抽帧（片段内与边界两侧）", "url_pattern": None,
                                               "acquired_by": "INSPECT/get_frames", "must_cite": True, "supports": ["REQ1", "REQ2"]}],
                        "required_evidence_rule": "all",
                        "checks": [{"kind": "interval_iou", "params": {"start": start, "end": end, "min_iou": 0.5},
                                    "note": "QVHighlights relevant_windows（片段内秒数）", "requirement": "REQ1"}],
                        "reference_path": f"../data/qvhighlights/{pick['qid']}/reference.json", "used_for_revision": [],
                        "notes": (f"g3 建卷（{now()}，scripts/g3/g3_media_p3.py qvh，模板同 walker_d2_cases.build_qvh）：标注片段 "
                                  f"[[{start}, {end}]]（视频 {pick.get('duration')} 秒，源 YouTube {source} 的 "
                                  f"{vid.rsplit('_', 2)[1]}–{vid.rsplit('_', 2)[2]} s）；上传者 / 频道：{uploader}；family 按源视频号分组；"
                                  f"媒体取自 QVHighlights 官方视频归档（不经 YouTube）。"),
                    }
                    d2.write_case(case)
                    g3c.g1c.record_made(case, "B", "qvhighlights", {"group": "P3_qvh", "window": [start, end], "channel": channel})
                    g3c.g1c.record_channel(cid, channel)
                    got.append({"qid": pick["qid"], "vid": vid, "source": source, "window": [start, end], "query": query[:120]})
                    log(f"  QVHighlights {pick['qid']}（{vid}）：[{start}, {end}]；{query[:80]}")
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {str(exc)[:300]}"
        log(f"[QVH] 出错：{error}")
    result = {"generated": now(), "archive": d2.QVH_ARCHIVE, "minutes_cap": minutes, "n_cap": n, "got": got, "stats": stats, "error": error}
    save_json(WORK / "p3_media_qvh.json", result)
    log(f"[QVH] 新建 {len(got)}；{json.dumps(stats, ensure_ascii=False)}")


def vcsl(n: int, browser: str | None) -> None:
    from foren_searcher.inspect.acquisition import ytdlp_download
    from foren_searcher.inspect.policy import InspectPolicy

    data = d2.VCSL_DATA
    urls = {r["uuid"]: r["url"] for r in csv.DictReader((data / "videos_url_uuid.csv").open(encoding="utf-8", newline=""))}
    frames = {r["uuid"]: int(r["frame_count"]) for r in csv.DictReader((data / "frames_all.csv").open(encoding="utf-8", newline=""))}
    labels = json.loads((data / "label_file_uuid_total.json").read_text(encoding="utf-8"))
    fams, evid = families("vcsl_"), d2._evid_text()
    used_q = {load_json(p, {}).get("query_id") for p in (MAIN / "experiments" / "sft_data" / "data" / "vcsl").glob("pair_*/reference.json")}
    pairs = []
    for row in csv.DictReader((data / "pair_file_train.csv").open(encoding="utf-8", newline="")):
        q, r = row["query_id"], row["reference_id"]
        segs = labels.get(f"{q}-{r}") or []
        if q in used_q or f"{q[:12]}" in fams or not (1 <= len(segs) <= 3):
            continue
        if not all("youtube.com" in urls.get(u, "") for u in (q, r)) or not all(15 <= frames.get(u, 0) <= 120 for u in (q, r)):
            continue
        if any(urls[u].split("v=")[-1] in evid for u in (q, r)):
            continue
        pairs.append((q, r, segs))
    order = {f"{q}-{r}": i for i, (q, r, _) in enumerate(pairs)}
    ranked = [pairs[order[k]] for k in d2._hash_order(list(order), "g3-vcsl")]
    tried, results, seen_q = 0, [], set()
    tmp = WORK / "vcsl_try"
    for q, r, segs in ranked:
        if tried >= n:
            break
        if q in seen_q:
            continue
        seen_q.add(q)
        tried += 1
        item: dict[str, Any] = {"query_id": q, "reference_id": r, "url_a": urls[q], "url_b": urls[r], "segments": segs,
                                "frames_a": frames.get(q), "frames_b": frames.get(r), "attempts": []}
        for label, opts in (("plain", None), ("cookies", browser)):
            if label == "cookies" and not browser:
                continue
            ok = True
            for side, url in (("A", urls[q]), ("B", urls[r])):
                target = tmp / f"{q[:8]}-{r[:8]}" / f"{side}.download"
                target.parent.mkdir(parents=True, exist_ok=True)
                started = time.monotonic()
                try:
                    if opts:                      # 工具层 ytdlp_download 不带 cookies：这一次直接用 yt_dlp（只为试下载，不改工具层）
                        from yt_dlp import YoutubeDL

                        with YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True, "cookiesfrombrowser": (opts,),
                                        "format": "bv*[height<=480][ext=mp4]/bv*[height<=480]",
                                        "outtmpl": str(target.with_suffix(".%(ext)s"))}) as ydl:
                            info = ydl.extract_info(url, download=True)
                        path = Path(ydl.prepare_filename(info))
                    else:
                        path = Path(ytdlp_download(url, target, InspectPolicy(), video_only_height=480).path)
                    item["attempts"].append({"how": label, "side": side, "ok": True, "seconds": round(time.monotonic() - started, 1),
                                             "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)})
                except Exception as exc:  # noqa: BLE001
                    ok = False
                    item["attempts"].append({"how": label, "side": side, "ok": False, "seconds": round(time.monotonic() - started, 1),
                                             "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
                    break
            if ok:
                item["downloaded"] = label
                break
        results.append(item)
        log(f"  VCSL {q[:8]}-{r[:8]}：{item.get('downloaded') or '下不来'}（{[a.get('error', 'ok')[:60] for a in item['attempts']]}）")
    save_json(WORK / "p3_media_vcsl.json", {"generated": now(), "cookies_browser": browser, "proxy": "本机未配置代理，无从试",
                                            "candidates": len(pairs), "tried": results})
    s2_list("vcsl", [{"query_id": i["query_id"], "reference_id": i["reference_id"], "url_a": i["url_a"], "url_b": i["url_b"],
                      "segments": i["segments"], "need": "两侧视频轨 ≤480p、各 15–120 s；建卷后本机 compare_media 预筛夹边界（对应段占比低 / 有锚点）"}
                     for i in results if not i.get("downloaded")] + [{"note": f"另有 {max(0, len(pairs) - tried)} 个候选对按同一筛法在 pair_file_train.csv 里，"
                                                                             "服务器可按 g3-vcsl 哈希顺序续下"}])


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("qvh")
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--minutes", type=float, default=20)
    p = sub.add_parser("vcsl")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--cookies-browser", default=None)
    args = parser.parse_args()
    if args.cmd == "qvh":
        qvh(args.n, args.minutes)
    else:
        vcsl(args.n, args.cookies_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
