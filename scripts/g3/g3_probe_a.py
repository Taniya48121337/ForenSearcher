"""g3 阶段 1：A 族（找原片）"分支可达"探测——复用 scripts/_branch_probe_common.py（walker 同一条 search 层
run_cases.search_service_for、Lens 硬关、图床关、Serper 余额前后核），只换候选池与输出位置。不建卷、不跑 walker。

    python scripts\\g3\\g3_probe_a.py fmnv            # FMNV 真新闻 YouTube 未用的全部（两半哈希都要），三种查询
    python scripts\\g3\\g3_probe_a.py fv              # FMNV CE 类（fv，删段假视频）：原片 = 同编号 YouTube，test 划分不取
    python scripts\\g3\\g3_probe_a.py g360            # GroundLie360 假帖子 + 原片都在本地的 49 对

每个候选各搜一次 output_type=video（与 walker A3 同一请求形状）：
- text_title：FMNV 标题 / GroundLie360 帖子标题（学生看不到标题，是上界——画面文字里常有标题条）；
- text_transcript：转写前 12 个内容词（学生 A2 ASR 后可见，S1 写查询词的保守替身）；
- text_image：转写词 + 中段一帧，以图搜只走 Yandex（= walker_run.r1_search_policy 给 A 族的 yandex）。
  只发 FMNV / GroundLie360 公开视频的帧；候选先排除 EVID 全文里出现过的编号，EVID 的帧 / 视频一律不碰。
两者（text→video 任一、text_image→video）任一把原片排进前 10 算可达。

FMNV：原片编号已知（= YouTube 编号），按编号判排名。
GroundLie360：原片只有数据集编号、没有网址——前 10 里的 YouTube 结果取缩略图（i.ytimg.com，只读），与本地原片均匀 12 帧比
指纹（walker 缩略图自核的 thumb_verify：去边 dHash / pHash，verdict same / similar），只作"疑似同一段素材"的预筛；temporal_edit 的配对还要下载疑似结果、用 compare_media 确认（另一步，
见 g3_confirm_g360.py），探测找到的链接不直接当 GT。

输出 claude/_g3_work/probe_a_fmnv.json、probe_a_g360.json。
"""

from __future__ import annotations

import ast
import asyncio
import csv
import gzip
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT.parent                                  # 主目录（.wt_g3 在主目录下），只读
sys.path.insert(0, str(ROOT / "scripts"))
import _branch_probe_common as pc  # noqa: E402

pc.OUT_DIR = ROOT / "claude" / "_g3_work"
pc.G1_CASES = MAIN / ".wt_g1" / "experiments" / "sft_data" / "cases"
pc.G2_CASES = MAIN / ".wt_g2" / "experiments" / "sft_data" / "cases"
CASE_DIRS = (MAIN / "experiments" / "sft_data" / "cases", pc.G1_CASES, pc.G2_CASES, ROOT / "experiments" / "sft_data" / "cases")
MODES = ["text_title", "text_transcript", "text_image"]


def used_ids() -> set[str]:
    """主目录 cases/（含未入库的）+ .wt_g1 + .wt_g2 + 本工作树案卷里出现过的编号（= pc.used_ids 的规则，多看两个目录）。"""

    out: set[str] = set()
    for folder in CASE_DIRS:
        if not folder.is_dir():
            continue
        for path in folder.glob("*.json"):
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            for value in (case.get("source_id"), str(case.get("family_key") or "").partition(":")[2], path.stem):
                v = str(value or "")
                if v:
                    out |= {v, re.sub(r"^(?:a|c|v|e|l)_", "", v),
                            re.sub(r"^(?:fmnv|groundlie360|averitec|averimatec)_(?:[ac]_|c_|v)?", "", v)}
    return out


pc.used_ids = used_ids


def evid_text() -> str:
    """EVID 案卷在主目录（工作树里没有 experiments/evid）；读不到就停，不许静默关掉 EVID 排除。"""

    folder = MAIN / "experiments" / "evid" / "cases"
    files = list(folder.glob("*.json"))
    if len(files) < 200:
        raise SystemExit(f"EVID 案卷只读到 {len(files)} 个（{folder}），不探测")
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in files)


pc.evid_text = evid_text


# ---------------------------------------------------------------- FMNV

def pool_fmnv() -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows, evid, used, split = pc.fmnv_rows(), pc.evid_text(), used_ids(), pc.fmnv_split()
    real = [v for v, r in rows.items() if r.get("label") == "true"]
    yt = [v for v in real if re.fullmatch(r"[\w-]{11}", v) and not v.isdigit()]
    noevid = [v for v in yt if v not in evid]
    free = [v for v in noevid if v not in used]
    notest = [v for v in free if split.get(v) != "test"]           # g1 建卷只取 FMNV 的 train / val（test 划分留着）
    funnel = {"real": len(real), "real_youtube": len(yt), "not_evid": len(noevid), "unused": len(free), "not_fmnv_test": len(notest)}
    # 探测两部分都探（免费）：test 划分能不能建卷由负责人定，汇总里分开计
    ranked = pc.ordered(notest, "g3-fmnv-a") + pc.ordered([v for v in free if v not in set(notest)], "g3-fmnv-a")
    return ([{"pool": "fmnv_real", "video_id": v, "title": rows[v].get("title"), "subject": rows[v].get("subject"),
              "transcript": rows[v].get("audio_transcript") or "", "fmnv_split": split.get(v),
              "hash_half": "A" if int(hashlib.sha1(("split-a:" + v).encode()).hexdigest(), 16) % 2 == 0 else "C"}
             for v in ranked], funnel)


async def probe_fmnv() -> None:
    path = pc.OUT_DIR / "probe_a_fmnv.json"
    old = {r["video_id"]: r for r in (json.loads(path.read_text(encoding="utf-8")).get("rows") or [])} if path.is_file() else {}
    cands, funnel = pool_fmnv()
    before = pc.serper_balance()
    for i, cand in enumerate(cands, 1):
        if cand["video_id"] in old and "error" not in old[cand["video_id"]]:
            continue
        started = time.monotonic()
        try:
            row = await pc.a_probe_one(cand, MODES, allow_yandex=True)
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
            row = {**{k: v for k, v in cand.items() if k != "transcript"}, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        row["seconds"] = round(time.monotonic() - started, 1)
        row["probed_at"] = pc.now()
        old[cand["video_id"]] = row
        print(f"[fmnv {i}/{len(cands)}] {cand['video_id']}: 原片排名 {row.get('original_in_top10')}；{row['seconds']:.0f}s", flush=True)
        pc.write_json(path, {"generated": pc.now(), "funnel": funnel, "rows": list(old.values())})
    after = pc.serper_balance()
    rows = [old[c["video_id"]] for c in cands]
    hit = lambda r, m: bool((r.get("original_in_top10") or {}).get(m))  # noqa: E731
    summary = {**funnel, **{part: part_summary([r for r in rows if (r.get("fmnv_split") == "test") == (part == "fmnv_test")], hit)
                            for part in ("train_val", "fmnv_test")}}
    pc.write_json(path, {"generated": pc.now(), "modes": MODES, "serper_balance_before": before, "serper_balance_after": after,
                         "summary": summary,
                         "reachable": [r["video_id"] for r in rows if any((r.get("original_in_top10") or {}).values())],
                         "rows": rows})
    print(json.dumps(summary, ensure_ascii=False))


def part_summary(rows: list[dict[str, Any]], hit: Any) -> dict[str, int]:
    return {
        "probed": len(rows), "errors": sum(1 for r in rows if "error" in r),
        "text_title_hit": sum(hit(r, "text_title") for r in rows),
        "text_transcript_hit": sum(hit(r, "text_transcript") for r in rows),
        "text_video_hit": sum(hit(r, "text_title") or hit(r, "text_transcript") for r in rows),
        "text_image_hit": sum(hit(r, "text_image") for r in rows),
        "reachable": sum(any((r.get("original_in_top10") or {}).values()) for r in rows),
    }


def pool_fmnv_fv() -> tuple[list[dict[str, Any]], dict[str, int]]:
    """FMNV CE 类（class fv，"关键片段被删"）：数据集作者从 YouTube 原片删段做成的假视频，zip 里有假视频、编号沿用原片的
    YouTube 编号（原片本机没有）。材料 = 假视频本身（真实加工），GT 原片 = youtube.com/watch?v=<同一编号>。test 划分不取。"""

    rows, evid, used, split = pc.fmnv_rows(), pc.evid_text(), used_ids(), pc.fmnv_split()
    fv = [v for v, r in rows.items() if r.get("class") == "fv"]
    yt = [v for v in fv if re.fullmatch(r"[\w-]{11}", v) and not v.isdigit()]
    free = [v for v in yt if v not in evid and v not in used]
    notest = [v for v in free if split.get(v) != "test"]
    funnel = {"fv": len(fv), "fv_youtube": len(yt), "unused_not_evid": len(free), "not_fmnv_test": len(notest)}
    return ([{"pool": "fmnv_fv", "video_id": v, "title": rows[v].get("title"), "subject": rows[v].get("subject"),
              "transcript": rows[v].get("audio_transcript") or "", "fmnv_split": split.get(v)}
             for v in pc.ordered(notest, "g3-fmnv-fv")], funnel)


async def probe_fmnv_fv() -> None:
    """同 probe_fmnv：标题 / 转写词 / 转写词 + 假视频中段一帧（zip 预抽帧）各搜一次视频，原片编号进前 10 算可达。"""

    path = pc.OUT_DIR / "probe_a_fmnv_fv.json"
    old = {r["video_id"]: r for r in (json.loads(path.read_text(encoding="utf-8")).get("rows") or [])} if path.is_file() else {}
    cands, funnel = pool_fmnv_fv()
    before = pc.serper_balance()
    for i, cand in enumerate(cands, 1):
        if cand["video_id"] in old and "error" not in old[cand["video_id"]]:
            continue
        started = time.monotonic()
        try:
            row = await pc.a_probe_one(cand, MODES, allow_yandex=True)
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
            row = {**{k: v for k, v in cand.items() if k != "transcript"}, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        row["seconds"] = round(time.monotonic() - started, 1)
        row["probed_at"] = pc.now()
        old[cand["video_id"]] = row
        print(f"[fv {i}/{len(cands)}] {cand['video_id']}: 原片排名 {row.get('original_in_top10')}；{row['seconds']:.0f}s", flush=True)
        pc.write_json(path, {"generated": pc.now(), "funnel": funnel, "rows": list(old.values())})
    after = pc.serper_balance()
    rows = [old[c["video_id"]] for c in cands]
    hit = lambda r, m: bool((r.get("original_in_top10") or {}).get(m))  # noqa: E731
    summary = {**funnel, **part_summary(rows, hit)}
    pc.write_json(path, {"generated": pc.now(), "modes": MODES, "serper_balance_before": before, "serper_balance_after": after,
                         "summary": summary,
                         "reachable": [r["video_id"] for r in rows if any((r.get("original_in_top10") or {}).values())],
                         "rows": rows})
    print(json.dumps(summary, ensure_ascii=False))


# ---------------------------------------------------------------- GroundLie360

G360 = pc.G360_CSV.parent


def g360_pairs() -> list[dict[str, Any]]:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with gzip.open(pc.G360_CSV, "rt", encoding="utf-8", newline="") as stream:
        rows = {str(r["video_id"]): r for r in csv.DictReader(stream)}
    index = json.loads(pc.G360_INDEX.read_text(encoding="utf-8"))
    evid, used = pc.evid_text(), used_ids()

    def lit(x: str) -> Any:
        try:
            return ast.literal_eval(x) if x and x.strip() else []
        except (ValueError, SyntaxError):
            return []

    out = []
    for v, r in rows.items():
        origs = [str(x) for x in lit(r.get("original_videoid") or "") if str(x) in index]
        if v not in index or not origs:
            continue
        flags = (["post_used"] if v in used else []) + (["orig_used"] if any(o in used for o in origs) else []) + \
                (["evid"] if v in evid or any(o in evid for o in origs) else [])
        second = lit(r.get("second_level_annotation") or "")
        out.append({"pool": "groundlie360", "post": v, "orig": origs, "post_path": str(G360 / index[v]),
                    "orig_paths": [str(G360 / index[o]) for o in origs], "label": r.get("first_level_annotation"),
                    "second": second, "temporal_edit": "temporal_edit" in second, "platform": r.get("platform"),
                    "title": " ".join((r.get("title") or "").split()), "date": r.get("video_date"),
                    "transcript": r.get("video_transcript") or "", "flags": flags})
    return out


def video_frames(path: str, n: int) -> list[Any]:
    """PyAV 均匀取 n 帧（PIL 图）。"""

    import av

    out = []
    with av.open(path) as box:
        stream = box.streams.video[0]
        duration = float(box.duration / 1e6) if box.duration else float(stream.duration * stream.time_base)
        for k in range(n):
            t = duration * (k + 0.5) / n
            box.seek(int(t / stream.time_base), stream=stream, any_frame=False, backward=True)
            for frame in box.decode(stream):
                if frame.time is not None and frame.time >= t - 0.5:
                    out.append(frame.to_image())
                    break
    return out


def fingerprint(img: Any) -> Any:
    """walker 缩略图自核同一套指纹（search/providers/thumb_verify：去边 + dHash / pHash）。"""

    from foren_searcher.search.providers import thumb_verify as tv

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return tv.fingerprint(buf.getvalue())


def thumb_print(yt: str, cache: dict[str, Any]) -> Any:
    if yt in cache:
        return cache[yt]
    import httpx

    from foren_searcher.search.providers import thumb_verify as tv

    value = None
    for name in ("hqdefault.jpg", "mqdefault.jpg"):
        try:
            r = httpx.get(f"https://i.ytimg.com/vi/{yt}/{name}", timeout=20)
            if r.status_code == 200 and r.content:
                value = tv.fingerprint(r.content)          # trim_borders 去掉 hqdefault 的上下黑边
                break
        except Exception:  # noqa: BLE001
            continue
    cache[yt] = value
    return value


def thumb_vs_orig(th: Any, origs: list[Any]) -> dict[str, Any] | None:
    from foren_searcher.search.providers import thumb_verify as tv

    if th is None or not origs:
        return None
    best = min((tv.distances(th, o) for o in origs), key=lambda d: (d["phash"] + d["dhash"]))
    return {**best, "verdict": tv.verdict(best)}


async def g360_probe_one(pair: dict[str, Any], thumbs: dict[str, Any]) -> dict[str, Any]:
    from foren_searcher.search.contracts import SearchQueryRequest

    row = {k: v for k, v in pair.items() if k != "transcript"}
    q_tr = pc.transcript_query(pair["transcript"])
    q_title = pair["title"][:300]
    row["queries"] = {"title": q_title, "transcript": q_tr}
    orig_prints = [fingerprint(im) for p in pair["orig_paths"] for im in video_frames(p, 12)]
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        frames = video_frames(pair["post_path"], 1)
        frame = None
        if frames:
            frames[0].convert("RGB").save(tmpdir / "frame.jpg", quality=90)
            from check_unanswerable import query_jpeg

            frame = query_jpeg(tmpdir / "frame.jpg", tmpdir / "query.jpg")
        case = pc.make_case("A", {"video_id": f"g360_{pair['post']}"})
        search, store, _desc = await pc.open_search(case, tmpdir, "yandex" if frame else "off")
        searches: dict[str, Any] = {}
        for mode, req, img in (("text_title", SearchQueryRequest("text", q_title, "video"), None),
                               ("text_transcript", SearchQueryRequest("text", q_tr, "video") if q_tr else None, None),
                               ("text_image", SearchQueryRequest("text_image", {"evidence_id": "E1", "text": q_tr or q_title}, "video")
                                if frame else None, frame)):
            if req is None:
                searches[mode] = {"skipped": "no transcript" if mode == "text_transcript" else "no frame"}
                continue
            entry = await pc.video_search(search, req, img)
            for res in entry.get("results") or []:
                yt = res.get("yt")
                res["thumb_vs_orig"] = thumb_vs_orig(thumb_print(yt, thumbs), orig_prints) if yt else None
            searches[mode] = entry
        await pc.close_search(search, store)
    row["searches"] = searches
    best: dict[str, Any] = {}
    for mode, entry in searches.items():
        for res in entry.get("results") or []:
            d = res.get("thumb_vs_orig")
            if not d or not res.get("yt"):
                continue
            score = d["phash"] + d["dhash"]
            if res["yt"] not in best or score < best[res["yt"]]["score"]:
                best[res["yt"]] = {"yt": res["yt"], "url": res["url"], "title": res.get("title"), "score": round(score, 4), **d,
                                   "modes": sorted({m for m, e in searches.items() for x in e.get("results") or [] if x.get("yt") == res["yt"]}),
                                   "rank": {m: x["rank"] for m, e in searches.items() for x in e.get("results") or [] if x.get("yt") == res["yt"]}}
    row["youtube_candidates"] = sorted(best.values(), key=lambda x: x["score"])[:5]
    row["likely_match"] = [c for c in row["youtube_candidates"] if c["verdict"] in ("same", "similar")]
    return row


async def probe_g360() -> None:
    path = pc.OUT_DIR / "probe_a_g360.json"
    old = {r["post"]: r for r in (json.loads(path.read_text(encoding="utf-8")).get("rows") or [])} if path.is_file() else {}
    pairs = g360_pairs()
    thumbs: dict[str, Any] = {}
    before = pc.serper_balance()
    for i, pair in enumerate(pairs, 1):
        if pair["flags"]:
            old[pair["post"]] = {k: v for k, v in pair.items() if k != "transcript"} | {"skipped": "used/evid"}
            continue
        if pair["post"] in old and "error" not in old[pair["post"]] and "skipped" not in old[pair["post"]]:
            continue
        started = time.monotonic()
        try:
            row = await g360_probe_one(pair, thumbs)
        except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
            row = {**{k: v for k, v in pair.items() if k != "transcript"}, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        row["seconds"] = round(time.monotonic() - started, 1)
        row["probed_at"] = pc.now()
        old[pair["post"]] = row
        print(f"[g360 {i}/{len(pairs)}] {pair['post']}（{'temporal_edit' if pair['temporal_edit'] else 'false_title 等'}）："
              f"疑似 {[(c['yt'], c['verdict']) for c in row.get('likely_match') or []]}；"
              f"最近 {[(c['yt'], c['phash'], c['dhash']) for c in (row.get('youtube_candidates') or [])[:2]]}；{row['seconds']:.0f}s", flush=True)
        pc.write_json(path, {"generated": pc.now(), "rows": list(old.values())})
    after = pc.serper_balance()
    rows = list(old.values())
    probed = [r for r in rows if "skipped" not in r]
    summary = {"pairs_local": len(pairs), "excluded_used_or_evid": sum(1 for r in rows if "skipped" in r), "probed": len(probed),
               "errors": sum(1 for r in probed if "error" in r),
               "temporal_edit": sum(1 for r in probed if r.get("temporal_edit")),
               "any_youtube_result": sum(1 for r in probed if r.get("youtube_candidates")),
               "likely_match": sum(1 for r in probed if r.get("likely_match")),
               "temporal_edit_likely_match": sum(1 for r in probed if r.get("temporal_edit") and r.get("likely_match"))}
    pc.write_json(path, {"generated": pc.now(), "modes": MODES, "serper_balance_before": before, "serper_balance_after": after,
                         "match_rule": "前 10 的 YouTube 结果缩略图 vs 本地原片均匀 12 帧，walker 缩略图自核同一套指纹（thumb_verify：去边 dHash / pHash），verdict same / similar = 疑似（只预筛，不当 GT）",
                         "summary": summary, "rows": rows})
    print(json.dumps(summary, ensure_ascii=False))


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    pc.no_lens_env()
    if which == "fmnv":
        asyncio.run(probe_fmnv())
    elif which == "fv":
        asyncio.run(probe_fmnv_fv())
    elif which == "g360":
        asyncio.run(probe_g360())
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
