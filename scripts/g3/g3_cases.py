"""g3 建卷：P1 A 族（找原片）。模板照 scripts/g1/g1_cases.build_fmnv_a（= walker_d2_cases.build_provenance，wk7 的做法：
原片 42% 处起删一段（原片时长 15%，夹在 6–20 s），PyAV 重编码作材料；GT = 原片 URL 模式 + 被删区间 interval_iou），只换：

- 候选只取 g3 探测（scripts/g3/g3_probe_a.py → claude/_g3_work/probe_a_fmnv.json）里"可达"的（text→video 或 text_image→video
  任一把原片排进前 10）；不再按哈希顺序从全池取；
- FMNV 自己的 test 划分默认不取（同 g1）；--allow-fmnv-test 才取（负责人定）；
- 去重：主目录 cases（含未入库）+ .wt_g1 + .wt_g2 + 本工作树案卷的编号 / family_key / 媒体与 notes 全文；EVID 全文读主目录
  experiments/evid/cases（工作树里没有这个目录，读不到就停）；
- C6 三重去重建卷时先做一遍：编号（EVID 全文）、频道（yt-dlp 元数据 vs EVID 131–222 频道表；取不到不挡，同 g1）、
  感知哈希（材料 vs EVID 131–222 视频，用门 C6 同一个 _dhash_frames 与阈值）；上传者 / 频道写进案卷 notes 与 g3_channels.json；
- 媒体放 cases/media/g3/，记录 g3_cases_made.json，日志 claude/_g3_work/g3_cases.log。

    python scripts\\g3\\g3_cases.py fmnv_a [--n 40] [--allow-fmnv-test]
    python scripts\\g3\\g3_cases.py fv_a --n 10   # FMNV CE 类（fv）找原片：URL-only GT，compare_media 只作建卷过滤与参考
    python scripts\\g3\\g3_cases.py p4         # P4 C 族续读 / 页内视频（p4_picks.json，22 个挑选 → 21 个案卷）
    python scripts\\g3\\g3_cases.py list       # 写 experiments/sft_data/walker_g3_cases.json（按 P 分组）
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts" / "g1"))
sys.path.insert(0, str(ROOT / "scripts" / "g3"))
sys.path.insert(0, str(ROOT / "scripts"))
import g1_cases as g1c  # noqa: E402
import walker_d2_cases as d2  # noqa: E402

SFT, CASES = g1c.SFT, g1c.CASES
WORK = ROOT / "claude" / "_g3_work"
g1c.MEDIA = CASES / "media" / "g3"
g1c.LOG = WORK / "g3_cases.log"
g1c.CHANNELS_FILE = SFT / "g3_channels.json"
g1c.MADE_FILE = SFT / "g3_cases_made.json"
G2_CASES = MAIN / ".wt_g2" / "experiments" / "sft_data" / "cases"
EVID_CASES = MAIN / "experiments" / "evid" / "cases"
EVID_VIDEOS = ROOT / "plan-harness-1" / "datasets" / "EVID-Bench" / "videos"
PROBE = WORK / "probe_a_fmnv.json"
LIST_FILE = SFT / "walker_g3_cases.json"
log, now, load_json, save_json = g1c.log, g1c.now, g1c.load_json, g1c.save_json


def evid_text() -> str:
    files = list(EVID_CASES.glob("*.json"))
    if len(files) < 200:
        raise SystemExit(f"EVID 案卷只读到 {len(files)} 个（{EVID_CASES}），不建卷")
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in files)


d2._evid_text = evid_text


def existing() -> dict[str, Any]:
    ex = g1c.existing()                                   # 主目录 cases + 本工作树 cases（已含合入的 g1 案卷）
    fams, names, blob = set(ex["families"]), set(ex["names"]), [ex["blob"]]
    for path in G2_CASES.glob("*.json"):
        names.add(path.stem)
        case = load_json(path, {})
        fams.add(str(case.get("family_key")))
        blob.append(json.dumps(case.get("media") or [], ensure_ascii=False))
        blob.append(str(case.get("notes") or ""))
    return {"families": fams, "names": names, "blob": "\n".join(blob), "evid": ex["evid"]}


_EVID_HASHES: dict[str, list[int]] | None = None


def phash_hits(material: Path, max_distance: int = 6) -> list[dict[str, Any]]:
    """= export_gate C6 之三：材料与 EVID 131–222 视频各取 8 帧 dHash，≥ 3 帧在对方找到汉明距离 ≤ 6 的近似帧算重复。"""

    global _EVID_HASHES
    from export_gate import _dhash_frames

    if _EVID_HASHES is None:
        _EVID_HASHES = {}
        for num in range(131, 223):
            path = EVID_VIDEOS / f"video_{num:03d}.mp4"
            if path.is_file():
                _EVID_HASHES[path.stem] = _dhash_frames(path)
        if len(_EVID_HASHES) < 80:
            raise SystemExit(f"EVID 131–222 视频只读到 {len(_EVID_HASHES)} 个（{EVID_VIDEOS}），感知哈希去重做不了")
    mine = _dhash_frames(material)
    hits = []
    for vid, theirs in _EVID_HASHES.items():
        close = [min((bin(a ^ b).count("1") for b in theirs), default=64) for a in mine]
        matched = sum(1 for d in close if d <= max_distance)
        if mine and theirs and matched >= min(3, len(mine)):
            hits.append({"evid": vid, "matched_frames": matched, "of": len(mine)})
    return hits


def build_fmnv_a(n: int, allow_test: bool) -> None:
    from _d2_media import cut_segments, probe_duration

    probe = load_json(PROBE, {})
    reachable = list(probe.get("reachable") or [])
    by_id = {r["video_id"]: r for r in probe.get("rows") or []}
    if not reachable:
        raise SystemExit(f"{PROBE} 里没有可达候选（先跑 g3_probe_a.py fmnv）")
    ex = existing()
    made = load_json(g1c.MADE_FILE, {})
    funnel = {"reachable": len(reachable), "skipped_fmnv_test": 0, "skipped_used": 0, "not_in_zip": 0, "duration_out": 0,
              "evid_channel": 0, "evid_phash": 0, "built": 0, "already_built": 0}
    rows = d2.fmnv_rows()
    with zipfile.ZipFile(g1c.FMNV_ZIP) as zf:
        split_of = g1c.fmnv_split_ids(zf)
        for vid in reachable:
            sid = f"a_{vid}".replace("-", "_")
            name = f"fmnv_{sid}"
            if (CASES / f"{name}.json").exists():
                funnel["already_built"] += 1
                continue
            if sum(1 for v in made.values() if v.get("family") == "A") >= n:
                break
            if split_of.get(vid) == "test" and not allow_test:
                funnel["skipped_fmnv_test"] += 1
                continue
            if vid in ex["evid"] or f"fmnv:{vid}" in ex["families"] or vid in ex["blob"] or name in ex["names"]:
                funnel["skipped_used"] += 1
                log(f"    {vid}：已用或在 EVID 里，跳过")
                continue
            original = g1c.fmnv_extract(zf, vid, f"fmnv_{vid}_orig")
            if original is None:
                funnel["not_in_zip"] += 1
                continue
            duration = probe_duration(original)
            if not duration or not (35.0 <= duration <= g1c.C_MAX_S):
                original.unlink(missing_ok=True)
                funnel["duration_out"] += 1
                log(f"    {vid}：原片 {duration} s 不在 35–{g1c.C_MAX_S:.0f} s，跳过")
                continue
            ok, channel = g1c.channel_check(f"fmnv:{sid}", g1c.youtube_url(vid))
            if not ok:
                original.unlink(missing_ok=True)
                funnel["evid_channel"] += 1
                continue
            length = round(min(20.0, max(6.0, 0.15 * duration)), 1)
            start = round(0.42 * duration, 1)
            end = round(start + length, 1)
            material = g1c.MEDIA / "fmnv" / f"fmnv_{vid}_cut.mp4"
            info = cut_segments(original, material, [(start, end)])
            original.unlink(missing_ok=True)          # 原片不交给模型（它在网上，walker 要自己找）；只留剪过的材料
            hits = phash_hits(material)
            if hits:
                material.unlink(missing_ok=True)
                funnel["evid_phash"] += 1
                log(f"    {vid}：材料与 EVID 感知哈希重复 {hits}，跳过")
                continue
            probe_row = by_id.get(vid) or {}
            ranks = probe_row.get("original_in_top10") or {}
            uploader = f"上传者 / 频道：{channel[0]}（{channel[1]}）" if channel and len(channel) > 1 else \
                (f"上传者 / 频道：{channel[0]}" if channel else "上传者 / 频道：取不到（yt-dlp 元数据）")
            url_id = vid.replace("[", "[[]").replace("?", "[?]")
            case = {
                "case_id": f"fmnv:{sid}", "dataset": "fmnv", "source_id": sid, "split": "dev_train", "task_kind": "provenance",
                "task_text": d2.PROVENANCE_ZH, "task_text_en": d2.PROVENANCE_EN, "family_key": f"fmnv:{vid}",
                "media": [{"ref": "I0", "path": g1c.rel_media(material), "media_type": "video/mp4"}], "claims": [],
                "source_policy": {"mode": "unrestricted", "excluded_keys": [], "date_cutoff": None},
                "requirements": [
                    {"id": "REQ1", "text": "给定视频的原始来源视频（地址、频道或标题），以及据以认定的画面对应证据", "must_cite": True,
                     "text_en": "The original source video of the given video (address, channel or title), and the picture correspondence that identifies it"},
                    {"id": "REQ2", "text": "给定视频与原片的差异：删掉 / 添加 / 调换 / 改变了哪些部分，两个视频中的时间", "must_cite": True,
                     "text_en": "How the given video differs from the original: which parts were removed, added, reordered or altered, with times in both videos"},
                    {"id": "REQ3", "text": "未能确定的内容（未比对的模态、边界精度等）", "must_cite": False,
                     "text_en": "What could not be determined (modalities not compared, boundary precision, etc.)"},
                ],
                "required_evidence": [
                    {"key": "original_video", "description": f"原片：YouTube {vid}（FMNV 真新闻，标题：{rows[vid].get('title', '')[:120]}）",
                     "url_pattern": f"youtube.com/watch[?]v={url_id}*", "must_cite": True, "supports": ["REQ1"], "acquired_by": None},
                    {"key": "comparison", "description": "材料与原片的 compare_media 比对结果（对应段、删掉的原片段）",
                     "acquired_by": "INSPECT/compare_media", "must_cite": True, "supports": ["REQ2"], "url_pattern": None},
                    {"key": "frames", "description": "剪点两侧的材料帧与原片帧", "acquired_by": "INSPECT/get_frames", "must_cite": True,
                     "supports": ["REQ2"], "url_pattern": None},
                ],
                "required_evidence_rule": "all:original_video; any_one_of:comparison,frames",
                "checks": [{"kind": "interval_iou", "params": {"start": start, "end": end, "min_iou": 0.5},
                            "note": f"原片 {start}–{end} s 被删（材料剪点在 {start} s）；区间由制作过程确定", "requirement": "REQ2"}],
                "reference_path": None, "used_for_revision": [],
                "notes": (f"g3 建卷（{now()}，scripts/g3/g3_cases.py，模板同 g1_cases.build_fmnv_a / walker_d2_cases.build_provenance）："
                          f"FMNV 真新闻 {vid}（{rows[vid].get('subject', '')}；FMNV 划分 {split_of.get(vid)}）。{uploader}。"
                          f"原片取自 FMNV.zip（与 YouTube 原片同一视频），去掉 {start}–{end} s（{length} s）后 PyAV 重编码"
                          f"（{info['video_codec']}，{info['size'][0]}×{info['size'][1]}）；材料时长 {info['duration_out']:.2f} s，"
                          f"剪点在材料 {start} s；原片时长 {duration:.2f} s。g3 探测（walker 同一条 search 层、Lens 关）原片排名："
                          f"{json.dumps(ranks, ensure_ascii=False)}。C6 建卷时自查：EVID 编号 0、频道 {'不在' if channel else '取不到，未查'}"
                          f" EVID 131–222 频道表、感知哈希 0。"),
            }
            d2.write_case(case)
            g1c.record_made(case, "A", "fmnv", {"duration_s": round(duration, 1), "cut": [start, end], "media": g1c.rel_media(material),
                                                "fmnv_split": split_of.get(vid), "probe_ranks": ranks, "channel": channel,
                                                "group": "P1_A_fmnv"})
            made = load_json(g1c.MADE_FILE, {})
            g1c.record_channel(case["case_id"], channel)
            funnel["built"] += 1
            log(f"  FMNV A：{vid}（{split_of.get(vid)}），原片 {duration:.0f} s，删 {start}–{end} s；频道 {channel}；探测 {ranks}")
    log(f"[FMNV A] {json.dumps(funnel, ensure_ascii=False)}")
    save_json(WORK / "build_a_fmnv_funnel.json", funnel)


def build_g360_a() -> None:
    """GroundLie360 temporal_edit 配对（负责人 10-18：只有这类能做 A 族）：材料 = 假帖子视频（数据集原文件，原地引用、不重编码）；
    GT URL = g3_confirm_g360.py 里两步都过的 YouTube 结果——本地原片 vs 结果 compare_media source_match，且帖子 vs 结果也
    source_match（walker 拿帖子去对，帖子画面得在结果里）。数据集只给剪辑点（temporal_edit_location 按镜头边界标 0/1），
    没有"原片哪段被删"的区间，所以不设 interval_iou，剪辑点写进 notes；答案层 GT 只有原片 URL。"""

    import csv
    import gzip

    from _d2_media import probe_duration

    confirm = load_json(WORK / "g360_confirm.json", {})
    checks = {(c["post"], c["yt"]): c for c in confirm.get("temporal_edit_checks") or []}
    # 入选：本地原片 vs 结果覆盖 ≥ 60%，且结果比帖子多出原片的画面（结果时长 > 帖子 + 5 s）——结果与帖子一样长、只含帖子那段时，
    # 分不清它是原片还是假帖子的转发（17399402 的结果就是帖子本身，只占原片 37%），不当 GT
    picks: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    for p in confirm.get("post_checks") or []:
        match_p = (p.get("post_vs_youtube") or {}).get("match") or {}
        match_o = ((checks.get((p["post"], p["yt"])) or {}).get("compare") or {}).get("match") or {}
        frac_o = match_o.get("fraction_of_a_matched") or 0
        longer = (match_p.get("b_duration_s") or 0) > (match_p.get("a_duration_s") or 0) + 5
        reason = ("帖子画面不在结果里" if not p.get("post_in_youtube") else
                  f"原片覆盖 {frac_o:.0%} < 60%" if frac_o < 0.6 else
                  f"结果（{match_p.get('b_duration_s')} s）不比帖子（{match_p.get('a_duration_s')} s）长：分不清原片与帖子转发" if not longer else None)
        if reason:
            rejected.append({"post": p["post"], "yt": p["yt"], "reason": reason})
            continue
        cur = picks.get(p["post"])
        if cur is None or frac_o > cur["_frac"]:
            picks[p["post"]] = {**p, "_frac": frac_o}
    save_json(WORK / "g360_a_selection.json", {"picked": {k: v["yt"] for k, v in picks.items()}, "rejected": rejected})
    for item in rejected:
        log(f"    groundlie360 {item['post']} vs {item['yt']}：不入选（{item['reason']}）")
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with gzip.open(g1c.G360_DIR / "data.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        rows = {str(r["video_id"]): r for r in csv.DictReader(stream)}
    index = load_json(g1c.G360_DIR / "videos_index.json", {})
    ex = existing()
    for post, pick in sorted(picks.items()):
        sid = f"a_{post}"
        name = f"groundlie360_{sid}"
        if (CASES / f"{name}.json").exists():
            continue
        if post in ex["evid"] or f"groundlie360:{post}" in ex["families"] or name in ex["names"]:
            log(f"    groundlie360 {post}：已用或在 EVID 里，跳过")
            continue
        r, yt = rows[post], pick["yt"]
        material = g1c.G360_DIR / index[post]
        duration = probe_duration(material)
        ok, channel = g1c.channel_check(f"groundlie360:{sid}", f"https://www.youtube.com/watch?v={yt}")
        if not ok:
            continue
        hits = phash_hits(material)
        if hits:
            log(f"    groundlie360 {post}：帖子视频与 EVID 感知哈希重复 {hits}，跳过")
            continue
        orig_check = checks.get((post, yt)) or {}
        line_o = ((orig_check.get("compare") or {}).get("summary_lines") or [""])[0]
        line_p = ((pick.get("post_vs_youtube") or {}).get("summary_lines") or [""])[0]
        uploader = f"{channel[0]}（{channel[1]}）" if channel and len(channel) > 1 else (channel[0] if channel else "取不到")
        rel = Path("..", "..", "..", "plan-harness-1", "datasets", "GroundLie360", index[post]).as_posix()
        case = {
            "case_id": f"groundlie360:{sid}", "dataset": "groundlie360", "source_id": sid, "split": "dev_train",
            "task_kind": "provenance", "task_text": d2.PROVENANCE_ZH, "task_text_en": d2.PROVENANCE_EN,
            "family_key": f"groundlie360:{post}",
            "media": [{"ref": "I0", "path": rel, "media_type": "video/mp4"}], "claims": [],
            "source_policy": {"mode": "unrestricted", "excluded_keys": [], "date_cutoff": None},
            "requirements": [
                {"id": "REQ1", "text": "给定视频的原始来源视频（地址、频道或标题），以及据以认定的画面对应证据", "must_cite": True,
                 "text_en": "The original source video of the given video (address, channel or title), and the picture correspondence that identifies it"},
                {"id": "REQ2", "text": "给定视频与原片的差异：删掉 / 添加 / 调换 / 改变了哪些部分，两个视频中的时间", "must_cite": True,
                 "text_en": "How the given video differs from the original: which parts were removed, added, reordered or altered, with times in both videos"},
                {"id": "REQ3", "text": "未能确定的内容（未比对的模态、边界精度等）", "must_cite": False,
                 "text_en": "What could not be determined (modalities not compared, boundary precision, etc.)"},
            ],
            "required_evidence": [
                {"key": "original_video", "description": f"原片：YouTube {yt}（与 GroundLie360 原片 {r.get('original_videoid')} 同一段素材，"
                                                          f"compare_media 确认）", "url_pattern": f"youtube.com/watch[?]v={yt}*",
                 "must_cite": True, "supports": ["REQ1"], "acquired_by": None},
                {"key": "comparison", "description": "材料与原片的 compare_media 比对结果（对应段、剪掉 / 改动的部分）",
                 "acquired_by": "INSPECT/compare_media", "must_cite": True, "supports": ["REQ2"], "url_pattern": None},
                {"key": "frames", "description": "剪辑点两侧的材料帧与原片帧", "acquired_by": "INSPECT/get_frames", "must_cite": True,
                 "supports": ["REQ2"], "url_pattern": None},
            ],
            "required_evidence_rule": "all:original_video; any_one_of:comparison,frames",
            "checks": [],
            "reference_path": None, "used_for_revision": [],
            "notes": (f"g3 建卷（{now()}，scripts/g3/g3_cases.py g360_a）：GroundLie360 假帖子 {post}（{r.get('platform')}，"
                      f"{r.get('first_level_annotation')}，{r.get('second_level_annotation')}；发布日期 {r.get('video_date')}；"
                      f"帖子标题：{' '.join((r.get('title') or '').split())[:200]}），数据集原片编号 {r.get('original_videoid')}。"
                      f"材料 = 帖子视频原文件（{duration and round(duration, 1)} s，不重编码）。YouTube 原片 {yt}，上传者 / 频道：{uploader}；"
                      f"g3 探测排名 {orig_check.get('rank')}。确认：本地原片 vs {yt}——{line_o}；帖子 vs {yt}——{line_p}。"
                      f"数据集剪辑点（temporal_edit_location，按镜头边界标 1）：{r.get('temporal_edit_location')}；镜头切分（帧号）："
                      f"{' '.join((r.get('video_scene_segmentation') or '').split())[:300]}。没有被删区间，不设 interval_iou。"
                      f"C6 建卷时自查：EVID 编号 0、频道{'不在' if channel else '取不到，未查'} EVID 131–222 频道表、感知哈希 0。"),
        }
        d2.write_case(case)
        g1c.record_made(case, "A", "groundlie360", {"post": post, "youtube": yt, "channel": channel, "group": "P1_A_groundlie360",
                                                    "media": rel})
        g1c.record_channel(case["case_id"], channel)
        log(f"  GroundLie360 A：{post} → YouTube {yt}；频道 {channel}")


FV_PROBE = WORK / "probe_a_fmnv_fv.json"
FV_DL = WORK / "fv_dl"


def fv_compare(fake: Path, original: Path) -> dict[str, Any]:
    """建卷时只用一次的 compare_media（画面引擎 SourceComparer，同 merged.compare_media 的 visual 部分）：A = 假视频，B = 原片。
    结果只作过滤与参考，不是 GT。"""

    from threading import Event

    from foren_searcher.inspect.source_compare import CompareConfig, SourceComparer

    comparer = SourceComparer(None, CompareConfig(time_budget_s=600.0))
    try:
        report, _ = comparer.compare(fake, original, Event())
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    match = report.get("match") or {}
    runs = report.get("correspondences") or []
    b_dur = float(match.get("b_duration_s") or 0) or None
    b_used = sum(max(0.0, float(r["b_range_s"][1]) - float(r["b_range_s"][0])) for r in runs if r.get("b_range_s"))
    lines = list(report.get("summary_lines") or [])
    structural = sum(len(report.get(k) or []) for k in ("segments_not_in_b", "b_cuts", "reorders", "repeats"))
    max_area = max((float(r.get("area") or 0) for r in report.get("changed_regions") or []), default=0.0)
    return {"source_match": bool(report.get("source_match")), "fake_in_original": match.get("fraction_of_a_matched"),
            "fake_s": match.get("a_duration_s"), "original_s": match.get("b_duration_s"),
            "original_not_in_fake": round(1 - b_used / b_dur, 3) if b_dur else None,
            # "完全没差异"（负责人 10-19）：既没有缺失 / 多出 / 调序 / 重复段，像素差异也在 same 档（变化区域面积 ≤ 画面 2%）；
            # 只有像素差异、没有缺失段的（画面内改动）要保留——那正是 fv 类与自剪删段案例不同的地方。原片两端没用到的部分不算缺失段
            "structural_diffs": structural, "max_region_area": max_area,
            "no_difference": structural == 0 and (max_area or 0.0) <= 0.02,
            "segments": [{k: r.get(k) for k in ("kind", "a_range_s", "b_range_s")} for r in runs[:8]],
            "segments_not_in_b": report.get("segments_not_in_b"), "b_cuts": report.get("b_cuts"),
            "changed_regions": [{k: r.get(k) for k in ("area", "peak_t") if k in r} for r in (report.get("changed_regions") or [])[:6]],
            "summary_lines": lines[:6]}


def build_fv_a(n: int) -> None:
    """FMNV CE 类（class fv）找原片（负责人 10-19）：材料 = 数据集作者删段做的假视频（zip 原文件，不重编码）；GT 只有原片 URL：
    答案层核两件事——原片 URL 必须引用；报告指出至少一处与原片的差异并引用比对记录（存在性，不做区间 IoU）。
    compare_media 只在建卷时用一次作过滤与参考：配准失败、差异 > 35%（假视频不在原片里的份额，或原片没用到的份额）、完全没差异的剔除；
    量到的差异段写进 notes，标"工具参考测量，非 GT，不参与核验"。案卷 schema 不收未知字段，manipulation_type = fv、interval 留空
    写进 notes 与 g3_cases_made.json。候选 = g3_probe_a.py fv 的可达清单（原片在前 10），文字 + 帧也能找到的排前。"""

    import shutil

    from _d2_media import probe_duration

    probe = load_json(FV_PROBE, {})
    rows = {r["video_id"]: r for r in probe.get("rows") or []}
    reach = [v for v in probe.get("reachable") or []]
    reach.sort(key=lambda v: (not (rows[v].get("original_in_top10") or {}).get("text_image"),
                              min(x for x in (rows[v].get("original_in_top10") or {}).values() if x)))
    ex = existing()
    data = d2.fmnv_rows()
    made = load_json(g1c.MADE_FILE, {})
    funnel = {"reachable": len(reach), "tried": 0, "duration_out": 0, "evid_channel": 0, "download_failed": 0, "compare_error": 0,
              "no_registration": 0, "difference_over_35": 0, "no_difference": 0, "evid_phash": 0, "built": 0}
    rejected: list[dict[str, Any]] = []
    with zipfile.ZipFile(g1c.FMNV_ZIP) as zf:
        split_of = g1c.fmnv_split_ids(zf)
        for vid in reach:
            if sum(1 for v in made.values() if v.get("group") == "P1_A_fmnv_fv") >= n:
                break
            sid = f"fv_{vid}".replace("-", "_")
            name = f"fmnv_{sid}"
            if (CASES / f"{name}.json").exists() or vid in ex["evid"] or f"fmnv:{vid}" in ex["families"] or vid in ex["blob"]:
                continue
            funnel["tried"] += 1
            fake = g1c.fmnv_extract(zf, vid, f"fmnv_fv_{vid}")
            if fake is None:
                continue
            final = g1c.MEDIA / "fmnv_fv" / fake.name
            final.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(fake), final)
            fake = final
            duration = probe_duration(fake)

            def drop(reason: str, key: str, extra: dict[str, Any] | None = None) -> None:
                fake.unlink(missing_ok=True)
                funnel[key] += 1
                rejected.append({"video_id": vid, "reason": reason, **(extra or {})})
                log(f"    fv {vid}：{reason}")

            if not duration or not (20.0 <= duration <= g1c.C_MAX_S):
                drop(f"假视频 {duration} s 不在 20–{g1c.C_MAX_S:.0f} s", "duration_out")
                continue
            url = f"https://www.youtube.com/watch?v={vid}"
            ok, channel = g1c.channel_check(f"fmnv:{sid}", url)
            if not ok:
                drop(f"频道 {channel} 在 EVID 131–222 频道表", "evid_channel")
                continue
            from g3_confirm_g360 import download  # 同一个 yt-dlp 下载（≤480p、≤20 min），只为比对，比完删

            g3_dl = sys.modules["g3_confirm_g360"]
            g3_dl.DL = FV_DL
            original, err = download(vid)
            if original is None:
                drop(f"原片下不来（{err}）", "download_failed")
                continue
            cmp = fv_compare(fake, original)
            shutil.rmtree(original.parent, ignore_errors=True)
            if cmp.get("error"):
                drop(f"比对出错 {cmp['error']}", "compare_error")
                continue
            not_in_orig = 1 - float(cmp.get("fake_in_original") or 0)
            not_in_fake = cmp.get("original_not_in_fake")
            if not cmp["source_match"]:
                drop("配准失败（compare_media 判不是同一来源）", "no_registration", {"compare": cmp})
                continue
            # "比不上"= 假视频有 > 35% 在原片里找不到。原片没用到的份额不算比不上：YouTube 原片常比 FMNV 的片段长得多（FMNV 先截取、再删段），
            # 10-19 第一遍按两侧都卡 35%，62 个候选里 50 个因"原片没用到 45–70%"被剔、0 个入选；只记进 notes 作参考
            if not_in_orig > 0.35:
                drop(f"比不上：假视频有 {not_in_orig:.0%} 在原片里找不到（> 35%）", "difference_over_35", {"compare": cmp})
                continue
            if cmp["no_difference"]:
                drop("完全没差异（无缺失 / 多出 / 调序 / 重复段，变化区域 ≤ 2%）", "no_difference", {"compare": cmp})
                continue
            hits = phash_hits(fake)
            if hits:
                drop(f"感知哈希与 EVID 重复 {hits}", "evid_phash")
                continue
            ranks = (rows.get(vid) or {}).get("original_in_top10") or {}
            uploader = f"{channel[0]}（{channel[1]}）" if channel and len(channel) > 1 else (channel[0] if channel else "取不到（yt-dlp 元数据）")
            measured = "；".join(cmp["summary_lines"][:4])
            case = {
                "case_id": f"fmnv:{sid}", "dataset": "fmnv", "source_id": sid, "split": "dev_train", "task_kind": "provenance",
                "task_text": d2.PROVENANCE_ZH, "task_text_en": d2.PROVENANCE_EN, "family_key": f"fmnv:{vid}",
                "media": [{"ref": "I0", "path": g1c.rel_media(fake), "media_type": "video/mp4"}], "claims": [],
                "source_policy": {"mode": "unrestricted", "excluded_keys": [], "date_cutoff": None},
                "requirements": [
                    {"id": "REQ1", "text": "给定视频的原始来源视频（地址、频道或标题），以及据以认定的画面对应证据", "must_cite": True,
                     "text_en": "The original source video of the given video (address, channel or title), and the picture correspondence that identifies it"},
                    {"id": "REQ2", "text": "给定视频与原片的差异：至少指出一处删掉 / 添加 / 调换 / 改变的部分，并引用比对记录", "must_cite": True,
                     "text_en": "How the given video differs from the original: name at least one part that was removed, added, reordered or altered, citing the comparison"},
                    {"id": "REQ3", "text": "未能确定的内容（未比对的模态、边界精度等）", "must_cite": False,
                     "text_en": "What could not be determined (modalities not compared, boundary precision, etc.)"},
                ],
                "required_evidence": [
                    {"key": "original_video", "description": f"原片：YouTube {vid}（FMNV CE 类假视频的来源，标题：{data[vid].get('title', '')[:120]}）",
                     "url_pattern": f"youtube.com/watch[?]v={vid.replace('[', '[[]').replace('?', '[?]')}*", "must_cite": True,
                     "supports": ["REQ1"], "acquired_by": None},
                    {"key": "comparison", "description": "材料与原片的 compare_media 比对结果（报告据以指出至少一处差异）",
                     "acquired_by": "INSPECT/compare_media", "must_cite": True, "supports": ["REQ2"], "url_pattern": None},
                ],
                "required_evidence_rule": "all",
                "checks": [{"kind": "evidence_key_required", "params": {"key": "original_video"}, "note": "原片 URL 必须引用", "requirement": "REQ1"},
                           {"kind": "evidence_key_required", "params": {"key": "comparison"},
                            "note": "报告指出至少一处与原片的差异并引用比对记录（存在性，不做区间 IoU）", "requirement": "REQ2"}],
                "reference_path": None, "used_for_revision": [],
                "notes": (f"g3 建卷（{now()}，scripts/g3/g3_cases.py fv_a）：manipulation_type = fv（FMNV CE 类，'关键片段被删'，数据集作者从 YouTube "
                          f"原片删段制作）；interval = 空（数据集没给被删区间，不设 interval_iou）。FMNV {vid}（{data[vid].get('subject', '')}；"
                          f"FMNV 划分 {split_of.get(vid)}）。材料 = 假视频原文件（zip，{duration:.1f} s，不重编码）。上传者 / 频道：{uploader}。"
                          f"g3 探测（walker 同一条 search 层、Lens 关）原片排名：{json.dumps(ranks, ensure_ascii=False)}。"
                          f"建卷时 compare_media 一次（A = 假视频、B = 原片），只作过滤与参考：假视频在原片中的覆盖率 "
                          f"{cmp.get('fake_in_original'):.0%}（工具参考，非 GT）；原片未用的份额 {not_in_fake:.0%}（工具参考，非 GT；"
                          f"含 FMNV 截取掉的两端）；差异摘要（工具参考，非 GT，不参与核验）：{measured}。"
                          f"C6 建卷时自查：EVID 编号 0、频道{'不在' if channel else '取不到，未查'} EVID 131–222 频道表、感知哈希 0。"),
            }
            d2.write_case(case)
            g1c.record_made(case, "A", "fmnv", {"group": "P1_A_fmnv_fv", "manipulation_type": "fv", "interval": None,
                                                "media": g1c.rel_media(fake), "fmnv_split": split_of.get(vid), "probe_ranks": ranks,
                                                "channel": channel, "reference_compare": cmp})
            made = load_json(g1c.MADE_FILE, {})
            g1c.record_channel(case["case_id"], channel)
            funnel["built"] += 1
            log(f"  FMNV fv：{vid}（{split_of.get(vid)}），{duration:.0f} s；频道 {channel}；探测 {ranks}；"
                f"假视频在原片 {cmp.get('fake_in_original')}、原片没用到 {not_in_fake}")
    log(f"[FMNV fv] {json.dumps(funnel, ensure_ascii=False)}")
    save_json(WORK / "build_fv_funnel.json", {"funnel": funnel, "rejected": rejected})


P4_PICKS = WORK / "p4_picks.json"
AVERITEC_TRAIN = MAIN / "experiments" / "sft_data" / "data" / "annotations" / "averitec_train.json"


def build_averitec_c(index: int, row: dict[str, Any], groups: list[str], pages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """AVeriTeC 训练集文字声明的 C 族案卷。r1 起的 C 族训练案卷里没有 AVeriTeC 模板（主目录 averitec_* 是 9 月手写的 dev_select 案卷），
    所以照最接近的现行模板 walker_d2_cases.build_claim_text（GroundLie360 文字声明：CLAIM_TEXT + 声明日期、必需证据 external、label_expected），
    来源策略照旧 AVeriTeC 案卷排除原核查文章（exclude_original，键 = check_unanswerable.norm(fact_checking_article)，同探测）。
    标签：Supported → true、Refuted → false（别名同 build_claim_text）；Not Enough Evidence / Conflicting 不设 label_expected，标签写 notes。"""

    from check_unanswerable import norm

    claim = " ".join(str(row.get("claim") or "").split())
    if not claim:
        return None
    d, m, y = (str(row.get("claim_date") or "").split("-") + ["", "", ""])[:3]
    date = f"{y}-{int(m):02d}-{int(d):02d}" if y and m.isdigit() and d.isdigit() else "unknown"
    label = row.get("label")
    checks = []
    if label == "Supported":
        checks = [{"kind": "label_expected", "params": {"label": "true", "aliases": d2.TRUE_FORMS + ["accurate"]},
                   "note": "AVeriTeC label = Supported"}]
    elif label == "Refuted":
        checks = [{"kind": "label_expected", "params": {"label": "false", "aliases": d2.FAKE_FORMS + ["misleading"]},
                   "note": "AVeriTeC label = Refuted"}]
    article = row.get("fact_checking_article") or ""
    policy = ({"mode": "exclude_original", "excluded_keys": [norm(article)], "date_cutoff": None} if article
              else {"mode": "unrestricted", "excluded_keys": [], "date_cutoff": None})
    why = {"P4_C8": "检索前 5 条里有正文带视频链接的页（C8 打开页内视频）", "P4_read_more": "GT 来源页是长页、检索前 5 条有长页（续读）"}
    return {
        "case_id": f"averitec:c_{index}", "dataset": "averitec", "source_id": f"c_{index}", "split": "dev_train",
        "task_kind": "claim_check", "task_text": d2.CLAIM_TEXT_ZH.format(claim=claim, date=date),
        "task_text_en": d2.CLAIM_TEXT_EN.format(claim=claim, date=date), "family_key": f"averitec:{index}",
        "media": [], "claims": [claim], "source_policy": policy,
        "requirements": [
            {"id": "REQ1", "text": "外部可靠来源对声明所述事件的记载（谁、何时、何地、发生了什么）", "must_cite": True,
             "text_en": "What reliable external sources record about the event the claim describes (who, when, where, what happened)"},
            {"id": "REQ2", "text": "未能确定的内容与结论的适用范围", "must_cite": False,
             "text_en": "What could not be determined and the scope of the conclusion"},
        ],
        "required_evidence": [
            {"key": "external", "description": "检索打开的外部页面（报道该事件的来源）", "acquired_by": "SEARCH/open",
             "must_cite": True, "supports": ["REQ1"], "url_pattern": None},
        ],
        "required_evidence_rule": "all",
        "checks": checks,
        "reference_path": None, "used_for_revision": [],
        "notes": (f"g3 建卷（{now()}，scripts/g3/g3_cases.py p4，模板同 walker_d2_cases.build_claim_text）：AVeriTeC 训练集 #{index}，"
                  f"label = {label}{'（不设 label_expected）' if not checks else ''}，声明日期 {date}；排除原核查文章 {article}。"
                  f"g3 入选理由：{'；'.join(why[g] for g in groups)}；探测命中页：{json.dumps(pages, ensure_ascii=False)[:600]}。"),
    }


def build_p4() -> None:
    """P4（C 族续读 / 页内视频）：p4_picks.json 的 22 个挑选（averitec 1253 两类都在 → 21 个案卷）。
    FMNV-C / GroundLie360 用 g1 的现行模板（g1_cases.build_fmnv_c / build_g360 --ids，滤规则照旧，不过的照实记下）；AVeriTeC 见 build_averitec_c。"""

    picks = load_json(P4_PICKS, {})
    groups: dict[str, list[str]] = {}
    pages: dict[str, list[dict[str, Any]]] = {}
    for group in ("P4_C8", "P4_read_more"):
        for item in picks.get(group) or []:
            cid = item["case_id"]
            groups.setdefault(cid, []).append(group)
            pages.setdefault(cid, []).extend(item.get("pages") or item.get("top_long_pages") or [])
    by_pool: dict[str, list[str]] = {}
    for cid in groups:
        pool, _, sid = cid.partition(":")
        by_pool.setdefault(pool, []).append(sid.removeprefix("probe_"))
    before = set(load_json(g1c.MADE_FILE, {}))
    if by_pool.get("fmnv_c"):
        g1c.build_fmnv_c(0, 0, set(by_pool["fmnv_c"]))
    if by_pool.get("groundlie360"):
        g1c.build_g360(0, 0, set(by_pool["groundlie360"]))
    ex = existing()
    rows = json.loads(AVERITEC_TRAIN.read_text(encoding="utf-8"))
    for sid in by_pool.get("averitec") or []:
        cid = f"averitec:probe_{sid}"
        name = f"averitec_c_{sid}"
        if (CASES / f"{name}.json").exists():
            continue
        if f"averitec:{sid}" in ex["families"]:
            log(f"    averitec {sid}：family 已用，跳过")
            continue
        case = build_averitec_c(int(sid), rows[int(sid)], groups[cid], pages[cid])
        if case:
            d2.write_case(case)
            g1c.record_made(case, "C", "averitec", {"label": rows[int(sid)].get("label")})
            log(f"  AVeriTeC C：{sid}（{rows[int(sid)].get('label')}）{case['claims'][0][:80]}")
    # 给新建的标 P4 组（g1 模板记的是 family / dataset）；没建成的照实记
    made = load_json(g1c.MADE_FILE, {})
    want, name_cid = {}, {}
    for cid, gs in groups.items():
        pool, _, sid = cid.partition(":")
        sid = sid.removeprefix("probe_")
        name = {"fmnv_c": f"fmnv_c_{sid}".replace("-", "_"), "groundlie360": f"groundlie360_v{sid}", "averitec": f"averitec_c_{sid}"}[pool]
        want[name] = gs
        name_cid[name] = cid
    missing = []
    for name, gs in want.items():
        if name in made and (CASES / f"{name}.json").exists():
            made[name]["group"] = gs[0]
            made[name]["also_groups"] = gs[1:]
            made[name]["p4_probe_pages"] = pages[name_cid[name]][:4]
        else:
            missing.append(name)
    save_json(g1c.MADE_FILE, made)
    new = sorted(set(made) - before)
    log(f"[P4] 新建 {len(new)}：{new}；没建成 {missing}")
    save_json(WORK / "build_p4_result.json", {"built": new, "not_built": missing, "groups": want})


def write_list() -> None:
    made = load_json(g1c.MADE_FILE, {})
    groups: dict[str, list[str]] = {}
    for name, info in made.items():
        if (CASES / f"{name}.json").exists():
            for g in [info.get("group") or f"{info.get('family')}_{info.get('dataset')}", *(info.get("also_groups") or [])]:
                groups.setdefault(g, []).append(name)
    # P3：r0 时建好、从没跑过的 QVH 三条（媒体在主目录 data/qvhighlights/0–2，本工作树用 junction 只读引用）
    groups.setdefault("P3_qvh", [])
    groups["P3_qvh"] = [n for n in ("qvhighlights_0", "qvhighlights_1", "qvhighlights_2") if (CASES / f"{n}.json").exists()] + \
        [n for n in groups["P3_qvh"] if n not in ("qvhighlights_0", "qvhighlights_1", "qvhighlights_2")]
    # P2：V9 离线模拟（g3_v9_sim.py）在交报告前确实触发 Q1 的底案，每个一次 V9 run；名字 <底案>_v9（g3_run 按它加 --override）
    sim = load_json(WORK / "v9_sim.json", {})
    groups["P2_V9"] = [f"{r['base']}_v9" for r in sim.get("results") or [] if r.get("verdict") == "trigger"]
    payload = {"round": "r1g3", "batch": "g3", "made_by": "scripts/g3/g3_cases.py", "written_at": now(),
               "total": sum(len(v) for v in groups.values()), "groups": groups}
    save_json(LIST_FILE, payload)
    log(f"[清单] {LIST_FILE.name}：{ {k: len(v) for k, v in groups.items()} }")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fmnv_a")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--allow-fmnv-test", action="store_true")
    sub.add_parser("g360_a")
    p = sub.add_parser("fv_a")
    p.add_argument("--n", type=int, default=10)
    sub.add_parser("p4")
    sub.add_parser("list")
    args = parser.parse_args()
    if args.cmd == "fmnv_a":
        build_fmnv_a(args.n, args.allow_fmnv_test)
    elif args.cmd == "g360_a":
        build_g360_a()
    elif args.cmd == "p4":
        build_p4()
    elif args.cmd == "fv_a":
        build_fv_a(args.n)
    elif args.cmd == "list":
        write_list()
    return 0


if __name__ == "__main__":
    sys.exit(main())
