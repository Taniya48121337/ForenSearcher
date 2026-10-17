"""g1（10-18）建卷：C 族（FMNV 60 / GroundLie360 40）、B 族（VideoSham + VCSL）、A 族（FMNV 找原片，先 5 条冒烟）。

    python scripts\\g1\\g1_cases.py fmnv_c --n-fake 30 --n-real 30
    python scripts\\g1\\g1_cases.py g360 --n-fake 20 --n-real 20
    python scripts\\g1\\g1_cases.py videosham --n 14
    python scripts\\g1\\g1_cases.py vcsl --n 20
    python scripts\\g1\\g1_cases.py fmnv_a --n 5
    python scripts\\g1\\g1_cases.py list            # 写 experiments/sft_data/walker_g1_cases.json（按族 / 数据集分组）

任务模板、需求、证据、label_expected 的写法照 scripts/walker_d2_cases.py（r0g / r1 案卷同一套）；不同处：
- 媒体：FMNV 从 D:\\BaiduNetdiskDownload\\FMNV.zip 只解压选中的那一个视频到 cases/media/g1/fmnv/（时长不合格立刻删）；
  GroundLie360 原地引用 plan-harness-1/datasets/GroundLie360/videos（已解压，不复制）；VideoSham 复制两段到
  cases/media/g1/videosham_<id>/；VCSL 只取视频轨下载到 cases/media/g1/vcsl_<pair>/。
- 去重：主目录 experiments/sft_data/cases（r0g / r1 / 更早的全部案卷，含未入库的）与本工作区的案卷 family / 编号都跳过；
  EVID（test，222 条）出现过的视频编号跳过；有频道的（YouTube / Vimeo）频道在 EVID 131–222 频道表里的跳过。
- GT 只进案卷的 checks / notes / reference.json（GT 区），task_text 只放声明原文与发布日期（学生可见）。
- 声明改写规则（建卷侧，冒烟自审后定，见报告）：在 claim_ok() 里。
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import walker_d2_cases as d2  # noqa: E402

SFT = ROOT / "experiments" / "sft_data"
CASES = SFT / "cases"
MEDIA = CASES / "media" / "g1"
MAIN_CASES = ROOT.parent / "experiments" / "sft_data" / "cases"          # 主目录（对话 3），只读
FMNV_ZIP = Path(r"D:\BaiduNetdiskDownload\FMNV.zip")
G360_DIR = ROOT / "plan-harness-1" / "datasets" / "GroundLie360"
LOG = ROOT / "claude" / "_g1_work" / "g1_cases.log"
CHANNELS_FILE = SFT / "g1_channels.json"
MADE_FILE = SFT / "g1_cases_made.json"
C_MAX_S = 180.0                       # 任务书：视频 ≤ 3 min
# 10-18 冒烟后逐条读 100 条声明，规则过不掉的非声明 / 重复（人工）：标题不是可核查的陈述、纯评论、非英文、同一事件的第二条帖子
DROP = {"fmnv_c_jGMEkJJj3iE", "fmnv_c_1865574071938752701", "fmnv_c_1894586189341970508", "fmnv_c_1896319604974289225",
        "fmnv_c_JFUwpAIhTzE", "groundlie360_v26269501", "groundlie360_v26113802", "groundlie360_v28240202",
        "groundlie360_v30280802", "groundlie360_v46433207", "groundlie360_v23648404", "groundlie360_v28381300",
        "groundlie360_v30317806", "groundlie360_v36906700", "groundlie360_v33040000", "groundlie360_v24670100",
        "groundlie360_v31017805", "groundlie360_v30448804", "groundlie360_v27585901"}
EN_WORDS = {"the", "of", "in", "on", "to", "is", "are", "was", "were", "for", "with", "by", "at", "and", "from", "after", "as",
            "has", "have", "his", "her", "their", "its", "into", "over", "during", "about", "be", "been", "will", "that", "this"}
OPINION_RE = re.compile(r"\b(?:so ugly|terrible|lovely|how dare|i think|my thoughts|shares (?:his|her|their) thoughts|in my opinion|"
                        r"lol|omg|wow)\b", re.I)
QUESTION_RE = re.compile(r"\?\s*$|^(?:is|are|was|were|do|does|did|can|could|will|would|should|has|have|who|what|why|how|when|where)\b", re.I)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(text: str) -> None:
    print(text, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as h:
        h.write(f"[{now()}] {text}\n")


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def record_made(case: dict[str, Any], family: str, dataset: str, extra: dict[str, Any] | None = None) -> None:
    made = load_json(MADE_FILE, {})
    made[case["case_id"].replace(":", "_")] = {"family": family, "dataset": dataset, "built_at": now(), **(extra or {})}
    save_json(MADE_FILE, made)


def record_channel(case_id: str, channel: list[str] | None) -> None:
    if not channel:
        return
    table = load_json(CHANNELS_FILE, {})
    table[case_id] = channel
    save_json(CHANNELS_FILE, table)


def existing() -> dict[str, set[str]]:
    """主目录 + 本工作区所有案卷的 family_key、文件名、媒体路径里出现的编号；EVID 全文。"""

    fams, names, blob = set(), set(), []
    for folder in (MAIN_CASES, CASES):
        for path in folder.glob("*.json"):
            names.add(path.stem)
            case = load_json(path, {})
            fams.add(str(case.get("family_key")))
            blob.append(json.dumps(case.get("media") or [], ensure_ascii=False))
            blob.append(str(case.get("notes") or ""))
    return {"families": fams, "names": names, "blob": "\n".join(blob), "evid": d2._evid_text()}


def claim_ok(text: str) -> str | None:
    """声明文字合格就返回规范化后的声明，否则 None。规则（建卷侧）：去多余空白；20–220 字符；不是问句；
    以 ASCII 为主（英文任务）；不含网址。"""

    claim = re.sub("�+", "'", str(text or ""))               # FMNV data.json 里弯引号被存成 U+FFFD（"India��s"）
    claim = "".join(ch for ch in claim if ord(ch) < 0x2190 or 0x3000 <= ord(ch) < 0xFE00)   # 去 emoji / 符号
    claim = " ".join(claim.split())
    claim = re.split(r"\s*\|\s*", claim)[0].strip()   # 末尾的 "| ABC News" / "| US Election 2024" 是频道 / 栏目名，不是声明
    if "..." in claim or "…" in claim:            # 帖子标题被截断
        return None
    if not (20 <= len(claim) <= 220) or QUESTION_RE.search(claim) or "http" in claim.lower():
        return None
    if not claim or sum(ch.isascii() for ch in claim) / len(claim) < 0.95:
        return None
    if "!" in claim or "#" in claim or len(claim.split()) < 6 or OPINION_RE.search(claim):
        return None
    return claim


def hash_order(ids: list[str], seed: str) -> list[str]:
    return sorted(ids, key=lambda x: hashlib.sha1((seed + x).encode("utf-8")).hexdigest())


# ---------------------------------------------------------------------------- FMNV（zip 里取视频）

def fmnv_split_ids(zf: zipfile.ZipFile) -> dict[str, str]:
    out = {}
    for split in ("train", "val", "test"):
        for line in zf.read(f"FMNV/data-split/vid_{split}.txt").decode("utf-8").split():
            out[line.strip()] = split
    return out


def fmnv_extract(zf: zipfile.ZipFile, video_id: str, stem: str) -> Path | None:
    name = f"FMNV/videos/{video_id}.mp4"
    try:
        info = zf.getinfo(name)
    except KeyError:
        return None
    target = MEDIA / "fmnv" / f"{stem}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or target.stat().st_size != info.file_size:
        with zf.open(info) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
    return target


def rel_media(path: Path) -> str:
    return path.relative_to(CASES).as_posix()


def fmnv_candidates(kind: str, ex: dict[str, set[str]], split_of: dict[str, str]) -> list[tuple[str, dict[str, Any], str]]:
    rows = d2.fmnv_rows()
    out = []
    for vid, row in rows.items():
        if kind == "fake" and row.get("class") != "ft":
            continue
        if kind == "real" and row.get("label") != "true":
            continue
        if split_of.get(vid) == "test" or vid in ex["evid"] or f"fmnv:{vid}" in ex["families"] or vid in ex["blob"]:
            continue
        claim = claim_ok(row.get("title"))
        if claim:
            out.append((vid, row, claim))
    order = {v: i for i, (v, _, _) in enumerate(out)}
    ranked = [out[order[v]] for v in hash_order(list(order), f"g1-fmnv-{kind}")]
    # YouTube 编号（新闻机构频道，标题是新闻标题）排前面，Twitter 编号（帖子文字，常是评论）排后面
    return [x for x in ranked if not x[0].isdigit()] + [x for x in ranked if x[0].isdigit()]


def channel_check(case_id: str, url: str | None) -> tuple[bool, list[str] | None]:
    if not url:
        return True, None
    channel = d2.channel_of(url)
    if d2.evid_channel_hit(channel):
        log(f"    {case_id}：频道 {channel} 在 EVID 131–222 频道表里，换下一个")
        return False, channel
    return True, channel


def youtube_url(vid: str) -> str | None:
    return None if vid.isdigit() else f"https://www.youtube.com/watch?v={vid}"


def build_fmnv_c(n_fake: int, n_real: int, ids: set[str] | None = None) -> None:
    from _d2_media import probe_duration

    ex = existing()
    with zipfile.ZipFile(FMNV_ZIP) as zf:
        split_of = fmnv_split_ids(zf)
        for kind, n in (("fake", n_fake), ("real", n_real)):
            got = sum(1 for v in load_json(MADE_FILE, {}).values() if v.get("dataset") == "fmnv" and v.get("kind") == kind)
            for vid, row, claim in fmnv_candidates(kind, ex, split_of):
                if ids is not None and vid not in ids:
                    continue
                if got >= n and ids is None:
                    break
                sid = f"c_{vid}".replace("-", "_")
                if (CASES / f"fmnv_{sid}.json").exists() or f"fmnv_{sid}" in DROP:
                    continue
                path = fmnv_extract(zf, vid, f"fmnv_{vid}")
                if path is None:
                    continue
                duration = probe_duration(path)
                if not duration or not (15.0 <= duration <= C_MAX_S):
                    path.unlink(missing_ok=True)
                    log(f"  FMNV {vid}：时长 {duration} 不在 15–{C_MAX_S:.0f} s，删掉、换下一个")
                    continue
                ok, channel = channel_check(f"fmnv:{sid}", youtube_url(vid))
                if not ok:
                    path.unlink(missing_ok=True)
                    continue
                fake = kind == "fake"
                label = ({"label": "misleading", "aliases": d2.FAKE_FORMS} if fake else {"label": "accurate", "aliases": d2.TRUE_FORMS})
                ctx = {"event": "事件", "object": "对象", "action": "动作"}.get(str(row.get("contextual_class") or ""), "")
                note = (f"FMNV CD 类（class=ft，标题与画面不符，替换的是{ctx}）：视频是 FMNV {vid} 原样，标题是数据集生成的假标题。"
                        if fake else f"FMNV 真新闻：视频是 FMNV {vid} 原样，标题是它自己的标题。")
                case = {
                    "case_id": f"fmnv:{sid}", "dataset": "fmnv", "source_id": sid, "split": "dev_train", "task_kind": "claim_check",
                    "task_text": d2.CLAIM_VIDEO_ZH.format(claim=claim), "task_text_en": d2.CLAIM_VIDEO_EN.format(claim=claim),
                    "family_key": f"fmnv:{vid}",
                    "media": [{"ref": "I0", "path": rel_media(path), "media_type": "video/mp4"}], "claims": [claim],
                    "source_policy": {"mode": "unrestricted", "excluded_keys": [], "date_cutoff": None},
                    "requirements": c_video_requirements(),
                    "required_evidence": c_video_evidence(),
                    "required_evidence_rule": "any_one_of:frames,transcript; all:external",
                    "checks": [{"kind": "label_expected", "params": label,
                                "note": f"FMNV label={row.get('label')}，class={row.get('class') or 'real'}；"
                                        f"{'答案应说标题与视频不符 / 不实' if fake else '答案应说标题属实'}"}],
                    "reference_path": None, "used_for_revision": [],
                    "notes": f"g1 建卷（{now()}，scripts/g1/g1_cases.py）：{note}FMNV subject：{row.get('subject', '')}；"
                             f"FMNV 划分 {split_of.get(vid, '?')}；视频 {duration:.1f} s（FMNV.zip 原文件，未重编码）。",
                }
                d2.write_case(case)
                record_made(case, "C", "fmnv", {"kind": kind, "duration_s": round(duration, 1), "media": rel_media(path)})
                record_channel(case["case_id"], channel)
                got += 1
                log(f"  FMNV C {kind} {got}/{n}：{vid}，{duration:.0f} s，{claim[:80]}")
            log(f"[FMNV C {kind}] {got}/{n}")


def c_video_requirements() -> list[dict[str, Any]]:
    return [
        {"id": "REQ1", "text": "给定视频实际显示和说出的内容（画面、字幕、语音）与声明是否相符", "must_cite": True,
         "text_en": "What the given video actually shows and says (pictures, on-screen text, speech) and whether it matches the claim"},
        {"id": "REQ2", "text": "外部来源对声明所说事件的记载", "must_cite": True,
         "text_en": "What external sources record about the event the claim describes"},
        {"id": "REQ3", "text": "未能确定的内容与结论的适用范围", "must_cite": False,
         "text_en": "What could not be determined and the scope of the conclusion"},
    ]


def c_video_evidence() -> list[dict[str, Any]]:
    return [
        {"key": "frames", "description": "视频画面（抽帧）", "acquired_by": "INSPECT/get_frames", "must_cite": True,
         "supports": ["REQ1"], "url_pattern": None},
        {"key": "transcript", "description": "视频语音转写", "acquired_by": "INSPECT/asr", "must_cite": True,
         "supports": ["REQ1"], "url_pattern": None},
        {"key": "external", "description": "检索打开的外部页面（报道该事件的来源）", "acquired_by": "SEARCH/open",
         "must_cite": True, "supports": ["REQ2"], "url_pattern": None},
    ]


# ---------------------------------------------------------------------------- GroundLie360（已解压的视频，原地引用）

def g360_date(row: dict[str, Any]) -> str:
    raw = str(row.get("video_date") or "").split(".")[0]
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}" if len(raw) == 8 else "unknown"


def build_g360(n_fake: int, n_real: int, ids: set[str] | None = None) -> None:
    from _d2_media import probe_duration

    ex = existing()
    rows = d2.g360_rows()
    index = load_json(G360_DIR / "videos_index.json", {})
    for kind, n in (("fake", n_fake), ("real", n_real)):
        got = sum(1 for v in load_json(MADE_FILE, {}).values() if v.get("dataset") == "groundlie360" and v.get("kind") == kind)
        pool = []
        for pid, row in rows.items():
            if pid not in index or str(row.get("platform")) not in ("youtube", "twitter", "x", "facebook", "tiktok", "instagram"):
                continue
            first = row.get("first_level_annotation")
            if kind == "fake" and not (first == "Fake" and "false_title" in str(row.get("second_level_annotation"))):
                continue
            if kind == "real" and first != "Real":
                continue
            if f"groundlie360:{pid}" in ex["families"] or pid in ex["evid"]:
                continue
            claim = claim_ok(row.get("title"))
            if claim:
                pool.append((pid, row, claim))
        order = {p: i for i, (p, _, _) in enumerate(pool)}
        for pid in hash_order(list(order), f"g1-g360-{kind}"):
            if ids is not None and pid not in ids:
                continue
            if got >= n and ids is None:
                break
            pid, row, claim = pool[order[pid]]
            sid = f"v{pid}"
            if (CASES / f"groundlie360_{sid}.json").exists() or f"groundlie360_{sid}" in DROP:
                continue
            video = G360_DIR / index[pid]
            duration = probe_duration(video) if video.is_file() else None
            if not duration or not (8.0 <= duration <= C_MAX_S):
                continue
            fake = kind == "fake"
            date = g360_date(row)
            label = ({"label": "false", "aliases": d2.FAKE_FORMS + ["misleading"]} if fake
                     else {"label": "true", "aliases": d2.TRUE_FORMS + ["accurate"]})
            media_path = Path("..", "..", "..", "plan-harness-1", "datasets", "GroundLie360", *Path(index[pid]).parts).as_posix()
            case = {
                "case_id": f"groundlie360:{sid}", "dataset": "groundlie360", "source_id": sid, "split": "dev_train",
                "task_kind": "claim_check", "task_text": d2.CLAIM_VIDEO_ZH.format(claim=claim) + f"（发布日期：{date}）",
                "task_text_en": d2.CLAIM_VIDEO_EN.format(claim=claim) + f" (posted: {date})", "family_key": f"groundlie360:{pid}",
                "media": [{"ref": "I0", "path": media_path, "media_type": "video/mp4"}], "claims": [claim],
                "source_policy": {"mode": "unrestricted", "excluded_keys": [], "date_cutoff": None},
                "requirements": c_video_requirements(), "required_evidence": c_video_evidence(),
                "required_evidence_rule": "any_one_of:frames,transcript; all:external",
                "checks": [{"kind": "label_expected", "params": label,
                            "note": f"GroundLie360 first_level={row.get('first_level_annotation')}，"
                                    f"second_level={row.get('second_level_annotation')}；"
                                    f"标题里的不实片段：{str(row.get('title_third_level_annotation'))[:200]}"}],
                "reference_path": None, "used_for_revision": [],
                "notes": f"g1 建卷（{now()}，scripts/g1/g1_cases.py）：GroundLie360 帖子 {pid}（{row.get('platform')}，{date}，"
                         f"{duration:.0f} s），视频原地引用已解压的 GroundLie360 videos/（Snopes 核查；核查站点被检索护栏屏蔽）。",
            }
            d2.write_case(case)
            record_made(case, "C", "groundlie360", {"kind": kind, "duration_s": round(duration, 1), "platform": row.get("platform")})
            got += 1
            log(f"  G360 C {kind} {got}/{n}：{pid}（{row.get('platform')}），{duration:.0f} s，{claim[:80]}")
        log(f"[G360 C {kind}] {got}/{n}")


# ---------------------------------------------------------------------------- B 族：VideoSham（本机数据集）

def build_videosham(n: int) -> None:
    ex = existing()
    rows = {Path(r["Video-Name"]).stem: r for r in
            csv.DictReader((d2.VIDEOSHAM / "VideoSourceandEdits.csv").open(encoding="utf-8-sig", newline=""))}
    used_groups = {rows[s]["Name/Id in source dataset"].strip() for s in
                   (n.split("_", 1)[1] for n in ex["names"] if n.startswith("videosham_")) if s in rows}
    got = sum(1 for v in load_json(MADE_FILE, {}).values() if v.get("dataset") == "videosham")
    stems = [p.name.split("_")[0] for p in (d2.VIDEOSHAM / "media" / "edited").glob("*_processed.mp4")]
    for sid in hash_order([s for s in stems if s in rows], "g1-videosham"):
        if got >= n:
            break
        row = rows[sid]
        group = row["Name/Id in source dataset"].strip()
        if f"videosham_{sid}" in ex["names"] or group in used_groups or group in ex["evid"]:
            continue
        if not (d2.VIDEOSHAM / "media" / "source" / f"{sid}_unedited.mp4").is_file():
            continue
        ok, channel = channel_check(f"videosham:{sid}", group if group.startswith("http") else None)
        if not ok:
            continue
        folder = MEDIA / f"videosham_{sid}"
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(d2.VIDEOSHAM / "media" / "edited" / f"{sid}_processed.mp4", folder / "A.mp4")
        shutil.copyfile(d2.VIDEOSHAM / "media" / "source" / f"{sid}_unedited.mp4", folder / "B.mp4")
        record = {k: v for k, v in row.items()}
        save_json(folder / "reference.json", {"dataset": "VideoSham", "source_record": record,
                                              "input_assignment": {"A.mp4": "processed", "B.mp4": "unedited"},
                                              "reference_status": "official_edit_description_requires_media_confirmation",
                                              "split": "development", "condition": "provided_candidate_pair"})
        case = d2.build_videosham({"id": sid, **row})
        base = f"media/g1/videosham_{sid}"
        case["media"] = [{"ref": "I0", "path": f"{base}/A.mp4", "media_type": "video/mp4"},
                         {"ref": "I1", "path": f"{base}/B.mp4", "media_type": "video/mp4"}]
        case["reference_path"] = f"{base}/reference.json"
        case["notes"] = case["notes"].replace("r0g 扩案例", "g1 建卷").replace("walker_d2_cases.py 程序建卷", "scripts/g1/g1_cases.py，模板同 walker_d2_cases.build_videosham")
        d2.write_case(case)
        used_groups.add(group)
        record_made(case, "B", "videosham", {"manipulation": row.get("Manipulation")})
        record_channel(case["case_id"], channel)
        got += 1
        log(f"  VideoSham {got}/{n}：{sid}（{row.get('Manipulation')}）")
    # build_videosham 会在 SFT/data/cases 下复制一份 inputs：删掉，只留 cases/media/g1 那份
    for sid in stems:
        extra = SFT / "data" / "cases" / f"videosham_{sid}"
        if extra.is_dir() and (CASES / f"videosham_{sid}.json").is_file():
            shutil.rmtree(extra)
    log(f"[VideoSham] {got}/{n}")


# ---------------------------------------------------------------------------- B 族：VCSL（训练集对，YouTube 只取视频轨）

def build_vcsl(n: int) -> None:
    from foren_searcher.inspect.acquisition import ytdlp_download
    from foren_searcher.inspect.policy import InspectPolicy

    ex = existing()
    data = d2.VCSL_DATA
    urls = {r["uuid"]: r["url"] for r in csv.DictReader((data / "videos_url_uuid.csv").open(encoding="utf-8", newline=""))}
    frames = {r["uuid"]: int(r["frame_count"]) for r in csv.DictReader((data / "frames_all.csv").open(encoding="utf-8", newline=""))}
    labels = json.loads((data / "label_file_uuid_total.json").read_text(encoding="utf-8"))
    used_q = set()
    for ref in list((ROOT.parent / "experiments" / "sft_data" / "data" / "vcsl").glob("pair_*/reference.json")) + \
            list(MEDIA.glob("vcsl_*/reference.json")):
        used_q.add(load_json(ref, {}).get("query_id"))
    pairs = []
    for row in csv.DictReader((data / "pair_file_train.csv").open(encoding="utf-8", newline="")):
        q, r = row["query_id"], row["reference_id"]
        segs = labels.get(f"{q}-{r}") or []
        if q in used_q or not (1 <= len(segs) <= 3) or f"vcsl:{q[:12]}" in ex["families"]:
            continue
        if not all("youtube.com" in urls.get(u, "") for u in (q, r)) or not all(15 <= frames.get(u, 0) <= 120 for u in (q, r)):
            continue
        if any(urls[u].split("v=")[-1] in ex["evid"] for u in (q, r)):
            continue
        pairs.append((q, r, segs))
    order = {f"{q}-{r}": i for i, (q, r, _) in enumerate(pairs)}
    made = load_json(MADE_FILE, {})
    got = sum(1 for v in made.values() if v.get("dataset") == "vcsl")
    tried = set()
    for key in hash_order(list(order), "g1-vcsl"):
        if got >= n:
            break
        q, r, segs = pairs[order[key]]
        if q in tried:
            continue
        tried.add(q)
        name = f"pair_g1_{got + 1:02d}"
        cid = f"vcsl:{name}"
        ua, ub = urls[q], urls[r]
        ok_a, ch_a = channel_check(cid, ua)
        ok_b, ch_b = channel_check(cid + "#B", ub) if ok_a else (False, None)
        if not (ok_a and ok_b):
            continue
        folder = MEDIA / f"vcsl_{name}"
        folder.mkdir(parents=True, exist_ok=True)
        try:
            for side, url in (("A", ua), ("B", ub)):
                if not (folder / f"{side}.mp4").is_file():
                    media = ytdlp_download(url, folder / f"{side}.download", InspectPolicy(), video_only_height=480)
                    Path(media.path).replace(folder / f"{side}.mp4")
        except Exception as exc:  # noqa: BLE001
            log(f"  VCSL {q[:8]}-{r[:8]}：下载失败（{type(exc).__name__}: {str(exc)[:160]}），删掉、换下一个")
            shutil.rmtree(folder, ignore_errors=True)
            continue
        for tmp in folder.glob("*.download*"):
            shutil.rmtree(tmp, ignore_errors=True) if tmp.is_dir() else tmp.unlink(missing_ok=True)
        save_json(folder / "reference.json", {"query_id": q, "reference_id": r, "urls": [ua, ub],
                                              "seconds_approx": [frames[q], frames[r]], "segments": segs, "dataset": "VCSL",
                                              "split": "train", "usage": "g1"})
        seg_text = "；".join(f"A {a0}–{a1} 秒 ↔ B {b0}–{b1} 秒" for a0, b0, a1, b1 in segs)
        base = f"media/g1/vcsl_{name}"
        case = {
            "case_id": cid, "dataset": "vcsl", "source_id": name, "split": "dev_train", "task_kind": "copy_detect",
            "task_text": d2.VCSL_ZH, "task_text_en": d2.VCSL_EN, "family_key": f"vcsl:{q[:12]}",
            "media": [{"ref": "I0", "path": f"{base}/A.mp4", "media_type": "video/mp4"},
                      {"ref": "I1", "path": f"{base}/B.mp4", "media_type": "video/mp4"}],
            "claims": [], "source_policy": {"mode": "provided_media", "excluded_keys": [], "date_cutoff": None},
            "requirements": [
                {"id": "REQ1", "text": "A 与 B 中内容对应（相同或经变换）的片段，以及两侧各自的时间范围", "must_cite": True,
                 "text_en": "Segments of A and B whose content corresponds (identical or transformed), with the time range on each side"},
                {"id": "REQ2", "text": "对应关系的画面依据：两侧对应帧的内容，以及画面包装、裁剪、叠加等变换", "must_cite": True,
                 "text_en": "Visual grounds for the correspondence: the content of corresponding frames on both sides, and transformations such as framing, cropping or overlays"},
                {"id": "REQ3", "text": "未能确定的内容：边界精度、未检查的区间；不把内容对应推断为原创归属或传播先后", "must_cite": False,
                 "text_en": "What could not be determined: boundary precision and unexamined intervals; a content correspondence is not taken as original authorship or order of spread"},
            ],
            "required_evidence": [{"key": "frames", "description": "两段视频的对应帧", "url_pattern": None,
                                   "acquired_by": "INSPECT/get_frames", "must_cite": True, "supports": ["REQ1", "REQ2"]}],
            "required_evidence_rule": "all", "checks": [], "reference_path": f"{base}/reference.json", "used_for_revision": [],
            "notes": (f"g1 建卷（{now()}，scripts/g1/g1_cases.py，模板同 walker_d2_cases.build_vcsl）：VCSL 训练对 {q} ↔ {r}，"
                      f"标注对应段 {seg_text}（格式按 [A起, B起, A止, B止] 解读）。不写 interval_iou，由充分性层判。"),
        }
        d2.write_case(case)
        record_made(case, "B", "vcsl", {"segments": len(segs)})
        record_channel(cid, (ch_a or []) + (ch_b or []))
        got += 1
        log(f"  VCSL {got}/{n}：{name} {ua} ↔ {ub}；{seg_text}")
    log(f"[VCSL] {got}/{n}")


# ---------------------------------------------------------------------------- A 族：FMNV 找原片（zip 里的原片剪掉一段）

def build_fmnv_a(n: int) -> None:
    from _d2_media import cut_segments, probe_duration

    ex = existing()
    with zipfile.ZipFile(FMNV_ZIP) as zf:
        split_of = fmnv_split_ids(zf)
        made = load_json(MADE_FILE, {})
        c_vids = {v.get("media", "").rsplit("/", 1)[-1].removeprefix("fmnv_").removesuffix(".mp4") for v in made.values()}
        rows = d2.fmnv_rows()
        pool = [v for v, r in rows.items() if r.get("label") == "true" and not v.isdigit() and split_of.get(v) != "test"
                and v not in ex["evid"] and f"fmnv:{v}" not in ex["families"] and v not in ex["blob"] and v not in c_vids]
        got = sum(1 for v in made.values() if v.get("dataset") == "fmnv" and v.get("family") == "A")
        for vid in hash_order(pool, "g1-fmnv-a"):
            if got >= n:
                break
            sid = f"a_{vid}".replace("-", "_")
            if (CASES / f"fmnv_{sid}.json").exists():
                continue
            original = fmnv_extract(zf, vid, f"fmnv_{vid}_orig")
            if original is None:
                continue
            duration = probe_duration(original)
            if not duration or not (35.0 <= duration <= C_MAX_S):
                original.unlink(missing_ok=True)
                continue
            ok, channel = channel_check(f"fmnv:{sid}", youtube_url(vid))
            if not ok:
                original.unlink(missing_ok=True)
                continue
            length = round(min(20.0, max(6.0, 0.15 * duration)), 1)
            start = round(0.42 * duration, 1)
            end = round(start + length, 1)
            material = MEDIA / "fmnv" / f"fmnv_{vid}_cut.mp4"
            info = cut_segments(original, material, [(start, end)])
            original.unlink(missing_ok=True)          # 原片不交给模型（它在网上，walker 要自己找）；只留剪过的材料
            url_id = vid.replace("[", "[[]").replace("?", "[?]")
            case = {
                "case_id": f"fmnv:{sid}", "dataset": "fmnv", "source_id": sid, "split": "dev_train", "task_kind": "provenance",
                "task_text": d2.PROVENANCE_ZH, "task_text_en": d2.PROVENANCE_EN, "family_key": f"fmnv:{vid}",
                "media": [{"ref": "I0", "path": rel_media(material), "media_type": "video/mp4"}], "claims": [],
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
                "notes": (f"g1 建卷（{now()}，scripts/g1/g1_cases.py，模板同 walker_d2_cases.build_provenance）：FMNV 真新闻 {vid}"
                          f"（{rows[vid].get('subject', '')}）。原片取自 FMNV.zip（与 YouTube 原片同一视频），去掉 {start}–{end} s"
                          f"（{length} s）后 PyAV 重编码（{info['video_codec']}，{info['size'][0]}×{info['size'][1]}）；材料时长 "
                          f"{info['duration_out']:.2f} s，剪点在材料 {start} s；原片时长 {duration:.2f} s。"),
            }
            d2.write_case(case)
            record_made(case, "A", "fmnv", {"duration_s": round(duration, 1), "cut": [start, end], "media": rel_media(material)})
            record_channel(case["case_id"], channel)
            got += 1
            log(f"  FMNV A {got}/{n}：{vid}，原片 {duration:.0f} s，删 {start}–{end} s")
        log(f"[FMNV A] {got}/{n}")


def fix_claims() -> None:
    """冒烟后定的声明规则补到已建的 C 案卷上：DROP 里的、按新规则不合格的删掉（连媒体、记录）；规范化后变了的改写声明。"""

    made = load_json(MADE_FILE, {})
    for name in list(made):
        info = made[name]
        path = CASES / f"{name}.json"
        if info.get("family") != "C" or not path.is_file():
            continue
        case = load_json(path, {})
        old = case["claims"][0]
        new = claim_ok(old)
        if name in DROP or new is None:
            path.unlink()
            if info.get("dataset") == "fmnv" and info.get("media"):
                (CASES / info["media"]).unlink(missing_ok=True)
            made.pop(name)
            log(f"  删掉 {name}：{'人工 DROP' if name in DROP else '新规则不合格'}（{old[:80]!r}）")
        elif new != old:
            for key in ("task_text", "task_text_en"):
                case[key] = case[key].replace(old, new)
            case["claims"] = [new]
            d2.write_case(case)
            log(f"  改写 {name}：{old[:80]!r} → {new[:80]!r}")
    save_json(MADE_FILE, made)


# ---------------------------------------------------------------------------- 清单

def write_list() -> None:
    made = load_json(MADE_FILE, {})
    groups: dict[str, list[str]] = {}
    for name, info in made.items():
        if (CASES / f"{name}.json").is_file():
            groups.setdefault(f"{info['family']}_{info['dataset']}", []).append(name)
    out = {"round": "r1g1", "made_by": "scripts/g1/g1_cases.py", "written_at": now(),
           "groups": {k: sorted(v) for k, v in sorted(groups.items())}, "total": sum(len(v) for v in groups.values())}
    save_json(SFT / "walker_g1_cases.json", out)
    print(json.dumps({k: len(v) for k, v in out["groups"].items()}, ensure_ascii=False), out["total"])


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fmnv_c"); p.add_argument("--n-fake", type=int, default=30); p.add_argument("--n-real", type=int, default=30); p.add_argument("--ids", nargs="*")
    p = sub.add_parser("g360"); p.add_argument("--n-fake", type=int, default=20); p.add_argument("--n-real", type=int, default=20); p.add_argument("--ids", nargs="*")
    p = sub.add_parser("videosham"); p.add_argument("--n", type=int, default=14)
    p = sub.add_parser("vcsl"); p.add_argument("--n", type=int, default=20)
    p = sub.add_parser("fmnv_a"); p.add_argument("--n", type=int, default=5)
    sub.add_parser("list")
    sub.add_parser("fix")
    args = parser.parse_args()
    d2.LOG = LOG
    if args.cmd == "fmnv_c":
        build_fmnv_c(args.n_fake, args.n_real, set(args.ids) if args.ids else None)
    elif args.cmd == "g360":
        build_g360(args.n_fake, args.n_real, set(args.ids) if args.ids else None)
    elif args.cmd == "videosham":
        build_videosham(args.n)
    elif args.cmd == "vcsl":
        build_vcsl(args.n)
    elif args.cmd == "fix":
        fix_claims()
    elif args.cmd == "fmnv_a":
        build_fmnv_a(args.n)
    write_list()
    return 0


if __name__ == "__main__":
    sys.exit(main())
