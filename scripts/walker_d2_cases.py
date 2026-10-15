"""walker D2 案卷（2026-10-13）：从 FMNV 与 GroundLie360 各挑几个案例，做 A 族（找原片）与 C 族（核查声明）。

    python scripts\\walker_d2_cases.py            # 下载 + 制作材料 + 写案卷（已有的跳过）
    python scripts\\walker_d2_cases.py --dry-run  # 只打印将要做的，不下载

A 族（找原片，task_kind=provenance）：FMNV 的真新闻视频（video_id 就是 YouTube 编号）。下载原片，去掉中间一段
  （约 15%，6–20 秒）后用 PyAV 重新编码，当作"网上流传的版本"交给模型；任务是找到原片并说明这一版改了什么。
  GT 由制作过程确定：原片 YouTube 编号、原片里被删掉的区间、材料里剪点的时刻。
C 族（核查声明，task_kind=claim_check）：
  - FMNV 的 CD 类假新闻（class=ft：真实视频配上与内容不符的标题）：下载视频原样不改，声明 = 那个假标题，标签 misleading；
  - FMNV 的真新闻 1 条：声明 = 视频自己的标题，标签 accurate；
  - GroundLie360（Snopes 核查过的帖子）：视频没有下载，只做纯文字声明核查，Fake → false、Real → true。

案卷写到 experiments/sft_data/cases/d2_*.json，媒体放 experiments/sft_data/media/d2/。下载用项目自己的 yt-dlp 封装
（inspect/acquisition.ytdlp_download：403 换客户端、没有 ffmpeg 时用 PyAV 封装分离流）。每类按候选顺序取前几个成功的。
数据集标注：plan-harness-1/datasets/FMNV/data.json、plan-harness-1/datasets/GroundLie360/data.csv.gz。
"""

from __future__ import annotations

import argparse
import ast
import gzip
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
SFT = ROOT / "experiments" / "sft_data"
CASES = SFT / "cases"
MEDIA = SFT / "media" / "d2"
FMNV = ROOT / "plan-harness-1" / "datasets" / "FMNV" / "data.json"
G360 = ROOT / "plan-harness-1" / "datasets" / "GroundLie360" / "data.csv.gz"
LOG = ROOT / "_walker_d2案卷日志.txt"

# 候选按顺序试，取前 N 个成功的（下载失败 / 时长不合适就换下一个）
A_POOL = ["3GwQdrhp__k", "1KkAwCyj7nI", "4u12EqPAHqU", "_V_W9ko2CTA", "DNkJSeY1jRg", "dKMlKAJiTrc", "4vbQKcShRXs",
          "CgK-2lfSASM"]
C_FAKE_POOL = ["RXR0B1SxiU0", "TIu19O02X2c", "C05vAdxXXEI", "S9uZ7tPnY80", "luSat_G-U64", "UkbuN8grJmY"]
C_REAL_POOL = ["eC8ctVxSglU", "FEbcNwkD7EU", "DQcHjdjCdMI"]
G360_PICKS = [("46819210", "Fake"), ("46581202", "Real")]
N_G360_VIDEO_FAKE, N_G360_VIDEO_REAL = 2, 1        # 10-13：GroundLie360 视频解压后，另做带视频的声明核查
N_AVERIMATEC = 6                                     # 10-13：E 族（图片最早出处），AVerImaTeC 训练集
AVM_USED = {423, 441, 63, 652, 660, 699, 98}         # 已在 dev_train 案卷里的，不再用
AVM_SOCIAL = ("x.com", "twitter.com", "facebook.com", "instagram.com", "perma.cc", "archive.ph", "archive.vn",
              "archive.is", "archive.today", "tiktok.com", "t.me", "whatsapp.com", "reddit.com", "t.co")
# 10-13 L 族（照片拍摄地）：声明点名了地点 / 地标、标注问"这张图拍于哪里"的条目（人工挑；地名取自标注答案）
# (编号, 实际地点, 来源/结论里的等价写法)
L_PICKS = [(208, "Peru", ["Cusco", "Cuzco"]), (623, "Kansas", ["Topeka", "Battle of Mine Creek"]),
           (739, "Russia", ["Russian"]), (101, "Kolkata", ["Calcutta"])]
AVM_IMAGE_ZIP = "https://huggingface.co/datasets/Rui4416/AVerImaTeC/resolve/main/images.zip"
N_A, N_C_FAKE, N_C_REAL = 3, 2, 1
A_DURATION = (35.0, 300.0)          # 太短剪不出一段、太长下载与比对都慢
C_DURATION = (15.0, 420.0)

PROVENANCE_EN = ("The given video circulated online. Find its original source video and determine whether and how this "
                 "version differs from the original: which parts were removed, added, reordered or altered, with times in "
                 "both videos. Cite specific evidence, and state what could not be determined.")
PROVENANCE_ZH = ("给定视频在网上流传。找到它的原始来源视频，并确定这一版与原片相比是否有改动、改在哪里：删掉、添加、调换或改变了哪些"
                 "部分，给出两个视频中的时间。引用具体证据，说明未能确定的内容。")
CLAIM_VIDEO_EN = ("Fact-check the following claim, which circulated as the title of the given video, examining the video and "
                  "searching for external evidence. Answer based on the evidence actually obtained, and state the scope of "
                  "the conclusion and its uncertainties: {claim}")
CLAIM_VIDEO_ZH = "核查以下声明（它是给定视频流传时的标题），检查视频并搜索外部证据。依据实际证据回答，说明结论范围与不确定性：{claim}"
ORIGIN_EN = ("The given image circulated with the following claim: \"{claim}\" (claim date: {date}). Find where and when this "
             "image was first published, as far as the evidence allows, and whether its original context matches the claim. "
             "Cite the sources and dates you rely on, and state what could not be determined.")
ORIGIN_ZH = "给定图片随以下声明流传：\"{claim}\"（声明日期：{date}）。依据证据尽量查明这张图片最早在哪里、何时发布，以及它的原始语境是否与声明相符。引用所依据的来源与日期，说明未能确定的内容。"
PLACE_EN = ("The given image circulated with the following claim: \"{claim}\" (claim date: {date}). Check whether the picture "
            "really shows the place the claim names, and find where it was actually taken, as far as the evidence allows. "
            "Cite the sources you rely on, and state what could not be determined.")
PLACE_ZH = "给定图片随以下声明流传：\"{claim}\"（声明日期：{date}）。核查这张图片是否真的拍于声明所说的地点，并依据证据尽量查明它实际拍于哪里。引用所依据的来源，说明未能确定的内容。"
CLAIM_TEXT_EN = ("Fact-check the following claim by searching for external evidence. Answer based on the evidence actually "
                 "obtained, and state the scope of the conclusion and its uncertainties: {claim} (claim date: {date})")
CLAIM_TEXT_ZH = "核查以下声明，搜索外部证据。依据实际证据回答，说明结论范围与不确定性：{claim}（声明日期：{date}）"

FAKE_FORMS = ["false", "inaccurate", "not accurate", "does not match", "mismatch", "unsupported", "not supported",
              "misrepresents", "fabricated", "not true", "untrue"]
TRUE_FORMS = ["true", "correct", "accurately describes", "consistent with", "supported by", "confirmed", "authentic", "genuine"]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(text: str) -> None:
    print(text, flush=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(text + "\n")


def download(video_id: str, stem: str) -> Path:
    """YouTube 原片 → MEDIA/<stem>.<实际扩展名>（yt-dlp 可能给 mp4 或 PyAV 封装的 mkv）。已存在就直接用。"""

    for existing in MEDIA.glob(stem + ".*"):
        if existing.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov") and existing.stat().st_size > 0:
            return existing
    from foren_searcher.inspect.acquisition import ytdlp_download
    from foren_searcher.inspect.policy import InspectPolicy

    tmp = MEDIA / f"{stem}.download"
    media = ytdlp_download(f"https://www.youtube.com/watch?v={video_id}", tmp, InspectPolicy())
    final = MEDIA / f"{stem}{media.suffix or '.mp4'}"
    Path(media.path).replace(final)
    return final


def fmnv_rows() -> dict[str, dict[str, Any]]:
    return {row["video_id"]: row for row in json.loads(FMNV.read_text(encoding="utf-8"))}


def g360_rows() -> dict[str, dict[str, Any]]:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))     # 逐词转写一栏很长
    with gzip.open(G360, "rt", encoding="utf-8", newline="") as stream:
        return {str(row["video_id"]): row for row in csv.DictReader(stream)}


def rel(path: Path) -> str:
    return Path("..", *path.relative_to(SFT).parts).as_posix()


def write_case(case: dict[str, Any]) -> Path:
    from foren_searcher.datagen.cases import Case

    Case.from_dict(case)                                   # 字段不合法在这里报错
    path = CASES / f"{case['case_id'].replace(':', '_')}.json"
    path.write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------- A 族：找原片

def build_provenance(video_id: str, row: dict[str, Any]) -> dict[str, Any] | None:
    from _d2_media import cut_segments, probe_duration

    original = download(video_id, f"fmnv_{video_id}_orig")
    duration = probe_duration(original)
    if not duration or not (A_DURATION[0] <= duration <= A_DURATION[1]):
        log(f"  {video_id}：时长 {duration} s 不在 {A_DURATION}，换下一个")
        return None
    length = round(min(20.0, max(6.0, 0.15 * duration)), 1)
    start = round(0.42 * duration, 1)
    end = round(start + length, 1)
    material = MEDIA / f"fmnv_{video_id}_cut.mp4"
    info = cut_segments(original, material, [(start, end)])
    sid = f"a_{video_id}".replace("-", "_")
    log(f"  {video_id}：原片 {duration:.1f} s，去掉 {start}–{end} s → 材料 {info['duration_out']:.1f} s（剪点在材料 {start} s）")
    url_id = video_id.replace("[", "[[]").replace("?", "[?]")
    return {
        "case_id": f"fmnv:{sid}", "dataset": "fmnv", "source_id": sid, "split": "dev_train", "task_kind": "provenance",
        "task_text": PROVENANCE_ZH, "task_text_en": PROVENANCE_EN, "family_key": f"fmnv:{video_id}",
        "media": [{"ref": "I0", "path": rel(material), "media_type": "video/mp4"}], "claims": [],
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
            {"key": "original_video", "description": f"原片：YouTube {video_id}（FMNV 真新闻，标题：{row.get('title', '')[:120]}）",
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
        "notes": (f"D2 制作（{now()}，scripts/walker_d2_cases.py）：FMNV 真新闻 {video_id}（{row.get('subject', '')}）。"
                  f"材料 = 原片去掉 {start}–{end} s（{length} s）后 PyAV 重编码（{info['video_codec']}，{info['size'][0]}×{info['size'][1]}），"
                  f"其余不变；材料时长 {info['duration_out']:.2f} s，剪点在材料 {start} s。原片时长 {duration:.2f} s。"),
    }


# ---------------------------------------------------------------------------- C 族：核查声明

def build_claim_video(video_id: str, row: dict[str, Any], fake: bool) -> dict[str, Any] | None:
    from _d2_media import cut_segments, probe_duration

    source = download(video_id, f"fmnv_{video_id}_orig")
    duration = probe_duration(source)
    if not duration or not (C_DURATION[0] <= duration <= C_DURATION[1]):
        log(f"  {video_id}：时长 {duration} s 不在 {C_DURATION}，换下一个")
        return None
    path = MEDIA / f"fmnv_{video_id}.mp4"
    if not path.exists():
        cut_segments(source, path, [])               # 不剪，只统一成 mp4（H.264 + AAC），内容与原片相同
    claim = str(row["title"]).strip()
    sid = f"c_{video_id}".replace("-", "_")
    label = ({"label": "misleading", "aliases": FAKE_FORMS} if fake else {"label": "accurate", "aliases": TRUE_FORMS})
    kind = {"event": "事件", "object": "对象", "action": "动作"}.get(str(row.get("contextual_class") or ""), "")
    note = (f"FMNV CD 类（标题与画面不符，替换的是{kind}）：视频是 YouTube {video_id} 原样，标题是数据集生成的假标题。"
            if fake else f"FMNV 真新闻：视频是 YouTube {video_id} 原样，标题是它自己的标题。")
    log(f"  {video_id}：{'假标题' if fake else '真标题'}，视频 {duration:.1f} s")
    return {
        "case_id": f"fmnv:{sid}", "dataset": "fmnv", "source_id": sid, "split": "dev_train", "task_kind": "claim_check",
        "task_text": CLAIM_VIDEO_ZH.format(claim=claim), "task_text_en": CLAIM_VIDEO_EN.format(claim=claim),
        "family_key": f"fmnv:{video_id}",
        "media": [{"ref": "I0", "path": rel(path), "media_type": "video/mp4"}], "claims": [claim],
        "source_policy": {"mode": "unrestricted", "excluded_keys": [], "date_cutoff": None},
        "requirements": [
            {"id": "REQ1", "text": "给定视频实际显示和说出的内容（画面、字幕、语音）与声明是否相符", "must_cite": True,
             "text_en": "What the given video actually shows and says (pictures, on-screen text, speech) and whether it matches the claim"},
            {"id": "REQ2", "text": "外部来源对声明所说事件的记载", "must_cite": True,
             "text_en": "What external sources record about the event the claim describes"},
            {"id": "REQ3", "text": "未能确定的内容与结论的适用范围", "must_cite": False,
             "text_en": "What could not be determined and the scope of the conclusion"},
        ],
        "required_evidence": [
            {"key": "frames", "description": "视频画面（抽帧）", "acquired_by": "INSPECT/get_frames", "must_cite": True,
             "supports": ["REQ1"], "url_pattern": None},
            {"key": "transcript", "description": "视频语音转写", "acquired_by": "INSPECT/asr", "must_cite": True,
             "supports": ["REQ1"], "url_pattern": None},
            {"key": "external", "description": "检索打开的外部页面（报道该事件的来源）", "acquired_by": "SEARCH/open",
             "must_cite": True, "supports": ["REQ2"], "url_pattern": None},
        ],
        "required_evidence_rule": "any_one_of:frames,transcript; all:external",
        "checks": [{"kind": "label_expected", "params": label,
                    "note": f"FMNV label={row.get('label')}，class={row.get('class') or 'real'}；{'答案应说标题与视频不符 / 不实' if fake else '答案应说标题属实'}"}],
        "reference_path": None, "used_for_revision": [],
        "notes": f"D2 制作（{now()}）：{note}FMNV subject：{row.get('subject', '')}。",
    }


def build_claim_text(post_id: str, row: dict[str, Any], label_name: str) -> dict[str, Any]:
    claim = " ".join(str(row["title"]).split())
    raw_date = str(row.get("video_date") or "").split(".")[0]
    date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}" if len(raw_date) == 8 else "unknown"
    fake = label_name == "Fake"
    sid = f"c_{post_id}"
    try:
        types = ast.literal_eval(row.get("second_level_annotation") or "[]")
    except (ValueError, SyntaxError):
        types = []
    label = ({"label": "false", "aliases": FAKE_FORMS + ["misleading"]} if fake else {"label": "true", "aliases": TRUE_FORMS + ["accurate"]})
    log(f"  GroundLie360 {post_id}：{label_name}，{claim[:80]}")
    return {
        "case_id": f"groundlie360:{sid}", "dataset": "groundlie360", "source_id": sid, "split": "dev_train",
        "task_kind": "claim_check", "task_text": CLAIM_TEXT_ZH.format(claim=claim, date=date),
        "task_text_en": CLAIM_TEXT_EN.format(claim=claim, date=date), "family_key": f"groundlie360:{post_id}",
        "media": [], "claims": [claim],
        "source_policy": {"mode": "unrestricted", "excluded_keys": [], "date_cutoff": None},
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
        "checks": [{"kind": "label_expected", "params": label,
                    "note": f"GroundLie360 first_level={label_name}，second_level={types}（Snopes 核查；核查站点被检索护栏屏蔽）"}],
        "reference_path": None, "used_for_revision": [],
        "notes": f"D2 制作（{now()}）：GroundLie360 帖子 {post_id}（{row.get('platform')}，{date}）。视频未下载，只核查文字声明。",
    }


# ---------------------------------------------------------------------------- E 族：图片最早出处（AVerImaTeC）

def _strip_archive(url: str) -> str:
    import re as _re

    url = _re.sub(r"^https?://web\.archive\.org/web/\d+[a-z_]*/", "", url.strip())
    url = _re.sub(r"^https?://", "", url)
    return _re.sub(r"^(www|m|mobile)\.", "", url).split("#", 1)[0].rstrip("/")


def averimatec_picks() -> list[tuple[int, dict[str, Any], dict[str, Any], str]]:
    """训练集里问"图片最早何时 / 在哪发布"、答案带具体日期、最早来源不是社交平台或存档站的条目（按编号顺序）。"""

    import re as _re
    sys.path.insert(0, str(ROOT / "src"))
    from foren_searcher.datagen.verify.lang import dates_in

    rows = json.loads((SFT / "data" / "annotations" / "averimatec_train.json").read_text(encoding="utf-8"))
    pat = _re.compile(r"first|earliest|originally|when was (?:the|this) (?:image|photo|picture)", _re.I)
    out = []
    for index, row in enumerate(rows):
        if index in AVM_USED or len(row.get("claim_images") or []) != 1 or row.get("label") not in ("Refuted", "Supported"):
            continue
        for q in row.get("questions") or []:
            if not pat.search(q.get("question") or ""):
                continue
            for answer in q.get("answers") or []:
                dates = dates_in(answer.get("answer_text") or "")
                source = _strip_archive(answer.get("source_url") or "")
                domain = source.split("/", 1)[0].lower()
                if dates and source and not any(domain == d or domain.endswith("." + d) for d in AVM_SOCIAL):
                    out.append((index, row, {**answer, "question": q.get("question"), "date": dates[0]}, source))
                    break
            else:
                continue
            break
    return out


def build_averimatec(index: int, row: dict[str, Any], answer: dict[str, Any], source: str, archive: Any) -> dict[str, Any]:
    from datetime import date, timedelta

    member = row["claim_images"][0]
    names = getattr(archive, "_names", None) or archive.namelist()
    archive._names = names
    name = next(n for n in names if n == member or n.endswith("/" + member))
    image = MEDIA.parent / "d3" / f"averimatec_e{index}.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    if not image.exists():
        image.write_bytes(archive.read(name))
    gt = date.fromisoformat(answer["date"])
    low, high = (gt - timedelta(days=31)).isoformat(), (gt + timedelta(days=183)).isoformat()
    claim = " ".join(str(row["claim_text"]).split())
    fake = row.get("label") == "Refuted"
    label = ({"label": "false", "aliases": FAKE_FORMS + ["misleading", "out of context", "miscaptioned", "unrelated"]}
             if fake else {"label": "true", "aliases": TRUE_FORMS + ["accurate"]})
    article = _strip_archive(row.get("article") or "")
    log(f"  AVerImaTeC {index}：{row['label']}；最早 {answer['date']} {source[:70]}；{claim[:70]}")
    return {
        "case_id": f"averimatec:e{index}", "dataset": "averimatec", "source_id": f"e{index}", "split": "dev_train",
        "task_kind": "provenance", "task_text": ORIGIN_ZH.format(claim=claim, date=row.get("date")),
        "task_text_en": ORIGIN_EN.format(claim=claim, date=row.get("date")), "family_key": f"averimatec:{index}",
        "media": [{"ref": "I0", "path": rel(image), "media_type": "image/jpeg"}], "claims": [claim],
        "source_policy": ({"mode": "exclude_original", "excluded_keys": [article], "date_cutoff": None} if article
                          else {"mode": "unrestricted", "excluded_keys": [], "date_cutoff": None}),
        "requirements": [
            {"id": "REQ1", "text": "这张图片能查到的最早一份（网站、页面）及其日期，以及日期的性质（页面声明 / 上传 / 存档，只能说明\"不晚于\"）",
             "must_cite": True, "text_en": "The earliest copy of the image that can be found (site, page) and its date, with what kind of date it is (declared by the page, upload, archive capture; it only shows 'no later than')"},
            {"id": "REQ2", "text": "最早那份所说的图片内容（地点、时间、事件）与声明是否相符", "must_cite": True,
             "text_en": "Whether what the earliest copy says the picture shows (place, time, event) matches the claim"},
            {"id": "REQ3", "text": "未能确定的内容与结论的适用范围", "must_cite": False,
             "text_en": "What could not be determined and the scope of the conclusion"},
        ],
        "required_evidence": [
            {"key": "dates", "description": "SEARCH 日期查询的结果（页面声明 / 上传 / 最早存档）", "acquired_by": "SEARCH/dates",
             "must_cite": True, "supports": ["REQ1"], "url_pattern": None},
            {"key": "external", "description": "检索打开的、载有这张图片的页面", "acquired_by": "SEARCH/open", "must_cite": True,
             "supports": ["REQ2"], "url_pattern": None},
            {"key": "earliest_source", "description": f"标注的最早来源：{source[:150]}（{answer['date']}）",
             "url_pattern": source.lower().split("?", 1)[0][:120] + "*", "acquired_by": None, "must_cite": False,
             "supports": ["REQ1"]},
        ],
        "required_evidence_rule": "all:dates,external; any_one_of:earliest_source,dates",
        "checks": [
            {"kind": "date_in_range", "params": {"min": low, "max": high},
             "note": f"标注：最早发布 {answer['date']}（{answer.get('question')}）；范围放宽到前 1 个月、后 6 个月（找到的可能是较晚的转载）"},
            {"kind": "label_expected", "params": label, "note": f"AVerImaTeC label={row.get('label')}"},
        ],
        "reference_path": None, "used_for_revision": [],
        "notes": (f"D3 制作（{now()}）：AVerImaTeC 训练集第 {index} 条；图片取自 HuggingFace Rui4416/AVerImaTeC images.zip 的 {name}。"
                  f"标注最早来源 {source}，日期 {answer['date']}（答案原文：{str(answer.get('answer_text'))[:120]}）。"
                  f"排除原核查文章 {article}。"),
    }


def build_place(index: int, row: dict[str, Any], place: str, aliases: list[str], archive: Any) -> dict[str, Any]:
    """L 族：给定照片 + 声明（点名了地点），问是否真拍于那里、实际拍于哪里。GT：标签（都是 Refuted）+ 实际地点实体。"""

    member = row["claim_images"][0]
    names = getattr(archive, "_names", None) or archive.namelist()
    archive._names = names
    name = next(n for n in names if n == member or n.endswith("/" + member))
    image = MEDIA.parent / "d3" / f"averimatec_l{index}.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    if not image.exists():
        image.write_bytes(archive.read(name))
    claim = " ".join(str(row["claim_text"]).split())
    fake = row.get("label") == "Refuted"
    label = ({"label": "false", "aliases": FAKE_FORMS + ["misleading", "out of context", "miscaptioned", "not taken in", "elsewhere"]}
             if fake else {"label": "true", "aliases": TRUE_FORMS + ["accurate"]})
    question = next((q for q in row.get("questions") or [] if re.search(r"where|location|place|city", q.get("question") or "", re.I)), {})
    answer = (question.get("answers") or [{}])[0]
    article = _strip_archive(row.get("article") or "")
    log(f"  AVerImaTeC L{index}：{row['label']}；实际地点 {place}；{claim[:70]}")
    return {
        "case_id": f"averimatec:l{index}", "dataset": "averimatec", "source_id": f"l{index}", "split": "dev_train",
        "task_kind": "claim_check", "task_text": PLACE_ZH.format(claim=claim, date=row.get("date")),
        "task_text_en": PLACE_EN.format(claim=claim, date=row.get("date")), "family_key": f"averimatec:{index}",
        "media": [{"ref": "I0", "path": rel(image), "media_type": "image/jpeg"}], "claims": [claim],
        "source_policy": ({"mode": "exclude_original", "excluded_keys": [article], "date_cutoff": None} if article
                          else {"mode": "unrestricted", "excluded_keys": [], "date_cutoff": None}),
        "requirements": [
            {"id": "REQ1", "text": "图片是否拍于声明所说的地点（与该地点的参考照片当面比对：建筑、地标、地形、标牌）", "must_cite": True,
             "text_en": "Whether the picture shows the place the claim names (compared with reference photos of that place: buildings, landmarks, terrain, signs)"},
            {"id": "REQ2", "text": "载有这张图片的来源说它实际拍于哪里", "must_cite": True,
             "text_en": "Where the sources that carry the picture say it was actually taken"},
            {"id": "REQ3", "text": "未能确定的内容与结论的适用范围", "must_cite": False,
             "text_en": "What could not be determined and the scope of the conclusion"},
        ],
        "required_evidence": [
            {"key": "external", "description": "检索打开的、载有这张图片或该地点照片的页面", "acquired_by": "SEARCH/open",
             "must_cite": True, "supports": ["REQ1", "REQ2"], "url_pattern": None},
        ],
        "required_evidence_rule": "all:external",
        "checks": [
            {"kind": "label_expected", "params": label, "note": f"AVerImaTeC label={row.get('label')}"},
            {"kind": "entity_in_evidence", "params": {"text": place, "aliases": aliases, "statement_aliases": aliases},
             "requirement": "REQ2", "note": f"标注：{question.get('question')} → {str(answer.get('answer_text'))[:160]}"},
        ],
        "reference_path": None, "used_for_revision": [],
        "notes": (f"D3 制作（{now()}）：AVerImaTeC 训练集第 {index} 条（L 族，照片拍摄地）；图片取自 images.zip 的 {name}。"
                  f"标注来源 {answer.get('source_url')}。排除原核查文章 {article}。"),
    }


def g360_video_picks(rows: dict[str, dict[str, Any]], index: dict[str, str]) -> list[tuple[str, dict[str, Any]]]:
    picks: list[tuple[str, dict[str, Any]]] = []
    fakes = [r for r in rows.values() if r.get("first_level_annotation") == "Fake" and str(r["video_id"]) in index
             and "false_title" in str(r.get("second_level_annotation")) and str(r.get("platform")) in ("youtube", "twitter", "x")]
    reals = [r for r in rows.values() if r.get("first_level_annotation") == "Real" and str(r["video_id"]) in index
             and str(r.get("platform")) in ("youtube", "twitter", "x")]
    for pool, n in ((sorted(fakes, key=lambda r: int(r["video_id"])), N_G360_VIDEO_FAKE),
                    (sorted(reals, key=lambda r: int(r["video_id"])), N_G360_VIDEO_REAL)):
        picks += [(str(r["video_id"]), r) for r in pool[:n]]
    return picks


def build_g360_video(post_id: str, row: dict[str, Any], video_rel: str) -> dict[str, Any] | None:
    from _d2_media import probe_duration

    path = ROOT / "plan-harness-1" / "datasets" / "GroundLie360" / video_rel
    duration = probe_duration(path) if path.is_file() else None
    if not duration or duration > 600:
        log(f"  GroundLie360 {post_id}：视频时长 {duration}，跳过")
        return None
    fake = row.get("first_level_annotation") == "Fake"
    claim = " ".join(str(row["title"]).split())
    raw_date = str(row.get("video_date") or "").split(".")[0]
    date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}" if len(raw_date) == 8 else "unknown"
    sid = f"v{post_id}"
    label = ({"label": "false", "aliases": FAKE_FORMS + ["misleading"]} if fake else {"label": "true", "aliases": TRUE_FORMS + ["accurate"]})
    log(f"  GroundLie360 视频 {post_id}：{'Fake' if fake else 'Real'}，{duration:.0f} s，{claim[:70]}")
    media_path = Path("..", "..", "..", *Path("plan-harness-1", "datasets", "GroundLie360", video_rel).parts).as_posix()
    return {
        "case_id": f"groundlie360:{sid}", "dataset": "groundlie360", "source_id": sid, "split": "dev_train",
        "task_kind": "claim_check", "task_text": CLAIM_VIDEO_ZH.format(claim=claim) + f"（发布日期：{date}）",
        "task_text_en": CLAIM_VIDEO_EN.format(claim=claim) + f" (posted: {date})", "family_key": f"groundlie360:{post_id}",
        "media": [{"ref": "I0", "path": media_path, "media_type": "video/mp4"}], "claims": [claim],
        "source_policy": {"mode": "unrestricted", "excluded_keys": [], "date_cutoff": None},
        "requirements": [
            {"id": "REQ1", "text": "给定视频实际显示和说出的内容（画面、字幕、语音）与声明是否相符", "must_cite": True,
             "text_en": "What the given video actually shows and says (pictures, on-screen text, speech) and whether it matches the claim"},
            {"id": "REQ2", "text": "外部来源对声明所说事件的记载", "must_cite": True,
             "text_en": "What external sources record about the event the claim describes"},
            {"id": "REQ3", "text": "未能确定的内容与结论的适用范围", "must_cite": False,
             "text_en": "What could not be determined and the scope of the conclusion"},
        ],
        "required_evidence": [
            {"key": "frames", "description": "视频画面（抽帧）", "acquired_by": "INSPECT/get_frames", "must_cite": True,
             "supports": ["REQ1"], "url_pattern": None},
            {"key": "transcript", "description": "视频语音转写", "acquired_by": "INSPECT/asr", "must_cite": True,
             "supports": ["REQ1"], "url_pattern": None},
            {"key": "external", "description": "检索打开的外部页面", "acquired_by": "SEARCH/open", "must_cite": True,
             "supports": ["REQ2"], "url_pattern": None},
        ],
        "required_evidence_rule": "any_one_of:frames,transcript; all:external",
        "checks": [{"kind": "label_expected", "params": label,
                    "note": f"GroundLie360 first_level={row.get('first_level_annotation')}，second_level={row.get('second_level_annotation')}；"
                            f"标题里的不实片段：{str(row.get('title_third_level_annotation'))[:200]}"}],
        "reference_path": None, "used_for_revision": [],
        "notes": f"D3 制作（{now()}）：GroundLie360 帖子 {post_id}（{row.get('platform')}，{date}），视频来自负责人下载的 vid.zip。",
    }


def build_d3(made: list[str]) -> None:
    """E 族（AVerImaTeC 图片最早出处）+ GroundLie360 带视频的声明核查（视频已解压时）。"""

    picks = averimatec_picks()
    todo = [(i, r, a, s) for i, r, a, s in picks if not (CASES / f"averimatec_e{i}.json").exists()]
    have = [f"averimatec_e{i}" for i, *_ in picks if (CASES / f"averimatec_e{i}.json").exists()]
    made += have[:N_AVERIMATEC]
    need = N_AVERIMATEC - len(have[:N_AVERIMATEC])
    log(f"[E 图片最早出处] 可用 {len(picks)} 条，已有 {len(have)}，还要做 {max(0, need)}")
    if need > 0:
        sys.path.insert(0, str(SFT))
        import zipfile
        from collect_media import RangeReader  # type: ignore

        with RangeReader(AVM_IMAGE_ZIP) as reader, zipfile.ZipFile(reader) as archive:
            for index, row, answer, source in todo:
                if need <= 0:
                    break
                try:
                    path = write_case(build_averimatec(index, row, answer, source, archive))
                except Exception as exc:  # noqa: BLE001
                    log(f"  AVerImaTeC {index}：失败（{type(exc).__name__}: {str(exc)[:160]}）")
                    continue
                made.append(path.stem)
                need -= 1
    # L 族（照片拍摄地，10-13）：同一个图片压缩包
    rows = json.loads((SFT / "data" / "annotations" / "averimatec_train.json").read_text(encoding="utf-8"))
    l_todo = [(i, place, al) for i, place, al in L_PICKS if not (CASES / f"averimatec_l{i}.json").exists()]
    made += [f"averimatec_l{i}" for i, *_ in L_PICKS if (CASES / f"averimatec_l{i}.json").exists()]
    log(f"[L 照片拍摄地] {len(L_PICKS)} 条，还要做 {len(l_todo)}")
    if l_todo:
        sys.path.insert(0, str(SFT))
        import zipfile
        from collect_media import RangeReader  # type: ignore

        with RangeReader(AVM_IMAGE_ZIP) as reader, zipfile.ZipFile(reader) as archive:
            for index, place, aliases in l_todo:
                try:
                    made.append(write_case(build_place(index, rows[index], place, aliases, archive)).stem)
                except Exception as exc:  # noqa: BLE001
                    log(f"  AVerImaTeC L{index}：失败（{type(exc).__name__}: {str(exc)[:160]}）")
    index_path = ROOT / "plan-harness-1" / "datasets" / "GroundLie360" / "videos_index.json"
    if not index_path.is_file():
        log("[C 带视频] GroundLie360 视频还没解压（没有 videos_index.json），跳过；先跑 scripts\\_datasets_prepare.cmd")
        return
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for post_id, row in g360_video_picks(g360_rows(), index):
        existing = CASES / f"groundlie360_v{post_id}.json"
        if existing.exists():
            made.append(existing.stem)
            continue
        case = build_g360_video(post_id, row, index[post_id])
        if case:
            made.append(write_case(case).stem)


def _evid_text() -> str:
    """EVID-Bench（test，锁定）案卷的全文：训练案例的视频编号出现在里面就不用（防止测试集泄漏进训练集）。"""

    folder = ROOT / "experiments" / "evid" / "cases"
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in folder.glob("*.json")) if folder.is_dir() else ""


def _ordered(ids: list[str], seed: str) -> list[str]:
    import hashlib

    return sorted(ids, key=lambda x: hashlib.sha1((seed + x).encode("utf-8")).hexdigest())


def expand_pools(rows: dict[str, dict[str, Any]]) -> None:
    """--pool all（10-14 晚，服务器放量）：A / C 候选从手挑名单扩到 FMNV 全集。真新闻按编号哈希一分为二（A 族找原片 /
    C 族真标题），同一视频不会同时进两族；CD 类（ft，标题与画面不符）全部进 C 族假标题。已下载成功的排前面（省下载）；
    EVID test 里出现过的编号跳过。"""

    global A_POOL, C_FAKE_POOL, C_REAL_POOL
    import hashlib

    downloaded = set()
    log_path = FMNV.parent / "yt_download_log.jsonl"
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if item.get("status") == "ok":
                downloaded.add(str(item.get("video_id")))
    evid = _evid_text()
    real = [v for v, r in rows.items() if r.get("label") == "true" and v not in evid]
    fake = [v for v, r in rows.items() if r.get("class") == "ft" and v not in evid]
    a_real = [v for v in real if int(hashlib.sha1(("split-a:" + v).encode()).hexdigest(), 16) % 2 == 0]
    c_real = [v for v in real if v not in set(a_real)]

    def order(ids: list[str], seed: str) -> list[str]:
        ranked = _ordered(ids, seed)
        return [v for v in ranked if v in downloaded] + [v for v in ranked if v not in downloaded]

    hand_c = set(C_FAKE_POOL) | set(C_REAL_POOL)
    A_POOL = list(dict.fromkeys(A_POOL + [v for v in order(a_real, "a") if v not in hand_c]))
    C_FAKE_POOL = list(dict.fromkeys(C_FAKE_POOL + order(fake, "cf")))
    C_REAL_POOL = list(dict.fromkeys(C_REAL_POOL + [v for v in order(c_real, "cr") if v not in set(A_POOL)]))
    log(f"[放量] 候选：A {len(A_POOL)}、C 假标题 {len(C_FAKE_POOL)}、C 真标题 {len(C_REAL_POOL)}（已下载的排前面；EVID test 编号已排除）")


def main() -> int:
    global N_A, N_C_FAKE, N_C_REAL, N_AVERIMATEC, N_G360_VIDEO_FAKE, N_G360_VIDEO_REAL
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    # 10-14 晚：服务器放量。默认值 = 原来的手挑规模（3 / 2 / 1 / 6 / 2 / 1），清单写 walker_d2_cases.json
    parser.add_argument("--pool", choices=["hand", "all"], default="hand", help="all：A / C 候选扩到 FMNV 全集")
    parser.add_argument("--n-a", type=int, default=N_A)
    parser.add_argument("--n-c-fake", type=int, default=N_C_FAKE)
    parser.add_argument("--n-c-real", type=int, default=N_C_REAL)
    parser.add_argument("--n-e", type=int, default=N_AVERIMATEC)
    parser.add_argument("--n-g360-fake", type=int, default=N_G360_VIDEO_FAKE)
    parser.add_argument("--n-g360-real", type=int, default=N_G360_VIDEO_REAL)
    parser.add_argument("--list", default="walker_d2_cases.json", help="案例清单文件名（experiments/sft_data 下）")
    args = parser.parse_args()
    N_A, N_C_FAKE, N_C_REAL, N_AVERIMATEC = args.n_a, args.n_c_fake, args.n_c_real, args.n_e
    N_G360_VIDEO_FAKE, N_G360_VIDEO_REAL = args.n_g360_fake, args.n_g360_real
    MEDIA.mkdir(parents=True, exist_ok=True)
    log(f"==== walker D2 案卷 {now()}")
    rows = fmnv_rows()
    if args.pool == "all":
        expand_pools(rows)
    made: list[str] = []
    plan = [("A 找原片", A_POOL, N_A, lambda vid: build_provenance(vid, rows[vid])),
            ("C 假标题", C_FAKE_POOL, N_C_FAKE, lambda vid: build_claim_video(vid, rows[vid], fake=True)),
            ("C 真标题", C_REAL_POOL, N_C_REAL, lambda vid: build_claim_video(vid, rows[vid], fake=False))]
    for name, pool, need, build in plan:
        got = 0
        for vid in pool:
            if got >= need:
                break
            prefix = "a" if name.startswith("A") else "c"
            existing = CASES / f"fmnv_{prefix}_{vid.replace('-', '_')}.json"
            if existing.exists():
                log(f"[{name}] {vid}：案卷已存在，跳过")
                made.append(existing.stem)
                got += 1
                continue
            if vid not in rows:
                log(f"[{name}] {vid}：标注里没有，跳过")
                continue
            if args.dry_run:
                log(f"[{name}] 将下载 {vid}：{rows[vid]['title'][:80]}")
                got += 1
                continue
            log(f"[{name}] {vid}：{rows[vid]['title'][:80]}")
            try:
                case = build(vid)
            except Exception as exc:  # noqa: BLE001 —— 下载 / 解码失败就换下一个候选
                log(f"  {vid}：失败（{type(exc).__name__}: {str(exc)[:200]}），换下一个")
                continue
            if case is None:
                continue
            path = write_case(case)
            made.append(path.stem)
            got += 1
        if got < need:
            log(f"[{name}] 只做成 {got}/{need} 个")
    grows = g360_rows()
    for post_id, label_name in G360_PICKS:
        row = grows.get(post_id)
        if row is None:
            log(f"[C 文字] GroundLie360 {post_id} 不在标注里，跳过")
            continue
        if args.dry_run:
            log(f"[C 文字] 将写 GroundLie360 {post_id}（{label_name}）")
            continue
        path = write_case(build_claim_text(post_id, row, label_name))
        made.append(path.stem)
    if not args.dry_run:
        build_d3(made)
    made = list(dict.fromkeys(made))
    (SFT / args.list).write_text(json.dumps(made, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"案卷 {len(made)} 个：{', '.join(made[:40])}{' …' if len(made) > 40 else ''}（清单 experiments\\sft_data\\{args.list}）")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
