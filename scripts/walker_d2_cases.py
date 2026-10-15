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
L_INDICES = {i for i, *_ in L_PICKS}
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
        if index in AVM_USED or index in L_INDICES or len(row.get("claim_images") or []) != 1                 or row.get("label") not in ("Refuted", "Supported"):
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


# ---------------------------------------------------------------------------- 10-15 r0g 扩案例（对话 7 前）：L 族程序筛、B 族、频道
# L 族：AVerImaTeC 训练集里 单图、Refuted、问"图拍于哪里"、答案来源不是社交 / 存档 / 地图站、声明点名了国家（或其形容词）或
# "in / at / near 地名" 的条目，按编号顺序取。地名照抄标注答案（与 L_PICKS 相同做法），表里没有的编号不建（打印出来补表）。
L_QUESTION = re.compile(r"\bwhere(?: and when)? (?:was|were) (?:the|this|these) (?:\d+-th )?(?:image|photo|picture|photograph)s? "
                        r"(?:taken|captured)\b", re.I)
L_EXTRA_SKIP = ("google.com/maps", "maps.google", "goo.gl/maps")
COUNTRY_WORDS = (
    "Afghanistan Albania Algeria Argentina Armenia Australia Austria Bangladesh Belarus Belgium Bolivia Brazil Bulgaria Cambodia "
    "Cameroon Canada Chile China Colombia Congo Croatia Cuba Denmark Egypt England Ethiopia Finland France Gaza Georgia Germany "
    "Ghana Greece Haiti Hungary India Indonesia Iran Iraq Ireland Israel Italy Japan Jordan Kenya Korea Lebanon Libya Malaysia "
    "Mexico Morocco Myanmar Nepal Netherlands Nigeria Norway Pakistan Palestine Peru Philippines Poland Portugal Qatar Romania "
    "Russia Rwanda Scotland Serbia Somalia Spain Sudan Sweden Switzerland Syria Taiwan Thailand Turkey Uganda Ukraine Venezuela "
    "Vietnam Wales Yemen Zimbabwe UK USA UAE "
    "Afghan American Argentine Australian Austrian Brazilian British Canadian Chinese Dutch Egyptian English French German Greek "
    "Indian Indonesian Iranian Iraqi Irish Israeli Italian Japanese Kenyan Mexican Nigerian Pakistani Palestinian Russian Spanish "
    "Swedish Syrian Thai Turkish Ukrainian").split()
L_PLACE_RE = re.compile(r"\b(?:" + "|".join(COUNTRY_WORDS) + r")\b|\b(?:in|at|near)\s+(?:the\s+)?[A-Z][a-z]{2,}")
# 编号 → (实际地点, 等价写法)，取自标注答案原文
L_PLACES = {124: ("Cannes", ["Cannes Film Festival"]), 167: ("Guizhou", ["Kaili", "Qiandongnan"]),
            244: ("Somalia", ["Mogadishu", "Somali"]), 247: ("Congo", ["DRC", "Congolese"]),
            249: ("Bolaang Mongondow", ["Indonesia", "Sulawesi", "Indonesian"]), 250: ("Thailand", ["Thai"]),
            274: ("Kremlin", ["Moscow", "St. Alexander Hall"]), 313: ("Kryvyi Rih", ["Krivoy Rog", "Kryvyi Rig", "Ukraine"]),
            316: ("Stone Mountain", ["Stone Mountain Park", "Confederate Memorial"]), 325: ("Japan", ["Japanese"]),
            341: ("Mandera", ["Kenya", "Kenyan"])}
N_L_AUTO = 0

# B 族（VideoSham / QVHighlights / VCSL）
VIDEOSHAM = Path(r"C:\Users\Zya\Documents\ChatGPT\ForenSearcher\claimunit_video_sham_1004\datasets\videosham")
VCSL_DATA = ROOT / "external" / "VCSL" / "data"
QVH_ARCHIVE = "https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz"
N_VIDEOSHAM, N_QVH, N_VCSL = 0, 0, 0
VIDEO_DIFF_EN = ("Compare the given video A with the candidate video B, and investigate the differences between them that the "
                 "materials can confirm and how those differences affect what the footage means. Cite specific evidence, and state "
                 "where the differences are and what could not be determined. Being given as a candidate does not mean B's source "
                 "identity has been confirmed; do not treat unexamined regions or modalities as having no difference.")
VIDEO_DIFF_ZH = ("比较给定视频 A 与候选视频 B，调查两者可由材料确认的差异及其对画面含义的影响。引用具体证据，说明差异所在位置和未能确定的内容。"
                 "给定候选不代表已经确认其来源身份；不要把未检查的区域或模态视为没有差异。")
QVH_EN = ("Locate the segment of the given video that matches the following description; give its time range in seconds with "
          "supporting evidence, and state the uncertainty of the localization: {query}")
QVH_ZH = "在给定视频中定位符合以下描述的片段，给出秒数范围及证据；说明定位的不确定性：{query}"
VCSL_EN = ("In the given videos A and B, find the segments that correspond to each other (identical or transformed), and give the "
           "time ranges on both sides with supporting evidence. Do not infer original authorship from a content correspondence. "
           "The input contains only the video track; do not assess audio.")
VCSL_ZH = "在给定视频 A 与 B 中查找相同或经变换的对应片段，给出两侧时间范围及证据。不要把内容对应推断成原创归属。输入仅包含视频轨，不评估音频。"
CHANNELS: dict[str, list[str]] = {}          # case_id → [频道名, 频道 ID]（新案卷建卷时取，写 train_channels.json）


def _excluded_families() -> set[str]:
    """dev_select / dev_confirm / dev_eval / test（及 contrast_only）案卷的 family：新案例不许落进去。"""

    out = set()
    for path in CASES.glob("*.json"):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if case.get("split") != "dev_train":
            out.add(str(case.get("family_key")))
    return out


def _evid_channels() -> tuple[set[str], set[str]]:
    """EVID 131–222 源视频的频道名（归一）与频道 ID（evid_channels_131_222.json 与 .detail.json）。"""

    names, ids = set(), set()
    table, detail = SFT / "evid_channels_131_222.json", SFT / "evid_channels_131_222.detail.json"
    if table.is_file():
        for value in json.loads(table.read_text(encoding="utf-8")).values():
            for v in ([value] if isinstance(value, str) else value or []):
                names.add(" ".join(str(v).casefold().split()))
    if detail.is_file():
        def walk(value: Any) -> None:
            if isinstance(value, dict):
                for key, sub in value.items():
                    if key in ("channel_id", "uploader_id") and isinstance(sub, str) and sub:
                        ids.add(sub.casefold())
                    elif key in ("channel", "uploader") and isinstance(sub, str) and sub:
                        names.add(" ".join(sub.casefold().split()))
                    else:
                        walk(sub)
            elif isinstance(value, list):
                for sub in value:
                    walk(sub)
        walk(json.loads(detail.read_text(encoding="utf-8")))
    return names, ids


_EVID_CH: tuple[set[str], set[str]] | None = None


def channel_of(url: str) -> list[str] | None:
    """yt-dlp 只取元数据（不下载）：[频道名, 频道 ID]；取不到（私有 / 下架）返回 None。"""

    try:
        from yt_dlp import YoutubeDL

        with YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # noqa: BLE001
        if "vimeo.com" in url:           # Vimeo 网页端要登录：退到公开的 oEmbed（author_name / author_url）
            try:
                import httpx

                data = httpx.get("https://vimeo.com/api/oembed.json", params={"url": url}, timeout=30).json()
                got = [x for x in (data.get("author_name"), str(data.get("author_url") or "").rstrip("/").rsplit("/", 1)[-1]) if x]
                if got:
                    return got
            except Exception:  # noqa: BLE001
                pass
        log(f"    频道：{url} 取不到（{type(exc).__name__}: {str(exc)[:120]}）")
        return None
    name = info.get("channel") or info.get("uploader") or ""
    cid = info.get("channel_id") or info.get("uploader_id") or ""
    return [x for x in (name, cid) if x]


def evid_channel_hit(channel: list[str] | None) -> bool:
    global _EVID_CH
    if not channel:
        return False
    if _EVID_CH is None:
        _EVID_CH = _evid_channels()
    names, ids = _EVID_CH
    return any(" ".join(c.casefold().split()) in names or c.casefold() in ids for c in channel)


def note_channel(case_id: str, url: str) -> bool:
    """取频道并记进 CHANNELS；频道是 EVID 131–222 的频道就返回 False（调用方换下一个候选）。"""

    channel = channel_of(url)
    if evid_channel_hit(channel):
        log(f"    频道 {channel} 在 EVID 131–222 频道表里，换下一个")
        return False
    if channel:
        CHANNELS[case_id] = channel
    return True


def l_auto_picks(exclude: set[int]) -> list[int]:
    rows = json.loads((SFT / "data" / "annotations" / "averimatec_train.json").read_text(encoding="utf-8"))
    out = []
    for index, row in enumerate(rows):
        if index in exclude or index in AVM_USED or len(row.get("claim_images") or []) != 1 or row.get("label") != "Refuted":
            continue
        if not L_PLACE_RE.search(" ".join(str(row.get("claim_text") or "").split())):
            continue
        for q in row.get("questions") or []:
            if not L_QUESTION.search(q.get("question") or ""):
                continue
            answer = (q.get("answers") or [{}])[0]
            source = _strip_archive(answer.get("source_url") or "")
            domain = source.split("/", 1)[0].lower()
            if (answer.get("answer_text") or "").strip() and source and not any(s in source.lower() for s in L_EXTRA_SKIP) \
                    and not any(domain == d or domain.endswith("." + d) for d in AVM_SOCIAL):
                out.append(index)
            break
    return out


def build_l_auto(made: list[str], e_indices: set[int]) -> None:
    have = [i for i, *_ in L_PICKS]
    picks = l_auto_picks(set(have) | e_indices)
    todo = [i for i in picks if not (CASES / f"averimatec_l{i}.json").exists()]
    existing = [i for i in picks if (CASES / f"averimatec_l{i}.json").exists()]
    made += [f"averimatec_l{i}" for i in existing[:N_L_AUTO]]
    need = N_L_AUTO - len(existing[:N_L_AUTO])
    log(f"[L 程序筛] 候选 {len(picks)} 条（前 12：{picks[:12]}），已有 {len(existing)}，还要做 {max(0, need)}")
    if need <= 0:
        return
    rows = json.loads((SFT / "data" / "annotations" / "averimatec_train.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(SFT))
    import zipfile
    from collect_media import RangeReader  # type: ignore

    with RangeReader(AVM_IMAGE_ZIP) as reader, zipfile.ZipFile(reader) as archive:
        for index in todo:
            if need <= 0:
                break
            if index not in L_PLACES:
                log(f"  AVerImaTeC L{index}：L_PLACES 里没有它的地名（取自标注答案），先补表；跳过")
                continue
            place, aliases = L_PLACES[index]
            try:
                case = build_place(index, rows[index], place, aliases, archive)
            except Exception as exc:  # noqa: BLE001
                log(f"  AVerImaTeC L{index}：失败（{type(exc).__name__}: {str(exc)[:160]}）")
                continue
            case["notes"] += " r0g 扩案例：按 walker_d2_cases.l_auto_picks 的规则程序筛（编号顺序），地名取自标注答案。"
            made.append(write_case(case).stem)
            need -= 1


def _hash_order(ids: list[str], seed: str) -> list[str]:
    return _ordered(ids, seed)


def videosham_picks() -> list[dict[str, str]]:
    with (VIDEOSHAM / "VideoSourceandEdits.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = {Path(r["Video-Name"]).stem: r for r in csv.DictReader(stream)}
    evid = _evid_text()
    used_groups = set()
    for path in CASES.glob("videosham_*.json"):
        sid = path.stem.split("_", 1)[1]
        if sid in rows:
            used_groups.add(rows[sid]["Name/Id in source dataset"].strip())
    stems = [p.name.split("_")[0] for p in (VIDEOSHAM / "media" / "edited").glob("*_processed.mp4")]
    out, groups = [], set(used_groups)
    for sid in _hash_order([s for s in stems if s in rows], "r0g-videosham"):
        row = rows[sid]
        group = row["Name/Id in source dataset"].strip()
        if (CASES / f"videosham_{sid}.json").exists() or group in groups or group in evid or f"videosham_{sid}" in evid:
            continue
        if not (VIDEOSHAM / "media" / "source" / f"{sid}_unedited.mp4").is_file():
            continue
        groups.add(group)
        out.append({"id": sid, **row})
    return out


def build_videosham(row: dict[str, str]) -> dict[str, Any]:
    import shutil

    sid = row["id"]
    folder = SFT / "data" / "cases" / f"videosham_{sid}"
    (folder / "inputs").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(VIDEOSHAM / "media" / "edited" / f"{sid}_processed.mp4", folder / "inputs" / "A.mp4")
    shutil.copyfile(VIDEOSHAM / "media" / "source" / f"{sid}_unedited.mp4", folder / "inputs" / "B.mp4")
    record = {k: v for k, v in row.items() if k != "id"}
    (folder / "reference.json").write_text(json.dumps(
        {"dataset": "VideoSham", "source_record": record, "input_assignment": {"A.mp4": "processed", "B.mp4": "unedited"},
         "reference_status": "official_edit_description_requires_media_confirmation", "split": "development",
         "condition": "provided_candidate_pair"}, ensure_ascii=False, indent=2), encoding="utf-8")
    task, manip = " ".join(row["Task"].split()), " ".join(row["Manipulation"].split())
    text_edit = "text" in manip.lower()
    evidence = [
        {"key": "frames", "description": "两段视频的抽帧（差异位置两侧的对应帧）", "url_pattern": None, "acquired_by": "INSPECT/get_frames",
         "must_cite": True, "supports": ["REQ1", "REQ3"]},
        {"key": "diff_images", "description": "diff_media 产出的逐帧差异图（差异时间线、差异框与成对放大），可替代成对抽帧作为差异位置的证据",
         "url_pattern": None, "acquired_by": "INSPECT/diff_media", "must_cite": True, "supports": ["REQ1", "REQ3"]},
        {"key": "media_profile", "description": "两段视频的媒体属性（时长、帧率、音轨数）", "url_pattern": None,
         "acquired_by": "INSPECT/media_profile", "must_cite": True, "supports": ["REQ2"]},
        {"key": "alignment", "description": "视觉或音频对齐结果（候选对应时间锚点，只作线索）", "url_pattern": None,
         "acquired_by": "INSPECT/align_media", "must_cite": False, "supports": ["REQ1"]},
    ]
    if text_edit:     # 文字类改动：两侧画面文字的 OCR 是直接证据（可选）
        evidence.append({"key": "ocr", "description": "两侧对应帧画面文字的 OCR", "url_pattern": None, "acquired_by": "INSPECT/ocr",
                         "must_cite": False, "supports": ["REQ1"]})
    log(f"  VideoSham {sid}：{manip}；{task[:90]}")
    return {
        "case_id": f"videosham:{sid}", "dataset": "videosham", "source_id": sid, "split": "dev_train", "task_kind": "video_diff",
        "task_text": VIDEO_DIFF_ZH, "task_text_en": VIDEO_DIFF_EN, "family_key": f"videosham:{sid}",
        "media": [{"ref": "I0", "path": f"../data/cases/videosham_{sid}/inputs/A.mp4", "media_type": "video/mp4"},
                  {"ref": "I1", "path": f"../data/cases/videosham_{sid}/inputs/B.mp4", "media_type": "video/mp4"}],
        "claims": [], "source_policy": {"mode": "provided_media", "excluded_keys": [], "date_cutoff": None},
        "requirements": [
            {"id": "REQ1", "text": f"A 是否有官方编辑描述所说的改动（{task}；须由画面确认），以及两侧对应位置", "must_cite": True,
             "text_en": f"Whether A has the edit the official description gives (\"{task}\"; to be confirmed from the footage), "
                        "and the corresponding positions on both sides"},
            {"id": "REQ2", "text": "音轨是否可取得；可确认的声音差异或无法核验的说明", "must_cite": True,
             "text_en": "Whether the audio tracks could be obtained; any confirmable audio difference, or a statement that it could not be verified"},
            {"id": "REQ3", "text": "已确认的差异对画面含义的影响（不把未检查的区域或模态当作没有差异）", "must_cite": True,
             "text_en": "How the confirmed differences affect what the footage means (unexamined regions or modalities are not treated as having no difference)"},
            {"id": "REQ4", "text": "未能确定的内容；不预设候选视频 B 的来源身份或制作先后", "must_cite": False,
             "text_en": "What could not be determined; the source identity or production order of candidate video B is not presupposed"},
        ],
        "required_evidence": evidence, "required_evidence_rule": "any_one_of:frames,diff_images", "checks": [],
        "reference_path": f"../data/cases/videosham_{sid}/reference.json", "used_for_revision": [],
        "notes": (f"官方编辑描述（reference_status: 需由画面确认）：{task}；Manipulation: {manip}；r0g 扩案例（{now()}，"
                  f"walker_d2_cases.py 程序建卷）：A = {sid}_processed.mp4，B = {sid}_unedited.mp4；label_expected 不写（子串匹配表达不了官方描述）。"
                  f"源：{row['Name/Id in source dataset'].strip()}"),
    }


def build_b(made: list[str]) -> None:
    excluded = _excluded_families()
    # VideoSham
    got = 0
    for row in videosham_picks() if N_VIDEOSHAM else []:
        if got >= N_VIDEOSHAM:
            break
        url = row["Name/Id in source dataset"].strip()
        cid = f"videosham:{row['id']}"
        if url.startswith("http") and not note_channel(cid, url):
            continue
        made.append(write_case(build_videosham(row)).stem)
        got += 1
    if N_VIDEOSHAM:
        log(f"[B VideoSham] 新做 {got}/{N_VIDEOSHAM}")
    if N_QVH:
        build_qvh(made, excluded)
    if N_VCSL:
        build_vcsl(made, excluded)


def build_qvh(made: list[str], excluded: set[str]) -> None:
    """QVHighlights：从官方视频归档流式读（只读前缀，≤1 GiB），取训练集里只有一个相关窗口（10–60 s）的片段，
    源 YouTube 编号不在已有案卷 / 非 dev_train family / EVID 案卷里。"""

    import tarfile

    import httpx

    sys.path.insert(0, str(SFT))
    from fetch_qvhighlights import StreamReader  # type: ignore

    rows: dict[str, list[dict[str, Any]]] = {}
    for line in (SFT / "data" / "annotations" / "qvhighlights_train.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows.setdefault(row["vid"], []).append(row)
    seen = set()
    for path in CASES.glob("qvhighlights_*.json"):
        seen.add(str(json.loads(path.read_text(encoding="utf-8")).get("family_key")).partition(":")[2])
    evid = _evid_text()
    need, got = N_QVH, 0
    with httpx.Client(timeout=90, follow_redirects=True) as client, client.stream("GET", QVH_ARCHIVE) as response:
        response.raise_for_status()
        with tarfile.open(fileobj=StreamReader(response.iter_bytes(65536)), mode="r|gz") as archive:
            for member in archive:
                vid = Path(member.name).stem
                source = vid.rsplit("_", 2)[0]
                if not member.isfile() or vid not in rows or source in seen or f"qvhighlights:{source}" in excluded or source in evid:
                    continue
                pick = next((r for r in rows[vid] if len(r.get("relevant_windows") or []) == 1
                             and 10 <= r["relevant_windows"][0][1] - r["relevant_windows"][0][0] <= 60), None)
                if pick is None:
                    continue
                cid = f"qvhighlights:{pick['qid']}"
                if not note_channel(cid, f"https://www.youtube.com/watch?v={source}"):
                    seen.add(source)
                    continue
                folder = SFT / "data" / "qvhighlights" / str(pick["qid"])
                (folder / "inputs").mkdir(parents=True, exist_ok=True)
                data = archive.extractfile(member)
                (folder / "inputs" / "A.mp4").write_bytes(data.read() if data else b"")
                (folder / "reference.json").write_text(json.dumps(pick, ensure_ascii=False), encoding="utf-8")
                seen.add(source)
                start, end = pick["relevant_windows"][0]
                query = " ".join(str(pick["query"]).split())
                log(f"  QVHighlights {pick['qid']}（{vid}）：[{start}, {end}]；{query[:80]}")
                case = {
                    "case_id": cid, "dataset": "qvhighlights", "source_id": str(pick["qid"]), "split": "dev_train",
                    "task_kind": "temporal_locate", "task_text": QVH_ZH.format(query=query), "task_text_en": QVH_EN.format(query=query),
                    "family_key": f"qvhighlights:{source}",
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
                    "notes": (f"r0g 扩案例（{now()}，walker_d2_cases.py）：标注片段 [[{start}, {end}]]（视频 {pick.get('duration')} 秒，"
                              f"源 YouTube {source} 的 {vid.rsplit('_', 2)[1]}–{vid.rsplit('_', 2)[2]} s）；family 按源视频号分组。"),
                }
                made.append(write_case(case).stem)
                got += 1
                if got >= need:
                    break
    log(f"[B QVHighlights] 新做 {got}/{need}")


def build_vcsl(made: list[str], excluded: set[str]) -> None:
    """VCSL：训练集对，两侧都是 YouTube、各 15–120 s（frames_all 按 1 fps 计）、标注 1–3 段；query 视频不在已有 family。
    下载只取视频轨（与任务"输入仅包含视频轨"一致）。"""

    from foren_searcher.inspect.acquisition import ytdlp_download
    from foren_searcher.inspect.policy import InspectPolicy

    urls = {}
    with (VCSL_DATA / "videos_url_uuid.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            urls[row["uuid"]] = row["url"]
    frames = {}
    with (VCSL_DATA / "frames_all.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            frames[row["uuid"]] = int(row["frame_count"])
    labels = json.loads((VCSL_DATA / "label_file_uuid_total.json").read_text(encoding="utf-8"))
    families = {str(json.loads(p.read_text(encoding="utf-8")).get("family_key")) for p in CASES.glob("vcsl_*.json")}
    used_queries = set()
    for ref in (SFT / "data" / "vcsl").glob("pair_*/reference.json"):
        used_queries.add(json.loads(ref.read_text(encoding="utf-8")).get("query_id"))
    pairs = []
    with (VCSL_DATA / "pair_file_train.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            q, r = row["query_id"], row["reference_id"]
            segs = labels.get(f"{q}-{r}") or []
            if q in used_queries or not (1 <= len(segs) <= 3):
                continue
            if not all("youtube.com" in urls.get(u, "") for u in (q, r)) or not all(15 <= frames.get(u, 0) <= 120 for u in (q, r)):
                continue
            pairs.append((q, r, segs))
    evid = _evid_text()
    order = {f"{q}-{r}": i for i, (q, r, _) in enumerate(pairs)}
    ranked = [p for key in _hash_order(list(order), "r0g-vcsl") for p in [pairs[order[key]]]]
    n_pair = max([int(p.name.split("_")[1]) for p in (SFT / "data" / "vcsl").glob("pair_*") if p.name.split("_")[1].isdigit()] + [0])
    got, tried_queries = 0, set()
    for q, r, segs in ranked:
        if got >= N_VCSL:
            break
        if q in tried_queries or f"vcsl:{q[:12]}" in families or f"vcsl:{q[:12]}" in excluded:
            continue
        ua, ub = urls[q], urls[r]
        if any(u.split("v=")[-1] in evid for u in (ua, ub)):
            continue
        tried_queries.add(q)
        name = f"pair_{n_pair + 1}"
        folder = SFT / "data" / "vcsl" / name / "inputs"
        folder.mkdir(parents=True, exist_ok=True)
        cid = f"vcsl:{name}"
        if not note_channel(cid, ua) or not note_channel(cid + "#B", ub):
            CHANNELS.pop(cid, None), CHANNELS.pop(cid + "#B", None)
            continue
        try:
            for side, url in (("A", ua), ("B", ub)):
                if not (folder / f"{side}.mp4").is_file():
                    media = ytdlp_download(url, folder / f"{side}.download", InspectPolicy(), video_only_height=480)
                    Path(media.path).replace(folder / f"{side}.mp4")
        except Exception as exc:  # noqa: BLE001
            log(f"  VCSL {q[:8]}-{r[:8]}：下载失败（{type(exc).__name__}: {str(exc)[:160]}），换下一个")
            CHANNELS.pop(cid, None), CHANNELS.pop(cid + "#B", None)
            continue
        if cid + "#B" in CHANNELS:
            CHANNELS[cid] = CHANNELS.get(cid, []) + CHANNELS.pop(cid + "#B")
        n_pair += 1
        (folder.parent / "reference.json").write_text(json.dumps(
            {"query_id": q, "reference_id": r, "urls": [ua, ub], "seconds_approx": [frames[q], frames[r]], "segments": segs,
             "dataset": "VCSL", "split": "train", "usage": "r0g_expansion"}, ensure_ascii=False), encoding="utf-8")
        seg_text = "；".join(f"A {a0}–{a1} 秒 ↔ B {b0}–{b1} 秒" for a0, b0, a1, b1 in segs)
        log(f"  VCSL {name}：{ua} ↔ {ub}；{seg_text}")
        case = {
            "case_id": cid, "dataset": "vcsl", "source_id": name, "split": "dev_train", "task_kind": "copy_detect",
            "task_text": VCSL_ZH, "task_text_en": VCSL_EN, "family_key": f"vcsl:{q[:12]}",
            "media": [{"ref": "I0", "path": f"../data/vcsl/{name}/inputs/A.mp4", "media_type": "video/mp4"},
                      {"ref": "I1", "path": f"../data/vcsl/{name}/inputs/B.mp4", "media_type": "video/mp4"}],
            "claims": [], "source_policy": {"mode": "provided_media", "excluded_keys": [], "date_cutoff": None},
            "requirements": [
                {"id": "REQ1", "text": "A 与 B 中内容对应（相同或经变换）的片段，以及两侧各自的时间范围", "must_cite": True,
                 "text_en": "Segments of A and B whose content corresponds (identical or transformed), with the time range on each side"},
                {"id": "REQ2", "text": "对应关系的画面依据：两侧对应帧的内容，以及画面包装、裁剪、叠加等变换", "must_cite": True,
                 "text_en": "Visual grounds for the correspondence: the content of corresponding frames on both sides, and transformations such as framing, cropping or overlays"},
                {"id": "REQ3", "text": "未能确定的内容：边界精度、未检查的区间；不把内容对应推断为原创归属或传播先后", "must_cite": False,
                 "text_en": "What could not be determined: boundary precision and unexamined intervals; a content correspondence is not taken as original authorship or order of spread"},
            ],
            "required_evidence": [{"key": "frames", "description": "两段视频的对应帧", "url_pattern": None, "acquired_by": "INSPECT/get_frames",
                                   "must_cite": True, "supports": ["REQ1", "REQ2"]}],
            "required_evidence_rule": "all", "checks": [],
            "reference_path": f"../data/vcsl/{name}/reference.json", "used_for_revision": [],
            "notes": (f"r0g 扩案例（{now()}，walker_d2_cases.py）：VCSL 训练对 {q} ↔ {r}，标注对应段 {seg_text}"
                      "（格式按 [A起, B起, A止, B止] 解读）。与第二批相同，不写 interval_iou，由充分性层判。"),
        }
        made.append(write_case(case).stem)
        got += 1
    log(f"[B VCSL] 新做 {got}/{N_VCSL}")


def write_channels() -> None:
    """train_channels.json：{case_id: [频道名, 频道 ID]}，门的 --train-channels 读；已有的表合并，不覆盖旧项。"""

    path = SFT / "train_channels.json"
    table = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    table.update({k: v for k, v in CHANNELS.items() if "#" not in k and (CASES / f"{k.replace(':', '_')}.json").exists()})
    path.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[频道] train_channels.json：{len(table)} 个案卷（本次新取 {len(CHANNELS)}）")


def main() -> int:
    global N_A, N_C_FAKE, N_C_REAL, N_AVERIMATEC, N_G360_VIDEO_FAKE, N_G360_VIDEO_REAL
    global N_L_AUTO, N_VIDEOSHAM, N_QVH, N_VCSL
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
    # 10-15 r0g 扩案例：L 族程序筛、B 族三个数据集（默认 0 = 不做，行为与以前相同）
    parser.add_argument("--n-l", type=int, default=0, help="L 族程序筛的条数（在 L_PICKS 4 条之外）")
    parser.add_argument("--n-videosham", type=int, default=0)
    parser.add_argument("--n-qvh", type=int, default=0)
    parser.add_argument("--n-vcsl", type=int, default=0)
    args = parser.parse_args()
    N_A, N_C_FAKE, N_C_REAL, N_AVERIMATEC = args.n_a, args.n_c_fake, args.n_c_real, args.n_e
    N_G360_VIDEO_FAKE, N_G360_VIDEO_REAL = args.n_g360_fake, args.n_g360_real
    N_L_AUTO, N_VIDEOSHAM, N_QVH, N_VCSL = args.n_l, args.n_videosham, args.n_qvh, args.n_vcsl
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
            if args.pool == "all" and not name.startswith("A") and                     not note_channel(f"fmnv:{prefix}_{vid.replace('-', '_')}", f"https://www.youtube.com/watch?v={vid}"):
                continue                     # 10-15：频道在 EVID 131–222 频道表里的不用
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
        if N_L_AUTO:
            build_l_auto(made, {int(m[len("averimatec_e"):]) for m in made if m.startswith("averimatec_e")})
        build_b(made)
        if CHANNELS:
            write_channels()
    made = list(dict.fromkeys(made))
    (SFT / args.list).write_text(json.dumps(made, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"案卷 {len(made)} 个：{', '.join(made[:40])}{' …' if len(made) > 40 else ''}（清单 experiments\\sft_data\\{args.list}）")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
