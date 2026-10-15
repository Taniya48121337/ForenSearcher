"""Other video-misinformation benchmarks besides EVID (2026-10-12): GroundLie360 and FMNV.

Both are run with the same agent setup as the EVID full run (Qwen3.8-27B Q4 without thinking + DETECT v2.4, 3 rounds,
hi@0.7.0 / gia@0.8.0, compact_16k, fact-check sites blocked). Their labels are classifications, not EVID's free-text
forgery points, so the task asks for an explanation followed by fixed answer lines that are parsed for scoring.

GroundLie360 (Snopes-checked real posts, ACL 2026; github.com/yangbingjian/GroundLie360)
    input  : the video + its title, platform and date (FakeMark, the paper's zero-shot agent, gets title + video + ASR;
             our agent transcribes itself)
    labels : Real / Fake; six fake types (multi-label); false title words; false-speech time spans; edited scenes
    paper  : zero-shot FakeMark (GPT-4o-mini + InternVL-8B) on all 2,044: Acc 62.90, P 67.81, R 52.81, F1 59.38;
             fake-type macro-F1 29.23; supervised FakingRecipe Acc 76.14 / F1 74.94 (on their 15% test split)
FMNV (media-published news videos, ICIC 2025; github.com/DennisIW/FMNV)
    input  : the video + its title
    labels : Real / Fake; fake class ft (contextual dishonesty: title vs video), fv (cherry-picked editing: shots
             removed), fa (synthetic voiceover), fc (contrived absurdity)
    note   : every video_id is a YouTube ID; for fakes it is the authentic video they were made from, so like EVID the
             original can be searched for and compared.
"""

from __future__ import annotations

import ast
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "plan-harness-1" / "datasets"
BENCH = ROOT / "experiments" / "bench"
VIDEO_SUFFIXES = (".mp4", ".webm", ".mkv", ".mov", ".m4v", ".avi", ".flv")

G360_TYPES = ("false_title", "false_speech", "temporal_edit", "CGI", "contradictory_content", "unsupported_content")
FMNV_TYPES = {"ft": "CD", "fv": "CE", "fa": "SV", "fc": "CA"}       # dataset class -> letter code in the task text
FMNV_CODES = ("CD", "CE", "SV", "CA")
PLATFORMS = {"twitter": "Twitter", "x": "X (Twitter)", "youtube": "YouTube", "tiktok": "TikTok", "facebook": "Facebook",
             "instagram": "Instagram", "reddit": "Reddit", "foxnews": "Fox News"}


@dataclass(frozen=True)
class Sample:
    dataset: str          # "g360" | "fmnv"
    sample_id: str
    label: str            # "Real" | "Fake"
    types: tuple[str, ...]
    title: str
    platform: str = ""
    date: str = ""
    gt: dict | None = None

    @property
    def case_id(self) -> str:
        return f"{self.dataset}:{self.sample_id}"


# ---------------------------------------------------------------- annotations

def _lit(value, default):
    if value in (None, ""):
        return default
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return default


def load_g360(root: Path | None = None) -> list[Sample]:
    root = root or DATASETS / "GroundLie360"
    csv.field_size_limit(1 << 30)
    out = []
    import gzip

    plain = root / "data.csv"
    opener = (lambda: plain.open(encoding="utf-8", newline="")) if plain.is_file() else \
        (lambda: gzip.open(root / "data.csv.gz", "rt", encoding="utf-8", newline=""))
    with opener() as stream:
        for row in csv.DictReader(stream):
            date = ""
            raw = (row.get("video_date") or "").split(".")[0]
            if re.fullmatch(r"\d{8}", raw):
                date = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
            label = "Fake" if row["first_level_annotation"].strip().lower() == "fake" else "Real"
            types = tuple(t for t in _lit(row.get("second_level_annotation"), []) if t in G360_TYPES)
            gt = {"title_spans": [x.get("value", "") for x in _lit(row.get("title_third_level_annotation"), [])
                                  if isinstance(x, dict)],
                  "false_speech_s": _lit(row.get("false_speech_temporal"), []),
                  "scene_frames": _scene_frames(row.get("video_scene_segmentation") or ""),
                  "edited_scenes": _lit(row.get("temporal_edit_location"), [])}
            out.append(Sample("g360", str(row["video_id"]).strip(), label, types, row.get("title") or "",
                              row.get("platform") or "", date, gt))
    return out


def _scene_frames(text: str) -> list[list[int]]:
    """'[[ 1 184]\\n [185 222] ...]' (numpy repr) -> [[1, 184], [185, 222], ...]."""

    nums = [int(n) for n in re.findall(r"\d+", text)]
    return [nums[i:i + 2] for i in range(0, len(nums) - 1, 2)]


def load_fmnv(root: Path | None = None) -> list[Sample]:
    root = root or DATASETS / "FMNV"
    out = []
    for row in json.loads((root / "data.json").read_text(encoding="utf-8")):
        label = "Fake" if row["label"] == "false" else "Real"
        types = (FMNV_TYPES[row["class"]],) if row.get("class") in FMNV_TYPES else ()
        out.append(Sample("fmnv", str(row["video_id"]), label, types, row.get("title") or "",
                          gt={"subject": row.get("subject"), "contextual_class": row.get("contextual_class")}))
    return out


LOADERS = {"g360": load_g360, "fmnv": load_fmnv}
DIRS = {"g360": "GroundLie360", "fmnv": "FMNV"}


def video_index(dataset: str) -> dict[str, Path]:
    """Stem -> file, searched recursively under datasets/<name>/ (the download layout is not fixed)."""

    found: dict[str, Path] = {}
    base = DATASETS / DIRS[dataset]
    if base.is_dir():
        for path in base.rglob("*"):
            if path.suffix.lower() in VIDEO_SUFFIXES and path.is_file():
                found.setdefault(path.stem, path)
    return found


# ---------------------------------------------------------------- sampling

def pick(dataset: str, samples: list[Sample], available: set[str], n: int, seed: int = 20261012) -> list[Sample]:
    """Deterministic stratified sample among samples whose video is on disk: half Real, half Fake, Fakes spread
    evenly over fake types (GroundLie360: by first listed type; a later, larger n keeps the earlier picks)."""

    rng = random.Random(seed)
    pool = [s for s in samples if s.sample_id in available]
    real = [s for s in pool if s.label == "Real"]
    fake = [s for s in pool if s.label == "Fake"]
    rng.shuffle(real)
    rng.shuffle(fake)
    n_fake = (n + 1) // 2
    groups: dict[str, list[Sample]] = {}
    for s in fake:
        groups.setdefault(s.types[0] if s.types else "other", []).append(s)
    order = sorted(groups)
    chosen_fake: list[Sample] = []
    while len(chosen_fake) < n_fake and any(groups.values()):
        for key in order:
            if groups[key] and len(chosen_fake) < n_fake:
                chosen_fake.append(groups[key].pop(0))
    chosen = chosen_fake + real[: n - len(chosen_fake)]
    return sorted(chosen, key=lambda s: s.sample_id)


# ---------------------------------------------------------------- task text

def task_text(sample: Sample) -> str:
    if sample.dataset == "g360":
        where = PLATFORMS.get(sample.platform.lower(), sample.platform or "social media")
        when = f" on {sample.date}" if sample.date else ""
        return (
            f"Material A is a video posted on {where}{when} with this title: \"{sample.title.strip()}\"\n\n"
            "Determine whether this post (the title together with the video and what is said in it) is misinformation. "
            "Watch the whole video and check what is said, look online for the original footage, earlier copies and "
            "reports of the event, and compare them with the post. Fact-checking websites are not available as sources.\n\n"
            "The post is Fake if any of these holds (more than one may apply):\n"
            "- false_title: the title makes a demonstrably false claim;\n"
            "- false_speech: what is said in the video makes a demonstrably false claim;\n"
            "- temporal_edit: the footage was cut, reordered or spliced to distort the chronology or build a deceptive "
            "narrative;\n"
            "- CGI: the video contains digitally manipulated or AI-generated imagery;\n"
            "- contradictory_content: the title and the video contradict each other;\n"
            "- unsupported_content: the title claims something the video does not show or support.\n"
            "Otherwise the post is Real.\n\n"
            "Final answer, in English: first explain your findings in a short numbered list, saying what you observed in "
            "Material A and what you found online. Then end with exactly these five lines:\n"
            "VERDICT: Real or Fake\n"
            "TYPES: the applicable types from the list above, comma-separated, or none\n"
            "FALSE_TITLE_WORDS: the exact misleading words of the title, or none\n"
            "FALSE_SPEECH_TIME: start-end seconds of each false statement in the video (for example 12.5-20; 41-48), or none\n"
            "EDITED_TIME: start-end seconds of each edited or generated part of the video, or none")
    return (
        f"Material A is a news video published online with this title: \"{sample.title.strip()}\"\n\n"
        "Determine whether this news video is fake. Watch the whole video and listen to its audio, look online for the "
        "original broadcast or footage and reports of the event, and compare them with Material A. Fact-checking "
        "websites are not available as sources.\n\n"
        "The video is Fake if one of these holds:\n"
        "- CD (contextual dishonesty): the title misrepresents the events actually shown in the video;\n"
        "- CE (cherry-picked editing): key segments of the original footage were removed to create a biased or false "
        "narrative;\n"
        "- SV (synthetic voiceover): the original audio was replaced with AI-generated speech that contradicts or "
        "fabricates the visual content;\n"
        "- CA (contrived absurdity): the video looks consistent but makes exaggerated, illogical or common-sense-defying "
        "claims.\n"
        "Otherwise it is Real.\n\n"
        "Final answer, in English: first explain your findings in a short numbered list, saying what you observed in "
        "Material A and what you found online. Then end with exactly these two lines:\n"
        "VERDICT: Real or Fake\n"
        "TYPE: CD, CE, SV, CA, or none")


# ---------------------------------------------------------------- answer parsing

def _line(text: str, key: str) -> str | None:
    hits = re.findall(rf"(?im)^\W*{key}\s*[:：]\s*(.+?)\s*$", text)
    return hits[-1].strip().strip("*").strip() if hits else None


def parse_spans(value: str | None) -> list[list[float]]:
    if not value or value.strip().lower() in ("none", "n/a", "-"):
        return []
    out = []
    for a, b in re.findall(r"(\d+(?:\.\d+)?)\s*(?:s|sec)?\s*[-–~to]+\s*(\d+(?:\.\d+)?)", value):
        lo, hi = float(a), float(b)
        if hi > lo:
            out.append([lo, hi])
    return out


def parse_answer(dataset: str, answer: str) -> dict:
    """Fixed answer lines -> prediction. 'parsed' is False when no VERDICT line was found."""

    verdict_raw = _line(answer, "VERDICT")
    verdict = None
    if verdict_raw:
        low = verdict_raw.lower()
        verdict = "Fake" if "fake" in low else "Real" if "real" in low else None
    pred = {"parsed": verdict is not None, "verdict": verdict}
    if dataset == "g360":
        raw = (_line(answer, "TYPES") or "").lower()
        pred["types"] = [t for t in G360_TYPES if t.lower() in raw.replace(" ", "_") or t.lower().replace("_", " ") in raw]
        words = _line(answer, "FALSE_TITLE_WORDS")
        pred["title_words"] = "" if not words or words.lower() in ("none", "n/a") else words.strip("\"'“”")
        pred["false_speech_s"] = parse_spans(_line(answer, "FALSE_SPEECH_TIME"))
        pred["edited_s"] = parse_spans(_line(answer, "EDITED_TIME"))
    else:
        raw = (_line(answer, "TYPE") or "").upper()
        pred["type"] = next((c for c in FMNV_CODES if re.search(rf"\b{c}\b", raw)), None)
    return pred


EXTRACT_PROMPT = {
    "g360": ("You read the final answer of a video fact-checking system and extract its conclusion. Do not judge "
             "correctness. Return JSON: verdict (\"Real\", \"Fake\" or \"unknown\" if the answer gives no conclusion), "
             "types (subset of " + ", ".join(G360_TYPES) + " that the answer says apply), title_words (the title words "
             "the answer calls false, or \"\"), false_speech_s and edited_s (lists of [start, end] seconds the answer "
             "gives, or [])."),
    "fmnv": ("You read the final answer of a news-video fact-checking system and extract its conclusion. Do not judge "
             "correctness. Return JSON: verdict (\"Real\", \"Fake\" or \"unknown\"), type (one of CD, CE, SV, CA, or "
             "\"none\"), where CD = title misrepresents the video, CE = footage cut to mislead, SV = synthetic or "
             "replaced voiceover, CA = absurd or exaggerated claims."),
}


# ---------------------------------------------------------------- metrics

def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def binary_metrics(rows: list[dict]) -> dict:
    """Unparsed / unfinished answers count as wrong (predicted the opposite of the label)."""

    n = len(rows)
    if not n:
        return {}
    correct = sum(1 for r in rows if r["pred_verdict"] == r["label"])
    out = {"n": n, "acc": correct / n}
    f1s = []
    for pos in ("Fake", "Real"):
        tp = sum(1 for r in rows if r["label"] == pos and r["pred_verdict"] == pos)
        fp = sum(1 for r in rows if r["label"] != pos and r["pred_verdict"] == pos)
        fn = sum(1 for r in rows if r["label"] == pos and r["pred_verdict"] != pos)
        p, rr, f = prf(tp, fp, fn)
        out[f"{pos.lower()}_p"], out[f"{pos.lower()}_r"], out[f"{pos.lower()}_f1"] = p, rr, f
        f1s.append(f)
    out["macro_f1"] = sum(f1s) / 2
    out["parsed"] = sum(1 for r in rows if r["pred_verdict"] in ("Real", "Fake")) / n
    return out


def type_metrics(rows: list[dict], types: tuple[str, ...], key_gt: str, key_pred: str) -> dict:
    """Multi-label per-type P/R/F1 over all samples (Real samples have no types); macro over types present in GT."""

    per = {}
    for t in types:
        tp = sum(1 for r in rows if t in r[key_gt] and t in r[key_pred])
        fp = sum(1 for r in rows if t not in r[key_gt] and t in r[key_pred])
        fn = sum(1 for r in rows if t in r[key_gt] and t not in r[key_pred])
        if tp + fn:
            per[t] = dict(zip(("p", "r", "f1"), prf(tp, fp, fn)), support=tp + fn)
    macro = sum(v["f1"] for v in per.values()) / len(per) if per else 0.0
    return {"per_type": per, "macro_f1": macro}


def token_f1(pred: str, gold: list[str]) -> float | None:
    gold_tokens = [w for g in gold for w in re.findall(r"\w+", g.lower())]
    if not gold_tokens:
        return None
    pred_tokens = re.findall(r"\w+", (pred or "").lower())
    if not pred_tokens:
        return 0.0
    common = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in set(gold_tokens))
    if not common:
        return 0.0
    p, r = common / len(pred_tokens), common / len(gold_tokens)
    return 2 * p * r / (p + r)


def span_f1(pred: list[list[float]], gold: list[list[float]], step: float = 0.1) -> float | None:
    """Time-level F1 on a 0.1 s grid (the paper uses frame-level F1)."""

    if not gold:
        return None
    def cells(spans):
        out = set()
        for a, b in spans:
            out.update(range(int(round(float(a) / step)), int(round(float(b) / step))))
        return out
    g, p = cells(gold), cells(pred)
    if not p:
        return 0.0
    tp = len(g & p)
    return prf(tp, len(p - g), len(g - p))[2]
