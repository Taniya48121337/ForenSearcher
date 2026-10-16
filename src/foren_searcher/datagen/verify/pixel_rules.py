"""verifier@0.4.0 第 3 处：像素检查只对句子所指的那张图（r0g2 金标 F006 / F007 / F009 / F012）。

第二层逐对 (Finding, E) 判定时，一句话引了几张图，每张图都被拿去核整句：
- F012：句子说"下方中间那位参与者的成对裁剪图里 A 有青色帽子"，核验模型拿 E21（左上参与者的整帧对照面板）去核，判"不是青色帽子"；
- F006 / F009：句子说"比对结果报告 A 的半秒样本 100% 在 B 里找到"（compare_media 原话，有归因），差异图里红框有一处 A 有 B 无，
  就被判"与 100% 匹配相反"；
- F007：句子描述的是给定图片 E1，却把三张参考图（外部网页上完好的高速公路、信息图）也列进了引用，每张都判一遍、记 5 条 critical。

这里的程序规则只作用于第二层像素层（method=pixels）的反驳类错误（wrong_citation / sloppy_numeric / misread_fine_text），不调模型：
1. tool_measurement：被反驳的部分是归因给工具的测量值（"the comparison reports 100% …""audio matches in 9 of 9 windows"），
   且同句所引的工具文本记录支持它 → 降为 minor。像素只能反驳对画面内容的直接描述；
2. other_image_referent：一句引了两张以上图，按句子里的图类型（裁剪图 / 对照面板）或取帧时间，这部分说的是同句引的另一张图 → 降为 minor；
3. cited_unrelated_image：这张图来自外部网页，而同句引的输入材料（给定图片 / 视频帧）已支持这部分 → 句子描述的是材料，外部图只是
   引错了；同一 Finding 里这类反驳合并成一条 minor wrong_citation。
都只降级、合并并留痕（extra.program_rule），不删错误。
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Iterable

from ..errors import ErrorRecord, ErrorType, Severity, error
from .lang import grams, split_sentences

_CONTRA = (ErrorType.WRONG_CITATION, ErrorType.SLOPPY_NUMERIC, ErrorType.MISREAD_FINE_TEXT)
_PUNCT = re.compile(r"[\s　，。、；：:,.;!?！？（）()\[\]【】“”\"'‘’《》<>—–\-·…]+")

# 句子里指图的类型词
_CROP = re.compile(r"\bcrop(?:s|ped)?\b|close-?ups?\b|裁剪|特写", re.I)
_PANEL = re.compile(r"\bpanels?\b|comparison (?:images?|panels?|frames?)|difference (?:images?|maps?|panels?)|\bdiff images?\b"
                    r"|whole-frame|差异图|对照图|对照面板", re.I)
# 归因给工具的测量值
_TOOL_SUBJECT = re.compile(r"\b(?:comparison|compare_media|alignment|align_media|tool|result|retrieval|locate_frames|analysis|ocr|asr"
                           r"|detector|search)\b|比对|工具|检索", re.I)
_TOOL_VERB = re.compile(r"\b(?:reports?|reported|finds?|found|match(?:es|ed)?|shows?|says|lists?|flags?|flagged|gives?|returns?"
                        r"|returned|scores?|measures?|estimates?)\b|according to|报告|显示|给出|返回", re.I)
_MEASURE = re.compile(r"\d+(?:\.\d+)?\s*%|\b\d+\s+(?:of|/)\s+\d+\b|\bsamples?\b|\bwindows?\b|\bscores?\b|\bsimilarity\b|相似度|样本|窗口", re.I)
_SECONDS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?|秒)\b", re.I)
_FRAME_TIME = re.compile(r"第\s*(\d+(?:\.\d+)?)\s*秒|at\s+(\d+(?:\.\d+)?)\s*s\b", re.I)


def _norm(text: str) -> str:
    return _PUNCT.sub("", text or "").lower()


def _grams(text: str) -> set[str]:
    return set(grams(text or "", _norm))


def overlap(part: str, texts: Iterable[str]) -> float:
    """part 的实词（中文为字符二元组）有多大比例出现在 texts 里。"""

    mine = _grams(part)
    if not mine:
        return 0.0
    theirs: set[str] = set()
    for t in texts:
        theirs |= _grams(t)
    return len(mine & theirs) / len(mine)


def image_type(j: Any) -> str:
    """crop / diff_panel / frame / material / web / other（按 E 的标题与来源归属）。"""

    label = str(getattr(j, "source_label", "") or "").lower()
    provenance = str(getattr(j, "provenance", "") or "")
    if "cropped region" in label or re.search(r"\(crop", label):
        return "crop"
    if "region in red differs" in label or "differs from b" in label or re.search(r"\bpanel\b", label):
        return "diff_panel"
    if "frame of video" in label or "的帧" in label:
        return "frame"
    if provenance.startswith("来源材料") and "原始材料" in provenance:
        return "material"
    if provenance.startswith("来源：外部网页"):
        return "web"
    return "other"


def frame_time(j: Any) -> float | None:
    """单帧图的取帧时间（media_note 的"第 X 秒"或标题里的"at X s"）；对照面板这类区间图返回 None。"""

    for text in (str(getattr(j, "media_note", "") or ""), str(getattr(j, "source_label", "") or "")):
        m = _FRAME_TIME.search(text)
        if m:
            return float(m.group(1) or m.group(2))
    return None


def sentence_of(statement: str, part: str) -> str:
    """part 所在的那一句（找不到就按实词重合最多的一句）；statement 只有一句时就是整句。"""

    sentences = split_sentences(statement or "") or [statement or ""]
    target = _norm(part)
    for s in sentences:
        if target and target in _norm(s):
            return s
    return max(sentences, key=lambda s: overlap(part, [s]))


def referent_types(text: str) -> set[str]:
    out = set()
    if _CROP.search(text or ""):
        out.add("crop")
    if _PANEL.search(text or ""):
        out.add("diff_panel")
    return out


def is_tool_measurement(part: str, sentence: str) -> bool:
    """被反驳的部分是不是归因给工具的测量值：句中有工具主语 + 报告类动词，部分本身带测量值（百分比、N of M、样本 / 窗口）。"""

    text = f"{sentence} {part}"
    return bool(_TOOL_SUBJECT.search(text) and _TOOL_VERB.search(text) and _MEASURE.search(part or ""))


def _other_referent(j: Any, part: str, sentence: str, images: list[Any]) -> list[str]:
    others = [k for k in images if k is not j]
    types = referent_types(part) or (referent_types(sentence) if len(referent_types(sentence)) == 1 else set())
    if types and image_type(j) not in types:
        hits = [k.evidence_id for k in others if image_type(k) in types]
        if hits:
            return hits
    times = [float(x) for x in _SECONDS.findall(f"{part} {sentence}")]
    mine = frame_time(j)
    if times and mine is not None and not any(abs(t - mine) <= 0.15 for t in times):
        hits = [k.evidence_id for k in others if (kt := frame_time(k)) is not None and any(abs(t - kt) <= 0.15 for t in times)]
        if hits:
            return hits
    return []


def _material_support(part: str, j: Any, images: list[Any], threshold: float = 0.6) -> list[str]:
    if image_type(j) != "web":
        return []
    return [k.evidence_id for k in images if k is not j and image_type(k) in ("material", "frame", "crop")
            and str(getattr(k, "provenance", "")).startswith("来源材料") and getattr(k, "supported", None)
            and overlap(part, [p.part for p in k.supported]) >= threshold]


def _text_support(part: str, judgments: list[Any], threshold: float = 0.5) -> list[str]:
    return [k.evidence_id for k in judgments if not k.is_image and k.supported
            and overlap(part, [f"{p.part} {p.quote}" for p in k.supported]) >= threshold]


def _demote(e: ErrorRecord, rule: str, note: str, **info: Any) -> ErrorRecord:
    return replace(e, severity=Severity.MINOR, detail=f"（程序规则：{note}，仅备注）" + e.detail,
                   extra={**e.extra, "program_rule": rule, "original_severity": e.severity.value, **info})


def apply_pixel_rules(errors: list[ErrorRecord], statement: str, judgments: list[Any]) -> list[ErrorRecord]:
    """一条 Finding 的错误表 → 应用三条规则后的错误表（顺序不变；cited_unrelated_image 合并成的一条放在最后）。"""

    by_id = {j.evidence_id: j for j in judgments}
    images = [j for j in judgments if j.is_image]
    out: list[ErrorRecord] = []
    unrelated: list[tuple[ErrorRecord, list[str]]] = []
    for e in errors:
        j = by_id.get(e.evidence_ids[0]) if len(e.evidence_ids) == 1 else None
        if j is None or e.method != "pixels" or e.type not in _CONTRA or e.severity is Severity.MINOR:
            out.append(e)
            continue
        part = str(e.span or "")
        sentence = sentence_of(statement, part)
        tools = _text_support(part, judgments) if is_tool_measurement(part, sentence) else []
        if tools:
            out.append(_demote(e, "tool_measurement", "归因给工具的测量值，像素不反驳", supported_by=tools))
            continue
        refs = _other_referent(j, part, sentence, images) if len(images) >= 2 else []
        if refs:
            out.append(_demote(e, "other_image_referent", "这部分说的是同句所引的另一张图", refers_to=refs))
            continue
        material = _material_support(part, j, images)
        if material:
            unrelated.append((e, material))
            continue
        out.append(e)
    if unrelated:
        eids = list(dict.fromkeys(e.evidence_ids[0] for e, _ in unrelated))
        refs = list(dict.fromkeys(k for _, ks in unrelated for k in ks))
        worst = max((e.severity for e, _ in unrelated), key=lambda s: ("minor", "major", "critical").index(s.value))
        finding = unrelated[0][0].extra.get("finding")
        out.append(error(ErrorType.WRONG_CITATION,
                         f"引用了与该句所述画面无关的图 {'、'.join(eids)}（句子描述的是 {'、'.join(refs)}）；合并 {len(unrelated)} 条像素反驳，仅记一条",
                         span=statement, evidence_ids=eids, method="pixels", severity=Severity.MINOR,
                         extra={"finding": finding, "program_rule": "cited_unrelated_image", "original_severity": worst.value,
                                "merged": len(unrelated), "described": refs,
                                "quotes": [str(e.extra.get("quote") or "")[:200] for e, _ in unrelated]}))
    return out


__all__ = ["apply_pixel_rules", "frame_time", "image_type", "is_tool_measurement", "overlap", "referent_types", "sentence_of"]
