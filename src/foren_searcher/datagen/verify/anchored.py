"""第二层：证据锚定的模型核验。逐条 Finding × 逐条所引 E，三态判定 + 必须给引语，引语程序回查。

核验模型只看"这句话"和"这条 E 的全文/像素"，不看演员当时的桌面、不看任务背景，避免被上下文带偏
（CaRR / Rulers 的做法）。它不打分，只回答：原文支持吗、反驳吗、还是根本没讲？
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..errors import ErrorRecord, ErrorType, Severity, error
from .context import EvidenceLookup, EvidenceText

MAX_SOURCE_CHARS = 20_000
VERDICTS = ("supported", "contradicted", "not_found", "needs_pixels")

ANCHOR_SYSTEM_PROMPT = """你是证据核对员。只依据给出的"来源全文"（或图片）判断"待核句"是否成立，不使用任何外部知识。
输出一个 JSON 对象：
{"verdict":"supported|contradicted|not_found",
 "quote":"来源中逐字支持或反驳待核句的一段原文（必须是原文的连续片段，不要改写、不要翻译；图片来源则写出你看到的具体文字/画面）",
 "source_reports_claim":true|false,   // 来源只是在转述别人的说法（"据称""某人表示"），而非自身断言
 "statement_asserts_fact":true|false, // 待核句把该内容当作事实断言，而不是写成"来源称"
 "error_hint":null|"misread_fine_text"|"scope_overreach"|"sloppy_numeric"}
规则：
- supported：来源明确支持待核句的全部可核部分；只支持一部分算 not_found，并在 quote 里写出支持的那部分。
- contradicted：来源与待核句相反或数字/日期/主体不一致。
- not_found：来源没有谈到这件事，或者只有相关但不能支撑的内容。
- 待核句由几条来源共同支持时，只判断这一条来源能支持的部分。
- 来源是图片时，只报告能直接看到的内容；看不清的细小文字不要猜，写 not_found。
- error_hint：数字口径/单位不一致填 sloppy_numeric；用少数帧/局部推断整体填 scope_overreach；图片细字读错填 misread_fine_text。"""


class JsonJudge(Protocol):
    """核验模型接口：给 system + payload（+ 可选图片），返回 JSON 对象。与教师模型分离。"""

    async def judge(self, *, system_prompt: str, payload: dict[str, Any], images: list[Path]) -> dict[str, Any]: ...


@dataclass
class Judgment:
    finding_id: str
    evidence_id: str
    verdict: str
    quote: str = ""
    quote_verified: bool | None = None       # None = 图片来源，无法程序回查
    source_reports_claim: bool = False
    statement_asserts_fact: bool = True
    error_hint: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "raw"}


@dataclass
class AnchoredResult:
    errors: list[ErrorRecord] = field(default_factory=list)
    judgments: list[Judgment] = field(default_factory=list)
    pending: list[dict[str, Any]] = field(default_factory=list)     # 需要看像素但核验模型看不了的

    def to_dict(self) -> dict[str, Any]:
        return {"errors": [e.to_dict() for e in self.errors], "judgments": [j.to_dict() for j in self.judgments],
                "pending": list(self.pending)}


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def quote_in_source(quote: str, content: str) -> bool:
    """引语回查：去掉所有空白后做子串匹配（引语长度 ≥ 6 个字符才算数）。"""

    q = _squash(quote)
    return len(q) >= 6 and q in _squash(content)


def _source_payload(text: EvidenceText) -> dict[str, Any]:
    content = text.content
    truncated = False
    if len(content) > MAX_SOURCE_CHARS:
        content = content[:MAX_SOURCE_CHARS]
        truncated = True
    return {"evidence_id": text.evidence_id, "modalities": list(text.modalities), "label": text.label,
            "source_url": text.source_url, "content": content, "content_truncated": truncated}


async def judge_pair(judge: JsonJudge, finding_id: str, statement: str, text: EvidenceText) -> Judgment:
    images = [text.image_path] if text.is_image and text.image_path else []
    payload = {"statement": statement, "source": _source_payload(text), "image_attached": bool(images)}
    if text.is_image and not images:
        return Judgment(finding_id, text.evidence_id, "needs_pixels")
    raw = await judge.judge(system_prompt=ANCHOR_SYSTEM_PROMPT, payload=payload, images=images)
    verdict = str(raw.get("verdict") or "not_found")
    if verdict not in VERDICTS:
        verdict = "not_found"
    quote = str(raw.get("quote") or "")
    verified: bool | None
    if text.is_text and verdict in ("supported", "contradicted"):
        verified = quote_in_source(quote, text.content)
        if not verified:
            verdict = "not_found"            # 引语对不上原文：不采信
    else:
        verified = None
    return Judgment(finding_id, text.evidence_id, verdict, quote=quote, quote_verified=verified,
                    source_reports_claim=bool(raw.get("source_reports_claim")),
                    statement_asserts_fact=bool(raw.get("statement_asserts_fact", True)),
                    error_hint=raw.get("error_hint") if raw.get("error_hint") in ("misread_fine_text", "scope_overreach", "sloppy_numeric") else None,
                    raw=dict(raw))


def _errors_for_finding(finding_id: str, statement: str, judgments: list[Judgment]) -> list[ErrorRecord]:
    out: list[ErrorRecord] = []
    supported = [j for j in judgments if j.verdict == "supported"]
    contradicted = [j for j in judgments if j.verdict == "contradicted"]
    not_found = [j for j in judgments if j.verdict == "not_found"]
    pending = [j for j in judgments if j.verdict == "needs_pixels"]
    base = {"span": statement, "extra": {"finding": finding_id}}

    def method(j: Judgment) -> str:
        return "pixels" if j.quote_verified is None and j.verdict != "needs_pixels" else "model"

    for j in contradicted:
        hint = {"misread_fine_text": ErrorType.MISREAD_FINE_TEXT, "sloppy_numeric": ErrorType.SLOPPY_NUMERIC}.get(j.error_hint or "")
        out.append(error(hint or ErrorType.WRONG_CITATION, f"{j.evidence_id} 的原文与该句相反：{j.quote[:200]}",
                         evidence_ids=[j.evidence_id], method=method(j), **base))
    if not supported and not contradicted:
        if pending and not not_found:
            return out                          # 只剩待看像素的，先不判
        detail = "所引 E 均未支持该句" + ("（另有图片证据待看像素）" if pending else "")
        out.append(error(ErrorType.UNSUPPORTED_ASSERTION, detail, evidence_ids=[j.evidence_id for j in not_found], method="model", **base))
        return out
    if supported and not_found:
        out.append(error(ErrorType.WRONG_CITATION, "引用里混入了不支持该句的 E：" + ", ".join(j.evidence_id for j in not_found),
                         severity=Severity.MINOR, evidence_ids=[j.evidence_id for j in not_found], method="model", **base))
    if any(j.source_reports_claim and j.statement_asserts_fact for j in supported):
        out.append(error(ErrorType.ATTRIBUTION_DRIFT, "来源只是转述说法，该句却写成了事实",
                         evidence_ids=[j.evidence_id for j in supported if j.source_reports_claim], method="model", **base))
    if any(j.error_hint == "scope_overreach" for j in supported):
        out.append(error(ErrorType.SCOPE_OVERREACH, "核验模型提示：由局部/少数样本推断了整体",
                         evidence_ids=[j.evidence_id for j in supported if j.error_hint == "scope_overreach"], method="model", **base))
    return out


async def verify_anchored(report: dict[str, Any], lookup: EvidenceLookup, judge: JsonJudge, *,
                          skip_evidence_ids: set[str] | None = None) -> AnchoredResult:
    """对每条 Finding 的每条可读 E 调用核验模型；skip_evidence_ids 是第一层已判为不存在/不可见的 E。"""

    result = AnchoredResult()
    skip = skip_evidence_ids or set()
    for index, finding in enumerate(report.get("findings") or []):
        if not isinstance(finding, dict):
            continue
        finding_id = str(finding.get("finding_id") or f"finding[{index}]")
        statement = str(finding.get("statement") or "")
        judgments: list[Judgment] = []
        for eid in finding.get("evidence_ids") or []:
            if not isinstance(eid, str) or eid in skip or not lookup.has(eid):
                continue
            text = lookup.get(eid)
            if not text.is_text and not text.is_image:
                continue                         # 视频/音频原件：本层不判，靠派生的帧/转写
            judgment = await judge_pair(judge, finding_id, statement, text)
            judgments.append(judgment)
            if judgment.verdict == "needs_pixels":
                result.pending.append({"finding": finding_id, "evidence_id": eid, "reason": "image_without_pixels"})
        result.judgments.extend(judgments)
        if judgments:
            result.errors.extend(_errors_for_finding(finding_id, statement, judgments))
    return result


__all__ = ["ANCHOR_SYSTEM_PROMPT", "AnchoredResult", "JsonJudge", "Judgment", "judge_pair", "quote_in_source", "verify_anchored"]
