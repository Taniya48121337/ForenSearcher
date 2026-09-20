"""第二层：证据锚定的模型核验（v2：按"部分"判定，再按 Finding 汇总覆盖）。

第一版对每条 (Finding, E) 只给一个三态，规则"只支持一部分算 not_found"在真实报告上失灵：报告里的句子几乎
都是复合句（"41 秒可见餐盒；48 秒女子举盒；51.48 秒食物不可见"），每条 E（一帧、一页）天然只支持其中一部分，
结果 85% 的判定是 not_found、92% 的报告被判负。这不是报告的问题，是核验器的问题。

现在分两步：
1. 逐对 (Finding, E)：核验模型列出这条 E **支持的部分**和**反驳的部分**，每一部分都带原文引语（文本来源
   程序回查引语；图片来源引语是它看到的画面描述）。句子里讲别的材料的部分，这条 E 管不着，不算反驳。
2. 逐 Finding 汇总：把所有 E 支持的部分（引语已核实）交给模型，问"句子里还有哪些实质部分没有任何来源支持"。
   有 → unsupported_assertion，span 是那一部分而不是整句；有被反驳的部分 → wrong_citation（或按提示改记
   misread_fine_text / sloppy_numeric）；转述写成事实 → attribution_drift；以偏概全提示 → scope_overreach。

核验模型只看"这句话"和"这条 E 的全文/像素"，不看任务、不看演员桌面（CaRR / Rulers 的做法）。
模型输出坏 JSON 时重试一次，再坏就记 pending，不让整个 run 报废。
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..errors import ErrorRecord, ErrorType, Severity, error
from .context import EvidenceLookup, EvidenceText

MAX_SOURCE_CHARS = 20_000

PAIR_SYSTEM_PROMPT = """你是证据核对员。只依据给出的"来源"（全文或图片）核对"待核句"，不使用任何外部知识。
待核句通常是复合句，这条来源可能只覆盖其中一部分；你的任务是说清楚它覆盖了哪些部分、怎么覆盖的。
输出一个 JSON 对象（所有字符串里的双引号用中文引号“”代替，避免破坏 JSON）：
{"supported_parts":[{"part":"待核句中被这条来源支持的那一部分（尽量逐字摘自待核句）","quote":"来源中支持它的原文片段（文本来源必须逐字连续摘录，不改写不翻译；图片来源写出你实际看到的具体内容）"}],
 "contradicted_parts":[{"part":"待核句中被这条来源反驳的那一部分","quote":"来源中与之相反的原文/画面","hint":null|"misread_fine_text"|"sloppy_numeric"}],
 "source_reports_claim":true|false,   // 来源只是转述别人的说法（“据称”“某人表示”），不是自身断言
 "statement_asserts_fact":true|false, // 待核句把这些内容当作事实断言，而不是写成“来源称”
 "scope_overreach":true|false,        // 待核句用这条来源（少数帧/局部片段）推断了整体或连续区间
 "notes":"一句话"}
规则：
- 只判断这条来源能覆盖的内容。待核句里讲别的材料、别的时刻、别的页面的部分，不写进 supported 也不写进 contradicted。
- contradicted 只在来源与待核句关于同一对象的说法相反（不同数字、不同日期、不同主体、相反的画面）时使用；
  来源“没提到”不是反驳。来源是视频某一帧时，只能反驳关于这一帧/这个时刻的说法。
- 图片来源：只报告能直接看到的内容；看不清的细小文字不要猜，不放进 supported。
- 数字口径/单位不一致填 hint=sloppy_numeric；图片细字读错填 misread_fine_text。
- 把这条来源能支持的**每一个**部分都列出来（时间、数字、地点、人物、谁说了什么、画面内容各算一部分），不要只列最明显的一个；
  漏列的部分后面会被当成"没有支持"。
- 两个列表都可以为空。"""

COVERAGE_SYSTEM_PROMPT = """你是报告审核员。给你一句调查结论（待核句）和它引用的各条来源已经核实的支持片段。
判断待核句里还有哪些**实质性的可核部分**没有任何来源支持。输出 JSON（字符串里的双引号用“”）：
{"unsupported_parts":["待核句中没有任何支持片段覆盖的实质部分，尽量逐字摘自待核句"],
 "grounded":true|false,   // 待核句的全部实质内容是否都被支持片段覆盖
 "notes":"一句话"}
规则：
- 实质部分 = 可以被证据证实或证伪的事实断言（时间、数字、地点、谁说了什么、画面里有什么）。
- 以下不算实质部分，不要列：限定语和保守措辞（“约”“尚不能确定”“未核验”）、对已支持内容的合理归纳、
  由已支持的数字直接算出的结果、对来源类型的描述（“报道”“帧”）。
- 支持片段已经由程序核实确实出现在来源里，请信任它们；不要因为片段短就认为不支持。
- 拿不准的部分列出来，但 grounded 只在确实缺少支持时才为 false。"""


def _obj(properties: dict[str, Any]) -> dict[str, Any]:
    """OpenRouter strict json_schema：所有字段必填、不允许多余键。"""

    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


_PART = _obj({"part": {"type": "string"}, "quote": {"type": "string"}})
_CPART = _obj({"part": {"type": "string"}, "quote": {"type": "string"},
               "hint": {"type": ["string", "null"], "enum": ["misread_fine_text", "sloppy_numeric", None]}})
PAIR_SCHEMA = _obj({
    "supported_parts": {"type": "array", "items": _PART},
    "contradicted_parts": {"type": "array", "items": _CPART},
    "source_reports_claim": {"type": "boolean"}, "statement_asserts_fact": {"type": "boolean"},
    "scope_overreach": {"type": "boolean"}, "notes": {"type": "string"},
})
COVERAGE_SCHEMA = _obj({"unsupported_parts": {"type": "array", "items": {"type": "string"}},
                        "grounded": {"type": "boolean"}, "notes": {"type": "string"}})


class JsonJudge(Protocol):
    """核验模型接口：给 system + payload（+ 可选图片、+ 可选输出 schema），返回 JSON 对象。与教师模型分离。"""

    async def judge(self, *, system_prompt: str, payload: dict[str, Any], images: list[Path],
                    schema: dict[str, Any] | None = None) -> dict[str, Any]: ...


@dataclass
class Part:
    part: str
    quote: str
    verified: bool | None = None      # None = 图片来源，无法程序回查
    hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"part": self.part, "quote": self.quote, "verified": self.verified, "hint": self.hint}


@dataclass
class Judgment:
    """一对 (Finding, E) 的判定。verdict 是汇总：supported（有核实的支持部分）/ contradicted / not_found / needs_pixels / failed。"""

    finding_id: str
    evidence_id: str
    verdict: str
    supported: list[Part] = field(default_factory=list)
    contradicted: list[Part] = field(default_factory=list)
    rejected: list[Part] = field(default_factory=list)      # 模型给了但引语对不上原文、没采信的部分（排查用）
    source_reports_claim: bool = False
    statement_asserts_fact: bool = True
    scope_overreach: bool = False
    is_image: bool = False
    notes: str = ""
    source_label: str = ""            # 来源标签 + URL，汇总时让模型知道"E14 就是 WFAA 那篇报道"
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"finding_id": self.finding_id, "evidence_id": self.evidence_id, "verdict": self.verdict, "source": self.source_label,
                "supported": [p.to_dict() for p in self.supported], "contradicted": [p.to_dict() for p in self.contradicted],
                "rejected": [p.to_dict() for p in self.rejected], "source_reports_claim": self.source_reports_claim, "statement_asserts_fact": self.statement_asserts_fact,
                "scope_overreach": self.scope_overreach, "is_image": self.is_image, "notes": self.notes}


@dataclass
class AnchoredResult:
    errors: list[ErrorRecord] = field(default_factory=list)
    judgments: list[Judgment] = field(default_factory=list)
    coverage: list[dict[str, Any]] = field(default_factory=list)     # 每条 Finding 的汇总结果
    pending: list[dict[str, Any]] = field(default_factory=list)      # 需要看像素 / 模型输出坏掉 的项

    def to_dict(self) -> dict[str, Any]:
        return {"errors": [e.to_dict() for e in self.errors], "judgments": [j.to_dict() for j in self.judgments],
                "coverage": list(self.coverage), "pending": list(self.pending)}


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text).replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'").lower()


def quote_in_source(quote: str, content: str) -> bool:
    """引语回查：去掉所有空白、统一引号后做子串匹配（引语 ≥ 6 字符）；不是子串时允许按顺序的公共片段合计 ≥ 85%（容忍省略号、标点小改）。"""

    q = _squash(quote)
    if len(q) < 6:
        return False
    c = _squash(content)
    if q in c:
        return True
    if len(q) > 400:
        q = q[:400]
    matcher = difflib.SequenceMatcher(None, c, q, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks() if block.size >= 4)   # 只算 ≥4 字符的连续片段，防碎片凑数
    return matched >= 0.85 * len(q)


def _source_payload(text: EvidenceText) -> dict[str, Any]:
    content = text.content
    truncated = False
    if len(content) > MAX_SOURCE_CHARS:
        content = content[:MAX_SOURCE_CHARS]
        truncated = True
    return {"evidence_id": text.evidence_id, "modalities": list(text.modalities), "label": text.label,
            "derived_from": list(text.derived_from), "source_url": text.source_url,
            "content": content if text.is_text else "（图片来源：请看附图）", "content_truncated": truncated}


async def _call(judge: JsonJudge, system_prompt: str, payload: dict[str, Any], images: list[Path],
                schema: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """调核验模型；坏 JSON 重试一次；仍失败返回 None（由调用方记 pending）。"""

    for _attempt in range(2):
        try:
            raw = await judge.judge(system_prompt=system_prompt, payload=payload, images=images, schema=schema)
        except Exception:  # noqa: BLE001 —— 模型输出坏掉不应让整个 run 报废
            continue
        if isinstance(raw, dict):
            return raw
    return None


def _parts(items: Any, text: EvidenceText) -> list[Part]:
    out = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        part = str(item.get("part") or "").strip()
        quote = str(item.get("quote") or "").strip()
        if not part:
            continue
        verified: bool | None = quote_in_source(quote, text.content) if text.is_text else None
        hint = item.get("hint") if item.get("hint") in ("misread_fine_text", "sloppy_numeric") else None
        out.append(Part(part=part, quote=quote, verified=verified, hint=hint))
    return out


async def judge_pair(judge: JsonJudge, finding_id: str, statement: str, text: EvidenceText) -> Judgment:
    images = [text.image_path] if text.is_image and text.image_path else []
    if text.is_image and not images:
        return Judgment(finding_id, text.evidence_id, "needs_pixels", is_image=True)
    payload = {"statement": statement, "source": _source_payload(text), "image_attached": bool(images)}
    raw = await _call(judge, PAIR_SYSTEM_PROMPT, payload, images, PAIR_SCHEMA)
    if raw is None:
        return Judgment(finding_id, text.evidence_id, "failed", is_image=text.is_image)
    all_supported = _parts(raw.get("supported_parts"), text)
    all_contradicted = _parts(raw.get("contradicted_parts"), text)
    supported = [p for p in all_supported if p.verified is not False]        # 文本来源引语对不上：不采信
    contradicted = [p for p in all_contradicted if p.verified is not False]
    rejected = [p for p in all_supported + all_contradicted if p.verified is False]
    verdict = "contradicted" if contradicted else "supported" if supported else "not_found"
    label = " ".join(x for x in (text.label, text.source_url) if x)
    return Judgment(finding_id, text.evidence_id, verdict, supported=supported, contradicted=contradicted, rejected=rejected,
                    source_label=label,
                    source_reports_claim=bool(raw.get("source_reports_claim")),
                    statement_asserts_fact=bool(raw.get("statement_asserts_fact", True)),
                    scope_overreach=bool(raw.get("scope_overreach")), is_image=text.is_image,
                    notes=str(raw.get("notes") or ""), raw=dict(raw))


async def judge_coverage(judge: JsonJudge, finding_id: str, statement: str, judgments: list[Judgment]) -> dict[str, Any] | None:
    support = [{"evidence_id": j.evidence_id, "source": j.source_label, "part": p.part, "quote": p.quote}
               for j in judgments for p in j.supported]
    payload = {"statement": statement, "supported_parts": support}
    raw = await _call(judge, COVERAGE_SYSTEM_PROMPT, payload, [], COVERAGE_SCHEMA)
    if raw is None:
        return None
    parts = [str(p).strip() for p in (raw.get("unsupported_parts") or []) if isinstance(p, (str, int, float)) and str(p).strip()]
    # 程序兜底：模型列出的"无支持部分"若已逐字出现在某个已核实的支持片段（或其引语）里，不算
    covered_text = _squash(" ".join(f"{p.part} {p.quote}" for j in judgments for p in j.supported))
    parts = [p for p in parts if _squash(p) not in covered_text]
    return {"finding_id": finding_id, "grounded": bool(raw.get("grounded", not parts)) and not parts,
            "unsupported_parts": parts, "notes": str(raw.get("notes") or "")}


def _errors_for_finding(finding_id: str, statement: str, judgments: list[Judgment], coverage: dict[str, Any] | None,
                        *, allow_unsupported: bool = True) -> list[ErrorRecord]:
    out: list[ErrorRecord] = []
    base_extra = {"finding": finding_id}

    def method(j: Judgment) -> str:
        return "pixels" if j.is_image else "model"

    for j in judgments:
        for p in j.contradicted:
            kind = {"misread_fine_text": ErrorType.MISREAD_FINE_TEXT, "sloppy_numeric": ErrorType.SLOPPY_NUMERIC}.get(p.hint or "", ErrorType.WRONG_CITATION)
            out.append(error(kind, f"{j.evidence_id} 与该部分相反：{p.quote[:200]}", span=p.part, evidence_ids=[j.evidence_id],
                             method=method(j), extra={**base_extra, "quote": p.quote[:500]}))
    if coverage is not None and coverage["unsupported_parts"]:
        cited = [j.evidence_id for j in judgments]
        for part in coverage["unsupported_parts"]:
            out.append(error(ErrorType.UNSUPPORTED_ASSERTION, "所引 E 均未支持这一部分", span=part, evidence_ids=cited, method="model",
                             extra={**base_extra, "notes": coverage.get("notes", "")}))
    elif (allow_unsupported and coverage is None and judgments and not any(j.supported for j in judgments)
          and all(j.verdict in ("not_found", "contradicted") for j in judgments)):
        out.append(error(ErrorType.UNSUPPORTED_ASSERTION, "所引 E 均未支持该句", span=statement,
                         evidence_ids=[j.evidence_id for j in judgments], method="model", extra=base_extra))
    supporting = [j for j in judgments if j.supported]
    hedged = re.search(r"据|称|报道|引述|表示|声称|According to|reported|said|claims?", statement) is not None
    if not hedged and any(j.source_reports_claim and j.statement_asserts_fact for j in supporting):
        out.append(error(ErrorType.ATTRIBUTION_DRIFT, "来源只是转述说法，该句却写成了事实", span=statement,
                         evidence_ids=[j.evidence_id for j in supporting if j.source_reports_claim], method="model", extra=base_extra))
    if any(j.scope_overreach for j in supporting):
        out.append(error(ErrorType.SCOPE_OVERREACH, "核验模型提示：由局部/少数样本推断了整体", span=statement,
                         evidence_ids=[j.evidence_id for j in supporting if j.scope_overreach], method="model", extra=base_extra))
    return out


async def verify_anchored(report: dict[str, Any], lookup: EvidenceLookup, judge: JsonJudge, *,
                          skip_evidence_ids: set[str] | None = None) -> AnchoredResult:
    """对每条 Finding：逐条可读 E 判"部分"，再汇总覆盖。skip_evidence_ids 是第一层已判为不存在/不可见的 E。"""

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
            elif judgment.verdict == "failed":
                result.pending.append({"finding": finding_id, "evidence_id": eid, "reason": "judge_output_invalid"})
        result.judgments.extend(judgments)
        usable = [j for j in judgments if j.verdict in ("supported", "contradicted", "not_found")]
        if not usable:
            continue
        if any(j.verdict == "failed" for j in judgments):
            # 有一条 E 没判成，就说不清"没被支持的部分"是真没支持还是那条 E 没看；只报反驳，不报无据
            result.errors.extend(_errors_for_finding(finding_id, statement, usable, None, allow_unsupported=False))
            continue
        coverage = await judge_coverage(judge, finding_id, statement, usable)
        if coverage is None:
            result.pending.append({"finding": finding_id, "reason": "coverage_output_invalid"})
        else:
            result.coverage.append(coverage)
        result.errors.extend(_errors_for_finding(finding_id, statement, usable, coverage))
    return result


__all__ = ["COVERAGE_SCHEMA", "COVERAGE_SYSTEM_PROMPT", "PAIR_SCHEMA", "PAIR_SYSTEM_PROMPT", "AnchoredResult", "JsonJudge", "Judgment", "Part", "judge_coverage",
           "judge_pair", "quote_in_source", "verify_anchored"]
