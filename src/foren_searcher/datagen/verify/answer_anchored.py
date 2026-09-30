"""answer 单元的逐句锚定（接核验模型后替代 units.check_answer 里的报告级代理）。

大白话：GIA 的 FINISH 答案是几段话，每句后面挂着"（E24、E21）"/"[E3][E6]"/"【E5】"这样的引用。这里把答案拆成句子，
把每句和它挂的 E 配成 (句, E) 对，然后复用第二层 anchored 的"部分支持 + 覆盖汇总"核验：
每条 E 只看自己能覆盖的部分、引语程序回查，最后问"这句还有哪部分没人撑"。
产物写回 answer 单元 Verdict：anchored.method 从 proxy_report_judgments 变成 judge_sentences，
proxy 的 flag 错误被模型判定取代，处置重算。

引用作用域：一组"（E…）"覆盖它紧跟的那句，以及往前直到上一组引用为止的同段句子（教师写法就是
"两三句一组引用"）。只引 F 的句子，把 F 换成该 Finding 的 evidence_ids。什么都没引的句子记 uncited，
不做锚定，也不报错（"结论：…不实"这类总括句通常是这种）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import ErrorRecord, ErrorType
from .anchored import AnchoredResult, JsonJudge, verify_anchored
from .context import EvidenceLookup, EvidenceText
from .hindsight import UsefulEvidence
from .verdict import Verdict, decide

if TYPE_CHECKING:
    from ..runlog import Decision, RunLog

# 一个引用组 = 一对括号（（）/()/[]/【】）里的一串 E/F；**相邻**的几组（中间只隔空白，如 "[E3][E6][E10]"）合并成一组。
# group(1) 是整段（含括号），_ID.findall 取出其中全部 E/F。
_ONE_CITE = r"[（(\[【]\s*(?:[EF]\d+\s*[、,，;；\s]*)+[）)\]】]"
_CITE_GROUP = re.compile(rf"({_ONE_CITE}(?:\s*{_ONE_CITE})*)")
_ID = re.compile(r"[EF]\d+")
_TERMINATOR = re.compile(r"(?<=[。！？!?；;])")
_SKIP = re.compile(r"^(结论|判定|标签|总结)[:：]?", re.I)


@dataclass
class AnswerSentence:
    index: int
    text: str                                   # 去掉引用组后的句子
    evidence_ids: list[str] = field(default_factory=list)
    finding_ids: list[str] = field(default_factory=list)
    scope: str = "own"                          # own | carried_back | uncited
    paragraph: int = 0

    @property
    def sentence_id(self) -> str:
        return f"A{self.index}"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.sentence_id, "text": self.text, "evidence_ids": list(self.evidence_ids),
                "finding_ids": list(self.finding_ids), "scope": self.scope, "paragraph": self.paragraph}


def split_answer(answer: str, finding_evidence: dict[str, list[str]] | None = None) -> list[AnswerSentence]:
    """把答案拆成带引用的句子。finding_evidence: F → 该 Finding 的 evidence_ids（把只引 F 的句子换成 E）。"""

    finding_evidence = finding_evidence or {}
    out: list[AnswerSentence] = []
    for p_index, paragraph in enumerate(re.split(r"\n+", answer)):
        if not paragraph.strip():
            continue
        pieces = [s for s in _TERMINATOR.split(paragraph) if s.strip()]
        # 段内：句子 → 引用组附着；"（E…）"在句首时属于上一句
        pending: list[AnswerSentence] = []           # 尚未拿到引用的句子（等着被后面的引用组"回溯覆盖"）
        for piece in pieces:
            groups = _CITE_GROUP.findall(piece)
            leading = _CITE_GROUP.match(piece.strip())
            text = _CITE_GROUP.sub("", piece).strip()
            ids_here = [i for g in groups for i in _ID.findall(g)]
            if leading and pending:
                # 句首引用组 → 结算前面待引用的句子
                lead_ids = _ID.findall(leading.group(1))
                _settle(pending, lead_ids, finding_evidence)
                out.extend(pending)
                pending = []
                ids_here = [i for i in ids_here if i not in lead_ids] if len(groups) > 1 else []
            if not text:
                continue
            sentence = AnswerSentence(index=len(out) + len(pending) + 1, text=text, paragraph=p_index)
            pending.append(sentence)
            if ids_here:
                _settle(pending, ids_here, finding_evidence)
                out.extend(pending)
                pending = []
        for sentence in pending:
            sentence.scope = "uncited"
        out.extend(pending)
    for i, sentence in enumerate(out, start=1):
        sentence.index = i
    return out


def _settle(pending: list[AnswerSentence], ids: list[str], finding_evidence: dict[str, list[str]]) -> None:
    e_ids = [i for i in ids if i.startswith("E")]
    f_ids = [i for i in ids if i.startswith("F")]
    for f in f_ids:
        for e in finding_evidence.get(f, []):
            if e not in e_ids:
                e_ids.append(e)
    for k, sentence in enumerate(pending):
        sentence.evidence_ids = list(e_ids)
        sentence.finding_ids = list(f_ids)
        sentence.scope = "own" if k == len(pending) - 1 else "carried_back"


def as_report(sentences: list[AnswerSentence]) -> dict[str, Any]:
    """拼成 anchored 层能吃的伪报告：每句一条 Finding。"""

    return {"findings": [{"finding_id": s.sentence_id, "statement": s.text, "evidence_ids": list(s.evidence_ids)}
                         for s in sentences if s.evidence_ids]}


class _WithRequestImages:
    """Store 的 artifact 不在（比如 run 目录只带了 requests/ 没带 artifacts/）时，用教师当时实际收到的附图文件补上像素。"""

    def __init__(self, inner: EvidenceLookup, images: dict[str, Path]) -> None:
        self.inner, self.images = inner, images

    def has(self, evidence_id: str) -> bool:
        return self.inner.has(evidence_id)

    def get(self, evidence_id: str) -> EvidenceText:
        text = self.inner.get(evidence_id)
        if text.is_image and not text.image_path and evidence_id in self.images:
            return replace(text, image_path=self.images[evidence_id])
        return text


def request_images(log: RunLog) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in log.decisions:
        for eid, path in zip(d.image_evidence_ids, d.image_paths):
            if eid not in out and path.is_file():
                out[eid] = path
    return out


async def anchor_answer(decision: Decision, log: RunLog, useful: UsefulEvidence, judge: JsonJudge) -> tuple[list[AnswerSentence], AnchoredResult]:
    finding_evidence = {fid: list(row["evidence_ids"]) for fid, row in useful.clean_findings.items()}
    for rec in log.store.workspace_report_records():
        for finding in rec.to_dict().get("findings") or []:
            fid = str(finding.get("finding_id"))
            finding_evidence.setdefault(fid, [x for x in finding.get("evidence_ids") or [] if isinstance(x, str)])
    sentences = split_answer(str(decision.output.get("answer") or ""), finding_evidence)
    skip = {e for s in sentences for e in s.evidence_ids if e not in decision.visible_ids}
    lookup = _WithRequestImages(log.lookup, request_images(log))
    result = await verify_anchored(as_report(sentences), lookup, judge, skip_evidence_ids=skip)
    return sentences, result


def apply_answer_anchoring(verdict: Verdict, sentences: list[AnswerSentence], result: AnchoredResult, *,
                           judge_model: str | None) -> Verdict:
    """把逐句锚定结果写回 answer 单元：换掉 proxy 错误与 anchored 字段，重算处置。原地修改并返回。"""

    if verdict.unit != "answer":
        raise ValueError("apply_answer_anchoring 只用于 answer 单元")
    kept: list[ErrorRecord] = [e for e in verdict.errors if not (e.method == "exec" and e.extra.get("confidence") == "flag"
                                                                 and e.type is ErrorType.UNSUPPORTED_ASSERTION)]
    judged_errors: list[ErrorRecord] = []
    for e in result.errors:
        # 答案句的 scope_overreach 提示保持 minor；其余按 anchored 层默认严重度
        judged_errors.append(replace(e, extra={**e.extra, "sentence": e.extra.get("finding")}))
    verdict.errors = kept + judged_errors
    verdict.pending = [p for p in verdict.pending if p.get("reason") != "anchored_pending_judge"] + list(result.pending)
    verdict.judgments = [j.to_dict() for j in result.judgments]
    verdict.coverage = list(result.coverage)
    verdict.judge_model = judge_model
    verdict.layers_run = [x for x in verdict.layers_run if x != "proxy_anchored"] + ["anchored_sentences"]
    supported: dict[str, list[str]] = {}
    for j in result.judgments:
        if j.verdict == "supported":
            supported.setdefault(j.finding_id, []).append(j.evidence_id)
    unsupported_sentences = sorted({str(e.extra.get("sentence")) for e in judged_errors if e.type is ErrorType.UNSUPPORTED_ASSERTION})
    contradicted_sentences = sorted({str(e.extra.get("sentence")) for e in judged_errors
                                     if e.type in (ErrorType.WRONG_CITATION, ErrorType.MISREAD_FINE_TEXT, ErrorType.SLOPPY_NUMERIC)})
    old = verdict.extra.get("anchored") or {}
    verdict.extra["anchored"] = {
        "method": "judge_sentences", "cited": old.get("cited"), "pending_judge": bool(result.pending),
        "sentences": [s.to_dict() for s in sentences],
        "uncited_sentences": [s.sentence_id for s in sentences if s.scope == "uncited"],
        "supported_by_sentence": supported, "unsupported_sentences": unsupported_sentences,
        "contradicted_sentences": contradicted_sentences,
        "proxy_before": {k: old.get(k) for k in ("supported_via_report", "unsupported_cites")},
    }
    verdict.extra["pending_judge"] = [p for p in verdict.extra.get("pending_judge", []) if p.get("reason") != "anchored_pending_judge"]
    verdict.disposition = decide(verdict.errors, verdict.pending)
    return verdict


def summarize_answer(verdict: Verdict) -> dict[str, Any]:
    a = verdict.extra.get("anchored") or {}
    return {"method": a.get("method"), "disposition": verdict.disposition,
            "sentences": len(a.get("sentences") or []), "uncited": len(a.get("uncited_sentences") or []),
            "unsupported": a.get("unsupported_sentences"), "contradicted": a.get("contradicted_sentences"),
            "errors": [e.type.value for e in verdict.errors], "severities": [e.severity.value for e in verdict.errors]}


__all__ = ["AnswerSentence", "anchor_answer", "apply_answer_anchoring", "as_report", "request_images", "split_answer", "summarize_answer"]
