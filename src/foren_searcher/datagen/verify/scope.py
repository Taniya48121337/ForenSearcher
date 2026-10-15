"""工作区级"该拿什么"的范围（verifier@0.3.0）。

大白话：案卷的 required_evidence 是整个案子该拿到的清单，但一个工作区只领了其中一部分任务。09-23 的召回实验
证明了两头都不对：把整案清单压给单个工作区，每份报告都多出假的 omitted_required_fact；什么都不给，删掉一条
必需 Finding 也抓不到。所以先算出"这个工作区的任务对准哪几项要求（REQ）"，再由 Case.scoped_required_evidence
反查出该工作区该拿的必需证据，交给第一层（程序匹配 url_pattern / acquired_by）和第三层（模型看描述）。

REQ 从哪来（按优先级）：
1. plan 单元的模型评判 gap.judge_targets（complete_units 产出的 verdicts_units_judged.jsonl；W{r}-{i} ↔ T{i}）；
2. 没有就按任务文字与 REQ 文字的字符二元组重合率兜底（阈值 0.25），并把方法记进 extra 便于排查；
3. 案卷没有 requirements → None（不收窄，和以前一样）。
"""

from __future__ import annotations

import re
from typing import Any

from ..cases import Case
from .lang import content_words, is_cjk
from .verdict import Verdict

_WS_RE = re.compile(r"^W(\d+)-(\d+)$")
FALLBACK_THRESHOLD = 0.25


def _grams(text: str) -> set[str]:
    text = re.sub(r"[\s，。；：、“”\"'（）()【】\[\]《》,.;:!?！？]", "", text)
    return {text[i:i + 2] for i in range(len(text) - 1)}


def overlap(requirement_text: str, task_text: str) -> float:
    a, b = _grams(requirement_text), _grams(task_text)
    return len(a & b) / len(a) if a else 0.0


def requirement_overlap(requirement: Any, task_text: str) -> float:
    """英文任务（英文规格，10-07）对 REQ 的英文表述 text_en 按实词比；中文任务照旧按字符二元组比中文表述。"""

    text_en = getattr(requirement, "text_en", None)
    if text_en and not is_cjk(task_text):
        a, b = content_words(text_en), content_words(task_text)
        return len(a & b) / len(a) if a else 0.0
    return overlap(requirement.text, task_text)


def targets_by_text(case: Case, task_text: str, threshold: float = FALLBACK_THRESHOLD) -> set[str]:
    return {r.id for r in case.requirements if requirement_overlap(r, task_text) >= threshold}


def judge_targets_from_units(units: list[Verdict], run: str, workspace_id: str) -> set[str] | None:
    """从 plan 单元的模型评判里取该工作区对应任务的 targets；没有就 None。"""

    m = _WS_RE.match(workspace_id)
    if not m:
        return None
    round_index, task_index = int(m.group(1)), int(m.group(2))
    for v in units:
        if v.unit != "plan" or v.subject.get("run") != run or int(v.subject.get("round") or 0) != round_index:
            continue
        judge = (v.extra.get("gap") or {}).get("judge_targets") or {}
        targets = judge.get(f"T{task_index}")
        if isinstance(targets, list):
            return {str(t) for t in targets}
    return None


def requirements_for_workspace(case: Case | None, task_text: str, *, run: str, workspace_id: str,
                               units: list[Verdict] | None = None) -> tuple[set[str] | None, dict[str, Any]]:
    """返回 (requirement_ids | None, 说明)。None 表示不收窄。"""

    if case is None or not case.requirements:
        return None, {"method": "none", "reason": "案卷无 requirements"}
    judged = judge_targets_from_units(units or [], run, workspace_id)
    if judged is not None:
        return judged, {"method": "judge_targets", "requirement_ids": sorted(judged)}
    by_text = targets_by_text(case, task_text)
    return by_text, {"method": "text_overlap", "requirement_ids": sorted(by_text),
                     "scores": {r.id: round(requirement_overlap(r, task_text), 2) for r in case.requirements}}


__all__ = ["FALLBACK_THRESHOLD", "judge_targets_from_units", "overlap", "requirement_overlap", "requirements_for_workspace", "targets_by_text"]
