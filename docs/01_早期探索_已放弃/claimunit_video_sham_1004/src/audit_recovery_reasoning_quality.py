"""High-bar local audit for causal-recovery SFT reasoning text.

This audit never judges correctness from prose alone.  It checks whether the
already executed target action is explained with student-visible evidence,
stage-specific causality, and enough linguistic variation to be useful as SFT.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json
from shadow_execute_generated_actions import _jsonl


DEFAULT_INPUT = (
    ROOT
    / "outputs/student_external_contract_v1/recovery_batch_candidate_v2/recovery_sft_candidate.jsonl"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.parent / "reasoning_quality_audit.json"

EVIDENCE_ID = re.compile(r"\bEA-[A-Z0-9_-]+\b", re.I)
NUMBER = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")
SENTENCE_END = re.compile(r"[.!?](?:\s|$)")
BAD_MARKERS = ("�", "…", "<placeholder>", "TODO", "TBD")


def _assistant(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    message = row["messages"][-1]
    calls = message.get("tool_calls") or []
    if len(calls) != 1:
        return "", {}
    arguments = (calls[0].get("function") or {}).get("arguments") or {}
    return str(message.get("content") or "").strip(), arguments


def _mentioned(text: str, values: list[str]) -> bool:
    return any(value and value in text for value in values)


def _flatten_argument_ids(arguments: dict[str, Any]) -> list[str]:
    result: list[str] = []
    stack: list[Any] = [arguments]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        elif isinstance(value, str) and value.upper().startswith("EA-"):
            result.append(value)
    return result


def _stage_checks(
    stage: str, tool_name: str, text: str, arguments: dict[str, Any]
) -> tuple[bool, bool, list[str]]:
    failures: list[str] = []
    ids = _flatten_argument_ids(arguments)
    fields: list[str] = []
    for key in ("target_fields", "fields"):
        fields.extend(str(item) for item in arguments.get(key) or [])
    fields.extend(
        str(item.get("field"))
        for item in arguments.get("field_updates") or []
        if isinstance(item, dict) and item.get("field")
    )
    fields.extend(
        str(item.get("field"))
        for item in arguments.get("binding_items") or []
        if isinstance(item, dict) and item.get("field")
    )
    lower = text.lower()
    evidence_grounded = bool(EVIDENCE_ID.search(text))
    action_causal = False

    if stage == "challenge":
        evidence_grounded = _mentioned(text, ids)
        action_causal = "challenge" in lower and any(
            word in lower for word in ("fresh measurement", "fresh check", "reinspect")
        )
    elif stage == "fresh_measurement":
        evidence_grounded = bool(EVIDENCE_ID.search(text))
        action_causal = tool_name in text and any(
            word in lower for word in ("re-execute", "measurement", "comparison", "test")
        )
    elif stage == "local_reopen":
        evidence_grounded = bool(EVIDENCE_ID.search(text)) and _mentioned(text, fields)
        action_causal = any(
            word in lower for word in ("reopen", "to open")
        ) and any(word in lower for word in ("only", "unchanged", "independent", "preserve"))
    elif stage == "claim_revision":
        evidence_grounded = bool(EVIDENCE_ID.search(text)) and _mentioned(text, fields)
        action_causal = "revis" in lower and "binding" in lower and "commit" in lower
    elif stage == "evidence_assessment":
        evidence_grounded = _mentioned(text, ids)
        action_causal = any(word in lower for word in ("retain", "reject")) and any(
            word in lower for word in ("bind", "commit", "contradict")
        )
    elif stage == "evidence_adjudication":
        evidence_grounded = _mentioned(text, ids)
        action_causal = "partition" in lower and "bind" in lower
    elif stage == "fresh_binding":
        evidence_grounded = _mentioned(text, ids) and _mentioned(text, fields)
        action_causal = any(
            word in lower for word in ("bind", "attribution")
        ) and "commit" in lower
    elif stage == "recommit":
        evidence_grounded = _mentioned(text, fields)
        action_causal = "commit" in lower and any(
            word in lower for word in ("support", "binding", "close")
        )
    elif stage == "terminal_verification":
        evidence_grounded = any(
            word in lower for word in ("evidenceatom", "support", "obligation", "field")
        )
        action_causal = any(word in lower for word in ("terminal", "stop")) and any(
            word in lower
            for word in (
                "closed",
                "satisfied",
                "no remaining",
                "no open",
                "empty",
                "without",
            )
        )
    else:
        failures.append("unknown_stage")

    if not evidence_grounded:
        failures.append("missing_student_visible_grounding")
    if not action_causal:
        failures.append("missing_action_consequence")
    return evidence_grounded, action_causal, failures


def _normalize(text: str) -> str:
    value = EVIDENCE_ID.sub("<evidence>", text.lower())
    value = NUMBER.sub("<number>", value)
    return " ".join(value.split())


def audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    raw_texts: list[str] = []
    normalized: list[str] = []
    openings: list[str] = []
    by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        metadata = row.get("metadata") or {}
        stage = str(metadata.get("causal_recovery_stage") or "")
        dataset = str(metadata.get("dataset") or "")
        tool_name = str(metadata.get("target_tool") or "")
        text, arguments = _assistant(row)
        failures: list[str] = []
        if not text:
            failures.append("empty_reasoning")
        artifact_hits = [marker for marker in BAD_MARKERS if marker in text]
        if artifact_hits:
            failures.append("truncation_or_placeholder_artifact")
        sentence_count = len(SENTENCE_END.findall(text))
        if text and not 2 <= sentence_count <= 5:
            failures.append("sentence_count_outside_2_to_5")
        word_count = len(re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text))
        if text and not 18 <= word_count <= 180:
            failures.append("reasoning_length_outside_18_to_180_words")
        grounded, causal, stage_failures = _stage_checks(
            stage, tool_name, text, arguments
        )
        failures.extend(stage_failures)
        item = {
            "row_id": metadata.get("row_id"),
            "episode_id": metadata.get("episode_id"),
            "dataset": dataset,
            "stage": stage,
            "tool_name": tool_name,
            "nonempty": bool(text),
            "artifact_hits": artifact_hits,
            "sentence_count": sentence_count,
            "word_count": word_count,
            "student_visible_grounding": grounded,
            "action_consequence": causal,
            "passed": not failures,
            "failures": failures,
            "reasoning": text,
        }
        details.append(item)
        raw_texts.append(text)
        norm = _normalize(text)
        normalized.append(norm)
        openings.append(" ".join(norm.split()[:10]))
        by_stage[stage]["rows"] += 1
        by_dataset[dataset]["rows"] += 1
        if not failures:
            by_stage[stage]["passed"] += 1
            by_dataset[dataset]["passed"] += 1

    raw_counts = Counter(raw_texts)
    normalized_counts = Counter(normalized)
    opening_counts = Counter(openings)
    duplicate_rows = sum(count - 1 for count in raw_counts.values() if count > 1)
    normalized_duplicate_rows = sum(
        count - 1 for count in normalized_counts.values() if count > 1
    )
    total = len(rows)
    metrics = {
        "rows": total,
        "nonempty_reasoning": sum(item["nonempty"] for item in details),
        "artifact_free": sum(not item["artifact_hits"] for item in details),
        "student_visible_grounding": sum(
            item["student_visible_grounding"] for item in details
        ),
        "action_consequence": sum(item["action_consequence"] for item in details),
        "sentence_count_2_to_5": sum(
            2 <= item["sentence_count"] <= 5 for item in details
        ),
        "reasoning_length_18_to_180": sum(
            18 <= item["word_count"] <= 180 for item in details
        ),
        "row_passed": sum(item["passed"] for item in details),
        "exact_duplicate_fraction": round(duplicate_rows / max(total, 1), 4),
        "normalized_template_duplicate_fraction": round(
            normalized_duplicate_rows / max(total, 1), 4
        ),
        "max_opening_share": round(
            max(opening_counts.values(), default=0) / max(total, 1), 4
        ),
        "unique_normalized_templates": len(normalized_counts),
    }
    thresholds = {
        "nonempty_reasoning": total,
        "artifact_free": total,
        "student_visible_grounding": total,
        "action_consequence": total,
        "sentence_count_2_to_5_min_rate": 0.95,
        "reasoning_length_18_to_180_min_rate": 1.0,
        "exact_duplicate_fraction_max": 0.05,
        "normalized_template_duplicate_fraction_max": 0.35,
        "max_opening_share_max": 0.20,
    }
    checks = {
        "all_reasoning_nonempty": metrics["nonempty_reasoning"] == total,
        "all_reasoning_artifact_free": metrics["artifact_free"] == total,
        "all_rows_student_visible_grounded": metrics["student_visible_grounding"]
        == total,
        "all_rows_explain_action_consequence": metrics["action_consequence"] == total,
        "sentence_count_rate": metrics["sentence_count_2_to_5"]
        / max(total, 1)
        >= thresholds["sentence_count_2_to_5_min_rate"],
        "reasoning_length_rate": metrics["reasoning_length_18_to_180"]
        / max(total, 1)
        >= thresholds["reasoning_length_18_to_180_min_rate"],
        "exact_duplicate_rate": metrics["exact_duplicate_fraction"]
        <= thresholds["exact_duplicate_fraction_max"],
        "normalized_template_duplicate_rate": metrics[
            "normalized_template_duplicate_fraction"
        ]
        <= thresholds["normalized_template_duplicate_fraction_max"],
        "opening_concentration": metrics["max_opening_share"]
        <= thresholds["max_opening_share_max"],
    }
    return {
        "artifact": "causal_recovery_reasoning_quality_audit_v1",
        "audit_scope": (
            "Executed assistant targets only; local lexical/grounding/naturalness audit, "
            "not a substitute for blind human review or downstream student gain."
        ),
        "metrics": metrics,
        "thresholds": thresholds,
        "checks": checks,
        "passed": all(checks.values()),
        "by_stage": {key: dict(value) for key, value in sorted(by_stage.items())},
        "by_dataset": {
            key: dict(value) for key, value in sorted(by_dataset.items())
        },
        "top_normalized_templates": [
            {"count": count, "template": template}
            for template, count in normalized_counts.most_common(12)
        ],
        "failure_examples": [item for item in details if not item["passed"]][:30],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit_rows(_jsonl(args.input))
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
