"""Strengthen H7 with two failures missed by the original lexical audit.

This is not a new gate.  UTQR-9 H7 already rejects truncation artifacts and
reasoning that narrates process instead of the student-visible evidence.  The
v1 audit did not recognize unbalanced inline JSON or several workflow-log
phrases, so structurally broken text could still pass.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

from audit_recovery_reasoning_quality import audit_rows as audit_rows_v1
from pipeline import ROOT, write_json
from shadow_execute_generated_actions import _jsonl


DEFAULT_INPUT = (
    ROOT
    / "outputs/student_external_contract_v1/recovery_batch_candidate_v14_humanlike/recovery_sft_candidate.jsonl"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.parent / "reasoning_quality_audit_v2.json"

META_PROCESS = re.compile(
    r"(?:Harness can verify|last tool call|visible routing state|"
    r"Binding validation|terminal verification|tests? the complete state|"
    r"success from tool availability)",
    re.I,
)


def _assistant_text(row: dict[str, Any]) -> str:
    return str((row.get("messages") or [{}])[-1].get("content") or "").strip()


def audit_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report = copy.deepcopy(audit_rows_v1(rows))
    meta_examples: list[dict[str, Any]] = []
    unbalanced_examples: list[dict[str, Any]] = []
    failed_rows: set[str] = set()
    for index, row in enumerate(rows):
        metadata = row.get("metadata") or {}
        row_id = str(metadata.get("row_id") or f"row-{index + 1}")
        text = _assistant_text(row)
        meta = sorted({match.group(0) for match in META_PROCESS.finditer(text)})
        if meta:
            failed_rows.add(row_id)
            meta_examples.append(
                {
                    "row_id": row_id,
                    "stage": metadata.get("causal_recovery_stage"),
                    "hits": meta,
                    "reasoning": text,
                }
            )
        delimiters = {
            "curly": [text.count("{"), text.count("}")],
            "square": [text.count("["), text.count("]")],
            "round": [text.count("("), text.count(")")],
        }
        unbalanced = {
            name: counts for name, counts in delimiters.items() if counts[0] != counts[1]
        }
        if unbalanced:
            failed_rows.add(row_id)
            unbalanced_examples.append(
                {
                    "row_id": row_id,
                    "stage": metadata.get("causal_recovery_stage"),
                    "unbalanced": unbalanced,
                    "reasoning": text,
                }
            )

    total = len(rows)
    metrics = report["metrics"]
    metrics.update(
        {
            "meta_process_narration_hits": len(meta_examples),
            "unbalanced_inline_serialization_hits": len(unbalanced_examples),
            "domain_focused_reasoning": total - len(meta_examples),
            "balanced_inline_serialization": total - len(unbalanced_examples),
            "strict_additional_failure_rows": len(failed_rows),
            "strict_additional_check_passed_rows": total - len(failed_rows),
        }
    )
    checks = report["checks"]
    checks.update(
        {
            "all_reasoning_domain_focused": not meta_examples,
            "all_inline_serialization_balanced": not unbalanced_examples,
        }
    )
    report.update(
        {
            "artifact": "causal_recovery_reasoning_quality_audit_v2",
            "audit_extension": (
                "Recognizes unbalanced serialized evidence and workflow-log narration "
                "under the existing UTQR-9 H7 truncation/grounding contract."
            ),
            "passed": bool(report["passed"] and all(checks.values())),
            "v2_failure_examples": {
                "meta_process_narration": meta_examples[:30],
                "unbalanced_inline_serialization": unbalanced_examples[:30],
            },
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit_rows(_jsonl(args.input))
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "rows": report["metrics"]["rows"],
                "row_passed": report["metrics"]["row_passed"],
                "meta_process_narration_hits": report["metrics"][
                    "meta_process_narration_hits"
                ],
                "unbalanced_inline_serialization_hits": report["metrics"][
                    "unbalanced_inline_serialization_hits"
                ],
                "passed": report["passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
