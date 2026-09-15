"""Compile teacher-safe VCSL route selections from a local verifier audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json, write_jsonl
from vcsl_batch import LOCALIZER_SPECS


COMPLEXITY_ORDER = {
    name: index for index, name in enumerate(LOCALIZER_SPECS, start=1)
}


def _eligible(row: dict[str, Any]) -> bool:
    metrics = row.get("verifier_only_metrics")
    return bool(
        row.get("status") == "executed"
        and isinstance(metrics, dict)
        and float(metrics.get("precision", 0.0)) >= 0.85
        and float(metrics.get("recall", 0.0)) >= 0.85
        and float(metrics.get("f1", 0.0)) >= 0.60
    )


def _score(variant: str, row: dict[str, Any]) -> tuple[float, float, float, int]:
    metrics = row["verifier_only_metrics"]
    precision = float(metrics["precision"])
    recall = float(metrics["recall"])
    f1 = float(metrics["f1"])
    return min(precision, recall), f1, precision + recall, -COMPLEXITY_ORDER[variant]


def compile_selection(audit: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    student_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}
    for case in audit["cases"]:
        candidates = [
            (variant, row)
            for variant, row in case["variants"].items()
            if _eligible(row)
        ]
        selected = max(candidates, key=lambda item: _score(*item), default=None)
        audit_rows.append(
            {
                "case_index": case["case_index"],
                "pair_id": case["pair_id"],
                "accepted": selected is not None,
                "selected_localizer": None if selected is None else selected[0],
                "rejection_reason": None if selected is not None else "no_candidate_meets_local_verifier_thresholds",
                "candidates": [
                    {
                        "variant": variant,
                        "status": row.get("status"),
                        "verifier_only_metrics": row.get("verifier_only_metrics"),
                        "eligible": _eligible(row),
                    }
                    for variant, row in case["variants"].items()
                ],
            }
        )
        if selected is None:
            continue
        variant, row = selected
        spec = LOCALIZER_SPECS[variant]
        observation = dict(row["student_visible_observation"])
        student_rows.append(
            {
                "pair_id": case["pair_id"],
                "query_id": case["pair_id"].rsplit("-", 1)[0],
                "reference_id": case["pair_id"].rsplit("-", 1)[1],
                "selected_localizer": variant,
                "selected_tool": {
                    "name": spec["tool_name"],
                    "arguments": {
                        "query_ref": "query",
                        "reference_ref": "reference",
                        "method": spec["method"],
                    },
                },
                "tool_observation": observation,
                "provenance": {
                    "selection_policy": "label_blind_execution_then_local_hidden_verifier_v2",
                    "label_visible_to_localizer": False,
                    "teacher_receives_verifier_metrics": False,
                },
            }
        )
        selected_counts[variant] = selected_counts.get(variant, 0) + 1

    serialized = json.dumps(student_rows, ensure_ascii=False)
    leak_tokens = [
        token
        for token in ("oracle_boxes", "verifier_only_metrics", "strict_utqr", "query_url", "reference_url")
        if token in serialized
    ]
    report = {
        "artifact": "vcsl_local_hidden_verifier_selection_v2",
        "policy": "all localizers execute label-blind; local verifier selects; teacher receives only selected student-visible observation",
        "cases": audit_rows,
        "summary": {
            "requested": len(audit_rows),
            "accepted": len(student_rows),
            "rejected": len(audit_rows) - len(student_rows),
            "acceptance_rate": round(len(student_rows) / max(len(audit_rows), 1), 4),
            "selected_localizers": selected_counts,
            "student_export_leak_tokens": leak_tokens,
            "safe_to_send_selected_student_view": bool(student_rows) and not leak_tokens,
        },
    }
    return student_rows, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    student_rows, report = compile_selection(audit)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "selection_audit.json", report)
    write_jsonl(args.output_root / "accepted_localizations.student.jsonl", student_rows)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
