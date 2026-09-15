"""Compare reasoning quality while proving that executed action targets stayed fixed."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit_recovery_reasoning_quality_v2 import audit_rows
from pipeline import ROOT, write_json
from shadow_execute_generated_actions import _jsonl


BASE = ROOT / "outputs/student_external_contract_v1"
DEFAULT_BEFORE = (
    BASE
    / "recovery_batch_candidate_v13_expanded_catalog_u11_compact/recovery_sft_candidate.jsonl"
)
DEFAULT_AFTER = (
    BASE
    / "recovery_batch_candidate_v16_edge_aware_humanlike/recovery_sft_candidate.jsonl"
)
DEFAULT_OUTPUT = (
    BASE
    / "recovery_batch_candidate_v16_edge_aware_humanlike/humanlike_quality_comparison.json"
)


def _action_signatures(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    signatures: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metadata = row["metadata"]
        function = row["messages"][-1]["tool_calls"][0]["function"]
        signatures[str(metadata["episode_id"])].append(
            {
                "stage": metadata["causal_recovery_stage"],
                "tool": function["name"],
                "arguments": function["arguments"],
            }
        )
    return dict(signatures)


def compare(
    before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    before = audit_rows(before_rows)
    after = audit_rows(after_rows)
    before_signatures = _action_signatures(before_rows)
    after_signatures = _action_signatures(after_rows)
    edge_aware = [
        row
        for row in after_rows
        if row["metadata"]["causal_recovery_stage"] == "challenge"
        and "support edge is invalid" in row["messages"][-1]["content"]
    ]
    edge_by_dataset = Counter(row["metadata"]["dataset"] for row in edge_aware)
    return {
        "artifact": "recovery_humanlike_quality_comparison_v1",
        "comparison_scope": (
            "Same 55 development episodes and executed targets; only generic reasoning "
            "generation and the existing H7 audit extension changed."
        ),
        "before": {
            "rows": before["metrics"]["rows"],
            "base_row_passed": before["metrics"]["row_passed"],
            "strict_additional_check_passed_rows": before["metrics"][
                "strict_additional_check_passed_rows"
            ],
            "meta_process_narration_hits": before["metrics"][
                "meta_process_narration_hits"
            ],
            "unbalanced_inline_serialization_hits": before["metrics"][
                "unbalanced_inline_serialization_hits"
            ],
            "passed": before["passed"],
        },
        "after": {
            "rows": after["metrics"]["rows"],
            "base_row_passed": after["metrics"]["row_passed"],
            "strict_additional_check_passed_rows": after["metrics"][
                "strict_additional_check_passed_rows"
            ],
            "meta_process_narration_hits": after["metrics"][
                "meta_process_narration_hits"
            ],
            "unbalanced_inline_serialization_hits": after["metrics"][
                "unbalanced_inline_serialization_hits"
            ],
            "passed": after["passed"],
        },
        "executed_action_signatures_unchanged": before_signatures == after_signatures,
        "episode_ids_unchanged": set(before_signatures) == set(after_signatures),
        "edge_aware_binding_diagnoses": {
            "episodes": len(edge_aware),
            "by_dataset": dict(edge_by_dataset),
            "meaning": (
                "The trajectory identifies an invalid evidence-field support edge while "
                "preserving the EvidenceAtom's validity for its advertised fields."
            ),
        },
        "improvement_established": bool(
            not before["passed"]
            and after["passed"]
            and before_signatures == after_signatures
        ),
        "scope_limit": (
            "This establishes a deterministic text-quality and semantic-locality fix, "
            "not independent human preference or downstream student gain."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    parser.add_argument("--after", type=Path, default=DEFAULT_AFTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = compare(_jsonl(args.before), _jsonl(args.after))
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
