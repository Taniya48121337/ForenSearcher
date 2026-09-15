"""Posthoc verifier audit after adding a label-blind typed VidDiff candidate."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from pipeline import write_json
from viddiff_batch import _load_jsonl


TOOL_NAME = "compare_predicate_independent"


def reverify(
    audit: dict[str, Any], candidate_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    revised = copy.deepcopy(audit)
    candidates = {str(row["sample_key"]): row for row in candidate_rows}
    newly_accepted: list[str] = []
    for case in revised.get("cases") or []:
        sample_key = str(case["sample_key"])
        row = candidates.get(sample_key)
        if row is None:
            continue
        typed = next(
            (
                item
                for item in row.get("candidate_tools") or []
                if item.get("name") == TOOL_NAME
                and (item.get("provenance") or {}).get("artifact_id")
                == "viddiff_local_typed_predicate_observer_canary_v1"
            ),
            None,
        )
        if typed is None:
            continue
        observation = typed.get("tool_observation") or {}
        intervals = [
            observation.get("video_a_interval_s"),
            observation.get("video_b_interval_s"),
        ]
        interval_valid = all(
            isinstance(value, list)
            and len(value) == 2
            and all(isinstance(number, (int, float)) for number in value)
            and 0 <= float(value[0]) <= float(value[1])
            for value in intervals
        )
        oracle_match = observation.get("preferred_video") == case.get("oracle_direction")
        verifier_candidate = {
            "run_id": "viddiff_local_typed_predicate_observer_canary_v1",
            "tool_name": TOOL_NAME,
            "predicted_direction": observation.get("preferred_video"),
            "interval_valid": interval_valid,
            "oracle_match": oracle_match,
            "confidence": observation.get("confidence"),
            "order_consistent": observation.get("order_consistent"),
        }
        case["candidates"] = [
            item for item in case.get("candidates") or []
            if item.get("tool_name") != TOOL_NAME
        ]
        case["candidates"].append(verifier_candidate)
        admissible = (
            oracle_match
            and interval_valid
            and observation.get("order_consistent") is not False
        )
        if admissible and case.get("accepted") is not True:
            case["accepted"] = True
            case["selected_tool"] = TOOL_NAME
            case["rejection_reason"] = None
            newly_accepted.append(sample_key)
    cases = list(revised.get("cases") or [])
    accepted = sum(item.get("accepted") is True for item in cases)
    summary = revised.setdefault("summary", {})
    summary.update(
        {
            "accepted": accepted,
            "rejected": len(cases) - accepted,
            "acceptance_rate": round(accepted / max(len(cases), 1), 4),
            "typed_candidates_reverified": sum(
                any(candidate.get("tool_name") == TOOL_NAME for candidate in item.get("candidates") or [])
                for item in cases
            ),
            "newly_accepted_by_typed_candidate": newly_accepted,
        }
    )
    revised["policy_extension"] = (
        "local typed unary observations frozen before deterministic comparison; "
        "oracle used only by this posthoc verifier audit"
    )
    return revised


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-audit", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.selection_audit.read_text(encoding="utf-8"))
    report = reverify(audit, _load_jsonl(args.candidates))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
