"""Mechanistic one-case ablation for typed predicate evidence routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline import write_json


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _generic_decision(item: dict[str, Any]) -> dict[str, Any]:
    votes = {direction: 0.0 for direction in ("a", "b", "c")}
    used = []
    for candidate in item.get("candidate_tools") or []:
        if candidate.get("name") == "compare_predicate_independent":
            continue
        observation = candidate.get("tool_observation") or {}
        direction = observation.get("preferred_video")
        if direction not in votes or observation.get("order_consistent") is False:
            continue
        confidence = observation.get("confidence")
        weight = float(confidence) if isinstance(confidence, (int, float)) else 0.5
        votes[direction] += weight
        used.append(
            {
                "method": candidate.get("name"),
                "direction": direction,
                "confidence_weight": weight,
            }
        )
    decision = max(votes, key=lambda direction: (votes[direction], direction == "c"))
    return {"decision": decision, "weighted_votes": votes, "used_routes": used}


def _typed_decision(item: dict[str, Any]) -> dict[str, Any]:
    candidate = next(
        entry
        for entry in item.get("candidate_tools") or []
        if entry.get("name") == "compare_predicate_independent"
    )
    observation = candidate.get("tool_observation") or {}
    return {
        "decision": observation.get("preferred_video"),
        "method": candidate.get("name"),
        "decision_rule": observation.get("decision_rule"),
        "predicate_family": observation.get("predicate_family"),
        "reasons_a": observation.get("reasons_a"),
        "reasons_b": observation.get("reasons_b"),
        "labels_visible_when_decision_frozen": False,
    }


def run(candidates_path: Path, observer_report_path: Path, sample_key: str) -> dict[str, Any]:
    item = next(row for row in _rows(candidates_path) if row.get("sample_key") == sample_key)
    generic = _generic_decision(item)
    typed = _typed_decision(item)
    # The two decisions above are frozen before this post-hoc evaluation record is opened.
    observer_report = json.loads(observer_report_path.read_text(encoding="utf-8"))
    case = next(row for row in observer_report.get("cases") or [] if row.get("sample_key") == sample_key)
    oracle = case.get("oracle_direction")
    generic["posthoc_correct"] = generic["decision"] == oracle
    typed["posthoc_correct"] = typed["decision"] == oracle
    return {
        "audit": "viddiff_typed_predicate_mechanistic_ablation_v1",
        "sample_key": sample_key,
        "decisions_frozen_before_oracle_load": True,
        "external_service_calls": 0,
        "generic_routes_only": generic,
        "typed_proof_route": typed,
        "posthoc_evaluation": {"oracle_direction": oracle},
        "observed_effect": {
            "generic_to_typed_direction_change": f"{generic['decision']}->{typed['decision']}",
            "correctness_change": f"{generic['posthoc_correct']}->{typed['posthoc_correct']}",
        },
        "interpretation_limit": (
            "This is a one-case mechanistic ablation demonstrating causal route utility, not a "
            "population accuracy estimate; the holdout-scale ablation must be run later."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--observer-report", type=Path, required=True)
    parser.add_argument("--sample-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.candidates, args.observer_report, args.sample_key)
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
