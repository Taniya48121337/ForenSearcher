"""Compute strict pass^k from independent full-trajectory rollout reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from pipeline import write_json


def _gold(case: dict[str, Any]) -> bool:
    return (
        case.get("status") == "passed"
        and float(case.get("rubric_score", 0.0)) >= 92.0
        and not case.get("rubric_failed_gates", [])
        and not case.get("rubric_failed_criteria", [])
    )


def audit_pass_k(
    run_roots: list[Path],
    *,
    case_id: str | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    discovered_ids: set[str] = set()
    rollout_seeds: list[int | None] = []
    generation_modes: list[str | None] = []
    for repeat_index, root in enumerate(run_roots, start=1):
        root = Path(root)
        report = json.loads((root / "batch_report.json").read_text(encoding="utf-8"))
        cases = report.get("cases", [])
        if not cases and report.get("episodes"):
            cases = [
                {
                    "case_id": episode.get("case_id"),
                    "status": (
                        "passed"
                        if episode.get("quality", {}).get("accepted_for_high_quality_sft")
                        else "rubric_rejected"
                    ),
                    "rubric_score": episode.get("quality", {}).get("score"),
                    "rubric_failed_gates": episode.get("quality", {}).get("failed_gates", []),
                    "rubric_failed_criteria": list(
                        episode.get("quality", {}).get("failures", {}).keys()
                    ),
                    "repair_operator": episode.get("mode"),
                    "student_tool_sequence": episode.get("tool_sequence", []),
                }
                for episode in report.get("episodes", [])
            ]
        generation = report.get("generation_parameters", {})
        rollout_seeds.append(generation.get("base_seed"))
        generation_modes.append(report.get("generation_mode"))
        selected = [case for case in cases if case_id is None or case.get("case_id") == case_id]
        if len(selected) != 1:
            raise ValueError(f"expected exactly one matching case in {root}, found {len(selected)}")
        case = selected[0]
        discovered_ids.add(str(case["case_id"]))
        telemetry = case.get("tool_telemetry", {}).get("selected_calls", {})
        state_tools = {
            "inspect_media",
            "inspect_pair_features",
            "inspect_comparison_task",
            "create_candidate_claim",
            "adjudicate_evidence",
            "bind_evidence",
            "set_field_decision",
            "stop",
        }
        inferred_tool = next((name for name in telemetry if name not in state_tools), None)
        rows.append(
            {
                "repeat_index": repeat_index,
                "source_run": str(root),
                "case_id": case["case_id"],
                "gold": _gold(case),
                "status": case.get("status"),
                "rubric_score": case.get("rubric_score"),
                "rubric_failed_gates": case.get("rubric_failed_gates", []),
                "rubric_failed_criteria": case.get("rubric_failed_criteria", []),
                "repair_operator": case.get("repair_operator"),
                "selected_tool": case.get("selected_measurement_tool")
                or case.get("selected_localization_method")
                or case.get("selected_difference_tool")
                or inferred_tool,
                "student_tool_sequence": case.get("student_tool_sequence"),
                "generation_mode": report.get("generation_mode"),
                "base_seed": generation.get("base_seed"),
            }
        )
    if len(discovered_ids) != 1:
        raise ValueError(f"independent repeats must target one case, found {sorted(discovered_ids)}")
    n = len(rows)
    successes = sum(row["gold"] for row in rows)
    pass_k = {
        f"pass^{k}": round(math.comb(successes, k) / math.comb(n, k), 6)
        for k in range(1, n + 1)
    }
    pass_at_k = {
        f"pass@{k}": round(
            1.0 - (math.comb(n - successes, k) / math.comb(n, k))
            if n - successes >= k
            else 1.0,
            6,
        )
        for k in range(1, n + 1)
    }
    report = {
        "definitions": {
            "pass^k": "strict full-trajectory consistency: C(successes,k)/C(independent_rollouts,k)",
            "pass@k": "at least one accepted trajectory in k draws: 1-C(failures,k)/C(independent_rollouts,k)",
        },
        "case_id": next(iter(discovered_ids)),
        "independent_rollouts": n,
        "gold_rollouts": successes,
        "pass_k": pass_k,
        "pass_at_k": pass_at_k,
        "all_actions_must_pass_validator_and_utqr": True,
        "route_portfolio_attempts_count_as_independent": False,
        "generation_modes": generation_modes,
        "rollout_seeds": rollout_seeds,
        "distinct_recorded_seeds": (
            len({seed for seed in rollout_seeds if seed is not None}) == len(rollout_seeds)
            if all(seed is not None for seed in rollout_seeds)
            else None
        ),
        "rollouts": rows,
    }
    if output is not None:
        write_json(Path(output), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_pass_k(args.runs, case_id=args.case_id, output=args.output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
