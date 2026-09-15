"""Hard-negative calibration for the unified closed-loop quality audit."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Callable

from audit_student_closed_loop_quality import (
    DEFAULT_CLEAN,
    DEFAULT_REPAIR,
    DEFAULT_SFT,
    SEED_ROWS,
    _audit_episode,
)
from pipeline import ROOT, write_json
from shadow_execute_generated_actions import _jsonl


DEFAULT_OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1"
    / "student_closed_loop_rubric_hard_negative_calibration.json"
)


def _turn(episode: dict[str, Any], tool: str, *, occurrence: int = 0) -> dict[str, Any]:
    matches = [item for item in episode["turns"] if item.get("generated_tool") == tool]
    return matches[occurrence]


def _terminal_failure(episode: dict[str, Any]) -> None:
    episode["terminal_success"] = False
    episode["final_terminal"] = False
    episode["final_stop_decision"] = {
        "allowed": False,
        "reason_code": "open_high_priority_obligations",
    }


def _rejected_action(episode: dict[str, Any]) -> None:
    episode["turns"][1]["status"] = "rejected"


def _hallucinated_evidence(episode: dict[str, Any]) -> None:
    episode["turns"][1]["reasoning"] += " This is confirmed by EA-NOT-VISIBLE-999."


def _claim_id_drift(episode: dict[str, Any]) -> None:
    _turn(episode, "bind_evidence")["generated_arguments"]["claim_id"] = "claim_drifted"


def _unconsumed_measurement(episode: dict[str, Any]) -> None:
    episode["turns"][1]["environment_outcome"].setdefault(
        "created_evidence_ids", []
    ).append("EA-UNUSED-MEASUREMENT-999")


def _repair_overreach(episode: dict[str, Any]) -> None:
    _turn(episode, "set_field_decision", occurrence=0)["generated_arguments"]["fields"].append(
        "sample_key"
    )


def _dependency_inversion(episode: dict[str, Any]) -> None:
    bind_index = next(
        index for index, item in enumerate(episode["turns"])
        if item.get("generated_tool") == "bind_evidence"
    )
    commit_index = next(
        index for index, item in enumerate(episode["turns"])
        if item.get("generated_tool") == "set_field_decision"
        and (item.get("generated_arguments") or {}).get("decision_status") == "committed"
    )
    episode["turns"][bind_index], episode["turns"][commit_index] = (
        episode["turns"][commit_index],
        episode["turns"][bind_index],
    )


def _redundant_measurement(episode: dict[str, Any]) -> None:
    measurement = next(
        item for item in episode["turns"]
        if item.get("generated_tool") not in {
            "create_candidate_claim", "adjudicate_evidence", "bind_evidence", "set_field_decision", "stop"
        }
        and not str(item.get("generated_tool", "")).startswith("inspect")
    )
    duplicate = copy.deepcopy(measurement)
    duplicate["turn"] = 99
    insert_at = next(
        (index for index, item in enumerate(episode["turns"])
         if item.get("generated_tool") in {"create_candidate_claim", "bind_evidence"}),
        len(episode["turns"]) - 1,
    )
    episode["turns"].insert(insert_at, duplicate)


def _generic_reasoning(episode: dict[str, Any]) -> None:
    episode["turns"][1]["reasoning"] = "Do the next step."


def _context_truncation(episode: dict[str, Any]) -> None:
    episode["turns"][-1]["prompt_tokens_kept"] = max(
        0, int(episode["turns"][-1]["prompt_tokens_source"]) - 1
    )


MUTATIONS: list[tuple[str, str, str, Callable[[dict[str, Any]], None]]] = [
    ("terminal_with_open_obligation", "clean", "VideoSham", _terminal_failure),
    ("rejected_action_hidden_in_trace", "clean", "VCSL", _rejected_action),
    ("hallucinated_evidence_citation", "clean", "VidDiffBench", _hallucinated_evidence),
    ("cross_turn_claim_id_drift", "clean", "VideoSham", _claim_id_drift),
    ("measurement_created_but_never_consumed", "clean", "VCSL", _unconsumed_measurement),
    ("repair_mutates_unaffected_invariant", "repair", "VidDiffBench", _repair_overreach),
    ("bind_commit_dependency_inversion", "clean", "VideoSham", _dependency_inversion),
    ("redundant_measurement_call", "repair", "VCSL", _redundant_measurement),
    ("generic_non_grounded_reasoning", "repair", "VideoSham", _generic_reasoning),
    ("prompt_context_left_truncated", "repair", "VidDiffBench", _context_truncation),
]


def calibrate(clean_path: Path, repair_path: Path, sft_path: Path) -> dict[str, Any]:
    reports = {
        "clean": json.loads(clean_path.read_text(encoding="utf-8")),
        "repair": json.loads(repair_path.read_text(encoding="utf-8")),
    }
    episodes = {
        mode: {item["dataset"]: item for item in report["episodes"]}
        for mode, report in reports.items()
    }
    rows = {
        str(item["metadata"]["row_id"]): item for item in _jsonl(sft_path)
    }
    details = []
    for family, mode, dataset, mutate in MUTATIONS:
        negative = copy.deepcopy(episodes[mode][dataset])
        mutate(negative)
        seed = rows[SEED_ROWS[mode][dataset]]
        audit = _audit_episode(mode, negative, seed, generation_limit=384)
        details.append(
            {
                "family": family,
                "mode": mode,
                "dataset": dataset,
                "rejected": not audit["accepted_for_high_quality_sft"],
                "score": audit["score"],
                "failed_dimensions": sorted(audit["failures"]),
            }
        )
    rejected = sum(item["rejected"] for item in details)
    return {
        "audit": "student_closed_loop_rubric_hard_negative_calibration_v1",
        "families": len(details),
        "rejected": rejected,
        "missed": len(details) - rejected,
        "all_hard_negatives_rejected": rejected == len(details),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--repair", type=Path, default=DEFAULT_REPAIR)
    parser.add_argument("--sft", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = calibrate(args.clean, args.repair, args.sft)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
