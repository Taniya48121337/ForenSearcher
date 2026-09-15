"""Oracle-bounded best-of-tool-tree selection for VidDiffBench measurements.

The measurement tools never receive dataset labels.  This compiler uses the
existing label only after all candidate executions finish, keeps one grounded
correct measurement per case, and exports a student-visible file with every
oracle/verifier field removed.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json, write_jsonl


PROBE_ROOT = ROOT / "outputs" / "probes"
DEFAULT_REPORTS = [
    PROBE_ROOT / "viddiff_surgery_seed_20260818_n10" / "probe_report.json",
    PROBE_ROOT / "viddiff_v2_surgery_seed_20260818_n10" / "probe_report.json",
    PROBE_ROOT / "viddiff_v3_surgery_seed_20260818_n10" / "probe_report.json",
]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "selection"


METHODS = {
    "viddiff_surgery_seed_20260818_n10": {
        "tool_name": "compare_uniform_timeline",
        "complexity_rank": 1,
    },
    "viddiff_v2_surgery_seed_20260818_n10": {
        "tool_name": "compare_global_motion_symmetric",
        "complexity_rank": 2,
    },
    "viddiff_v3_surgery_seed_20260818_n10": {
        "tool_name": "compare_action_refined_symmetric",
        "complexity_rank": 3,
    },
}

_STUDENT_FORBIDDEN_KEYS = {
    "candidate_report",
    "differences_gt",
    "ground_truth",
    "local_path",
    "metrics",
    "oracle_boxes",
    "oracle_direction",
    "path",
    "retrieval_frames",
    "url",
    "urls",
    "verifier",
    "verifier_result",
    "video_path",
}


def _student_visible(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _student_visible(child)
            for key, child in value.items()
            if str(key).strip().lower() not in _STUDENT_FORBIDDEN_KEYS
        }
    if isinstance(value, list):
        return [_student_visible(child) for child in value]
    return copy.deepcopy(value)


def _method_for_run_id(run_id: str) -> dict[str, Any] | None:
    """Resolve the executed method independently of batch size/index suffix."""
    if run_id.startswith("viddiff_v3_surgery_seed_"):
        return {"tool_name": "compare_action_refined_symmetric", "complexity_rank": 3}
    if run_id.startswith("viddiff_v2_surgery_seed_"):
        return {"tool_name": "compare_global_motion_symmetric", "complexity_rank": 2}
    if run_id.startswith("viddiff_surgery_seed_"):
        return {"tool_name": "compare_uniform_timeline", "complexity_rank": 1}
    return None


def _confidence(measurement: dict[str, Any]) -> float:
    value = measurement.get("confidence", 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _candidate_is_admissible(candidate: dict[str, Any]) -> bool:
    """Require swap consistency whenever a route exposes that measurement."""
    return bool(candidate.get("oracle_match")) and candidate.get("order_consistent") is not False


def _student_observation(case: dict[str, Any], method: dict[str, Any], report_path: Path) -> dict[str, Any]:
    measurement = copy.deepcopy(case["measurement"])
    for key in ("usage",):
        measurement.pop(key, None)
    return {
        "case_id": case["case_id"],
        "sample_key": case["sample_key"],
        "task": {
            "action_name": case.get("action_name"),
            "action_description": case.get("action_description"),
            "comparison_statement": case["candidate_difference"],
            "visible_assets": _student_visible(
                case.get("visual_assets", case.get("sampling", case.get("targeted_assets")))
            ),
        },
        "selected_tool": {
            "name": method["tool_name"],
            "arguments": {
                "video_a_ref": "video_a",
                "video_b_ref": "video_b",
                "comparison_statement": case["candidate_difference"],
            },
        },
        "tool_observation": measurement,
        "provenance": {
            "candidate_run_id": report_path.parent.name,
            "measurement_id": measurement.get("measurement_id"),
            "selection_policy": "post_execution_oracle_bounded_rejection_sampling_v1",
            "label_visible_to_tool": False,
        },
    }


def _student_candidate_set(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Export every executed label-blind branch without revealing verifier decisions.

    The legacy export below is intentionally retained for reproducibility.  This
    adaptive export is the SFT-v2 input: the policy sees the same candidate tool
    surface for every case and must decide what to execute from observations.
    Hidden labels are used only after a complete rollout is produced.
    """
    first = candidates[0]
    case = first["case"]
    exported_tools = []
    for item in sorted(candidates, key=lambda value: value["method"]["complexity_rank"]):
        observation = copy.deepcopy(item["case"].get("measurement") or {})
        observation.pop("usage", None)
        exported_tools.append(
            {
                "name": item["method"]["tool_name"],
                "arguments": {
                    "video_a_ref": "video_a",
                    "video_b_ref": "video_b",
                    "comparison_statement": case["candidate_difference"],
                },
                "tool_observation": observation,
                "provenance": {
                    "candidate_run_id": item["run_id"],
                    "measurement_id": observation.get("measurement_id"),
                    "label_visible_to_tool": False,
                },
            }
        )
    return {
        "case_id": case["case_id"],
        "sample_key": case["sample_key"],
        "task": {
            "action_name": case.get("action_name"),
            "action_description": case.get("action_description"),
            "comparison_statement": case["candidate_difference"],
            "visible_assets": _student_visible(
                case.get("visual_assets", case.get("sampling", case.get("targeted_assets")))
            ),
        },
        "candidate_tools": exported_tools,
        "routing_contract": {
            "policy": "visible_observation_adaptive_branching_v2",
            "initial_candidates": [item["name"] for item in exported_tools],
            "hidden_label_may_select_next_action": False,
            "continue_when": [
                "low_confidence",
                "order_inconsistency",
                "cross_method_disagreement",
            ],
            "stop_measurement_when": [
                "one_high_confidence_order_consistent_result",
                "two_methods_agree",
                "all_candidates_exhausted",
            ],
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    reports = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in args.reports]
    seeds = {payload.get("seed") for _path, payload in reports}
    sizes = {payload.get("sample_size") for _path, payload in reports}
    if len(seeds) != 1 or len(sizes) != 1:
        raise ValueError("candidate reports must use the same seed and sample size")
    case_maps = [{item["sample_key"]: item for item in payload["cases"]} for _path, payload in reports]
    keys = list(case_maps[0])
    if any(set(mapping) != set(keys) for mapping in case_maps[1:]):
        raise ValueError("candidate reports do not contain the same cases")

    accepted_exports = []
    adaptive_exports = []
    audit_cases = []
    for sample_key in keys:
        candidates = []
        oracle_values = {mapping[sample_key].get("oracle_direction") for mapping in case_maps}
        if len(oracle_values) != 1:
            raise ValueError(f"inconsistent oracle direction across reports for {sample_key}")
        oracle = next(iter(oracle_values))
        for (report_path, report), mapping in zip(reports, case_maps):
            case = mapping[sample_key]
            method = _method_for_run_id(report["run_id"])
            if method is None:
                raise ValueError(f"unknown candidate method report: {report['run_id']}")
            measurement = case.get("measurement") or {}
            direction = measurement.get("preferred_video")
            interval_valid = case.get("interval_format_valid") is True
            correct = direction == oracle and interval_valid
            candidates.append(
                {
                    "report_path": report_path,
                    "run_id": report["run_id"],
                    "method": method,
                    "case": case,
                    "direction": direction,
                    "interval_valid": interval_valid,
                    "oracle_match": correct,
                    "confidence": _confidence(measurement),
                    "order_consistent": measurement.get("order_consistent"),
                }
            )
        valid = [item for item in candidates if _candidate_is_admissible(item)]
        adaptive_exports.append(_student_candidate_set(candidates))
        # Prefer higher confidence, then the simpler measurement path.  The
        # verifier decides eligibility only; it never edits the observation.
        selected = max(
            valid,
            key=lambda item: (item["confidence"], -item["method"]["complexity_rank"]),
            default=None,
        )
        audit_item = {
            "sample_key": sample_key,
            "oracle_direction": oracle,
            "candidates": [
                {
                    "run_id": item["run_id"],
                    "tool_name": item["method"]["tool_name"],
                    "predicted_direction": item["direction"],
                    "interval_valid": item["interval_valid"],
                    "oracle_match": item["oracle_match"],
                    "confidence": item["confidence"],
                    "order_consistent": item["order_consistent"],
                }
                for item in candidates
            ],
            "accepted": selected is not None,
            "selected_tool": selected["method"]["tool_name"] if selected else None,
            "rejection_reason": (
                None
                if selected
                else "no_label_blind_order_consistent_candidate_matched_oracle"
            ),
        }
        audit_cases.append(audit_item)
        if selected is not None:
            accepted_exports.append(
                _student_observation(selected["case"], selected["method"], selected["report_path"])
            )

    serialized = json.dumps(
        {"legacy": accepted_exports, "adaptive": adaptive_exports},
        ensure_ascii=False,
    )
    leak_tokens = [
        token
        for token in ("oracle_direction", "oracle_match", "direction_correct", "differences_gt", "retrieval_frames")
        if token in serialized
    ]
    private_leak_tokens = [
        token
        for token in (
            '"candidate_report"',
            '"local_path"',
            '"path"',
            '"url"',
            '"urls"',
            '"verifier"',
            '"video_path"',
        )
        if token in serialized
    ]
    output_root = args.output_root / f"viddiff_best_of_tree_seed_{next(iter(seeds))}_n{next(iter(sizes))}"
    output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": output_root.name,
        "dataset": "VidDiffBench",
        "seed": next(iter(seeds)),
        "sample_size": next(iter(sizes)),
        "policy": "label_blind_candidate_execution_then_oracle_bounded_rejection_sampling",
        "summary": {
            "accepted": len(accepted_exports),
            "rejected": len(keys) - len(accepted_exports),
            "acceptance_rate": round(len(accepted_exports) / max(len(keys), 1), 4),
            "student_export_oracle_leak_tokens": leak_tokens,
            "student_export_private_leak_tokens": private_leak_tokens,
            "safe_to_build_harness_sft": (
                len(accepted_exports) >= 8 and not leak_tokens and not private_leak_tokens
            ),
            "adaptive_candidate_sets": len(adaptive_exports),
            "safe_to_build_adaptive_harness": (
                len(adaptive_exports) == len(keys)
                and not leak_tokens
                and not private_leak_tokens
            ),
        },
        "cases": audit_cases,
    }
    write_json(output_root / "selection_audit.json", report)
    write_jsonl(output_root / "accepted_measurements.student.jsonl", accepted_exports)
    write_jsonl(output_root / "candidate_measurements.student.jsonl", adaptive_exports)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, nargs=3, default=DEFAULT_REPORTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
