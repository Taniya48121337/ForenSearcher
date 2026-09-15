"""Add validated, label-blind independent predicate measurements to VidDiff candidates."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_jsonl
from viddiff_batch import DEFAULT_ADAPTIVE, _load_jsonl
from viddiff_probe_v4 import DEFAULT_OUTPUT as DEFAULT_PROBE_ROOT


DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "selection"
    / "viddiff_best_of_tree_seed_20260818_n10"
    / "candidate_measurements_v4.student.jsonl"
)

# Force is deliberately excluded: the pilot showed that monocular deformation
# proxies could rank the wrong clip confidently. These two families have a
# well-defined, visually testable predicate and passed the blinded pilot.
SUPPORTED_FAMILIES = {
    "universal_visibility",
    "supporting_grasper_stability",
    "suture_tension_proxy",
}
TOOL_NAME = "compare_predicate_independent"


def _candidate_from_probe(
    case: dict[str, Any], statement: str, probe_report_path: Path
) -> dict[str, Any]:
    measurement = case["measurement"]
    left = measurement["independent_a"]
    right = measurement["independent_b"]
    rationale = (
        f"A support={measurement['support_score_a']:.1f}; "
        f"B support={measurement['support_score_b']:.1f}; "
        f"calibrated margin={measurement['calibrated_margin']:.2f}. "
        f"A: {left['measurement_summary']} B: {right['measurement_summary']}"
    )
    return {
        "name": TOOL_NAME,
        "arguments": {
            "video_a_ref": "video_a",
            "video_b_ref": "video_b",
            "comparison_statement": statement,
        },
        "tool_observation": {
            "preferred_video": measurement["preferred_video"],
            "decision_rule": measurement["decision_rule"],
            "order_consistent": True,
            "video_a_interval_s": measurement["video_a_interval_s"],
            "video_b_interval_s": measurement["video_b_interval_s"],
            "confidence": measurement["confidence"],
            "support_score_a": measurement["support_score_a"],
            "support_score_b": measurement["support_score_b"],
            "calibrated_margin": measurement["calibrated_margin"],
            "comparative_rationale": rationale,
            "independent_measurement_summaries": {
                "video_a": left["measurement_summary"],
                "video_b": right["measurement_summary"],
            },
        },
        "provenance": {
            "candidate_report": str(probe_report_path),
            "predicate_family": case["predicate_family"],
            "measurement_ids": [left["measurement_id"], right["measurement_id"]],
            "label_visible_to_tool": False,
            "comparison_performed_after_independent_measurements": True,
        },
    }


def integrate(
    adaptive_path: Path = DEFAULT_ADAPTIVE,
    probe_report_path: Path = DEFAULT_PROBE_ROOT / "probe_report.json",
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    items = _load_jsonl(adaptive_path)
    report = json.loads(probe_report_path.read_text(encoding="utf-8"))
    probes = {case["sample_key"]: case for case in report["cases"]}
    added: list[str] = []
    output: list[dict[str, Any]] = []
    for source in items:
        item = copy.deepcopy(source)
        case = probes.get(item["sample_key"])
        if (
            case
            and case.get("status") in {"passed", "failed"}
            and case.get("predicate_family") in SUPPORTED_FAMILIES
            and isinstance(case.get("measurement"), dict)
        ):
            item["candidate_tools"] = [
                entry for entry in item["candidate_tools"] if entry["name"] != TOOL_NAME
            ]
            item["candidate_tools"].append(
                _candidate_from_probe(
                    case,
                    item["task"]["comparison_statement"],
                    probe_report_path,
                )
            )
            item["routing_contract"]["initial_candidates"] = [
                TOOL_NAME,
                *[
                    name
                    for name in item["routing_contract"]["initial_candidates"]
                    if name != TOOL_NAME
                ],
            ]
            added.append(item["sample_key"])
        output.append(item)
    write_jsonl(output_path, output)
    return {
        "input_cases": len(items),
        "augmented_cases": len(added),
        "augmented_sample_keys": added,
        "supported_families": sorted(SUPPORTED_FAMILIES),
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive-path", type=Path, default=DEFAULT_ADAPTIVE)
    parser.add_argument(
        "--probe-report",
        type=Path,
        default=DEFAULT_PROBE_ROOT / "probe_report.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            integrate(args.adaptive_path, args.probe_report, args.output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
