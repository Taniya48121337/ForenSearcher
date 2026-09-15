"""Oracle-bounded selection of label-blind VCSL localization candidates.

Every localizer runs before labels are consulted.  The compiler then keeps only
strictly verified candidates and emits a student-visible JSONL with oracle boxes
and verifier metrics removed.  A separate local audit preserves the rejection
decisions for reproducibility.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json, write_jsonl


DEFAULT_PROBE_ROOT = ROOT / "outputs" / "scale30_vcsl_probes"
DEFAULT_REPORTS = sorted(DEFAULT_PROBE_ROOT.glob("*/probe_report.json"))
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "selection"


LOCALIZER_TO_TOOL = {
    "hv": ("hv", "localize_copy_segment", "HV", 1),
    "hv_segment": ("hv_segment", "localize_copy_segment_hv_segment", "HV-SEGMENT", 2),
    "local_alignment": (
        "local_alignment",
        "localize_copy_segment_local_alignment",
        "LOCAL-ALIGNMENT",
        3,
    ),
    "dp": ("dp", "localize_copy_segment_dp", "DP", 4),
    "temporal_network": (
        "temporal_network",
        "localize_copy_segment_temporal_network",
        "TEMPORAL-NETWORK",
        5,
    ),
    "temporal_stability": (
        "tn_stability",
        "localize_copy_segment_temporal_stability",
        "TN-STABILITY",
        6,
    ),
    "tn_stability": (
        "tn_stability",
        "localize_copy_segment_temporal_stability",
        "TN-STABILITY",
        6,
    ),
    "axis_saliency": (
        "axis_saliency",
        "localize_copy_segment_axis_saliency",
        "AXIS-SALIENCY",
        7,
    ),
    "spd_official": (
        "spd_official",
        "localize_copy_segment_spd_official",
        "SPD-OFFICIAL-ISC",
        8,
    ),
}


def _score(case: dict[str, Any], complexity_rank: int) -> tuple[float, float, float, int]:
    metrics = case.get("metrics") or {}
    precision = float(metrics.get("precision", 0.0))
    recall = float(metrics.get("recall", 0.0))
    f1 = float(metrics.get("f1", 0.0))
    return min(precision, recall, f1), f1, precision + recall, -complexity_rank


def _student_item(candidate: dict[str, Any]) -> dict[str, Any]:
    case = candidate["case"]
    canonical, tool_name, method, _rank = candidate["tool"]
    shape = copy.deepcopy(case.get("feature_shapes") or {})
    summary = copy.deepcopy(case.get("similarity_summary") or {})
    try:
        report_ref = str(candidate["report_path"].relative_to(ROOT))
    except ValueError:
        report_ref = candidate["report_path"].name
    return {
        "pair_id": case["pair_id"],
        "query_id": case["query_id"],
        "reference_id": case["reference_id"],
        "selected_localizer": canonical,
        "selected_tool": {
            "name": tool_name,
            "arguments": {
                "query_ref": "query",
                "reference_ref": "reference",
                "method": method,
            },
        },
        "tool_observation": {
            "relationship": "partial_copy",
            "copied_segments_frames": copy.deepcopy(case.get("predicted_boxes") or []),
            "localization_method": method,
            "similarity_shape": shape,
            "similarity_summary": summary,
            "routing_diagnostics": copy.deepcopy(case.get("routing_diagnostics")),
        },
        "provenance": {
            "candidate_report": report_ref,
            "selection_policy": "post_execution_oracle_bounded_rejection_sampling_v1",
            "label_visible_to_localizer": False,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    reports = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in args.reports]
    if not reports:
        raise ValueError("at least one VCSL probe report is required")
    seeds = {payload.get("seed") for _path, payload in reports}
    sizes = {payload.get("sample_size") for _path, payload in reports}
    if len(seeds) != 1 or len(sizes) != 1:
        raise ValueError("candidate reports must use the same seed and sample size")

    case_maps = [{case["pair_id"]: case for case in payload["cases"]} for _path, payload in reports]
    pair_ids = list(case_maps[0])
    if any(set(mapping) != set(pair_ids) for mapping in case_maps[1:]):
        raise ValueError("candidate reports do not contain the same VCSL pairs")

    student_items: list[dict[str, Any]] = []
    audit_cases: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}
    for pair_id in pair_ids:
        candidates: list[dict[str, Any]] = []
        oracle_values = {json.dumps(mapping[pair_id].get("oracle_boxes"), sort_keys=True) for mapping in case_maps}
        if len(oracle_values) != 1:
            raise ValueError(f"inconsistent oracle boxes across reports for {pair_id}")
        oracle_boxes = json.loads(next(iter(oracle_values)))

        for (report_path, _report), mapping in zip(reports, case_maps):
            case = mapping[pair_id]
            resolved = str(case.get("resolved_localizer") or "")
            if resolved not in LOCALIZER_TO_TOOL:
                raise ValueError(f"unknown resolved VCSL localizer: {resolved}")
            tool = LOCALIZER_TO_TOOL[resolved]
            candidates.append(
                {
                    "report_path": report_path,
                    "case": case,
                    "tool": tool,
                    "eligible": case.get("strict_utqr_gold_reachable") is True,
                    "score": _score(case, tool[3]),
                }
            )

        eligible = [candidate for candidate in candidates if candidate["eligible"]]
        selected = max(eligible, key=lambda candidate: candidate["score"], default=None)
        audit_cases.append(
            {
                "pair_id": pair_id,
                "oracle_boxes": oracle_boxes,
                "accepted": selected is not None,
                "selected_localizer": selected["tool"][0] if selected else None,
                "rejection_reason": None if selected else "no_strictly_verified_label_blind_candidate",
                "candidates": [
                    {
                        "report": str(candidate["report_path"]),
                        "resolved_localizer": candidate["case"].get("resolved_localizer"),
                        "canonical_localizer": candidate["tool"][0],
                        "predicted_boxes": candidate["case"].get("predicted_boxes"),
                        "metrics": candidate["case"].get("metrics"),
                        "strict_utqr_gold_reachable": candidate["eligible"],
                    }
                    for candidate in candidates
                ],
            }
        )
        if selected is not None:
            item = _student_item(selected)
            student_items.append(item)
            name = item["selected_localizer"]
            selected_counts[name] = selected_counts.get(name, 0) + 1

    serialized = json.dumps(student_items, ensure_ascii=False)
    leak_tokens = [token for token in ("oracle_boxes", "metrics", "strict_utqr") if token in serialized]
    seed = next(iter(seeds))
    size = next(iter(sizes))
    output_root = args.output_root / f"vcsl_best_of_tree_seed_{seed}_n{size}"
    output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": output_root.name,
        "dataset": "VCSL",
        "seed": seed,
        "sample_size": size,
        "policy": "label_blind_candidate_execution_then_oracle_bounded_rejection_sampling",
        "summary": {
            "accepted": len(student_items),
            "rejected": len(pair_ids) - len(student_items),
            "acceptance_rate": round(len(student_items) / max(len(pair_ids), 1), 4),
            "student_export_oracle_leak_tokens": leak_tokens,
            "safe_to_build_harness_sft": bool(student_items) and not leak_tokens,
            "selected_localizers": selected_counts,
        },
        "cases": audit_cases,
    }
    write_json(output_root / "selection_audit.json", report)
    write_jsonl(output_root / "accepted_localizations.student.jsonl", student_items)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, nargs="+", default=DEFAULT_REPORTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    report = run(parser.parse_args())
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
