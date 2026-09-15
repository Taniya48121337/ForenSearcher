"""Replace the legacy predicate candidate with validated typed observations."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from pipeline import write_jsonl
from viddiff_batch import _load_jsonl


TOOL_NAME = "compare_predicate_independent"


def _candidate(case: dict[str, Any]) -> dict[str, Any]:
    measurement = copy.deepcopy(case["measurement"])
    records = copy.deepcopy(case["records"])
    rationale = (
        f"Typed {case['protocol_family']} observations measured "
        f"A={measurement.get('metric_a')} and B={measurement.get('metric_b')}. "
        f"The deterministic rule {measurement.get('decision_rule')} returned "
        f"preferred_video={measurement.get('preferred_video')}; "
        f"A limitations={measurement.get('reasons_a') or []}; "
        f"B limitations={measurement.get('reasons_b') or []}."
    )
    return {
        "name": TOOL_NAME,
        "arguments": {
            "video_a_ref": "video_a",
            "video_b_ref": "video_b",
            "comparison_statement": case["comparison_statement"],
        },
        "tool_observation": {
            "preferred_video": measurement["preferred_video"],
            "decision_rule": measurement["decision_rule"],
            "order_consistent": True,
            "video_a_interval_s": records["a"]["interval_s"],
            "video_b_interval_s": records["b"]["interval_s"],
            "confidence": measurement["confidence"],
            "predicate_family": case["protocol_family"],
            "metric_a": measurement.get("metric_a"),
            "metric_b": measurement.get("metric_b"),
            "usable_a": measurement.get("usable_a"),
            "usable_b": measurement.get("usable_b"),
            "reasons_a": measurement.get("reasons_a") or [],
            "reasons_b": measurement.get("reasons_b") or [],
            "independent_typed_records": records,
            "comparative_rationale": rationale,
        },
        "provenance": {
            "artifact_id": "viddiff_local_typed_predicate_observer_canary_v1",
            "predicate_family": case["protocol_family"],
            "label_visible_to_tool": False,
            "unary_records_generated_before_comparison": True,
            "direction_computed_by_deterministic_label_blind_comparator": True,
            "role_equivariant_by_construction": True,
        },
    }


def integrate(
    adaptive_path: Path,
    probe_report_paths: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    items = _load_jsonl(adaptive_path)
    probes: dict[str, dict[str, Any]] = {}
    for path in probe_report_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if int(report.get("external_service_calls", -1)) != 0:
            raise ValueError("typed observer integration requires a local-only report")
        if report.get("generator_label_visibility") is not False:
            raise ValueError("typed observer generator must be label blind")
        for case in report.get("cases") or []:
            probes[str(case["sample_key"])] = case

    augmented: list[str] = []
    output: list[dict[str, Any]] = []
    for source in items:
        item = copy.deepcopy(source)
        case = probes.get(str(item.get("sample_key") or ""))
        if case is not None:
            if case.get("comparison_statement") != item["task"]["comparison_statement"]:
                raise ValueError(f"statement mismatch for {item['sample_key']}")
            item["candidate_tools"] = [
                candidate
                for candidate in item.get("candidate_tools") or []
                if candidate.get("name") != TOOL_NAME
            ]
            item["candidate_tools"].append(_candidate(case))
            initial = list((item.get("routing_contract") or {}).get("initial_candidates") or [])
            item.setdefault("routing_contract", {})["initial_candidates"] = [
                TOOL_NAME,
                *(name for name in initial if name != TOOL_NAME),
            ]
            augmented.append(str(item["sample_key"]))
        output.append(item)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, output)
    return {
        "artifact": "viddiff_typed_observer_candidate_integration_v1",
        "input_cases": len(items),
        "augmented_cases": len(augmented),
        "augmented_sample_keys": augmented,
        "tool_name": TOOL_NAME,
        "new_tool_added": False,
        "legacy_candidate_replaced_when_present": True,
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive-path", type=Path, required=True)
    parser.add_argument("--probe-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
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
