"""Audit a frozen VidDiff manifest for typed predicate-observation coverage."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline import write_json
from viddiff_predicate_protocol import compile_protocol


def build_report(selection_report: dict[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    families: Counter[str] = Counter()
    generic: list[str] = []
    for item in selection_report.get("cases") or []:
        sample_key = str(item.get("sample_key") or "")
        statement = str(
            item.get("candidate_difference")
            or (item.get("task") or {}).get("comparison_statement")
            or ""
        )
        protocol = compile_protocol(statement)
        family = str(protocol["family"])
        families[family] += 1
        if family == "generic_visual_comparison":
            generic.append(sample_key)
        cases.append(
            {
                "sample_key": sample_key,
                "comparison_statement": statement,
                "family": family,
                "quantifier": protocol["quantifier"],
                "target_variable": protocol["target_variable"],
                "metric_direction": protocol["metric_direction"],
                "route_priority": protocol["route_priority"],
                "minimum_valid_units": protocol["minimum_valid_units"],
                "proof_obligations": protocol["proof_obligations"],
            }
        )
    return {
        "artifact": "viddiff_typed_predicate_coverage_audit_v1",
        "seed": selection_report.get("seed"),
        "cases": len(cases),
        "family_counts": dict(sorted(families.items())),
        "generic_case_ids": generic,
        "all_cases_have_typed_protocol": not generic and bool(cases),
        "direction_or_label_fields_exported": False,
        "case_protocols": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.selection_report.read_text(encoding="utf-8"))
    report = build_report(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in (
        "cases", "family_counts", "generic_case_ids", "all_cases_have_typed_protocol"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
