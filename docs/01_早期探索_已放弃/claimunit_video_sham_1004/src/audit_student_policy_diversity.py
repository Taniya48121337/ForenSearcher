"""Advisory, batch-level audit for student trajectory policy diversity.

The per-episode rubric can prove that each trajectory is valid while missing a
distributional failure: many otherwise perfect episodes may copy the same
macro route.  This audit reports that failure mode without turning route
diversity into a hard acceptance gate.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


MEASUREMENT_PREFIXES = ("compare_", "detect_", "localize_")
SUBSTANTIVE_INSPECTION_TOOLS = {"inspect_media"}


def _is_measurement_action(tool: str) -> bool:
    return tool.startswith(MEASUREMENT_PREFIXES) or tool in SUBSTANTIVE_INSPECTION_TOOLS


def _predicate_family(statement: str) -> str:
    text = statement.lower()
    if any(token in text for token in ("number of", "more often", "how many", "stops")):
        return "count_or_frequency"
    if any(token in text for token in ("perpendicular", "radial", "closer to", "approximately", "2/3", "two-thirds")):
        return "geometry_or_position"
    if any(token in text for token in ("tube moves", "moves more", "movement of the tube")):
        return "object_motion"
    if any(token in text for token in ("force", "tension", "tight", "pull")):
        return "force_or_tension"
    if any(token in text for token in ("visible", "visibility", "occluded", "supported")):
        return "visibility_or_support"
    return "other"


def _case_files(batch_dirs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for batch_dir in batch_dirs:
        cases_dir = batch_dir / "cases"
        if cases_dir.exists():
            files.extend(sorted(cases_dir.rglob("*.json")))
    return files


def _episode_record(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episode = payload["episode"]
    quality = payload["quality"]
    sequence = list(episode.get("tool_sequence") or quality.get("tool_sequence") or [])
    measurement_route = [tool for tool in sequence if _is_measurement_action(tool)]

    statement = ""
    selected_method = ""
    selected_direction = ""
    for turn in episode.get("turns", []):
        arguments = turn.get("generated_arguments") or {}
        if not statement and isinstance(arguments.get("comparison_statement"), str):
            statement = arguments["comparison_statement"]
        if turn.get("generated_tool") == "create_candidate_claim":
            fields = arguments.get("field_values") or {}
            statement = fields.get("comparison_statement", statement)
            selected_method = fields.get("measurement_method", "")
            selected_direction = fields.get("preferred_video", "")

    return {
        "dataset": episode.get("dataset", payload.get("dataset", "unknown")),
        "case_id": episode.get("source_case_id", payload.get("case_id", path.stem)),
        "accepted": bool(quality.get("accepted_for_high_quality_sft")),
        "score": quality.get("score"),
        "statement": statement,
        "predicate_family": _predicate_family(statement),
        "tool_sequence": sequence,
        "measurement_route": measurement_route,
        "selected_method": selected_method,
        "selected_direction": selected_direction,
        "used_meta_abstention": "compare_symmetry_residual_abstention" in sequence,
    }


def audit_batches(batch_dirs: Iterable[Path]) -> dict[str, Any]:
    records = [_episode_record(path) for path in _case_files(batch_dirs)]
    route_counts = Counter(" -> ".join(row["measurement_route"]) for row in records)
    full_counts = Counter(" -> ".join(row["tool_sequence"]) for row in records)
    family_routes: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        family_routes[row["predicate_family"]][" -> ".join(row["measurement_route"])] += 1

    total = len(records)
    dominant_route_count = max(route_counts.values(), default=0)
    dominant_route_share = round(dominant_route_count / total, 4) if total else 0.0
    unique_families = len({row["predicate_family"] for row in records})
    warnings: list[dict[str, Any]] = []
    if total >= 5 and unique_families >= 2 and dominant_route_share > 0.8:
        warnings.append(
            {
                "code": "dominant_macro_route_across_predicate_families",
                "severity": "advisory",
                "message": (
                    "More than 80% of episodes share one measurement route across "
                    "multiple predicate families; inspect for policy-template collapse."
                ),
            }
        )

    by_dataset: dict[str, Any] = {}
    for dataset in sorted({row["dataset"] for row in records}):
        subset = [row for row in records if row["dataset"] == dataset]
        dataset_routes = Counter(" -> ".join(row["measurement_route"]) for row in subset)
        dataset_dominant = max(dataset_routes.values(), default=0)
        dataset_share = round(dataset_dominant / len(subset), 4) if subset else 0.0
        by_dataset[dataset] = {
            "episode_count": len(subset),
            "accepted_episode_count": sum(row["accepted"] for row in subset),
            "unique_measurement_routes": len(dataset_routes),
            "dominant_measurement_route_share": dataset_share,
            "measurement_route_counts": dict(dataset_routes.most_common()),
            "tools_used": dict(
                Counter(
                    tool
                    for row in subset
                    for tool in row["measurement_route"]
                ).most_common()
            ),
        }

    return {
        "audit_type": "student_policy_diversity_advisory_v1",
        "is_acceptance_gate": False,
        "episode_count": total,
        "accepted_episode_count": sum(row["accepted"] for row in records),
        "unique_full_tool_sequences": len(full_counts),
        "unique_measurement_routes": len(route_counts),
        "dominant_measurement_route_share": dominant_route_share,
        "deep_search_episode_count": sum(len(row["measurement_route"]) > 1 for row in records),
        "single_measurement_episode_count": sum(len(row["measurement_route"]) == 1 for row in records),
        "meta_abstention_episode_count": sum(row["used_meta_abstention"] for row in records),
        "predicate_family_count": unique_families,
        "measurement_route_counts": dict(route_counts.most_common()),
        "selected_method_counts": dict(Counter(row["selected_method"] for row in records).most_common()),
        "selected_direction_counts": dict(Counter(row["selected_direction"] for row in records).most_common()),
        "routes_by_predicate_family": {
            family: dict(counts.most_common()) for family, counts in sorted(family_routes.items())
        },
        "by_dataset": by_dataset,
        "warnings": warnings,
        "episodes": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = audit_batches(args.batch_dirs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
