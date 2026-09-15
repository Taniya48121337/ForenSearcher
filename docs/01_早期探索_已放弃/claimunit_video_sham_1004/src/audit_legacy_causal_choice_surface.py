"""Reassess recorded UTQR-7 packets under the evidence-choice autonomy criterion.

This is a read-only historical audit.  It reconstructs the eligible semantic
tool set at each recorded assistant action from the visible transcript; it does
not rerun a teacher or infer that missing telemetry passed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from causal_trajectory_generation import (
    control_plane_exposure_audit,
    is_semantic_measurement_tool,
)
from pipeline import ROOT, write_json


def _parsed(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _eligible_surfaces(value: Any) -> list[list[str]]:
    value = _parsed(value)
    surfaces: list[list[str]] = []
    if isinstance(value, dict):
        eligible = value.get("eligible_tools")
        if isinstance(eligible, list):
            surfaces.append([str(name) for name in eligible])
        for child in value.values():
            surfaces.extend(_eligible_surfaces(child))
    elif isinstance(value, list):
        for child in value:
            surfaces.extend(_eligible_surfaces(child))
    return surfaces


def _selected_tool(message: dict[str, Any]) -> str:
    calls = message.get("tool_calls") or []
    if not calls:
        return ""
    function = calls[0].get("function") or {}
    if isinstance(function, dict):
        return str(function.get("name") or "")
    return ""


def audit_episode(episode: dict[str, Any], *, episode_name: str) -> dict[str, Any]:
    messages = episode.get("messages") or []
    visible_prefix: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for turn_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            visible_prefix.append(message)
            continue
        selected = _selected_tool(message)
        surfaces: list[list[str]] = []
        for prior in visible_prefix:
            surfaces.extend(_eligible_surfaces(prior.get("content")))
        eligible = surfaces[-1] if surfaces else []
        semantic_eligible = [name for name in eligible if is_semantic_measurement_tool(name)]
        selected_is_measurement = selected in semantic_eligible
        router = control_plane_exposure_audit(visible_prefix, [])
        rows.append(
            {
                "turn_index": turn_index,
                "selected_tool": selected,
                "eligible_tools": eligible,
                "semantic_eligible_tools": semantic_eligible,
                "semantic_choice_count": len(semantic_eligible),
                "selected_is_semantic_measurement": selected_is_measurement,
                "nontrivial_semantic_choice": selected_is_measurement
                and len(semantic_eligible) >= 2,
                "categorical_router_answer_visible": router[
                    "categorical_router_answer_visible"
                ],
                "categorical_router_answer_paths": router[
                    "categorical_router_answer_paths"
                ],
            }
        )
        visible_prefix.append(message)
    measurement_rows = [row for row in rows if row["selected_is_semantic_measurement"]]
    return {
        "episode": episode_name,
        "trajectory_id": episode.get("trajectory_id"),
        "assistant_turns": len(rows),
        "semantic_measurement_turns": len(measurement_rows),
        "nontrivial_semantic_choice_turns": sum(
            bool(row["nontrivial_semantic_choice"]) for row in measurement_rows
        ),
        "measurement_turns_with_router_answer": sum(
            bool(row["categorical_router_answer_visible"]) for row in measurement_rows
        ),
        "rows": rows,
    }


def audit_packet(path: Path) -> dict[str, Any]:
    packet = json.loads(path.read_text(encoding="utf-8"))
    episodes = packet.get("raw_base_episodes") or {}
    results = [
        audit_episode(episode, episode_name=name)
        for name, episode in episodes.items()
        if isinstance(episode, dict) and episode.get("messages")
    ]
    semantic_turns = sum(result["semantic_measurement_turns"] for result in results)
    nontrivial_turns = sum(
        result["nontrivial_semantic_choice_turns"] for result in results
    )
    return {
        "path": str(path.relative_to(ROOT)),
        "dataset": packet.get("dataset"),
        "case_id": packet.get("case_id"),
        "semantic_measurement_turns": semantic_turns,
        "nontrivial_semantic_choice_turns": nontrivial_turns,
        "all_measurement_turns_nontrivial": semantic_turns > 0
        and semantic_turns == nontrivial_turns,
        "categorical_router_answer_visible_on_measurement": any(
            result["measurement_turns_with_router_answer"] > 0 for result in results
        ),
        "episodes": results,
    }


def audit_paths(paths: list[Path]) -> dict[str, Any]:
    packets = [audit_packet(path) for path in paths]
    semantic_turns = sum(row["semantic_measurement_turns"] for row in packets)
    nontrivial_turns = sum(row["nontrivial_semantic_choice_turns"] for row in packets)
    return {
        "artifact": "legacy_causal_choice_surface_reassessment_v1",
        "scope": "recorded_packets_only_no_teacher_rerun",
        "packet_count": len(packets),
        "semantic_measurement_turns": semantic_turns,
        "nontrivial_semantic_choice_turns": nontrivial_turns,
        "nontrivial_fraction": round(nontrivial_turns / semantic_turns, 4)
        if semantic_turns
        else 0.0,
        "packets_with_router_answer_on_measurement": sum(
            bool(row["categorical_router_answer_visible_on_measurement"])
            for row in packets
        ),
        "interpretation": (
            "Historical success does not establish evidence-choice autonomy when a semantic "
            "measurement turn had fewer than two executable semantic alternatives or a "
            "categorical router answer was visible. This audit does not rescore evidence correctness."
        ),
        "packets": packets,
    }


def audit_exposure_jsonl(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    measurement_rows: list[dict[str, Any]] = []
    by_dataset: dict[str, dict[str, int]] = {}
    for row in rows:
        target = str(row.get("target_tool") or "")
        if not is_semantic_measurement_tool(target):
            continue
        semantic_eligible = [
            str(name)
            for name in row.get("eligible_tools") or []
            if is_semantic_measurement_tool(str(name))
        ]
        nontrivial = len(semantic_eligible) >= 2
        measurement_rows.append(
            {
                "row_id": row.get("row_id"),
                "dataset": row.get("dataset"),
                "target_tool": target,
                "semantic_eligible_tools": semantic_eligible,
                "nontrivial_semantic_choice": nontrivial,
            }
        )
        summary = by_dataset.setdefault(
            str(row.get("dataset") or "unknown"),
            {"semantic_measurement_turns": 0, "nontrivial_semantic_choice_turns": 0},
        )
        summary["semantic_measurement_turns"] += 1
        summary["nontrivial_semantic_choice_turns"] += int(nontrivial)
    try:
        display_path = str(path.relative_to(ROOT))
    except ValueError:
        display_path = str(path)
    return {
        "path": display_path,
        "row_count": len(rows),
        "semantic_measurement_turns": len(measurement_rows),
        "nontrivial_semantic_choice_turns": sum(
            bool(row["nontrivial_semantic_choice"]) for row in measurement_rows
        ),
        "by_dataset": by_dataset,
        "rows": measurement_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "outputs/student_external_contract_v1/policy_choice_surface_v1/legacy_causal_packet_reassessment.json",
    )
    args = parser.parse_args()
    paths = sorted(ROOT.glob("outputs/**/utqr_v7_causal_packet.json"))
    report = audit_paths(paths)
    exposure_paths = [
        ROOT / "outputs/production_sft_v8_evidence_first_candidate/control_plane_exposure_rows.jsonl",
        ROOT / "outputs/student_external_contract_v1/evidence_first_sft_v6/control_plane_exposure_rows.jsonl",
    ]
    report["candidate_exports"] = [
        audit_exposure_jsonl(path) for path in exposure_paths if path.exists()
    ]
    write_json(args.output, report)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"packets", "candidate_exports"}
            },
            indent=2,
        )
    )
    for export in report["candidate_exports"]:
        print(
            json.dumps(
                {key: value for key, value in export.items() if key != "rows"},
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
