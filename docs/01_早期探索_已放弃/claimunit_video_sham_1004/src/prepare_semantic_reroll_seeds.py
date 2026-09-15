"""Materialize the exact local seed rows named by a semantic quarantine manifest."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from pipeline import write_json, write_jsonl
from run_student_closed_loop_batch import DEFAULT_SFT_ROOT, _case_id, _first_rows, _mode
from shadow_execute_generated_actions import _jsonl, _payload


def _episode_fallback_row(item: dict[str, Any]) -> dict[str, Any] | None:
    path_text = str(item.get("episode_path") or "")
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    episode = record.get("episode") or record
    turns = episode.get("turns") or []
    if not turns or not (turns[0].get("sft_datum") or {}):
        return None
    row = copy.deepcopy(turns[0]["sft_datum"])
    roles = [message.get("role") for message in row.get("messages") or []]
    if roles[:2] != ["system", "user"]:
        raise ValueError(f"episode fallback lacks system/user seed messages: {path}")
    dataset = str(item["dataset"])
    case_id = str(item["case_id"])
    mode = str(item["mode"])
    metadata = row.setdefault("metadata", {})
    source_seed_row_id = str(metadata.get("source_seed_row_id") or "")
    if case_id not in source_seed_row_id or not source_seed_row_id.endswith(
        f":{mode}:1"
    ):
        raise ValueError(
            f"episode fallback source identity mismatch for {dataset}/{case_id}/{mode}"
        )
    metadata.update(
        {
            "dataset": dataset,
            "case_id": case_id,
            "row_id": source_seed_row_id,
            "trajectory_type": episode.get("trajectory_type"),
            "seed_recovered_from_student_visible_turn1": True,
        }
    )
    if dataset == "VidDiffBench":
        second_eligible = list((turns[1] if len(turns) > 1 else {}).get(
            "visible_eligible_tools"
        ) or [])
        if len(second_eligible) > 1:
            metadata["routing_variant"] = "balanced"
        elif second_eligible:
            first = str(second_eligible[0])
            metadata["routing_variant"] = {
                "compare_uniform_timeline": "uniform_first",
                "compare_global_motion_symmetric": "global_first",
                "compare_action_refined_symmetric": "action_first",
                "compare_predicate_independent": "predicate_first",
            }.get(first, "balanced")
        else:
            metadata["routing_variant"] = "balanced"
    _payload(row)
    return row


def prepare(
    manifest: dict[str, Any],
    source_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested = [
        (
            str(item["dataset"]),
            str(item["case_id"]),
            str(item["mode"]),
        )
        for item in manifest.get("items") or []
    ]
    if len(requested) != len(set(requested)):
        raise ValueError("quarantine manifest contains duplicate dataset/case/mode entries")

    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in source_rows:
        key = (str(row["metadata"]["dataset"]), _case_id(row), _mode(row))
        by_key.setdefault(key, []).append(row)

    ambiguous = [key for key in requested if len(by_key.get(key) or []) > 1]
    if ambiguous:
        raise ValueError(
            f"reroll seed resolution failed; ambiguous={ambiguous}"
        )
    selected: list[dict[str, Any]] = []
    sources: list[str] = []
    missing: list[tuple[str, str, str]] = []
    for index, key in enumerate(requested):
        candidates = by_key.get(key) or []
        if candidates:
            selected.append(candidates[0])
            sources.append("source_sft_jsonl")
            continue
        fallback = _episode_fallback_row((manifest.get("items") or [])[index])
        if fallback is None:
            missing.append(key)
            continue
        selected.append(fallback)
        sources.append("student_visible_turn1_fallback")
    if missing:
        raise ValueError(f"reroll seed resolution failed; missing={missing}")
    report = {
        "artifact": "semantic_quarantine_exact_reroll_seed_report_v1",
        "requested_episodes": len(requested),
        "resolved_episodes": len(selected),
        "exact_match": len(selected) == len(requested),
        "teacher_or_external_service_called": False,
        "selection": [
            {
                "dataset": key[0],
                "case_id": key[1],
                "mode": key[2],
                "seed_row_id": selected[index]["metadata"]["row_id"],
                "resolution_source": sources[index],
                "prior_failure_codes": list(
                    (manifest.get("items") or [])[index].get("failure_codes") or []
                ),
            }
            for index, key in enumerate(requested)
        ],
    }
    return selected, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sft-root", type=Path, default=DEFAULT_SFT_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        action="append",
        help="Explicit source SFT JSONL; repeat to combine local splits.",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.source_jsonl:
        all_rows = [row for path in args.source_jsonl for row in _jsonl(path)]
        source_rows = [
            row
            for row in all_rows
            if str(row["metadata"]["row_id"]).endswith(":clean:1")
            or str(row["metadata"]["row_id"]).endswith(":repair:1")
        ]
    else:
        source_rows = _first_rows(args.sft_root, args.split)
    rows, report = prepare(manifest, source_rows)
    write_jsonl(args.output_jsonl, rows)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
