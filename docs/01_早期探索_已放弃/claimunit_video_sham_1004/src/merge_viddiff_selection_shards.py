"""Merge disjoint VidDiff rejection-sampling shards into one audited selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json, write_jsonl
from shadow_execute_generated_actions import _jsonl


FORBIDDEN_STUDENT_KEYS = {
    "candidate_report",
    "differences_gt",
    "ground_truth",
    "local_path",
    "metrics",
    "oracle_direction",
    "retrieval_frames",
    "url",
    "urls",
    "verifier",
    "video_path",
}


def _forbidden_keys(value: Any, prefix: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}"
            if str(key).lower() in FORBIDDEN_STUDENT_KEYS:
                found.append(child)
            found.extend(_forbidden_keys(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_forbidden_keys(item, f"{prefix}[{index}]"))
    return found


def merge(shard_dirs: list[Path], expected_cases: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates_by_key: dict[str, dict[str, Any]] = {}
    audits_by_key: dict[str, dict[str, Any]] = {}
    seeds: set[int] = set()

    for shard_dir in shard_dirs:
        candidate_path = shard_dir / "candidate_measurements.student.jsonl"
        audit_path = shard_dir / "selection_audit.json"
        if not candidate_path.is_file() or not audit_path.is_file():
            raise ValueError(f"selection shard is incomplete: {shard_dir}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("dataset") != "VidDiffBench":
            raise ValueError(f"unexpected dataset in {audit_path}: {audit.get('dataset')!r}")
        if isinstance(audit.get("seed"), int):
            seeds.add(audit["seed"])

        shard_candidates = _jsonl(candidate_path)
        shard_audits = audit.get("cases", [])
        if len(shard_candidates) != len(shard_audits):
            raise ValueError(f"candidate/audit count mismatch in {shard_dir}")
        for item in shard_candidates:
            key = str(item.get("sample_key"))
            if key in candidates_by_key:
                raise ValueError(f"duplicate VidDiff sample_key across shards: {key}")
            leaks = _forbidden_keys(item)
            if leaks:
                raise ValueError(f"private fields in student candidate {key}: {leaks}")
            candidates_by_key[key] = item
        for item in shard_audits:
            key = str(item.get("sample_key"))
            if key in audits_by_key:
                raise ValueError(f"duplicate VidDiff audit sample_key across shards: {key}")
            audits_by_key[key] = item

    if set(candidates_by_key) != set(audits_by_key):
        missing_candidates = sorted(set(audits_by_key) - set(candidates_by_key))
        missing_audits = sorted(set(candidates_by_key) - set(audits_by_key))
        raise ValueError(
            f"merged candidate/audit keys differ: missing_candidates={missing_candidates}, "
            f"missing_audits={missing_audits}"
        )
    if expected_cases is not None and len(candidates_by_key) != expected_cases:
        raise ValueError(
            f"expected {expected_cases} unique cases, found {len(candidates_by_key)}"
        )

    keys = sorted(candidates_by_key)
    candidates = [candidates_by_key[key] for key in keys]
    cases = [audits_by_key[key] for key in keys]
    accepted = sum(item.get("accepted") is True for item in cases)
    selected_tools: dict[str, int] = {}
    for item in cases:
        tool = item.get("selected_tool")
        if item.get("accepted") is True and isinstance(tool, str):
            selected_tools[tool] = selected_tools.get(tool, 0) + 1

    audit = {
        "run_id": "viddiff_external_holdout_merged_selection_v1",
        "dataset": "VidDiffBench",
        "seed": next(iter(seeds)) if len(seeds) == 1 else sorted(seeds),
        "sample_size": len(cases),
        "policy": "merge disjoint label-blind candidate shards after local verifier-only admission",
        "summary": {
            "shards": len(shard_dirs),
            "unique_cases": len(cases),
            "accepted": accepted,
            "rejected": len(cases) - accepted,
            "acceptance_rate": accepted / len(cases) if cases else 0.0,
            "selected_tools": selected_tools,
            "student_export_private_leak_tokens": [],
            "adaptive_candidate_sets": len(candidates),
            "safe_to_build_adaptive_harness": bool(candidates),
        },
        "cases": cases,
    }
    return candidates, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/selection/viddiff_external_holdout_merged_v1",
    )
    args = parser.parse_args()
    candidates, audit = merge(args.shard_dirs, args.expected_cases)
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "candidate_measurements.student.jsonl", candidates)
    accepted_keys = {
        str(item["sample_key"])
        for item in audit["cases"]
        if item.get("accepted") is True
    }
    write_jsonl(
        args.output / "accepted_measurements.student.jsonl",
        [item for item in candidates if str(item["sample_key"]) in accepted_keys],
    )
    write_json(args.output / "selection_audit.json", audit)
    print(json.dumps(audit["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
