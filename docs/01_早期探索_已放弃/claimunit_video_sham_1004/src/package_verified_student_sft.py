"""Package independently verified stitched SFT rows for training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline import write_json, write_jsonl
from validate_student_batch_sft import validate_rows


FORBIDDEN_STUDENT_VISIBLE_TOKENS = (
    '"candidate_report"',
    '"differences_gt"',
    '"ground_truth"',
    '"local_path"',
    '"oracle_direction"',
    '"retrieval_frames"',
    '"url"',
    '"urls"',
    '"verifier"',
    '"video_path"',
    "file://",
    "http://",
    "https://",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def package(
    paths: list[Path],
    *,
    semantic_audit_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [row for path in paths for row in _read_jsonl(path)]
    validation = validate_rows(rows)
    leaks: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        visible = json.dumps(
            {"messages": row.get("messages"), "tools": row.get("tools")},
            ensure_ascii=False,
        ).lower()
        found = [token for token in FORBIDDEN_STUDENT_VISIBLE_TOKENS if token in visible]
        if found:
            leaks.append(
                {
                    "index": index,
                    "row_id": (row.get("metadata") or {}).get("row_id"),
                    "tokens": found,
                }
            )
    cases = sorted({str((row.get("metadata") or {}).get("case_id")) for row in rows})
    packaged_identities = {
        (
            str((row.get("metadata") or {}).get("dataset")),
            str((row.get("metadata") or {}).get("case_id")),
            str((row.get("metadata") or {}).get("trajectory_mode")),
        )
        for row in rows
    }
    semantic_audit = None
    semantic_coverage_missing: list[list[str]] = []
    if semantic_audit_path is not None:
        semantic_audit = json.loads(semantic_audit_path.read_text(encoding="utf-8"))
        audited_identities = {
            (
                str(item.get("dataset")),
                str(item.get("case_id")),
                str(item.get("mode")),
            )
            for item in semantic_audit.get("episode_reports") or []
            if item.get("semantic_passed") is True
        }
        semantic_coverage_missing = [
            list(identity) for identity in sorted(packaged_identities - audited_identities)
        ]
    semantic_ready = (
        semantic_audit_path is None
        or (
            semantic_audit is not None
            and semantic_audit.get("all_passed") is True
            and not semantic_coverage_missing
        )
    )
    report = {
        "artifact": "verified_student_sft_package_v1",
        "inputs": [str(path) for path in paths],
        "cases": len(cases),
        "case_ids": cases,
        "rows": len(rows),
        "privacy_leaks": leaks,
        "validation": validation,
        "semantic_audit": {
            "provided": semantic_audit_path is not None,
            "path": str(semantic_audit_path) if semantic_audit_path is not None else None,
            "all_passed": semantic_audit.get("all_passed") if semantic_audit else None,
            "coverage_missing": semantic_coverage_missing,
        },
        "ready_for_training": validation["all_passed"] and not leaks and semantic_ready,
    }
    return rows if report["ready_for_training"] else [], report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--semantic-audit", type=Path)
    args = parser.parse_args()
    rows, report = package(args.input, semantic_audit_path=args.semantic_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ready_for_training"] else 1)


if __name__ == "__main__":
    main()
