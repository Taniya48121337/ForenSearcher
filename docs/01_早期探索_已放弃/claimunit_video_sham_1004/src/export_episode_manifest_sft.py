"""Export embedded, accepted per-turn SFT rows from an episode manifest."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from pipeline import write_json, write_jsonl
from validate_student_batch_sft import diversity_balance_rows, validate_rows


def export(
    manifest_path: Path,
    *,
    max_exact_reasoning_per_tool: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    failures = []
    identities = set()
    for item in manifest.get("items") or []:
        path = Path(str(item["episode_path"]))
        artifact = json.loads(path.read_text(encoding="utf-8"))
        episode = artifact.get("episode") or artifact
        quality = artifact.get("quality") or {}
        identity = (str(item["dataset"]), str(item["case_id"]), str(item["mode"]))
        identities.add(identity)
        if quality.get("accepted_for_high_quality_sft") is not True:
            failures.append({"identity": list(identity), "reason": "source_not_accepted"})
            continue
        for turn in episode.get("turns") or []:
            if turn.get("status") not in {"accepted", "executed"}:
                failures.append(
                    {"identity": list(identity), "turn": turn.get("turn"), "reason": "turn_not_executed"}
                )
                continue
            row = copy.deepcopy(turn.get("sft_datum") or {})
            if not row:
                failures.append(
                    {"identity": list(identity), "turn": turn.get("turn"), "reason": "missing_sft_datum"}
                )
                continue
            metadata = row.setdefault("metadata", {})
            metadata.setdefault("dataset", identity[0])
            metadata.setdefault("case_id", identity[1])
            metadata.setdefault("trajectory_mode", identity[2])
            metadata.setdefault("turn_index", turn.get("turn"))
            metadata["strict_closed_loop_score"] = quality.get("score")
            metadata["strict_closed_loop_accepted"] = True
            metadata.setdefault("teacher_action_used", False)
            metadata.setdefault("oracle_or_verifier_used", False)
            metadata.setdefault(
                "row_id",
                f"{identity[0]}:{identity[1]}:utqr-v6:{identity[2]}:{turn.get('turn')}",
            )
            rows.append(row)
    full_validation = validate_rows(rows)
    diversity_report = None
    export_rows = rows
    if max_exact_reasoning_per_tool is not None:
        export_rows, diversity_report = diversity_balance_rows(
            rows,
            max_exact_reasoning_per_tool=max_exact_reasoning_per_tool,
        )
    validation = validate_rows(export_rows)
    ready = not failures and validation.get("all_passed") is True
    report = {
        "artifact": "episode_manifest_embedded_sft_export_v1",
        "episodes": len(identities),
        "full_rows": len(rows),
        "rows": len(export_rows),
        "source_failures": failures,
        "full_validation": full_validation,
        "diversity_balance": diversity_report,
        "validation": validation,
        "ready": ready,
        "ready_for_training": ready,
    }
    return export_rows if report["ready"] else [], report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-exact-reasoning-per-tool", type=int)
    args = parser.parse_args()
    rows, report = export(
        args.manifest,
        max_exact_reasoning_per_tool=args.max_exact_reasoning_per_tool,
    )
    write_jsonl(args.output, rows)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ready"] else 1)


if __name__ == "__main__":
    main()
