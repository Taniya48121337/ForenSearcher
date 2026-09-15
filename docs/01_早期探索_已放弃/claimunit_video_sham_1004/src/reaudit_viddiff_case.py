"""Re-audit a saved VidDiff CLI rollout without calling the teacher again.

This is intentionally a replay-only recovery path: it rebuilds the per-turn
SFT targets from the saved transcript, reruns the causal/grounding audit, and
rewrites the case artifacts.  No model call or new measurement is performed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from llm_teacher_rollout import _accepted_turn_datums, _write_bundle
from pipeline import write_json, write_jsonl
from step_utility_auditor import apply_step_utility_filter, audit_step_utility
from viddiff_batch import DEFAULT_ADAPTIVE, DEFAULT_AUDIT, _load_jsonl, _validate


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _refresh_batch_outputs(
    case_dir: Path,
    bundle: dict[str, Any],
    report: dict[str, Any],
) -> None:
    run_root = case_dir.parent.parent
    batch_path = run_root / "batch_report.json"
    if not batch_path.exists():
        return
    batch = _read_json(batch_path)
    selected = Counter(
        turn["tool_name"]
        for episode_name in ("clean_episode", "repair_episode")
        for turn in bundle[episode_name]["turn_history"]
    )
    indispensable = Counter(
        entry["tool_name"]
        for entries in bundle["step_utility_report"]["episodes"].values()
        for entry in entries
        if entry["verdict"] == "indispensable"
    )
    for row in batch["cases"]:
        if row["case_id"] != bundle["case_id"]:
            continue
        row.update(
            {
                "status": "passed" if report["passed"] else "validation_failed",
                "failed_checks": [name for name, value in report["checks"].items() if not value],
                "selected_measurement_tool": (bundle["repair_episode"].get("final_claim") or {}).get(
                    "measurement_method"
                ),
                "repair_operator": bundle["execution_summary"].get("repair_operator"),
                "clean_turns": len(bundle["clean_episode"]["turn_history"]),
                "repair_turns": len(bundle["repair_episode"]["turn_history"]),
                "usage": bundle["execution_summary"]["api_usage"],
                "selected_calls": dict(selected),
                "indispensable_calls": dict(indispensable),
            }
        )
        row.pop("error", None)
        row.pop("error_type", None)
        break

    passed = sum(row["status"] == "passed" for row in batch["cases"])
    batch["summary"] = {
        "passed": passed,
        "failed_after_cli_start": sum(
            row["status"] in {"execution_failed", "validation_failed"} for row in batch["cases"]
        ),
        "safe_to_compile_sft": passed == batch["accepted_input_count"] and bool(batch["cases"]),
    }
    selected_total: Counter[str] = Counter()
    indispensable_total: Counter[str] = Counter()
    for row in batch["cases"]:
        selected_total.update(row.get("selected_calls", {}))
        indispensable_total.update(row.get("indispensable_calls", {}))
    batch["tool_telemetry"] = {
        "selected_calls": dict(selected_total),
        "indispensable_calls": dict(indispensable_total),
        "selected_but_not_indispensable": {
            name: count - indispensable_total.get(name, 0)
            for name, count in selected_total.items()
            if count > indispensable_total.get(name, 0)
        },
    }
    write_json(batch_path, batch)

    accepted_clean: list[dict[str, Any]] = []
    accepted_repair: list[dict[str, Any]] = []
    accepted_sidecar: list[dict[str, Any]] = []
    for candidate_dir in sorted((run_root / "cases").iterdir()):
        validation_path = candidate_dir / "llm_teacher_validation_report.json"
        if not validation_path.exists() or not _read_json(validation_path).get("passed"):
            continue
        accepted_clean.extend(_load_jsonl(candidate_dir / "llm_teacher_sft_clean.jsonl"))
        accepted_repair.extend(_load_jsonl(candidate_dir / "llm_teacher_sft_repair.jsonl"))
        accepted_sidecar.extend(_load_jsonl(candidate_dir / "llm_teacher_transition_sidecar.jsonl"))
    write_jsonl(run_root / "sft_clean_accepted.jsonl", accepted_clean)
    write_jsonl(run_root / "sft_repair_accepted.jsonl", accepted_repair)
    write_jsonl(run_root / "transition_sidecar.jsonl", accepted_sidecar)


def reaudit(case_dir: Path, adaptive_path: Path, audit_path: Path) -> dict[str, Any]:
    clean = _read_json(case_dir / "llm_teacher_clean_episode.json")
    repair = _read_json(case_dir / "llm_teacher_repair_episode.json")
    summary = _read_json(case_dir / "llm_teacher_execution_summary.json")
    initial_payload = json.loads(clean["messages"][1]["content"])
    sample_key = (repair.get("final_claim") or clean.get("final_claim") or {})["sample_key"]
    case_id = f"viddiff-{sample_key}"

    clean_sft, clean_excluded = _accepted_turn_datums(
        clean["messages"],
        clean["turn_history"],
        case_id=case_id,
        trajectory_id=clean["trajectory_id"],
        trajectory_type="llm_teacher_clean_acquisition",
    )
    repair_sft, repair_excluded = _accepted_turn_datums(
        repair["messages"],
        repair["turn_history"],
        case_id=case_id,
        trajectory_id=repair["trajectory_id"],
        trajectory_type=repair.get("trajectory_type", "llm_teacher_evidence_dropout_repair"),
    )
    summary["excluded_rejected_sft_turns"] = {
        "clean": clean_excluded,
        "repair": repair_excluded,
    }
    schema = initial_payload["claim_schema"]
    bundle = {
        "case_id": case_id,
        "clean_episode": clean,
        "repair_episode": repair,
        "clean_turn_sft": clean_sft,
        "repair_turn_sft": repair_sft,
        "execution_summary": summary,
        "harness_config": {
            "claim_schema": schema,
            "tool_registry": initial_payload["tool_registry"],
            "target_fields": schema["target_fields"],
            "invariant_fields": schema["invariant_fields"],
        },
    }
    step_report = audit_step_utility(bundle)
    apply_step_utility_filter(bundle, step_report)

    candidates = {item["sample_key"]: item for item in _load_jsonl(adaptive_path)}
    oracle_rows = _read_json(audit_path)["cases"]
    oracle = {item["sample_key"]: item["oracle_direction"] for item in oracle_rows}
    report = _validate(bundle, oracle[sample_key], candidates[sample_key])
    _write_bundle(bundle, report, promote=False, output_dir=case_dir)
    _refresh_batch_outputs(case_dir, bundle, report)
    return {
        "case_id": case_id,
        "passed": report["passed"],
        "checks": report["checks"],
        "step_utility_summary": step_report["summary"],
        "teacher_called": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the audits for one saved VidDiff teacher case")
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--adaptive-path", type=Path, default=DEFAULT_ADAPTIVE)
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    print(json.dumps(reaudit(args.case_dir, args.adaptive_path, args.audit_path), indent=2))


if __name__ == "__main__":
    main()
