"""Run UTQR-7 causal qualification on one saved VidDiff trajectory.

The clean and repair demonstrations are replayed from disk.  Only the fixed,
label-blind causal probes and optional blind process judges make new model calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from causal_probe_suite import qualify_and_write_causal_packet
from causal_trajectory_generation import CausalPolicyViewTeacher
from cli_teacher_rollout import CodexCLITeacher
from unified_trajectory_rubric import load_saved_bundle
from viddiff_batch import (
    DEFAULT_ADAPTIVE,
    DEFAULT_AUDIT,
    VidDiffAdaptiveMeasurementEnvironment,
    _load_jsonl,
    _restore_viddiff_harness_config,
    _validate,
)


def qualify_saved_case(
    case_dir: Path,
    *,
    adaptive_path: Path,
    audit_path: Path,
    output_dir: Path,
    model: str,
    max_probe_turns: int,
    blind_judges: bool,
) -> dict[str, Any]:
    bundle, _saved_validation = load_saved_bundle(case_dir)
    _restore_viddiff_harness_config(bundle)
    final_claim = bundle["repair_episode"].get("final_claim") or {}
    sample_key = str(final_claim["sample_key"])
    items = {item["sample_key"]: item for item in _load_jsonl(adaptive_path)}
    item = items[sample_key]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    oracle = {row["sample_key"]: row["oracle_direction"] for row in audit["cases"]}
    validation = _validate(bundle, oracle[sample_key], item)
    if not validation["passed"]:
        raise ValueError("saved base trajectory no longer passes local validation")

    routing_variant = str(
        bundle.get("execution_summary", {}).get("routing_variant") or "uniform_first"
    )
    teacher = CausalPolicyViewTeacher(
        CodexCLITeacher(model=model),
        candidates_per_turn=2,
    )

    def environment_factory() -> VidDiffAdaptiveMeasurementEnvironment:
        return VidDiffAdaptiveMeasurementEnvironment(
            item,
            routing_variant=routing_variant,
        )

    _packet, report = qualify_and_write_causal_packet(
        bundle=bundle,
        teacher=teacher,
        dataset="VidDiffBench",
        validation=validation,
        output_dir=output_dir,
        environment_factory=environment_factory,
        run_blind_judges=blind_judges,
        judge_model=model,
        max_probe_turns=max_probe_turns,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--adaptive-path", type=Path, default=DEFAULT_ADAPTIVE)
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--max-probe-turns", type=int, default=16)
    parser.add_argument("--blind-judges", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or args.case_dir / "full"
    report = qualify_saved_case(
        args.case_dir,
        adaptive_path=args.adaptive_path,
        audit_path=args.audit_path,
        output_dir=output_dir,
        model=args.model,
        max_probe_turns=args.max_probe_turns,
        blind_judges=args.blind_judges,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
