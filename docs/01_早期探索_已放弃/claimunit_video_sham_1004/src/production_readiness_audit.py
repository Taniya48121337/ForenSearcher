"""Advisory cross-dataset readiness audit for accepted executable SFT trajectories."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json


CURRENT_RUBRIC = ROOT / "rubric" / "unified_trajectory_quality_rubric.json"


DEFAULT_RUNS = {
    "VideoSham": [
        ROOT / "outputs" / "utqr_v2_videosham_17_consolidated",
    ],
    "VCSL": [
        ROOT / "outputs" / "scale30_vcsl_teacher_consolidated",
    ],
    "VidDiffBench": [
        ROOT / "outputs" / "scale30_viddiff_consolidated",
    ],
}
DEFAULT_HARD_NEGATIVES = ROOT / "outputs" / "rubric_v4_scale" / "hard_negative_scale_calibration.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "utqr_v2_readiness" / "cross_dataset_readiness.json"
DEFAULT_STUDENT_BATCH = (
    ROOT
    / "outputs/student_external_contract_v1/student_batch_test_v10_native/batch_report.json"
)
DEFAULT_STUDENT_SFT_VALIDATION = (
    ROOT
    / "outputs/student_external_contract_v1/student_batch_test_v10_native/sft_validation.json"
)
DEFAULT_ADDITIONAL_STUDENT_BATCHES = [
    (
        ROOT
        / "outputs/student_external_contract_v1/student_batch_videosham_external_holdout_v1/batch_report.json",
        ROOT
        / "outputs/student_external_contract_v1/student_batch_videosham_external_holdout_v1/sft_validation.json",
    ),
    (
        ROOT
        / "outputs/student_external_contract_v1/student_batch_vcsl_spd_external_holdout_v2/batch_report.json",
        ROOT
        / "outputs/student_external_contract_v1/student_batch_vcsl_spd_external_holdout_v2/sft_validation.json",
    ),
    (
        ROOT
        / "outputs/student_external_contract_v1/student_external_viddiff_strict9_consumption_contract_v3/batch_report.json",
        ROOT
        / "outputs/student_external_contract_v1/student_external_viddiff_strict9_consumption_contract_v3/sft_validation.json",
    ),
    (
        ROOT
        / "outputs/student_external_contract_v1/student_external_viddiff_symmetry4_seed1506_v1/batch_report.json",
        ROOT
        / "outputs/student_external_contract_v1/student_external_viddiff_symmetry4_seed1506_v1/sft_validation.json",
    ),
]
DEFAULT_CLOSED_LOOP_HARD_NEGATIVES = (
    ROOT
    / "outputs/student_external_contract_v1/student_closed_loop_rubric_hard_negative_calibration.json"
)
DEFAULT_QUALITY_RESCORES = {
    "VideoSham": ROOT / "outputs" / "rubric_v4_scale" / "videosham_rescore.json",
    "VCSL": ROOT / "outputs" / "rubric_v4_scale" / "vcsl_rescore.json",
    "VidDiffBench": ROOT / "outputs" / "rubric_v4_scale" / "viddiff_rescore.json",
}
DEFAULT_PASS_K_REPORTS = {
    "VideoSham": [ROOT / "outputs" / "pass_k_videosham_4054" / "pass_k_report.json"],
    "VCSL": [ROOT / "outputs" / "pass_k_vcsl_axis_case10" / "pass_k_report.json"],
    "VidDiffBench": [ROOT / "outputs" / "pass_k_viddiff_surgery159" / "pass_k_report.json"],
}
DEFAULT_STUDENT_PASS_K_REPORTS = [
    ROOT / "outputs/student_external_contract_v1/student_passk_videosham1007_v1/pass_k_report.json",
    ROOT / "outputs/student_external_contract_v1/student_passk_vcsl_localalign_v1/pass_k_report.json",
    ROOT / "outputs/student_external_contract_v1/student_passk_viddiff26_v1/pass_k_report.json",
    ROOT / "outputs/student_external_contract_v1/student_repair_passk_videosham2010_v1/pass_k_report.json",
    ROOT / "outputs/student_external_contract_v1/student_repair_passk_vcsl_dp_v1/pass_k_report.json",
    ROOT / "outputs/student_external_contract_v1/student_repair_passk_viddiff27_v1/pass_k_report.json",
    ROOT / "outputs/student_external_contract_v1/student_external_passk_0064_v1/pass_k_report.json",
    ROOT / "outputs/student_external_contract_v1/student_external_passk_0105_v1/pass_k_report.json",
    ROOT / "outputs/student_external_contract_v1/student_external_passk_1028_v1/pass_k_report.json",
    ROOT / "outputs/student_external_contract_v1/student_external_passk_4008_v1/pass_k_report.json",
    ROOT / "outputs/student_external_contract_v1/student_external_passk_4126_v1/pass_k_report.json",
    ROOT / "outputs/student_external_contract_v1/student_external_passk_1004_v1/pass_k_report.json",
    ROOT / "outputs/student_external_contract_v1/student_external_passk_4020_v1/pass_k_report.json",
    ROOT / "outputs/student_external_contract_v1/student_external_passk_4025_v1/pass_k_report.json",
    ROOT / "outputs/student_external_contract_v1/student_external_passk_4059_v1/pass_k_report.json",
    ROOT / "outputs/student_external_contract_v1/student_external_passk_4061_v1/pass_k_report.json",
    ROOT / "outputs/student_external_contract_v1/student_external_passk_4073_v1/pass_k_report.json",
    ROOT / "outputs/student_external_contract_v1/student_external_passk_4127_v1/pass_k_report.json",
    ROOT / "outputs/student_external_contract_v1/student_external_passk_4214_v1/pass_k_report.json",
    ROOT / "outputs/student_external_contract_v1/student_external_passk_vcsl_spd_086c_v1/pass_k_report.json",
    ROOT / "outputs/student_external_contract_v1/student_external_passk_vcsl_spd_5c2c_v1/pass_k_report.json",
    ROOT / "outputs/student_external_contract_v1/student_external_passk_viddiff_symmetry127_v1/pass_k_report.json",
    ROOT / "outputs/student_external_contract_v1/student_external_passk_viddiff_symmetry15_v1/pass_k_report.json",
    ROOT / "outputs/student_external_contract_v1/student_external_passk_viddiff_symmetry166_v1/pass_k_report.json",
    ROOT / "outputs/student_external_contract_v1/student_external_passk_viddiff_symmetry78_v1/pass_k_report.json",
]
DEFAULT_MEASUREMENT_POOLS = {
    "VideoSham": ROOT
    / "outputs"
    / "selection"
    / "videosham_measurement_probe_v7_streaming_rgb_all_local_n30"
    / "measurement_probe_report.json",
    "VCSL": ROOT
    / "outputs"
    / "selection"
    / "vcsl_best_of_tree_seed_20260818_n30"
    / "selection_audit.json",
    "VidDiffBench": ROOT
    / "outputs"
    / "selection"
    / "viddiff_external_holdout30_symmetry_abstention_v1"
    / "selection_audit.json",
}
DEFAULT_MEASUREMENT_POOL_SUPPLEMENTS = {
    "VCSL": [
        ROOT
        / "outputs/student_external_contract_v1/vcsl_official_spd_experiment_v1.json"
    ],
}
REPAIR_OPERATORS = ("evidence_dropout", "binding_invalidation", "decision_reopen")
STUDENT_HELD_OUT_TARGET_PER_DATASET = 30
STUDENT_DATASETS = ("VideoSham", "VCSL", "VidDiffBench")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _student_closed_loop_evidence(
    batch_path: Path,
    sft_validation_path: Path,
    hard_negative_path: Path,
    additional_batches: list[tuple[Path, Path]],
    pass_k_report_paths: list[Path],
) -> dict[str, Any]:
    sources = [(batch_path, sft_validation_path), *additional_batches]
    batches = [_load(batch) for batch, _validation in sources]
    validations = [_load(validation) for _batch, validation in sources]
    hard_negatives = _load(hard_negative_path)
    episodes = [episode for batch in batches for episode in batch.get("episodes", [])]
    episode_case_ids = {str(episode["case_id"]) for episode in episodes}
    pass_k_reports = [
        _load(path) for path in pass_k_report_paths if Path(path).exists()
    ]
    measured_pass_k = [
        report
        for report in pass_k_reports
        if str(report.get("case_id")) in episode_case_ids
        and int(report.get("independent_rollouts", 0)) >= 2
        and report.get("distinct_recorded_seeds") is True
    ]
    pass_k_case_ids = {str(report["case_id"]) for report in measured_pass_k}
    by_dataset: dict[str, set[str]] = defaultdict(set)
    for episode in episodes:
        by_dataset[str(episode["dataset"])].add(str(episode["case_id"]))
    held_out_cases = {name: len(case_ids) for name, case_ids in by_dataset.items()}
    held_out_shortfall = {
        name: max(
            STUDENT_HELD_OUT_TARGET_PER_DATASET - held_out_cases.get(name, 0),
            0,
        )
        for name in STUDENT_DATASETS
    }
    summaries = [batch.get("summary", {}) for batch in batches]
    episode_count = sum(int(summary.get("episodes", 0)) for summary in summaries)
    accepted_count = sum(
        int(summary.get("accepted_episodes", 0)) for summary in summaries
    )
    quality_passed = (
        batches[0].get("split") == "test"
        and episode_count > 0
        and accepted_count == episode_count
        and min(int(summary.get("minimum_score", 0)) for summary in summaries) >= 98
        and sum(int(summary.get("episodes_with_rejection", 0)) for summary in summaries) == 0
        and all(validation.get("all_passed") is True for validation in validations)
        and hard_negatives.get("all_hard_negatives_rejected") is True
    )
    scale_requirements = {
        "held_out_cases_per_dataset_at_least_30": bool(held_out_cases)
        and not any(held_out_shortfall.values()),
        "independent_rollouts_per_case_at_least_2": bool(episode_case_ids)
        and pass_k_case_ids == episode_case_ids,
    }
    return {
        "batch_sources": [str(batch) for batch, _validation in sources],
        "sft_validation_sources": [
            str(validation) for _batch, validation in sources
        ],
        "hard_negative_source": str(hard_negative_path),
        "splits": [batch.get("split") for batch in batches],
        "held_out_cases_by_dataset": held_out_cases,
        "held_out_target_per_dataset": STUDENT_HELD_OUT_TARGET_PER_DATASET,
        "held_out_shortfall_by_dataset": held_out_shortfall,
        "total_additional_held_out_cases_required": sum(held_out_shortfall.values()),
        "episodes": episode_count,
        "accepted_episodes": accepted_count,
        "actions": sum(int(summary.get("actions", 0)) for summary in summaries),
        "accepted_sft_rows": sum(
            int(summary.get("accepted_sft_rows", 0)) for summary in summaries
        ),
        "diversity_balanced_sft_rows": sum(
            int(summary.get("diversity_balanced_sft_rows", 0))
            for summary in summaries
        ),
        "minimum_score": min(
            int(summary.get("minimum_score", 0)) for summary in summaries
        ),
        "episodes_with_rejection": sum(
            int(summary.get("episodes_with_rejection", 0)) for summary in summaries
        ),
        "measured_episode_seconds": round(
            sum(float(summary.get("measured_episode_seconds", 0.0)) for summary in summaries),
            3,
        ),
        "batch_summaries": summaries,
        "sft_validation_passed": all(
            validation.get("all_passed", False) for validation in validations
        ),
        "sft_failed_rows": sum(
            int(validation.get("failed_rows", 0)) for validation in validations
        ),
        "exact_reasoning_duplicates": sum(
            int(validation.get("exact_reasoning_duplicates", 0))
            for validation in validations
        ),
        "closed_loop_hard_negative_families": hard_negatives.get("families", 0),
        "closed_loop_hard_negatives_rejected": hard_negatives.get("rejected", 0),
        "higher_order_pass_k": {
            "measured_cases": len(pass_k_case_ids),
            "evaluated_case_coverage": round(
                len(pass_k_case_ids) / len(episode_case_ids), 4
            )
            if episode_case_ids
            else 0.0,
            "reports": [
                {
                    "case_id": report["case_id"],
                    "independent_rollouts": report["independent_rollouts"],
                    "pass_k": report["pass_k"],
                    "rollout_seeds": report.get("rollout_seeds"),
                }
                for report in measured_pass_k
            ],
        },
        "quality_canary_passed": quality_passed,
        "scale_requirements": scale_requirements,
        "scale_validation_passed": all(scale_requirements.values()),
    }


def _p10(scores: list[float]) -> float:
    ordered = sorted(scores)
    return ordered[max(0, math.ceil(0.10 * len(ordered)) - 1)] if ordered else 0.0


def _family(dataset: str, case: dict[str, Any]) -> str:
    if dataset == "VideoSham":
        return str(case.get("capability_family", "unknown"))
    if dataset == "VCSL":
        return str(case.get("selected_localization_method", "unknown"))
    return str(case.get("selected_measurement_tool", "unknown"))


def _case_is_gold(case: dict[str, Any]) -> bool:
    return (
        case.get("status") == "passed"
        and float(case.get("rubric_score", 0.0)) >= 92.0
        and not case.get("rubric_failed_gates", [])
        and not case.get("rubric_failed_criteria", [])
    )


def _teacher_started(case: dict[str, Any]) -> bool:
    return case.get("rubric_score") is not None or bool(case.get("attempts"))


def _deduplicated_cases(dataset: str, run_roots: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    by_id: dict[str, dict[str, Any]] = {}
    sources: list[str] = []
    for run_root in run_roots:
        run_root = Path(run_root)
        report = _load(run_root / "batch_report.json")
        sources.append(str(run_root))
        for case in report.get("cases", []):
            if case.get("status") == "preflight_blocked":
                continue
            enriched = dict(case)
            enriched["source_run"] = str(run_root)
            by_id[str(case["case_id"])] = enriched
    return list(by_id.values()), sources


def _dataset_report(
    dataset: str,
    run_roots: list[Path],
    pass_k_report_paths: list[Path] | None = None,
    measurement_pool_path: Path | None = None,
    quality_rescore_path: Path | None = None,
    measurement_supplement_paths: list[Path] | None = None,
) -> dict[str, Any]:
    cases, sources = _deduplicated_cases(dataset, run_roots)
    rescore_source = None
    rescored_case_count = 0
    if quality_rescore_path is not None and Path(quality_rescore_path).exists():
        rescore_source = str(Path(quality_rescore_path))
        rescore = _load(Path(quality_rescore_path))
        rescored = {
            str(row["case_id"]): row
            for row in rescore.get("cases", [])
            if row.get("scored") and row.get("case_id") is not None
        }
        for case in cases:
            row = rescored.get(str(case.get("case_id")))
            if row is None:
                continue
            diagnosis = row.get("diagnosis", {})
            case["rubric_score"] = row.get("total_score")
            case["rubric_grade"] = row.get("grade")
            case["rubric_failed_gates"] = diagnosis.get("failed_gates", [])
            case["rubric_failed_criteria"] = diagnosis.get("failed_criteria", [])
            case["status"] = "passed" if row.get("accepted_for_sft") else "rubric_rejected"
            rescored_case_count += 1
    teacher_cases = [case for case in cases if _teacher_started(case)]
    scores = [float(case.get("rubric_score", 0.0)) for case in teacher_cases if case.get("rubric_score") is not None]
    gold = [case for case in teacher_cases if _case_is_gold(case)]
    repair_counts = Counter(str(case.get("repair_operator", "unknown")) for case in teacher_cases)
    repair_gold = Counter(str(case.get("repair_operator", "unknown")) for case in gold)
    family_counts = Counter(_family(dataset, case) for case in cases)
    pool_case_count = len(cases)
    pool_reachable_count = len(teacher_cases)
    pool_source = None
    pool_supplement_sources: list[str] = []
    if measurement_pool_path is not None and Path(measurement_pool_path).exists():
        pool = _load(measurement_pool_path)
        pool_source = str(Path(measurement_pool_path))
        if dataset == "VideoSham":
            pool_case_count = int(pool.get("selected_cases", pool_case_count))
            pool_reachable_count = int(pool.get("status_counts", {}).get("measurement_reachable", 0))
            family_counts = Counter({
                family: int(values.get("cases", 0))
                for family, values in pool.get("family_audit", {}).items()
            })
        elif dataset in {"VCSL", "VidDiffBench"}:
            pool_summary = pool.get("summary", {})
            pool_case_count = int(
                pool.get("sample_size", pool_summary.get("cases", pool_case_count))
            )
            pool_reachable_count = int(
                pool_summary.get(
                    "accepted",
                    pool_summary.get("revised_strict_accepted", 0),
                )
            )
            family_key = "selected_localizers" if dataset == "VCSL" else "selected_tools"
            family_counts = Counter(pool_summary.get(family_key, {}))
    for supplement_path in measurement_supplement_paths or []:
        supplement_path = Path(supplement_path)
        if not supplement_path.exists() or dataset != "VCSL":
            continue
        supplement = _load(supplement_path)
        supplement_cases = int(supplement.get("summary", {}).get("cases", 0))
        if supplement_cases != pool_case_count:
            raise ValueError(
                f"VCSL measurement supplement has a different case universe: "
                f"{supplement_cases} != {pool_case_count}"
            )
        combined = int(supplement.get("summary", {}).get("combined_reachable", 0))
        newly_recovered = int(supplement.get("summary", {}).get("newly_recovered", 0))
        pool_reachable_count = max(pool_reachable_count, combined)
        if newly_recovered:
            family_counts["spd_official"] += newly_recovered
        pool_supplement_sources.append(str(supplement_path))
    selected: Counter[str] = Counter()
    indispensable: Counter[str] = Counter()
    total_tokens = 0
    for case in cases:
        telemetry = case.get("tool_telemetry") or {
            "selected_calls": case.get("selected_calls", {}),
            "indispensable_calls": case.get("indispensable_calls", {}),
        }
        selected.update(telemetry.get("selected_calls", {}))
        indispensable.update(telemetry.get("indispensable_calls", {}))
        total_tokens += int(case.get("usage", {}).get("total_tokens", 0))

    repair_rates = {
        operator: {
            "cases": repair_counts.get(operator, 0),
            "gold": repair_gold.get(operator, 0),
            "gold_rate": (
                round(repair_gold.get(operator, 0) / repair_counts[operator], 4)
                if repair_counts.get(operator, 0)
                else None
            ),
        }
        for operator in REPAIR_OPERATORS
    }
    extra_calls = {
        name: count - indispensable.get(name, 0)
        for name, count in selected.items()
        if count > indispensable.get(name, 0)
    }
    case_ids = {str(case["case_id"]) for case in cases}
    pass_k_reports = [
        _load(path)
        for path in (pass_k_report_paths or [])
        if Path(path).exists()
    ]
    measured_pass_k = [
        report
        for report in pass_k_reports
        if str(report.get("case_id")) in case_ids
        and int(report.get("independent_rollouts", 0)) >= 2
    ]
    teacher_gold_rate = len(gold) / len(teacher_cases) if teacher_cases else 0.0
    measurement_reachability_rate = pool_reachable_count / pool_case_count if pool_case_count else 0.0
    quality_canary_passed = (
        bool(teacher_cases)
        and teacher_gold_rate >= 0.90
        and _p10(scores) >= 92
        and not extra_calls
    )
    scale_requirements = {
        "dataset_cases_at_least_30": pool_case_count >= 30,
        "measurement_candidate_reachability_at_least_90_percent": measurement_reachability_rate >= 0.90,
        "each_observed_family_at_least_30": bool(family_counts) and min(family_counts.values()) >= 30,
        "all_repair_operators_represented": all(repair_counts.get(operator, 0) > 0 for operator in REPAIR_OPERATORS),
        "each_repair_operator_at_least_85_percent_gold": all(
            repair_counts.get(operator, 0) > 0
            and repair_gold.get(operator, 0) / repair_counts[operator] >= 0.85
            for operator in REPAIR_OPERATORS
        ),
        # One hard-case repeat is useful evidence, but scale requires every
        # evaluated case; route portfolios never count as independent repeats.
        "multi_rollout_pass_k_measured": bool(teacher_cases) and len(measured_pass_k) == len(teacher_cases),
    }
    return {
        "dataset": dataset,
        "source_runs": sources,
        "quality_rescore_source": rescore_source,
        "quality_rescored_case_count": rescored_case_count,
        "case_count": pool_case_count,
        "measurement_pool_source": pool_source,
        "measurement_pool_supplement_sources": pool_supplement_sources,
        "trajectory_case_count": len(cases),
        "teacher_started_case_count": len(teacher_cases),
        "measurement_reachable_case_count": pool_reachable_count,
        "measurement_unreachable_case_count": pool_case_count - pool_reachable_count,
        "measurement_reachability_rate": round(measurement_reachability_rate, 4),
        "gold_count": len(gold),
        "gold_rate": round(len(gold) / pool_case_count, 4) if pool_case_count else 0.0,
        "teacher_started_gold_rate": round(teacher_gold_rate, 4),
        "score_median": round(statistics.median(scores), 3) if scores else 0.0,
        "score_p10": round(_p10(scores), 3),
        "hard_gate_failures_in_export_candidates": sum(bool(case.get("rubric_failed_gates")) for case in gold),
        "repair_operator_results": repair_rates,
        "capability_family_counts": dict(family_counts),
        "tool_calls": {
            "selected": dict(selected),
            "indispensable": dict(indispensable),
            "selected_but_not_indispensable": extra_calls,
        },
        "teacher_token_cost": {
            "total_tokens": total_tokens,
            "tokens_per_gold": round(total_tokens / len(gold)) if gold else None,
        },
        "observed_pass_power_1": round(teacher_gold_rate, 4),
        "higher_order_pass_k": {
            "measured_cases": len(measured_pass_k),
            "case_coverage": round(len(measured_pass_k) / len(teacher_cases), 4) if teacher_cases else 0.0,
            "reports": [
                {
                    "case_id": report["case_id"],
                    "independent_rollouts": report["independent_rollouts"],
                    "pass_k": report["pass_k"],
                }
                for report in measured_pass_k
            ],
            "route_portfolios_excluded": True,
        },
        "quality_canary_passed": quality_canary_passed,
        "scale_requirements": scale_requirements,
        "scale_validation_passed": all(scale_requirements.values()),
    }


def audit_readiness(
    runs: dict[str, list[Path]],
    *,
    hard_negative_path: Path = DEFAULT_HARD_NEGATIVES,
    pass_k_reports: dict[str, list[Path]] = DEFAULT_PASS_K_REPORTS,
    measurement_pools: dict[str, Path] = DEFAULT_MEASUREMENT_POOLS,
    measurement_pool_supplements: dict[str, list[Path]] = DEFAULT_MEASUREMENT_POOL_SUPPLEMENTS,
    quality_rescores: dict[str, Path] = DEFAULT_QUALITY_RESCORES,
    student_batch_path: Path = DEFAULT_STUDENT_BATCH,
    student_sft_validation_path: Path = DEFAULT_STUDENT_SFT_VALIDATION,
    closed_loop_hard_negative_path: Path = DEFAULT_CLOSED_LOOP_HARD_NEGATIVES,
    additional_student_batches: list[tuple[Path, Path]] = DEFAULT_ADDITIONAL_STUDENT_BATCHES,
    student_pass_k_reports: list[Path] = DEFAULT_STUDENT_PASS_K_REPORTS,
) -> dict[str, Any]:
    datasets = [
        _dataset_report(
            dataset,
            roots,
            pass_k_reports.get(dataset, []),
            measurement_pools.get(dataset),
            quality_rescores.get(dataset),
            measurement_pool_supplements.get(dataset, []),
        )
        for dataset, roots in runs.items()
    ]
    hard_negatives = _load(hard_negative_path)
    negative_rows = hard_negatives.get("hard_negatives", [])
    by_family: dict[str, list[bool]] = defaultdict(list)
    for row in negative_rows:
        by_family[str(row["family"])].append(bool(row["detected"]))
    family_rates = {
        family: round(sum(values) / len(values), 4) for family, values in sorted(by_family.items())
    }
    negative_rate = (
        round(sum(bool(row.get("detected")) for row in negative_rows) / len(negative_rows), 4)
        if negative_rows
        else 0.0
    )
    hard_negative_quality_passed = (
        negative_rate >= 0.90 and bool(family_rates) and min(family_rates.values()) >= 0.80
    )
    quality_passed = all(row["quality_canary_passed"] for row in datasets) and hard_negative_quality_passed
    scale_passed = all(row["scale_validation_passed"] for row in datasets)
    current_rubric_id = _load(CURRENT_RUBRIC)["rubric_id"]
    student_evidence = _student_closed_loop_evidence(
        student_batch_path,
        student_sft_validation_path,
        closed_loop_hard_negative_path,
        additional_student_batches,
        student_pass_k_reports,
    )
    held_out_counts = student_evidence["held_out_cases_by_dataset"]
    held_out_progress = ", ".join(
        f"{dataset}={count}" for dataset, count in sorted(held_out_counts.items())
    )
    evaluated_cases = sum(int(count) for count in held_out_counts.values())
    repeated_cases = int(student_evidence["higher_order_pass_k"]["measured_cases"])
    return {
        "audit_type": "advisory_not_a_runtime_gate",
        "rubric_id": current_rubric_id,
        "status": (
            "mass_production_evidence_complete"
            if quality_passed and scale_passed
            else "quality_canaries_passed_scale_validation_pending"
            if quality_passed
            else "quality_canary_repair_required"
        ),
        "quality_canaries_passed": quality_passed,
        "scale_validation_passed": scale_passed,
        "datasets": datasets,
        "student_closed_loop_evidence": student_evidence,
        "hard_negative_calibration": {
            "source_rubric_id": hard_negatives.get("rubric_id"),
            "count": len(negative_rows),
            "detected": sum(bool(row.get("detected")) for row in negative_rows),
            "detection_rate": negative_rate,
            "per_family_detection_rate": family_rates,
            "quality_threshold_passed": hard_negative_quality_passed,
            "scale_note": (
                "cross-dataset minimal edits cover every current Gold case; this validates detector behavior "
                "on the observed pool but is not a confidence interval for unseen perturbations"
            ),
        },
        "next_evidence_needed": [
            f"expand student free-run evaluation from current held-out counts "
            f"({held_out_progress}) to at least 30 per dataset",
            "expand teacher trajectories to at least 30 cases per dataset and capability family",
            "retain at least 85% Gold for each repair operator",
            f"expand repeated independent student rollouts from {repeated_cases}/{evaluated_cases} "
            "evaluated cases to every evaluated case",
            "expand minimal-edit hard negatives beyond one example per perturbation family",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hard-negatives", type=Path, default=DEFAULT_HARD_NEGATIVES)
    args = parser.parse_args()
    report = audit_readiness(DEFAULT_RUNS, hard_negative_path=args.hard_negatives)
    write_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "quality_canaries_passed": report["quality_canaries_passed"],
        "scale_validation_passed": report["scale_validation_passed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
