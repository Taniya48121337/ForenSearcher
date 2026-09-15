"""Hybrid UTQR-6 qualification for saved, executable cross-dataset episodes.

The machine layer never pretends to judge naturalness.  Blind judge records are
kept separate and are required before an episode is called Gold.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from pipeline import write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUBRIC = ROOT / "rubric" / "unified_trajectory_quality_rubric_v6.json"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _p10(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.10 * len(ordered)) - 1)] if ordered else 0.0


def _machine_episode_map(semantic: dict[str, Any], key: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_identity = {
        (str(item.get("dataset")), str(item.get("case_id")), str(item.get("mode"))): item
        for item in semantic.get("episode_reports") or []
    }
    result = {}
    for item in key.get("mapping") or []:
        identity = (str(item.get("dataset")), str(item.get("case_id")), str(item.get("mode")))
        result[str(item.get("blind_id"))] = by_identity.get(identity, {})
    return result


def _artifact_hits(packet_row: dict[str, Any], patterns: list[str]) -> list[str]:
    reasoning = "\n".join(
        str(turn.get("reasoning") or "").lower() for turn in packet_row.get("turns") or []
    )
    return [pattern for pattern in patterns if pattern.lower() in reasoning]


def _validate_judgment(
    row: dict[str, Any],
    dimension_ids: set[str],
) -> list[str]:
    problems = []
    scores = row.get("dimension_scores") or {}
    if set(scores) != dimension_ids:
        problems.append("dimension_set_mismatch")
    for dimension_id in dimension_ids:
        score = scores.get(dimension_id)
        if not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 5:
            problems.append(f"invalid_score:{dimension_id}")
    evidence = row.get("dimension_evidence") or {}
    if set(evidence) != dimension_ids:
        problems.append("dimension_evidence_set_mismatch")
    for dimension_id in dimension_ids:
        text = str(evidence.get(dimension_id) or "").strip()
        if len(text) < 12:
            problems.append(f"insufficient_evidence_note:{dimension_id}")
    if not isinstance(row.get("hard_fail"), bool):
        problems.append("hard_fail_not_boolean")
    if not isinstance(row.get("keep_for_sft"), bool):
        problems.append("keep_for_sft_not_boolean")
    if len(str(row.get("overall_comment") or "").strip()) < 20:
        problems.append("overall_comment_too_short")
    if not str(row.get("judge_id") or "").strip():
        problems.append("judge_id_missing")
    return problems


def evaluate(
    package_report_path: Path,
    semantic_audit_path: Path,
    blind_packet_path: Path,
    review_key_path: Path,
    judgment_paths: list[Path],
    *,
    rubric_path: Path = DEFAULT_RUBRIC,
) -> dict[str, Any]:
    rubric = _json(rubric_path)
    package = _json(package_report_path)
    semantic = _json(semantic_audit_path)
    packet = _jsonl(blind_packet_path)
    key = _json(review_key_path)
    dimensions = rubric["dimensions"]
    dimension_ids = {item["id"] for item in dimensions}
    weights = {item["id"]: float(item["weight"]) for item in dimensions}
    critical = {item["id"] for item in dimensions if item.get("critical")}
    acceptance = rubric["episode_acceptance"]
    patterns = list(rubric.get("known_process_artifacts") or [])
    machine_by_blind = _machine_episode_map(semantic, key)
    packet_by_id = {str(row.get("blind_id")): row for row in packet}
    identity_by_id = {str(row.get("blind_id")): row for row in key.get("mapping") or []}

    judgments = [row for path in judgment_paths for row in _jsonl(path)]
    judgment_problems: list[dict[str, Any]] = []
    valid_by_blind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_judge_pairs: set[tuple[str, str]] = set()
    for row in judgments:
        blind_id = str(row.get("blind_id") or "")
        judge_id = str(row.get("judge_id") or "")
        problems = _validate_judgment(row, dimension_ids)
        if blind_id not in packet_by_id:
            problems.append("unknown_blind_id")
        if (blind_id, judge_id) in seen_judge_pairs:
            problems.append("duplicate_judge_for_episode")
        seen_judge_pairs.add((blind_id, judge_id))
        if problems:
            judgment_problems.append(
                {"blind_id": blind_id, "judge_id": judge_id, "problems": problems}
            )
        else:
            valid_by_blind[blind_id].append(row)

    episode_reports = []
    for blind_id, packet_row in packet_by_id.items():
        identity = identity_by_id.get(blind_id, {})
        machine_episode = machine_by_blind.get(blind_id, {})
        artifact_hits = _artifact_hits(packet_row, patterns)
        machine_qualified = bool(
            package.get("ready_for_training") is True
            and semantic.get("all_passed") is True
            and machine_episode.get("semantic_passed") is True
            and machine_episode.get("terminal_success") is True
            and not artifact_hits
        )
        episode_judgments = valid_by_blind.get(blind_id, [])
        enough_judges = len({row["judge_id"] for row in episode_judgments}) >= int(
            acceptance["minimum_independent_blind_judges"]
        )
        means = {
            dimension_id: (
                sum(float(row["dimension_scores"][dimension_id]) for row in episode_judgments)
                / len(episode_judgments)
                if episode_judgments
                else None
            )
            for dimension_id in sorted(dimension_ids)
        }
        ranges = {
            dimension_id: (
                max(int(row["dimension_scores"][dimension_id]) for row in episode_judgments)
                - min(int(row["dimension_scores"][dimension_id]) for row in episode_judgments)
                if episode_judgments
                else None
            )
            for dimension_id in sorted(dimension_ids)
        }
        weighted_mean = (
            sum(weights[dimension_id] * float(means[dimension_id]) for dimension_id in dimension_ids)
            if episode_judgments
            else None
        )
        judge_ranges_ok = bool(episode_judgments) and all(
            value is not None and value <= int(acceptance["maximum_unadjudicated_judge_range"])
            for value in ranges.values()
        )
        judge_scores_ok = bool(episode_judgments) and all(
            float(means[dimension_id]) >= float(acceptance["each_critical_dimension_minimum"])
            for dimension_id in critical
        ) and all(
            float(value) >= float(acceptance["any_dimension_minimum"])
            for value in means.values()
            if value is not None
        )
        hard_fail_clear = bool(episode_judgments) and not any(
            row["hard_fail"] for row in episode_judgments
        )
        keep_majority = bool(episode_judgments) and sum(
            row["keep_for_sft"] for row in episode_judgments
        ) > len(episode_judgments) / 2
        blind_judge_qualified = bool(
            enough_judges
            and weighted_mean is not None
            and weighted_mean >= float(acceptance["weighted_mean_minimum"])
            and judge_ranges_ok
            and judge_scores_ok
            and hard_fail_clear
            and keep_majority
        )
        episode_reports.append(
            {
                "blind_id": blind_id,
                "dataset": identity.get("dataset"),
                "case_id": identity.get("case_id"),
                "mode": identity.get("mode"),
                "machine_qualified": machine_qualified,
                "machine_process_artifact_hits": artifact_hits,
                "valid_blind_judges": len(episode_judgments),
                "dimension_means": means,
                "dimension_ranges": ranges,
                "weighted_mean": round(weighted_mean, 4) if weighted_mean is not None else None,
                "blind_judge_qualified": blind_judge_qualified,
                "gold_for_sft": machine_qualified and blind_judge_qualified,
            }
        )

    dataset_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episode_reports:
        dataset_groups[str(row.get("dataset"))].append(row)
    dataset_reports = {}
    for dataset, rows in sorted(dataset_groups.items()):
        scores = [float(row["weighted_mean"]) for row in rows if row["weighted_mean"] is not None]
        gold = [row for row in rows if row["gold_for_sft"]]
        unique_cases = {str(row.get("case_id")) for row in rows}
        modes = defaultdict(int)
        for row in rows:
            modes[str(row.get("mode"))] += 1
        dataset_reports[dataset] = {
            "episodes": len(rows),
            "independent_cases": len(unique_cases),
            "modes": dict(modes),
            "gold_episodes": len(gold),
            "gold_rate": round(len(gold) / len(rows), 4) if rows else 0.0,
            "score_p10": round(_p10(scores), 4),
        }

    machine_qualified = bool(episode_reports) and all(
        row["machine_qualified"] for row in episode_reports
    ) and not judgment_problems
    blind_judging_complete = bool(episode_reports) and all(
        row["valid_blind_judges"] >= int(acceptance["minimum_independent_blind_judges"])
        for row in episode_reports
    )
    current_batch_gold = machine_qualified and blind_judging_complete and all(
        row["gold_for_sft"] for row in episode_reports
    )
    batch_spec = rubric["batch_readiness"]
    scale_checks = {
        "independent_cases_per_dataset": bool(dataset_reports) and all(
            row["independent_cases"] >= int(batch_spec["minimum_independent_cases_per_dataset"])
            for row in dataset_reports.values()
        ),
        "clean_and_repair_episodes_per_dataset": bool(dataset_reports) and all(
            row["modes"].get("clean", 0) >= int(batch_spec["minimum_clean_and_repair_episodes_per_dataset"])
            and row["modes"].get("repair", 0) >= int(batch_spec["minimum_clean_and_repair_episodes_per_dataset"])
            for row in dataset_reports.values()
        ),
        "gold_rate_per_dataset": bool(dataset_reports) and all(
            row["gold_rate"] >= float(batch_spec["minimum_episode_acceptance_rate_per_dataset"])
            for row in dataset_reports.values()
        ),
        "score_p10_per_dataset": bool(dataset_reports) and all(
            row["score_p10"] >= float(batch_spec["minimum_episode_score_p10_per_dataset"])
            for row in dataset_reports.values()
        ),
    }
    return {
        "rubric_id": rubric["rubric_id"],
        "machine_qualified": machine_qualified,
        "blind_judging_complete": blind_judging_complete,
        "current_batch_gold": current_batch_gold,
        "mass_production_ready": current_batch_gold and all(scale_checks.values()),
        "judgment_validation_problems": judgment_problems,
        "episodes": episode_reports,
        "datasets": dataset_reports,
        "scale_checks": scale_checks,
        "pending": [
            item
            for condition, item in (
                (blind_judging_complete, "two independent blind judgments per episode"),
                (scale_checks["independent_cases_per_dataset"], "at least 30 independent cases per dataset"),
                (scale_checks["clean_and_repair_episodes_per_dataset"], "at least 10 clean and 10 repair episodes per dataset"),
                (scale_checks["gold_rate_per_dataset"], "at least 90% Gold per dataset"),
                (scale_checks["score_p10_per_dataset"], "per-dataset score p10 at least 4.0"),
            )
            if not condition
        ],
        "scope_note": "pass@2, hard-negative calibration, repair-operator balance, and student free-run gain remain separate batch evidence and cannot be inferred from this episode packet.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-report", type=Path, required=True)
    parser.add_argument("--semantic-audit", type=Path, required=True)
    parser.add_argument("--blind-packet", type=Path, required=True)
    parser.add_argument("--review-key", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, action="append", default=[])
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        args.package_report,
        args.semantic_audit,
        args.blind_packet,
        args.review_key,
        args.judgments,
        rubric_path=args.rubric,
    )
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "rubric_id": report["rubric_id"],
                "machine_qualified": report["machine_qualified"],
                "blind_judging_complete": report["blind_judging_complete"],
                "current_batch_gold": report["current_batch_gold"],
                "mass_production_ready": report["mass_production_ready"],
                "pending": report["pending"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
