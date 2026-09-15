"""Unified, executable quality rubric for multi-dataset tool trajectories.

The scorer intentionally separates hidden-outcome verification from student-visible
process quality.  It operates on the common clean+repair bundle emitted by the
state-externalizing harness and can also re-score saved case artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from pipeline import write_json
from sft_context_compaction import compact_sft_row, validate_compact_sft_row


ROOT = Path(__file__).resolve().parents[1]
RUBRIC_PATH = ROOT / "rubric" / "unified_trajectory_quality_rubric.json"
EVIDENCE_ID_PATTERN = re.compile(r"EA-[A-Za-z0-9_-]+")
WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")
STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "into", "only", "still", "then",
    "there", "because", "before", "after", "current", "latest", "visible", "state",
    "observation", "established", "remaining", "correct", "next", "step", "action",
    "tool", "field", "fields", "claim", "evidence", "atom", "atoms", "required",
    "supported", "support", "open", "accepted", "now", "not", "all", "for", "are",
    "has", "have", "was", "were", "will", "can", "should", "would", "which", "than",
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fraction(flags: Iterable[bool]) -> float:
    values = list(flags)
    return sum(bool(value) for value in values) / len(values) if values else 1.0


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in WORD_PATTERN.findall(text or "")
        if len(token) >= 3 and token.lower() not in STOPWORDS
    }


def _word_count(text: str) -> int:
    return len(WORD_PATTERN.findall(text or ""))


def _criterion(spec: dict[str, Any], score: float, evidence: list[str], failures: list[str]) -> dict[str, Any]:
    normalized = _bounded(score)
    return {
        "criterion_id": spec["id"],
        "name": spec["name"],
        "normalized_score": round(normalized, 4),
        "earned": round(spec["weight"] * normalized, 3),
        "max_points": spec["weight"],
        "minimum": spec["minimum"],
        "passed_minimum": normalized >= spec["minimum"],
        "evidence": evidence,
        "failure_codes": failures,
    }


def infer_dataset(bundle: dict[str, Any], validation: dict[str, Any] | None = None) -> str:
    case_id = str(bundle.get("case_id", "")).lower()
    checks = set((validation or {}).get("checks", {}))
    if case_id.startswith("vcsl-") or "localization_recall_at_least_half" in checks:
        return "VCSL"
    if case_id.startswith("viddiff-") or "oracle_direction_match" in checks:
        return "VidDiffBench"
    return "VideoSham"


def _outcome_quality(dataset: str, validation: dict[str, Any]) -> tuple[float, bool, list[str]]:
    checks = validation.get("checks", {})
    failures: list[str] = []
    if dataset == "VCSL":
        metrics = validation.get("localization_metrics", {})
        precision = float(metrics.get("precision", 0.0))
        recall = float(metrics.get("recall", 0.0))
        f1 = float(metrics.get("f1", 0.0))
        # Q1's published minimum is literal: every localization metric must be
        # at least 0.85 for Gold.  Scaling by 0.85 here would make the effective
        # floor 0.85**2 (0.7225), allowing weak recall to hide behind a high
        # aggregate score.
        score = min(precision, recall, f1)
        gate = precision >= 0.50 and recall >= 0.50 and f1 >= 0.60
        if not gate:
            failures.append(f"localization_too_imprecise:p={precision:.3f},r={recall:.3f},f1={f1:.3f}")
        return _bounded(score), gate, [f"precision={precision:.4f}", f"recall={recall:.4f}", f"f1={f1:.4f}"]
    if dataset == "VidDiffBench":
        passed = checks.get("oracle_direction_match") is True
        if not passed:
            failures.append("direction_mismatch")
        return float(passed), passed, [f"oracle_direction_match={passed}"]
    outcome_keys = [
        key for key in ("oracle_label_match_post_execution", "oracle_window_match_post_execution")
        if key in checks
    ]
    passed = bool(outcome_keys) and all(checks[key] is True for key in outcome_keys)
    if not passed:
        failures.append("videosham_label_or_window_mismatch")
    return float(passed), passed, [f"{key}={checks.get(key)}" for key in outcome_keys]


def _episode_fields(episode: dict[str, Any]) -> dict[str, dict[str, Any]]:
    claims = episode.get("final_state", {}).get("candidate_claims", [])
    return claims[0].get("fields", {}) if claims else {}


def _evidence_grounding(bundle: dict[str, Any], validation: dict[str, Any]) -> tuple[float, list[str]]:
    flags: list[bool] = []
    failures: list[str] = []
    provenance_ids: set[str] = set()
    for episode_name in ("clean_episode", "repair_episode"):
        episode = bundle[episode_name]
        for event in episode.get("events", []):
            if event.get("action_type") != "external_tool_execution":
                continue
            observation = event.get("result", {}).get("observation", {})
            execution_id = observation.get("execution_id")
            if execution_id:
                provenance_ids.add(str(execution_id))
        state = episode.get("final_state", {})
        store = state.get("evidence_store", {})
        for field_name, field_state in _episode_fields(episode).items():
            support_ids = field_state.get("support_ids", [])
            committed = field_state.get("decision_status") == "committed"
            supported = field_state.get("evidence_status") == "supported"
            flags.extend([committed, supported, bool(support_ids)])
            if not (committed and supported and support_ids):
                failures.append(f"{episode_name}:{field_name}:unclosed_or_unsupported")
            for evidence_id in support_ids:
                atom = store.get(evidence_id)
                exists_active = isinstance(atom, dict) and atom.get("active", True) is True
                declares_field = exists_active and field_name in atom.get("supports_fields", [])
                generator = atom.get("provenance", {}).get("generated_by") if exists_active else None
                provenance_closed = bool(generator) and str(generator) in provenance_ids
                flags.extend([exists_active, declares_field, provenance_closed])
                if not exists_active:
                    failures.append(f"{episode_name}:{field_name}:missing_or_inactive:{evidence_id}")
                elif not declares_field:
                    failures.append(f"{episode_name}:{field_name}:undeclared_support:{evidence_id}")
                if not provenance_closed:
                    failures.append(f"{episode_name}:{field_name}:open_provenance:{evidence_id}")
    semantic_keys = [
        key for key in ("final_field_values_match_bound_evidence", "all_fields_supported_and_committed")
        if key in validation.get("checks", {})
    ]
    flags.extend(validation["checks"][key] is True for key in semantic_keys)
    return _fraction(flags), sorted(set(failures))


def _proposal_states(episode: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event for event in episode.get("events", [])
        if event.get("actor") == "policy" and str(event.get("action_type", "")).endswith("proposal")
    ]


def _planning_quality(bundle: dict[str, Any]) -> tuple[float, list[str]]:
    flags: list[bool] = []
    failures: list[str] = []
    for episode_name in ("clean_episode", "repair_episode"):
        episode = bundle[episode_name]
        turns = episode.get("turn_history", [])
        proposals = _proposal_states(episode)
        for index, turn in enumerate(turns):
            eligible = proposals[index].get("state_after", {}).get("eligible_tools", []) if index < len(proposals) else []
            legal = turn.get("tool_name") in eligible
            flags.append(legal)
            if not legal:
                failures.append(f"{episode_name}:turn_{index + 1}:action_not_prestate_eligible")
        tools = [turn.get("tool_name") for turn in turns]
        flags.extend([bool(tools), bool(tools) and tools[-1] == "stop"])
        if episode_name == "clean_episode":
            required = ["create_candidate_claim", "bind_evidence", "set_field_decision", "stop"]
            flags.extend(tool in tools for tool in required)
            if all(tool in tools for tool in required):
                positions = [tools.index(tool) for tool in required]
                ordered = positions == sorted(positions)
                flags.append(ordered)
                if not ordered:
                    failures.append("clean:claim_bind_commit_stop_dependency_violation")
                external_before_claim = any(
                    turn.get("observation", {}).get("authority") == "external_tool_environment"
                    for turn in turns[: tools.index("create_candidate_claim")]
                )
                flags.append(external_before_claim)
        elif "create_candidate_claim" in tools:
            flags.append(False)
            failures.append("repair:unnecessary_claim_recreation")
    return _fraction(flags), failures


def _utility_quality(bundle: dict[str, Any]) -> tuple[float, list[str]]:
    report = bundle.get("step_utility_report", {})
    flags: list[bool] = [report.get("passed") is True]
    failures: list[str] = []
    for episode_name, steps in report.get("episodes", {}).items():
        for step in steps:
            external = str(step.get("execution_status")) == "executed"
            flags.extend(
                [
                    step.get("accepted_for_sft") is True,
                    step.get("reasoning_grounded") is True,
                    step.get("downstream_consumed") is True,
                    step.get("causally_necessary") is True or step.get("branch_enabling_measurement") is True,
                    (not external) or step.get("observation_sensitive") is True,
                ]
            )
            if step.get("verdict") not in {"indispensable", "branch_enabling"}:
                failures.append(f"{episode_name}:turn_{step.get('turn_idx')}:nonessential:{step.get('verdict')}")
    return _fraction(flags), failures


def _reasoning_quality(bundle: dict[str, Any]) -> tuple[float, list[str], dict[str, Any]]:
    length_scores: list[float] = []
    tool_scores: list[float] = []
    gap_scores: list[float] = []
    specificity_scores: list[float] = []
    reference_scores: list[float] = []
    failures: list[str] = []
    all_reasonings: list[str] = []
    unsupported_refs = 0
    for episode_name in ("clean_episode", "repair_episode"):
        turns = bundle[episode_name].get("turn_history", [])
        for index, turn in enumerate(turns):
            reasoning = str(turn.get("reasoning", ""))
            all_reasonings.append(reasoning)
            count = _word_count(reasoning)
            length_score = 1.0 if 35 <= count <= 150 else 0.8 if 25 <= count <= 190 else 0.4 if count <= 240 else 0.0
            length_scores.append(length_score)
            tool = str(turn.get("tool_name", ""))
            tool_parts = {part for part in tool.lower().split("_") if len(part) >= 4}
            tool_scores.append(float(tool.lower() in reasoning.lower() or bool(tool_parts.intersection(_tokens(reasoning)))))
            gap_markers = ("gap", "remain", "missing", "need", "unresolved", "absence", "缺", "尚未", "需要")
            gap_scores.append(float(any(marker in reasoning.lower() for marker in gap_markers)))
            if index == 0:
                specificity = any(marker in reasoning.lower() for marker in ("no ", "absence", "only", "open", "没有", "尚无"))
            else:
                previous = json.dumps(turns[index - 1].get("observation", {}), ensure_ascii=False)
                overlap = _tokens(previous).intersection(_tokens(reasoning))
                specificity = len(overlap) >= 2 or bool(EVIDENCE_ID_PATTERN.findall(reasoning))
            specificity_scores.append(float(specificity))
            final_state = bundle[episode_name].get("final_state", {})
            visible_ids = set(final_state.get("evidence_store", {})).union(
                final_state.get("dropped_evidence_ids", [])
            )
            refs = set(EVIDENCE_ID_PATTERN.findall(reasoning))
            bad = refs - visible_ids
            unsupported_refs += len(bad)
            reference_scores.append(float(not bad))
            if length_score < 0.8:
                failures.append(f"{episode_name}:turn_{index + 1}:reasoning_length={count}")
            if not specificity:
                failures.append(f"{episode_name}:turn_{index + 1}:weak_observation_specificity")
            if bad:
                failures.append(f"{episode_name}:turn_{index + 1}:nonvisible_evidence={sorted(bad)}")
    pairwise: list[float] = []
    for left_index, left in enumerate(all_reasonings):
        left_tokens = _tokens(left)
        for right in all_reasonings[left_index + 1 :]:
            right_tokens = _tokens(right)
            union = left_tokens | right_tokens
            if union:
                pairwise.append(len(left_tokens & right_tokens) / len(union))
    max_similarity = max(pairwise, default=0.0)
    repetition_score = 1.0 if max_similarity <= 0.62 else 0.7 if max_similarity <= 0.72 else 0.2
    if repetition_score < 0.8:
        failures.append(f"cross_turn_reasoning_repetition={max_similarity:.3f}")
    average_words = sum(_word_count(text) for text in all_reasonings) / max(1, len(all_reasonings))
    dimension_scores = {
        "length_calibration": sum(length_scores) / max(1, len(length_scores)),
        "action_rationale": sum(tool_scores) / max(1, len(tool_scores)),
        "gap_identification": sum(gap_scores) / max(1, len(gap_scores)),
        "observation_specificity": sum(specificity_scores) / max(1, len(specificity_scores)),
        "visible_reference_integrity": sum(reference_scores) / max(1, len(reference_scores)),
        "cross_turn_non_repetition": repetition_score,
    }
    return (
        sum(dimension_scores.values()) / len(dimension_scores),
        failures,
        {
            "turns": len(all_reasonings),
            "average_reasoning_words": round(average_words, 2),
            "max_pairwise_token_jaccard": round(max_similarity, 4),
            "unsupported_evidence_references": unsupported_refs,
            "dimension_scores": {key: round(value, 4) for key, value in dimension_scores.items()},
        },
    )


def _repair_operator(bundle: dict[str, Any]) -> str:
    summary = bundle.get("execution_summary", {})
    mutation = bundle.get("repair_episode", {}).get("mutation", {})
    operator = summary.get("repair_operator") or mutation.get("operator")
    if operator:
        return str(operator)
    return "evidence_dropout" if mutation.get("removed_evidence_ids") else "unknown"


def _claims_equal(clean: dict[str, Any], repair: dict[str, Any]) -> bool:
    clean_claim = clean.get("final_claim") or {}
    repair_claim = repair.get("final_claim") or {}
    shared = set(clean_claim).intersection(repair_claim) - {"evidence_ids", "confidence"}
    return bool(shared) and all(clean_claim[key] == repair_claim[key] for key in shared)


def _repair_quality(bundle: dict[str, Any], validation: dict[str, Any]) -> tuple[float, list[str], str]:
    clean = bundle["clean_episode"]
    repair = bundle["repair_episode"]
    mutation = repair.get("mutation", {})
    operator = _repair_operator(bundle)
    tools = [turn.get("tool_name") for turn in repair.get("turn_history", [])]
    external = [
        turn for turn in repair.get("turn_history", [])
        if turn.get("observation", {}).get("authority") == "external_tool_environment"
    ]
    checks = validation.get("checks", {})
    flags: list[bool] = [
        mutation.get("media_changed") is False,
        mutation.get("claim_values_changed", mutation.get("claim_text_changed_at_injection", False)) is False,
        repair.get("invariant_fields_before") == repair.get("invariant_fields_after"),
        _claims_equal(clean, repair),
        "create_candidate_claim" not in tools,
        bool(tools) and tools[-1] == "stop",
    ]
    failures: list[str] = []
    if operator == "decision_reopen":
        flags.extend([not external, "bind_evidence" not in tools, len(tools) <= 2])
        if external or "bind_evidence" in tools:
            failures.append("decision_reopen_overrepair")
    elif operator == "binding_invalidation":
        flags.extend([not external, "bind_evidence" in tools, len(tools) <= 4])
        if external:
            failures.append("binding_invalidation_unnecessary_remeasurement")
    elif operator == "evidence_dropout":
        integrity_steps = sum(
            name in {"assess_evidence_candidate", "adjudicate_evidence"}
            for name in tools
        )
        flags.extend(
            [
                bool(external),
                "bind_evidence" in tools,
                len(tools) - integrity_steps <= 6,
            ]
        )
        if not external:
            failures.append("evidence_dropout_without_remeasurement")
    else:
        flags.append(False)
        failures.append("unknown_repair_operator")
    for key in ("off_target_invariance", "dropped_evidence_not_reused", "target_fields_repaired", "all_fields_supported_and_committed"):
        if key in checks:
            flags.append(checks[key] is True)
            if checks[key] is not True:
                failures.append(key)
    return _fraction(flags), failures, operator


def _measurement_alias_mentioned(method: str, text: str) -> bool:
    aliases = {
        "compare_uniform_timeline": ("compare_uniform_timeline", "uniform timeline", "uniform-timeline", "timeline pass"),
        "compare_global_motion_symmetric": (
            "compare_global_motion_symmetric",
            "global motion",
            "global-motion",
            "motion-symmetric",
        ),
        "compare_action_refined_symmetric": (
            "compare_action_refined_symmetric",
            "action refined",
            "action-refined",
            "refined action",
            "refined-action",
            "refined symmetric",
            "refined-symmetric",
            "refined atom",
            "refined alternative",
            "refined measurement",
            "action-focused",
        ),
    }
    candidates = aliases.get(method, (method.replace("_", " "), method))
    return any(candidate and candidate in text for candidate in candidates)


def _safe_confidence(atom: dict[str, Any]) -> float:
    value = atom.get("observation", {}).get("confidence")
    return float(value) if isinstance(value, (int, float)) else 1.0


def _uncertainty_quality(bundle: dict[str, Any], dataset: str) -> tuple[float, list[str]]:
    """Check semantic adjudication, not literal repetition of tool function names."""
    if dataset != "VidDiffBench":
        return 1.0, []
    clean = bundle["clean_episode"]
    final_state = clean.get("final_state", {})
    store = final_state.get("evidence_store", {})
    measurement_items = [
        (evidence_id, atom)
        for evidence_id, atom in store.items()
        if "preferred_video" in atom.get("supports_fields", []) and atom.get("active", True)
    ]
    if not measurement_items:
        return 0.0, ["viddiff_has_no_active_comparison_measurement"]

    preferences = {
        atom.get("observation", {}).get("preferred_video")
        for _evidence_id, atom in measurement_items
        if atom.get("observation", {}).get("preferred_video") in {"a", "b", "c"}
    }
    disagreement = len(preferences) > 1
    low_confidence = any(_safe_confidence(atom) < 0.65 for _evidence_id, atom in measurement_items)
    order_failure = any(
        atom.get("observation", {}).get("order_consistent") is False
        for _evidence_id, atom in measurement_items
    )
    if not (disagreement or low_confidence or order_failure):
        return 1.0, []

    turns = clean.get("turn_history", [])
    create_turn = next((turn for turn in turns if turn.get("tool_name") == "create_candidate_claim"), {})
    bind_turn = next((turn for turn in turns if turn.get("tool_name") == "bind_evidence"), {})
    create_reasoning = str(create_turn.get("reasoning", "")).lower()
    adjudication_reasoning = " ".join(
        str(turn.get("reasoning", "")).lower()
        for turn in turns
        if turn.get("tool_name") == "adjudicate_evidence"
    )
    decision_reasoning = " ".join(
        value
        for value in (
            create_reasoning,
            adjudication_reasoning,
            str(bind_turn.get("reasoning", "")).lower(),
        )
        if value
    )

    claims = final_state.get("candidate_claims", [])
    fields = claims[0].get("fields", {}) if claims else {}
    selected_ids = {
        support_id
        for field_name in ("preferred_video", "video_a_interval_s", "video_b_interval_s", "measurement_method")
        for support_id in fields.get(field_name, {}).get("support_ids", [])
        if support_id in store
    }
    measurement_ids = {evidence_id for evidence_id, _atom in measurement_items}
    selected_measurement_ids = selected_ids.intersection(measurement_ids)
    selected_methods = {
        str(store[evidence_id].get("observation", {}).get("measurement_method", ""))
        for evidence_id in selected_measurement_ids
    }
    selected_grounded = bool(selected_measurement_ids) and any(
        evidence_id.lower() in decision_reasoning
        or _measurement_alias_mentioned(method, decision_reasoning)
        for evidence_id in selected_measurement_ids
        for method in selected_methods
    )

    referenced_ids = {
        evidence_id
        for evidence_id in measurement_ids
        if evidence_id.lower() in decision_reasoning
    }
    referenced_methods = {
        str(atom.get("observation", {}).get("measurement_method", ""))
        for _evidence_id, atom in measurement_items
        if _measurement_alias_mentioned(
            str(atom.get("observation", {}).get("measurement_method", "")),
            decision_reasoning,
        )
    }
    alternatives_covered = (
        len(measurement_items) == 1
        or len(referenced_ids) >= 2
        or len(referenced_methods) >= 2
    )

    factor_groups = {
        "order": ("order", "sequence", "temporal"),
        "confidence": ("confidence", "certainty", "low-confidence", "high-confidence"),
        "abstention": ("abstain", "insufficient", "ambiguous", "unsupported"),
        "query_fit": ("exact statement", "exact query", "claim fit", "specific action"),
        "visible_basis": ("visible", "rationale", "cue", "proxy", "direct", "indirect", "deformation"),
        "comparison": ("weaker", "stronger", "alternative", "disagree", "conflict", "counter"),
    }
    observed_factors = {
        name for name, terms in factor_groups.items() if any(term in decision_reasoning for term in terms)
    }
    # A post-rejection-sampling trajectory can intentionally expose exactly
    # one verified measurement. In that case there is no visible alternative
    # to compare, so require calibration plus one non-confidence basis. Keep
    # the stricter three-factor requirement whenever multiple measurements
    # are visible and agreement/conflict must actually be adjudicated.
    substantive_comparison = (
        len(observed_factors) >= 2
        if len(measurement_items) == 1
        else len(observed_factors) >= 3
    )
    nonconfidence_basis = bool(
        observed_factors.intersection({"order", "abstention", "query_fit", "visible_basis"})
    )

    resolution_terms = (
        "strongest",
        "stronger",
        "weaker",
        "selected",
        "selecting",
        "chosen",
        "retained",
        "best",
        "resolve",
        "partition",
        "because",
        "therefore",
    )
    explicit_resolution = selected_grounded and any(term in decision_reasoning for term in resolution_terms)
    if disagreement:
        calibrated = any(
            term in decision_reasoning
            for term in ("disagree", "conflict", "alternative", "opposite", "different direction")
        )
    elif len(measurement_items) > 1:
        calibrated = any(
            term in decision_reasoning
            for term in (
                "agree",
                "consistent",
                "corroborat",
                "both support",
                "same abstaining",
                "again supports",
            )
        )
    else:
        calibrated = any(term in decision_reasoning for term in ("confidence", "uncertain", "ambiguous", "abstain"))

    flags = [
        "oracle" not in decision_reasoning,
        selected_grounded,
        alternatives_covered,
        substantive_comparison,
        explicit_resolution and calibrated,
        nonconfidence_basis,
    ]
    failure_codes = [
        code
        for passed, code in zip(
            flags,
            (
                "viddiff_oracle_mentioned_in_adjudication",
                "viddiff_selected_measurement_not_grounded_in_claim_reasoning",
                "viddiff_alternative_measurements_not_compared",
                "viddiff_adjudication_lacks_multiple_substantive_factors",
                "viddiff_conflict_or_uncertainty_not_explicitly_resolved",
                "viddiff_adjudication_relies_on_confidence_or_vote_only",
            ),
        )
        if not passed
    ]
    score = _fraction(flags)
    # Q7 is deliberately conjunctive: naming tools or citing confidence cannot
    # compensate for a missing alternative, resolution, or non-confidence basis.
    if not all(flags[1:]):
        score = min(score, 2.0 / 3.0)
    return score, failure_codes


def _turn_arguments(turn: dict[str, Any]) -> dict[str, Any]:
    for key in ("params", "arguments", "generated_arguments", "tool_arguments"):
        value = turn.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _turn_tool_name(turn: dict[str, Any]) -> str:
    return str(turn.get("tool_name") or turn.get("generated_tool") or "")


def _binding_support_ids(arguments: dict[str, Any], target_fields: set[str]) -> set[str]:
    support_ids: set[str] = set()
    for item in arguments.get("binding_items") or []:
        if not isinstance(item, dict) or str(item.get("field")) not in target_fields:
            continue
        support_ids.update(str(value) for value in item.get("support_ids") or [])
    for field_name, raw_support in (arguments.get("bindings") or {}).items():
        if str(field_name) not in target_fields:
            continue
        values = raw_support if isinstance(raw_support, list) else [raw_support]
        support_ids.update(str(value) for value in values if value)
    return support_ids


def _adjudication_quality(
    bundle: dict[str, Any],
) -> tuple[float, list[str], list[str], dict[str, Any]]:
    """Validate explicit conflict resolution when the new state action is present.

    Legacy trajectories remain scoreable through ``_uncertainty_quality``. Once a
    trajectory emits ``adjudicate_evidence``, however, its accepted observation is
    an executable audit record: every considered atom must be partitioned exactly
    once, every choice must be verbalized, and the next binding must honor it.
    """

    structural_failures: list[str] = []
    reasoning_failures: list[str] = []
    records = 0
    cited_atoms = 0
    considered_atoms = 0
    for episode_name in ("clean_episode", "repair_episode"):
        episode = bundle.get(episode_name) or {}
        turns = episode.get("turn_history") or episode.get("turns") or []
        store = (episode.get("final_state") or {}).get("evidence_store") or {}
        for index, turn in enumerate(turns):
            if _turn_tool_name(turn) != "adjudicate_evidence":
                continue
            records += 1
            arguments = _turn_arguments(turn)
            claim_id = str(arguments.get("claim_id") or "")
            target_fields = {str(value) for value in arguments.get("target_fields") or []}
            selected = [str(value) for value in arguments.get("selected_evidence_ids") or []]
            rejected_items = arguments.get("rejected_evidence") or []
            rejected_ids = [
                str(item.get("evidence_id"))
                for item in rejected_items
                if isinstance(item, dict) and item.get("evidence_id")
            ]
            partition = selected + rejected_ids
            observation = turn.get("observation") or turn.get("environment_outcome") or {}
            observed = observation.get("adjudication") or {}
            considered = [str(value) for value in observed.get("considered_evidence_ids") or []]
            if not considered:
                considered = partition
            considered_atoms += len(set(considered))

            prefix = f"{episode_name}:turn_{index + 1}"
            if not claim_id or not target_fields or not selected or not rejected_items:
                structural_failures.append(f"{prefix}:adjudication_missing_required_partition_fields")
            if len(partition) != len(set(partition)):
                structural_failures.append(f"{prefix}:adjudication_partition_has_duplicates_or_overlap")
            if set(partition) != set(considered):
                structural_failures.append(f"{prefix}:adjudication_omits_or_invents_competitor")
            for item in rejected_items:
                defect_codes = item.get("defect_codes") if isinstance(item, dict) else None
                if not defect_codes:
                    structural_failures.append(f"{prefix}:rejected_evidence_lacks_visible_defect")

            reasoning = str(turn.get("reasoning") or "").lower()
            methods = {
                evidence_id: str(
                    ((store.get(evidence_id) or {}).get("observation") or {}).get(
                        "measurement_method", ""
                    )
                ).lower()
                for evidence_id in partition
            }
            method_counts = {
                method: sum(value == method for value in methods.values())
                for method in set(methods.values())
                if method
            }
            for evidence_id in partition:
                method = methods.get(evidence_id, "")
                if evidence_id.lower() in reasoning or (
                    method
                    and method_counts.get(method) == 1
                    and _measurement_alias_mentioned(method, reasoning)
                ):
                    cited_atoms += 1
                else:
                    reasoning_failures.append(f"{prefix}:adjudication_reasoning_omits:{evidence_id}")

            next_bind = next(
                (
                    later
                    for later in turns[index + 1 :]
                    if _turn_tool_name(later) in {"bind_evidence", "adjudicate_evidence"}
                ),
                None,
            )
            if next_bind is None or _turn_tool_name(next_bind) != "bind_evidence":
                structural_failures.append(f"{prefix}:adjudication_not_consumed_by_following_bind")
            else:
                bind_args = _turn_arguments(next_bind)
                if claim_id and str(bind_args.get("claim_id") or "") != claim_id:
                    structural_failures.append(f"{prefix}:following_bind_targets_different_claim")
                bound_ids = _binding_support_ids(bind_args, target_fields)
                if not bound_ids or not bound_ids.issubset(set(selected)):
                    structural_failures.append(f"{prefix}:following_bind_bypasses_selected_evidence")

    if not records:
        return 1.0, [], [], {
            "explicit_adjudication_records": 0,
            "legacy_semantic_adjudication_only": True,
        }
    flags = [not structural_failures, not reasoning_failures]
    return _fraction(flags), structural_failures, reasoning_failures, {
        "explicit_adjudication_records": records,
        "considered_evidence_atoms": considered_atoms,
        "reasoning_cited_evidence_atoms": cited_atoms,
        "structural_failures": len(structural_failures),
        "reasoning_failures": len(reasoning_failures),
    }


def _sft_quality(bundle: dict[str, Any]) -> tuple[float, list[str]]:
    datums = bundle.get("clean_turn_sft", []) + bundle.get("repair_turn_sft", [])
    flags: list[bool] = [bool(datums)]
    failures: list[str] = []
    for index, datum in enumerate(datums):
        metadata = datum.get("metadata", {})
        messages = datum.get("messages", [])
        assistant_targets = [message for message in messages if message.get("role") == "assistant"]
        flags.extend(
            [
                metadata.get("target_execution_status") in {"accepted", "executed"},
                metadata.get("target_includes_observation") is False,
                metadata.get("loss_roles") == ["assistant"],
                metadata.get("loss_scope") == "last_assistant_action_only",
                metadata.get("step_utility_verdict") in {"indispensable", "branch_enabling"},
                bool(assistant_targets),
            ]
        )
        if metadata.get("target_execution_status") not in {"accepted", "executed"}:
            failures.append(f"datum_{index}:nonaccepted_target")
    clean_tools = {turn.get("tool_name") for turn in bundle["clean_episode"].get("turn_history", [])}
    registry = bundle.get("harness_config", {}).get("tool_registry", [])
    external_names = {item.get("name") for item in registry if item.get("domain") == "external_media"}
    if not external_names:
        external_names = {
            turn.get("tool_name") for turn in bundle["clean_episode"].get("turn_history", [])
            if turn.get("observation", {}).get("authority") == "external_tool_environment"
        }
    flags.extend(
        [
            bool(clean_tools.intersection(external_names)),
            "create_candidate_claim" in clean_tools,
            "bind_evidence" in clean_tools,
            "set_field_decision" in clean_tools,
            "stop" in clean_tools,
        ]
    )
    return _fraction(flags), failures


def _context_compaction_quality(bundle: dict[str, Any]) -> tuple[float, list[str], dict[str, Any]]:
    datums = bundle.get("clean_turn_sft", []) + bundle.get("repair_turn_sft", [])
    failures: list[str] = []
    flags: list[bool] = [bool(datums)]
    results: list[dict[str, Any]] = []
    for index, datum in enumerate(datums):
        try:
            compact = compact_sft_row(datum)
            result = validate_compact_sft_row(datum, compact)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            failures.append(f"datum_{index}:compaction_error:{type(error).__name__}")
            flags.append(False)
            continue
        results.append(result)
        for check_name, passed in result["checks"].items():
            flags.append(bool(passed))
            if not passed:
                failures.append(f"datum_{index}:{check_name}")
    source_chars = sum(item["source_visible_chars"] for item in results)
    compact_chars = sum(item["compact_visible_chars"] for item in results)
    ratios = sorted(item["context_char_ratio"] for item in results)
    stats = {
        "rows": len(datums),
        "rows_verified": sum(item["passed"] for item in results),
        "source_visible_chars": source_chars,
        "compact_visible_chars": compact_chars,
        "visible_char_reduction_rate": round(1.0 - compact_chars / max(source_chars, 1), 6),
        "context_char_ratio_p50": ratios[round((len(ratios) - 1) * 0.50)] if ratios else 0.0,
        "context_char_ratio_p90": ratios[round((len(ratios) - 1) * 0.90)] if ratios else 0.0,
        "private_locator_or_oracle_hits": sum(
            sum(item["privacy_hits"].values()) for item in results
        ),
    }
    return _fraction(flags), failures, stats


def _student_isolation(bundle: dict[str, Any], validation: dict[str, Any]) -> tuple[bool, list[str]]:
    payload = json.dumps(
        {"clean": bundle.get("clean_turn_sft", []), "repair": bundle.get("repair_turn_sft", [])},
        ensure_ascii=False,
    ).lower()
    tokens = ("oracle_direction", "oracle_boxes", "claim_oracle.json", "differences_gt", "retrieval_frames")
    leaks = [token for token in tokens if token in payload]
    leak_checks = {
        key: value for key, value in validation.get("checks", {}).items()
        if "leak" in key
    }
    passed = not leaks and all(value is True for value in leak_checks.values())
    return passed, leaks + [key for key, value in leak_checks.items() if value is not True]


def evaluate_bundle(
    bundle: dict[str, Any],
    validation: dict[str, Any],
    *,
    dataset: str | None = None,
) -> dict[str, Any]:
    rubric = _json(RUBRIC_PATH)
    dataset = dataset or infer_dataset(bundle, validation)
    specs = {item["id"]: item for item in rubric["criteria"]}

    q1, outcome_gate, q1_evidence = _outcome_quality(dataset, validation)
    q2, q2_failures = _evidence_grounding(bundle, validation)
    q3, q3_failures = _planning_quality(bundle)
    q4, q4_failures = _utility_quality(bundle)
    q5, q5_failures, reasoning_stats = _reasoning_quality(bundle)
    q6, q6_failures, operator = _repair_quality(bundle, validation)
    q7, q7_failures = _uncertainty_quality(bundle, dataset)
    adjudication_score, adjudication_structural, adjudication_reasoning, adjudication_stats = (
        _adjudication_quality(bundle)
    )
    if adjudication_stats["explicit_adjudication_records"]:
        q5 = min(q5, 1.0 if not adjudication_reasoning else 0.5)
        q5_failures.extend(adjudication_reasoning)
        q7 = min(q7, adjudication_score)
        q7_failures.extend(adjudication_structural + adjudication_reasoning)
    q8, q8_failures = _sft_quality(bundle)
    q9, q9_failures, compaction_stats = _context_compaction_quality(bundle)

    criteria = [
        _criterion(specs["Q1"], q1, q1_evidence, [] if outcome_gate else ["dataset_outcome_failure"]),
        _criterion(specs["Q2"], q2, ["clean+repair final field bindings and provenance"], q2_failures),
        _criterion(specs["Q3"], q3, ["pre-action eligible_tools and dependency order"], q3_failures),
        _criterion(specs["Q4"], q4, ["fixed-suffix removal and observation counterfactuals"], q4_failures),
        _criterion(
            specs["Q5"],
            q5,
            [
                json.dumps(reasoning_stats, ensure_ascii=False),
                json.dumps(adjudication_stats, ensure_ascii=False),
            ],
            q5_failures,
        ),
        _criterion(specs["Q6"], q6, [f"repair_operator={operator}"], q6_failures),
        _criterion(
            specs["Q7"],
            q7,
            [
                "visible confidence/order/conflict evidence only",
                json.dumps(adjudication_stats, ensure_ascii=False),
            ],
            q7_failures,
        ),
        _criterion(specs["Q8"], q8, ["per-turn assistant-only SFT metadata"], q8_failures),
        _criterion(
            specs["Q9"],
            q9,
            [json.dumps(compaction_stats, ensure_ascii=False)],
            q9_failures,
        ),
    ]
    isolation_pass, isolation_failures = _student_isolation(bundle, validation)
    turns = bundle["clean_episode"].get("turn_history", []) + bundle["repair_episode"].get("turn_history", [])
    executable = (
        bundle["clean_episode"].get("terminated") is True
        and bundle["repair_episode"].get("terminated") is True
        and all(turn.get("observation", {}).get("status") in {"accepted", "executed"} for turn in turns)
        and all(
            datum.get("metadata", {}).get("target_execution_status") in {"accepted", "executed"}
            for datum in bundle.get("clean_turn_sft", []) + bundle.get("repair_turn_sft", [])
        )
    )
    hard_gate_values = {
        "H1": outcome_gate,
        "H2": executable,
        "H3": (
            q2 >= specs["Q2"]["minimum"]
            and not q2_failures
            and not adjudication_structural
        ),
        "H4": isolation_pass,
        "H5": (
            q4 >= specs["Q4"]["minimum"]
            and bundle.get("step_utility_report", {}).get("passed") is True
            and not q4_failures
        ),
        "H6": q6 >= specs["Q6"]["minimum"] and not q6_failures,
        "H7": q9 >= specs["Q9"]["minimum"] and not q9_failures,
    }
    hard_gates = []
    for gate in rubric["hard_gates"]:
        passed = hard_gate_values[gate["id"]]
        details: list[str] = []
        if not passed:
            if gate["id"] == "H1":
                details = criteria[0]["failure_codes"]
            elif gate["id"] == "H3":
                details = q2_failures + adjudication_structural
            elif gate["id"] == "H4":
                details = isolation_failures
            elif gate["id"] == "H5":
                details = q4_failures
            elif gate["id"] == "H6":
                details = q6_failures
            elif gate["id"] == "H7":
                details = q9_failures
        hard_gates.append({**gate, "passed": passed, "details": details})

    total = round(sum(item["earned"] for item in criteria), 3)
    all_gates = all(item["passed"] for item in hard_gates)
    required_minima = {
        "Q1": 0.85, "Q2": 0.90, "Q3": 0.80, "Q4": 0.90,
        "Q5": 0.80, "Q6": 0.90, "Q7": 0.80, "Q8": 0.90, "Q9": 0.95,
    }
    criteria_ok = all(
        next(item for item in criteria if item["criterion_id"] == criterion_id)["normalized_score"] >= minimum
        for criterion_id, minimum in required_minima.items()
    )
    accepted = all_gates and total >= 92 and criteria_ok
    grade = "reject"
    if accepted:
        for band in rubric["aggregation"]["grades"]:
            if total >= band["min"]:
                grade = band["label"]
                break
    elif all_gates and total >= 86:
        # A high aggregate cannot be called Gold when a mandatory criterion
        # fails; keep it as a repair candidate instead of masking the defect.
        grade = "B_repair"
    failed_criteria = [item["criterion_id"] for item in criteria if not item["passed_minimum"]]
    return {
        "rubric_id": rubric["rubric_id"],
        "dataset": dataset,
        "case_id": bundle.get("case_id"),
        "trajectory_ids": [
            bundle["clean_episode"].get("trajectory_id"),
            bundle["repair_episode"].get("trajectory_id"),
        ],
        "total_score": total,
        "max_score": 100,
        "grade": grade,
        "accepted_for_sft": accepted,
        "hard_gates_passed": all_gates,
        "hard_gates": hard_gates,
        "criteria": criteria,
        "diagnosis": {
            "failed_gates": [item["id"] for item in hard_gates if not item["passed"]],
            "failed_criteria": failed_criteria,
            "priority_repairs": [
                item["criterion_id"] for item in sorted(criteria, key=lambda value: value["normalized_score"])
                if not item["passed_minimum"]
            ],
        },
        "repair_operator": operator,
        "reasoning_statistics": reasoning_stats,
        "evidence_adjudication_statistics": adjudication_stats,
        "judge_mode": "typed_deterministic_plus_counterfactual_replay",
        "llm_judge_used": False,
    }


def load_saved_bundle(artifact_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_dir = Path(artifact_dir)
    if not (artifact_dir / "llm_teacher_clean_episode.json").exists() and (
        artifact_dir / "outputs" / "llm_teacher_clean_episode.json"
    ).exists():
        artifact_dir = artifact_dir / "outputs"
    bundle = {
        "case_id": None,
        "clean_episode": _json(artifact_dir / "llm_teacher_clean_episode.json"),
        "repair_episode": _json(artifact_dir / "llm_teacher_repair_episode.json"),
        "execution_summary": _json(artifact_dir / "llm_teacher_execution_summary.json"),
        "step_utility_report": _json(artifact_dir / "llm_teacher_step_utility_report.json"),
        "clean_turn_sft": _jsonl(artifact_dir / "llm_teacher_sft_clean.jsonl"),
        "repair_turn_sft": _jsonl(artifact_dir / "llm_teacher_sft_repair.jsonl"),
    }
    all_datums = bundle["clean_turn_sft"] + bundle["repair_turn_sft"]
    if all_datums:
        bundle["case_id"] = all_datums[0].get("metadata", {}).get("case_id")
    validation = _json(artifact_dir / "llm_teacher_validation_report.json")
    return bundle, validation


def evaluate_artifact_dir(artifact_dir: Path, *, write_report: bool = True) -> dict[str, Any]:
    bundle, validation = load_saved_bundle(artifact_dir)
    report = evaluate_bundle(bundle, validation)
    if write_report:
        target = Path(artifact_dir)
        if not (target / "llm_teacher_clean_episode.json").exists() and (target / "outputs").exists():
            target = target / "outputs"
        write_json(target / "unified_trajectory_quality_report.json", report)
    return report


def _case_artifact_dir(run_root: Path, row: dict[str, Any], dataset: str) -> Path | None:
    source = row.get("source_case_dir") or row.get("source_artifact_dir")
    candidates: list[Path] = [Path(source)] if source else []
    case_id = str(row.get("case_id", ""))
    if dataset == "VideoSham":
        candidates.append(run_root / "cases" / case_id)
    elif dataset == "VCSL":
        candidates.append(run_root / "cases" / case_id.removeprefix("vcsl-"))
    else:
        candidates.append(run_root / "cases" / case_id.removeprefix("viddiff-"))
    for candidate in candidates:
        if (candidate / "llm_teacher_clean_episode.json").exists() or (
            candidate / "outputs" / "llm_teacher_clean_episode.json"
        ).exists():
            return candidate
    return None


def evaluate_run(run_root: Path, output_path: Path | None = None) -> dict[str, Any]:
    run_root = Path(run_root)
    batch_path = run_root / "batch_report.json"
    if not batch_path.exists():
        batch_path = run_root / "consolidated_report.json"
    batch = _json(batch_path)
    dataset = batch.get("dataset", "VidDiffBench" if "viddiff" in str(run_root).lower() else "unknown")
    rows: list[dict[str, Any]] = []
    for case in batch.get("cases", []):
        artifact_dir = _case_artifact_dir(run_root, case, dataset)
        if case.get("status") != "passed" or artifact_dir is None:
            rows.append(
                {
                    "case_id": case.get("case_id"),
                    "source_status": case.get("status"),
                    "scored": False,
                    "artifact_dir": str(artifact_dir) if artifact_dir else None,
                }
            )
            continue
        quality = evaluate_artifact_dir(artifact_dir, write_report=True)
        rows.append({**quality, "source_status": case.get("status"), "scored": True, "artifact_dir": str(artifact_dir)})
    scored = [row for row in rows if row.get("scored")]
    scores = sorted(float(row["total_score"]) for row in scored)
    gold = [row for row in scored if row.get("accepted_for_sft")]
    operators = sorted({str(row.get("repair_operator")) for row in scored})
    p10 = scores[max(0, math.ceil(0.10 * len(scores)) - 1)] if scores else 0.0
    report = {
        "rubric_id": _json(RUBRIC_PATH)["rubric_id"],
        "dataset": dataset,
        "source_run": str(run_root),
        "summary": {
            "source_cases": len(batch.get("cases", [])),
            "scored_cases": len(scored),
            "gold_cases": len(gold),
            "gold_rate": round(len(gold) / len(scored), 4) if scored else 0.0,
            "median_score": round(median(scores), 3) if scores else 0.0,
            "p10_score": round(p10, 3),
            "hard_gate_failure_cases": sum(not row.get("hard_gates_passed", False) for row in scored),
            "repair_operators_observed": operators,
            "production_ready": False,
            "production_readiness_note": (
                "Advisory only until >=30 held-out cases, three repair operators, hard-negative calibration, "
                "and multi-rollout reliability are measured."
            ),
        },
        "cases": rows,
    }
    if output_path:
        write_json(Path(output_path), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--artifact-dir", type=Path)
    group.add_argument("--run-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.artifact_dir:
        report = evaluate_artifact_dir(args.artifact_dir)
    else:
        report = evaluate_run(args.run_root, args.output)
    print(json.dumps(report.get("summary", report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
