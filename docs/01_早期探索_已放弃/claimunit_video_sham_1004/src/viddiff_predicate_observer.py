"""Typed, label-blind predicate observations and deterministic A/B adjudication.

The vision model records independent per-video facts. It never emits the dataset
direction. This module validates those facts against the predicate-specific proof
obligation and computes a conservative direction only when both records are
comparable.
"""

from __future__ import annotations

from typing import Any


CRITICAL_UNCERTAINTY_CODES = {
    "unresolved_landmarks",
    "noncomparable_scale",
    "hidden_quantity_no_proxy",
    "event_boundaries_ambiguous",
    "camera_motion_dominates",
    "target_action_not_visible",
    "projection_crossing_ambiguous",
    "boundary_skipped",
    "frame_skip",
    "partial_visibility",
}


def observation_record_schema() -> dict[str, Any]:
    interval = {
        "type": "array",
        "items": {"type": "number", "minimum": 0},
        "minItems": 2,
        "maxItems": 2,
    }
    nullable_number = {"type": ["number", "null"]}
    nullable_count = {"type": ["integer", "null"], "minimum": 0}
    return {
        "type": "object",
        "properties": {
            "observable": {"type": "boolean"},
            "measured_variable": {"type": "string", "minLength": 3, "maxLength": 120},
            "coverage_fraction": {"type": "number", "minimum": 0, "maximum": 1},
            "valid_units": {"type": "integer", "minimum": 0},
            "scalar_value": nullable_number,
            "event_count": nullable_count,
            "violation_count": nullable_count,
            "proxy_cue_count": {"type": "integer", "minimum": 0},
            "observed_units": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "time_s": {"type": "number", "minimum": 0},
                        "description": {"type": "string", "minLength": 8, "maxLength": 240},
                    },
                    "required": ["time_s", "description"],
                    "additionalProperties": False,
                },
                "maxItems": 32,
            },
            "cue_times_s": {
                "type": "array",
                "items": {"type": "number", "minimum": 0},
                "maxItems": 32,
            },
            "counterexample_times_s": {
                "type": "array",
                "items": {"type": "number", "minimum": 0},
                "maxItems": 32,
            },
            "uncertainty_codes": {
                "type": "array",
                "items": {"type": "string", "minLength": 3},
                "maxItems": 12,
                "uniqueItems": True,
            },
            "interval_s": interval,
            "summary": {"type": "string", "minLength": 20, "maxLength": 800},
        },
        "required": [
            "observable",
            "measured_variable",
            "coverage_fraction",
            "valid_units",
            "scalar_value",
            "event_count",
            "violation_count",
            "proxy_cue_count",
            "observed_units",
            "cue_times_s",
            "counterexample_times_s",
            "uncertainty_codes",
            "interval_s",
            "summary",
        ],
        "additionalProperties": False,
    }


def predicate_observer_schema() -> dict[str, Any]:
    record = observation_record_schema()
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "video_a": record,
            "video_b": record,
            "comparative_note": {"type": "string", "minLength": 30, "maxLength": 800},
        },
        "required": ["video_a", "video_b", "comparative_note"],
        "additionalProperties": False,
    }


def observation_problems(record: dict[str, Any], protocol: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    missing = [
        field
        for field in protocol.get("required_observation_record") or []
        if field not in record
    ]
    if missing:
        problems.append(f"missing_fields:{','.join(missing)}")
        return problems

    coverage = record.get("coverage_fraction")
    if not isinstance(coverage, (int, float)) or isinstance(coverage, bool) or not 0 <= coverage <= 1:
        problems.append("coverage_fraction_invalid")
    valid_units = record.get("valid_units")
    if not isinstance(valid_units, int) or isinstance(valid_units, bool) or valid_units < 0:
        problems.append("valid_units_invalid")
    for field in ("event_count", "violation_count"):
        value = record.get(field)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            problems.append(f"{field}_invalid")
    proxy_cues = record.get("proxy_cue_count")
    if not isinstance(proxy_cues, int) or isinstance(proxy_cues, bool) or proxy_cues < 0:
        problems.append("proxy_cue_count_invalid")
    measured_variable = record.get("measured_variable")
    if not isinstance(measured_variable, str) or not measured_variable.strip():
        problems.append("measured_variable_invalid")
    observed_units = record.get("observed_units")
    if not isinstance(observed_units, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("time_s"), (int, float))
        or isinstance(item.get("time_s"), bool)
        or float(item.get("time_s")) < 0
        or not isinstance(item.get("description"), str)
        or len(str(item.get("description")).strip()) < 8
        for item in observed_units or []
    ):
        problems.append("observed_units_invalid")
    interval = record.get("interval_s")
    if not (
        isinstance(interval, list)
        and len(interval) == 2
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in interval)
        and 0 <= float(interval[0]) <= float(interval[1])
    ):
        problems.append("interval_invalid")
    for field in ("cue_times_s", "counterexample_times_s"):
        values = record.get(field)
        if not isinstance(values, list) or any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0
            for value in values or []
        ):
            problems.append(f"{field}_invalid")
    uncertainties = record.get("uncertainty_codes")
    if not isinstance(uncertainties, list) or any(
        not isinstance(value, str) or not value.strip() for value in uncertainties or []
    ):
        problems.append("uncertainty_codes_invalid")
    summary = record.get("summary")
    if not isinstance(summary, str) or len(summary.strip()) < 20:
        problems.append("summary_too_short")
    return problems


def _usable(record: dict[str, Any], protocol: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = observation_problems(record, protocol)
    if reasons:
        return False, reasons
    if record.get("observable") is not True:
        reasons.append("not_observable")
    if str(record.get("measured_variable") or "") != str(
        protocol.get("target_variable") or ""
    ):
        reasons.append("measured_variable_mismatch")
    if float(record.get("coverage_fraction") or 0) < 0.5:
        reasons.append("insufficient_coverage")
    if int(record.get("valid_units") or 0) < int(protocol.get("minimum_valid_units") or 1):
        reasons.append("insufficient_valid_units")
    critical = sorted(
        set(str(value) for value in record.get("uncertainty_codes") or [])
        & CRITICAL_UNCERTAINTY_CODES
    )
    if critical:
        reasons.append(f"critical_uncertainty:{','.join(critical)}")
    if protocol.get("family") == "force_tension_proxy" and int(
        record.get("proxy_cue_count") or 0
    ) < int(protocol.get("minimum_proxy_cues") or 2):
        reasons.append("insufficient_visible_proxy_cues")
    if protocol.get("family") in {"event_count", "thread_topology"}:
        event_count = record.get("event_count")
        units = list(record.get("observed_units") or [])
        if event_count is None or int(event_count) != len(units):
            reasons.append("event_count_does_not_match_observed_units")
        keyword_groups = list(protocol.get("required_unit_keyword_groups") or [])
        for index, unit in enumerate(units):
            description = str(unit.get("description") or "").lower()
            if any(
                not any(str(keyword).lower() in description for keyword in group)
                for group in keyword_groups
            ):
                reasons.append(f"observed_unit_semantic_mismatch:{index}")
    return not reasons, reasons


def _metric(record: dict[str, Any], protocol: dict[str, Any]) -> float | None:
    family = str(protocol.get("family") or "")
    if family == "visibility_coverage":
        count = record.get("violation_count")
        units = int(record.get("valid_units") or 0)
        return None if count is None or units <= 0 else float(count) / units
    if family in {"event_count", "thread_topology"}:
        count = record.get("event_count")
        return None if count is None else float(count)
    value = record.get("scalar_value")
    return None if value is None else float(value)


def adjudicate_predicate_observations(
    protocol: dict[str, Any],
    video_a: dict[str, Any],
    video_b: dict[str, Any],
) -> dict[str, Any]:
    """Conservatively compare two independent typed records without labels."""

    usable_a, reasons_a = _usable(video_a, protocol)
    usable_b, reasons_b = _usable(video_b, protocol)
    metric_a = _metric(video_a, protocol) if usable_a else None
    metric_b = _metric(video_b, protocol) if usable_b else None
    family = str(protocol.get("family") or "generic_visual_comparison")

    if not usable_a or not usable_b or metric_a is None or metric_b is None:
        return {
            "preferred_video": "c",
            "decision_rule": "predicate_observation_insufficient_abstain",
            "confidence": 0.0,
            "metric_a": metric_a,
            "metric_b": metric_b,
            "usable_a": usable_a,
            "usable_b": usable_b,
            "reasons_a": reasons_a,
            "reasons_b": reasons_b,
        }

    if family == "visibility_coverage" and metric_a == 0 and metric_b == 0:
        return {
            "preferred_video": "c",
            "decision_rule": "universal_claim_no_observed_counterexample_abstain",
            "confidence": 0.25,
            "metric_a": metric_a,
            "metric_b": metric_b,
            "usable_a": True,
            "usable_b": True,
            "reasons_a": ["sparse_sampling_cannot_prove_never"],
            "reasons_b": ["sparse_sampling_cannot_prove_never"],
        }

    scale = max(abs(metric_a), abs(metric_b), 1.0)
    relative_margin = abs(metric_a - metric_b) / scale
    required_margin = float(protocol.get("minimum_relative_margin") or 0.15)
    if relative_margin < required_margin:
        return {
            "preferred_video": "c",
            "decision_rule": "predicate_metric_margin_insufficient_abstain",
            "confidence": round(min(0.5, 0.2 + relative_margin), 3),
            "metric_a": metric_a,
            "metric_b": metric_b,
            "relative_margin": round(relative_margin, 6),
            "usable_a": True,
            "usable_b": True,
            "reasons_a": [],
            "reasons_b": [],
        }

    if protocol.get("metric_direction") == "lower_is_more_support":
        preferred = "a" if metric_a < metric_b else "b"
    else:
        preferred = "a" if metric_a > metric_b else "b"
    coverage = min(
        float(video_a.get("coverage_fraction") or 0),
        float(video_b.get("coverage_fraction") or 0),
    )
    confidence_cap = 0.7 if family == "visibility_coverage" else 0.9
    confidence = min(confidence_cap, 0.45 + 0.35 * relative_margin * coverage)
    return {
        "preferred_video": preferred,
        "decision_rule": "typed_predicate_metric_margin",
        "confidence": round(confidence, 3),
        "metric_a": metric_a,
        "metric_b": metric_b,
        "relative_margin": round(relative_margin, 6),
        "usable_a": True,
        "usable_b": True,
        "reasons_a": [],
        "reasons_b": [],
    }


def swap_adjudication(result: dict[str, Any]) -> dict[str, Any]:
    """Map an adjudication made under swapped A/B roles back to original roles."""

    mapped = dict(result)
    mapped["preferred_video"] = {
        "a": "b",
        "b": "a",
        "c": "c",
    }[str(result.get("preferred_video") or "c")]
    mapped["metric_a"], mapped["metric_b"] = result.get("metric_b"), result.get("metric_a")
    mapped["usable_a"], mapped["usable_b"] = result.get("usable_b"), result.get("usable_a")
    mapped["reasons_a"], mapped["reasons_b"] = result.get("reasons_b"), result.get("reasons_a")
    return mapped
