"""Compile VidDiff comparison statements into label-blind visual protocols."""

from __future__ import annotations

import re
from typing import Any


PROTOCOLS: tuple[dict[str, Any], ...] = (
    {
        "family": "visibility_coverage",
        "pattern": r"never\s+out\s+of\s+view|occlud|out\s+of\s+frame|visible",
        "observation_unit": "sampled frame with both task-relevant instrument tips visibly present",
        "aggregation": "compare visible-frame coverage and the longest contiguous missing/occluded run",
        "asset_policy": "exhaustive_uniform_timeline_plus_targeted_detail",
        "abstain_rule": "choose c when tip identity or visibility cannot be judged in enough frames",
    },
    {
        "family": "event_count",
        "pattern": r"number\s+of|more\s+errors|stops?\s+more\s+often|corrected",
        "observation_unit": "complete, non-overlapping task event matching the statement",
        "aggregation": "count events independently in A and B over the full chronological timeline",
        "asset_policy": "exhaustive_uniform_timeline_with_motion_context",
        "abstain_rule": "choose c when event boundaries are skipped or cannot be distinguished",
    },
    {
        "family": "thread_topology",
        "pattern": r"thread\s+tangles?|suturing\s+thread\s+tangles?",
        "observation_unit": "new persistent crossing, loop lock, or visible tangle state",
        "aggregation": "compare the number and persistence of topology changes across the full timeline",
        "asset_policy": "exhaustive_uniform_timeline_plus_dense_local_detail",
        "abstain_rule": "choose c when crossings are caused by projection or tool occlusion",
    },
    {
        "family": "geometric_relation",
        "pattern": r"perpendicular|radial|closer\s+to\s+the\s+tip|2/3\s+from|accurately\s+hit|correct\s+position",
        "observation_unit": "task-relevant angle, normalized grasp position, or tip-to-target distance",
        "aggregation": "measure the relation in comparable contact frames and compare robust central tendency",
        "asset_policy": "high_resolution_targeted_detail_with_global_context",
        "abstain_rule": "choose c when the needle tip, incision, hoop, dot, or contact frame is not resolved",
    },
    {
        "family": "force_tension_proxy",
        "pattern": r"\bforce\b|\btension\b|taut|slack",
        "observation_unit": "visible deformation, slack, oscillation, contact displacement, or settling after load",
        "aggregation": "compare only visible proxy magnitude and temporal variability; never infer hidden force directly",
        "asset_policy": "full_timeline_plus_adjacent_frame_motion_and_targeted_detail",
        "abstain_rule": "choose c when the claimed force/tension has no reliable visible proxy",
    },
    {
        "family": "bimanual_support",
        "pattern": r"left\s+gras(?:p|er)|left\s+graser|second\s+grasper|both\s+graspers|stabili[sz]e|stabalize|supports?\s+the\s+right",
        "observation_unit": "sustained second-tool contact or stabilizing participation during the target action",
        "aggregation": "compare participation coverage and continuity, not mere tool presence",
        "asset_policy": "full_timeline_plus_targeted_contact_detail",
        "abstain_rule": "choose c when contact and causal stabilization cannot be visually separated",
    },
    {
        "family": "kinematic_progression",
        "pattern": r"speed|faster|more\s+fluid|tube\s+moves?|movement\s+of\s+the\s+needle|moves?\s+more",
        "observation_unit": "adjacent-frame displacement, pause, reversal, or smooth task progression",
        "aggregation": "compare normalized displacement/progression over matched local windows and the full action",
        "asset_policy": "motion_strip_plus_targeted_adjacent_frames_with_global_context",
        "abstain_rule": "choose c when duration, camera scale, or sparse sampling dominates the apparent motion",
    },
)


GENERIC_PROTOCOL = {
    "family": "generic_visual_comparison",
    "observation_unit": "directly visible statement-relevant event or relation",
    "aggregation": "describe A and B independently, then compare only matched visible evidence",
    "asset_policy": "full_timeline_plus_targeted_detail",
    "abstain_rule": "choose c when the statement is not reliably observable from the supplied views",
}


ROUTE_PRIORITY = {
    "visibility_coverage": ["exhaustive_timeline", "targeted_adjacent_strip"],
    "event_count": ["exhaustive_timeline", "motion_strip", "targeted_adjacent_strip"],
    "thread_topology": ["exhaustive_timeline", "targeted_adjacent_strip"],
    "geometric_relation": ["targeted_high_resolution", "exhaustive_timeline"],
    "force_tension_proxy": ["targeted_adjacent_strip", "motion_strip", "exhaustive_timeline"],
    "bimanual_support": ["targeted_adjacent_strip", "exhaustive_timeline"],
    "kinematic_progression": ["motion_strip", "targeted_adjacent_strip", "exhaustive_timeline"],
    "generic_visual_comparison": ["exhaustive_timeline", "targeted_adjacent_strip"],
}


def _semantic_contract(statement: str, family: str) -> dict[str, Any]:
    """Return the typed, label-blind proof obligation for one statement.

    Every metric is defined so that the deterministic comparator can infer a
    direction from two independently produced observation records. The teacher
    never receives or emits the dataset direction.
    """

    lowered = statement.lower()
    common = {
        "route_priority": ROUTE_PRIORITY[family],
        "minimum_relative_margin": 0.15,
        "required_observation_record": [
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
        "disallowed_shortcuts": [
            "video duration alone",
            "general skill impression",
            "subject identity",
            "A/B presentation order",
            "confidence without a typed observation",
        ],
    }
    if family == "visibility_coverage":
        return {
            **common,
            "quantifier": "universal_negative",
            "target_variable": "observed_visibility_violation_rate",
            "metric_direction": "lower_is_more_support",
            "minimum_valid_units": 8,
            "proof_obligations": [
                "Record every sampled occlusion or out-of-frame counterexample.",
                "If neither video has an observed violation, abstain: sparse sampling cannot prove never.",
                "Compare violation rates only over valid frames where both task-relevant tips can be judged.",
            ],
        }
    if family == "event_count":
        lower_is_support = bool(
            re.search(r"(?:number\s+of.{0,120}\blower\b|\bfewer\b|\bless\b)", lowered)
        )
        if re.search(r"number\s+of.{0,80}(?:arrange|needle).{0,80}thread", lowered):
            unit_definition = "discrete needle-repositioning movement before threading begins"
            keyword_groups = [["arrang", "reposition"], ["before thread", "pre-thread"]]
        elif re.search(r"errors?.{0,30}correct", lowered):
            unit_definition = "completed correction of a visible task error"
            keyword_groups = [["error", "mistake"], ["correct", "repair"]]
        else:
            unit_definition = "complete stop or pause used to plan the next task step"
            keyword_groups = [["stop", "pause"], ["plan", "next step"]]
        return {
            **common,
            "quantifier": "count",
            "target_variable": "complete_nonoverlapping_event_count",
            "observation_unit_definition": unit_definition,
            "required_unit_keyword_groups": keyword_groups,
            "metric_direction": (
                "lower_is_more_support" if lower_is_support else "higher_is_more_support"
            ),
            "minimum_valid_units": 1,
            "proof_obligations": [
                "List timestamped complete events before counting them.",
                "Do not count pauses, corrections, or movements unless they match the exact statement.",
                "Abstain when event boundaries are skipped, duplicated, or not visible.",
            ],
        }
    if family == "thread_topology":
        return {
            **common,
            "quantifier": "persistent_event_count",
            "target_variable": "persistent_tangle_event_count",
            "observation_unit_definition": "new crossing, loop lock, or topology change that persists across adjacent frames",
            "required_unit_keyword_groups": [["cross", "loop", "tangle"], ["persist", "adjacent"]],
            "metric_direction": "higher_is_more_support",
            "minimum_valid_units": 2,
            "proof_obligations": [
                "A tangle requires a new crossing, loop lock, or topology change that persists across adjacent frames.",
                "Projection crossings and temporary tool occlusions are counterevidence, not tangle events.",
            ],
        }
    if family == "geometric_relation":
        if "perpendicular" in lowered:
            target = "absolute_angle_error_from_90_degrees"
        elif "closer" in lowered or "2/3" in lowered or "correct position" in lowered:
            target = "normalized_position_error"
        elif "accurately hit" in lowered:
            target = "normalized_tip_to_target_distance"
        else:
            target = "absolute_radial_alignment_error"
        return {
            **common,
            "quantifier": "robust_geometric_measurement",
            "target_variable": target,
            "metric_direction": "lower_is_more_support",
            "minimum_valid_units": 3,
            "proof_obligations": [
                "Use only comparable contact frames with all required landmarks resolved.",
                "Normalize positions or distances by a visible object scale before comparing videos.",
                "Report multiple measurements and aggregate their median; a single pose is insufficient.",
            ],
        }
    if family == "force_tension_proxy":
        lower_is_support = bool(re.search(r"\blower\b|\bless\b", lowered))
        higher_is_support = bool(re.search(r"\bmore\s+force\b", lowered))
        return {
            **common,
            "quantifier": "visible_proxy_only",
            "target_variable": "visible_deformation_slack_or_settling_proxy",
            "metric_direction": (
                "lower_is_more_support"
                if lower_is_support
                else "higher_is_more_support"
                if higher_is_support
                else "higher_is_more_support"
            ),
            "minimum_valid_units": 3,
            "minimum_proxy_cues": 2,
            "proof_obligations": [
                "Name at least two timestamped visible deformation, slack, oscillation, displacement, or settling cues.",
                "Never infer hidden force or tension directly from tool pose, speed, or confidence.",
                "Abstain when a common visible proxy cannot be measured in both videos.",
            ],
        }
    if family == "bimanual_support":
        return {
            **common,
            "quantifier": "participation_coverage",
            "target_variable": "sustained_second_tool_contact_fraction",
            "metric_direction": "higher_is_more_support",
            "minimum_valid_units": 3,
            "proof_obligations": [
                "Count sustained contact or stabilizing participation, not passive tool presence.",
                "Require adjacent frames that show the second tool participating during the target action.",
            ],
        }
    if family == "kinematic_progression":
        fluid = "fluid" in lowered
        return {
            **common,
            "quantifier": "temporal_progression",
            "target_variable": (
                "pause_reversal_or_jerk_rate" if fluid else "normalized_progression_or_displacement_rate"
            ),
            "metric_direction": "lower_is_more_support" if fluid else "higher_is_more_support",
            "minimum_valid_units": 4,
            "proof_obligations": [
                "Use adjacent-frame displacement or matched action progression, never duration alone.",
                "Separate camera motion and scale change from task-object motion.",
                "For fluidity, measure pauses, reversals, or abrupt changes rather than raw speed.",
            ],
        }
    return {
        **common,
        "quantifier": "direct_visible_support",
        "target_variable": "statement_support_score",
        "metric_direction": "higher_is_more_support",
        "minimum_valid_units": 3,
        "proof_obligations": [
            "Record matched visible evidence independently for each video.",
            "Abstain when the statement lacks a shared observable variable.",
        ],
    }


def compile_protocol(statement: str) -> dict[str, Any]:
    normalized = str(statement).strip()
    for protocol in PROTOCOLS:
        if re.search(str(protocol["pattern"]), normalized, re.I):
            compiled = {
                key: value for key, value in protocol.items() if key != "pattern"
            }
            compiled.update(_semantic_contract(normalized, str(protocol["family"])))
            return compiled
    compiled = dict(GENERIC_PROTOCOL)
    compiled.update(_semantic_contract(normalized, "generic_visual_comparison"))
    return compiled
