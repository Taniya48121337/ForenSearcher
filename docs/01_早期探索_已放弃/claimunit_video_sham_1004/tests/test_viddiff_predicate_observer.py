from __future__ import annotations

import copy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from viddiff_predicate_observer import (  # noqa: E402
    adjudicate_predicate_observations,
    observation_problems,
    predicate_observer_schema,
    swap_adjudication,
)
from viddiff_predicate_protocol import compile_protocol  # noqa: E402
from viddiff_probe_protocol import ProtocolSymmetricComparator  # noqa: E402


def record(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "observable": True,
        "measured_variable": "complete_nonoverlapping_event_count",
        "coverage_fraction": 0.9,
        "valid_units": 12,
        "scalar_value": None,
        "event_count": None,
        "violation_count": None,
        "proxy_cue_count": 0,
        "observed_units": [],
        "cue_times_s": [1.0, 2.0],
        "counterexample_times_s": [],
        "uncertainty_codes": [],
        "interval_s": [0.0, 3.0],
        "summary": "Two independent timestamped observations are clearly visible.",
    }
    value.update(changes)
    return value


def test_observer_schema_does_not_request_a_direction_or_label() -> None:
    schema = predicate_observer_schema()
    rendered = str(schema).lower()
    assert "preferred_video" not in rendered
    assert "oracle" not in rendered
    assert "label" not in rendered


def test_visibility_without_counterexample_must_abstain() -> None:
    protocol = compile_protocol("The instrument tips are never out of view")
    a = record(
        measured_variable="observed_visibility_violation_rate",
        violation_count=0,
        cue_times_s=[],
    )
    b = record(
        measured_variable="observed_visibility_violation_rate",
        violation_count=0,
        cue_times_s=[],
    )
    result = adjudicate_predicate_observations(protocol, a, b)
    assert result["preferred_video"] == "c"
    assert result["decision_rule"] == "universal_claim_no_observed_counterexample_abstain"


def test_visibility_prefers_fewer_observed_violations_but_caps_confidence() -> None:
    protocol = compile_protocol("The instrument tips are never out of view")
    a = record(
        measured_variable="observed_visibility_violation_rate",
        violation_count=0,
        cue_times_s=[],
    )
    b = record(
        measured_variable="observed_visibility_violation_rate",
        violation_count=3,
        counterexample_times_s=[0.5, 1.5, 2.5],
    )
    result = adjudicate_predicate_observations(protocol, a, b)
    assert result["preferred_video"] == "a"
    assert result["confidence"] <= 0.7


def test_hidden_force_without_two_visible_proxy_cues_must_abstain() -> None:
    protocol = compile_protocol("The tension on the suturing thread is lower")
    a = record(
        measured_variable="visible_deformation_slack_or_settling_proxy",
        scalar_value=0.2,
        proxy_cue_count=1,
    )
    b = record(
        measured_variable="visible_deformation_slack_or_settling_proxy",
        scalar_value=0.8,
        proxy_cue_count=3,
    )
    result = adjudicate_predicate_observations(protocol, a, b)
    assert result["preferred_video"] == "c"
    assert "insufficient_visible_proxy_cues" in result["reasons_a"]


def test_lower_event_count_and_geometric_error_map_to_correct_role() -> None:
    count_protocol = compile_protocol(
        "The number of movements to arrange the needle before threading is lower"
    )
    count_result = adjudicate_predicate_observations(
        count_protocol,
        record(
            event_count=2,
            observed_units=[
                {"time_s": 1.0, "description": "Reposition before threading begins."},
                {"time_s": 2.0, "description": "Arrange the needle before threading."},
            ],
        ),
        record(
            event_count=3,
            observed_units=[
                {"time_s": 1.0, "description": "Reposition before threading begins."},
                {"time_s": 2.0, "description": "Arrange the needle before threading."},
                {"time_s": 2.5, "description": "Reposition before threading starts."},
            ],
        ),
    )
    assert count_result["preferred_video"] == "a"

    geometry_protocol = compile_protocol("The needle is grasped closer to the tip")
    geometry_result = adjudicate_predicate_observations(
        geometry_protocol,
        record(measured_variable="normalized_position_error", scalar_value=0.18),
        record(measured_variable="normalized_position_error", scalar_value=0.43),
    )
    assert geometry_result["preferred_video"] == "a"


def test_nearby_but_wrong_event_definition_cannot_produce_direction() -> None:
    protocol = compile_protocol(
        "The number of movements to arrange the needle before threading is lower"
    )
    wrong = record(
        event_count=1,
        observed_units=[
            {"time_s": 15.3, "description": "Needle passed through four hoops."}
        ],
    )
    valid = record(
        event_count=2,
        observed_units=[
            {"time_s": 1.0, "description": "Reposition before threading begins."},
            {"time_s": 2.0, "description": "Arrange the needle before threading."},
        ],
    )
    result = adjudicate_predicate_observations(protocol, wrong, valid)
    assert result["preferred_video"] == "c"
    assert "observed_unit_semantic_mismatch:0" in result["reasons_a"]


def test_role_swap_is_equivariant() -> None:
    protocol = compile_protocol("More errors are corrected")
    a = record(
        event_count=2,
        observed_units=[
            {"time_s": 1.0, "description": "Visible task error is corrected."},
            {"time_s": 2.0, "description": "A second mistake is repaired."},
        ],
    )
    b = record(
        event_count=1,
        observed_units=[
            {"time_s": 1.5, "description": "One visible task error is corrected."},
        ],
    )
    forward = adjudicate_predicate_observations(protocol, a, b)
    swapped = adjudicate_predicate_observations(protocol, b, a)
    assert swap_adjudication(swapped)["preferred_video"] == forward["preferred_video"]
    assert swap_adjudication(swapped)["metric_a"] == forward["metric_a"]
    assert swap_adjudication(swapped)["metric_b"] == forward["metric_b"]


def test_critical_uncertainty_or_malformed_record_cannot_win() -> None:
    protocol = compile_protocol("The dot is more accurately hit")
    bad = record(
        measured_variable="normalized_tip_to_target_distance",
        scalar_value=0.1,
        uncertainty_codes=["unresolved_landmarks"],
    )
    good = record(
        measured_variable="normalized_tip_to_target_distance",
        scalar_value=0.6,
    )
    result = adjudicate_predicate_observations(protocol, bad, good)
    assert result["preferred_video"] == "c"

    malformed = copy.deepcopy(good)
    malformed["coverage_fraction"] = 1.4
    assert "coverage_fraction_invalid" in observation_problems(malformed, protocol)


class FakeComparator(ProtocolSymmetricComparator):
    def __init__(self, outputs: list[dict[str, object]]) -> None:
        self.outputs = iter(outputs)

    def _one_pass(self, **_kwargs: object) -> dict[str, object]:
        return next(self.outputs)


def pass_output(a: dict[str, object], b: dict[str, object]) -> dict[str, object]:
    return {
        "video_a": a,
        "video_b": b,
        "comparative_note": "The two independent records are left for deterministic comparison.",
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


def test_symmetric_comparator_uses_typed_records_not_teacher_direction() -> None:
    protocol = compile_protocol("More errors are corrected")
    original_a = record(
        event_count=2,
        observed_units=[
            {"time_s": 1.0, "description": "Visible task error is corrected."},
            {"time_s": 2.0, "description": "A second mistake is repaired."},
        ],
    )
    original_b = record(
        event_count=1,
        observed_units=[
            {"time_s": 1.5, "description": "One visible task error is corrected."},
        ],
    )
    comparator = FakeComparator(
        [
            pass_output(original_a, original_b),
            pass_output(original_b, original_a),
        ]
    )
    result = comparator.compare(
        timeline_a=Path("a-timeline.jpg"),
        motion_a=Path("a-motion.jpg"),
        targeted_a=Path("a-targeted.jpg"),
        timeline_b=Path("b-timeline.jpg"),
        motion_b=Path("b-motion.jpg"),
        targeted_b=Path("b-targeted.jpg"),
        action_description="A neutral shared task.",
        candidate_difference="More errors are corrected",
        protocol=protocol,
        metadata_a={},
        metadata_b={},
    )
    assert result["preferred_video"] == "a"
    assert result["order_consistent"] is True
    assert result["decision_rule"] == "typed_predicate_order_consistent"
    assert "preferred_video" not in result["forward"]["raw_observations"]
