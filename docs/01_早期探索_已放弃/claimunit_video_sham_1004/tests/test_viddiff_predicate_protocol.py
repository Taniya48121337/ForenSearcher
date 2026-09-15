from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from viddiff_predicate_protocol import compile_protocol  # noqa: E402


def test_predicate_protocols_route_distinct_measurement_semantics() -> None:
    cases = {
        "The instrument tips are never out of view": "visibility_coverage",
        "The number of movements is lower": "event_count",
        "More errors are corrected": "event_count",
        "The suturing thread tangles.": "thread_topology",
        "The tension is lower": "force_tension_proxy",
        "The needle is more perpendicular": "geometric_relation",
        "The force is applied in a more radial way": "geometric_relation",
        "The second grasper is used to stabilize the target": "bimanual_support",
        "The passage is more fluid": "kinematic_progression",
    }
    for statement, family in cases.items():
        assert compile_protocol(statement)["family"] == family


def test_force_protocol_requires_visible_proxy_and_abstention() -> None:
    protocol = compile_protocol("The instrument applies more force to the tissue")
    assert protocol["family"] == "force_tension_proxy"
    assert "never infer hidden force" in protocol["aggregation"]
    assert "choose c" in protocol["abstain_rule"]


def test_unknown_statement_has_safe_generic_protocol() -> None:
    protocol = compile_protocol("The action is performed better")
    assert protocol["family"] == "generic_visual_comparison"
    assert "choose c" in protocol["abstain_rule"]


def test_new_external_holdout_predicates_all_compile_to_typed_contracts() -> None:
    statements = [
        "The tension on the suturing material and the tissue is better controlled",
        "The number of movements to arrange the needle before threading is lower",
        "The grasper is more precisely positioned to the correct position, which is approximately 2/3 from the needle tip",
        "The needle is grasped closer to the tip",
        "The needle is inserted in the fabric more perpendicular to the incision.",
        "The tension on the suturing thread is lower",
        "More errors are corrected",
        "The instrument tips are never out of view (occluded by instruments, or out of frame)",
        "The force is applied in a more radial way",
        "The suturing speed is higher",
        "The left graser supports the right grasper, by pressing down on the tissue.",
        "The tube moves more",
        "The dot is more accurately hit",
        "The suturing thread tangles.",
        "The movement of the needle through the hoop is more radial",
        "The passage of the needle between two hands is more fluid",
        "The second grasper is used to stabalize the target.",
        "The instrument applies more force to the tissue and needle",
        "The surgeon stops more often to plan next steps",
        "The movements are faster.",
        "The force on the target is lower",
    ]
    for statement in statements:
        protocol = compile_protocol(statement)
        assert protocol["family"] != "generic_visual_comparison", statement
        assert protocol["metric_direction"] in {
            "higher_is_more_support",
            "lower_is_more_support",
        }
        assert protocol["target_variable"]
        assert protocol["proof_obligations"]
        assert protocol["route_priority"]
        assert protocol["required_observation_record"]


def test_event_count_polarity_is_compiled_from_statement() -> None:
    lower = compile_protocol(
        "The number of movements to arrange the needle before threading is lower"
    )
    higher = compile_protocol("More errors are corrected")
    assert lower["metric_direction"] == "lower_is_more_support"
    assert higher["metric_direction"] == "higher_is_more_support"
