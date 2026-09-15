import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sft_learning_value import learning_value_profile, supervision_profile


def test_semantic_measurement_and_repair_receive_core_weight() -> None:
    measure = learning_value_profile(
        "compare_global",
        external_measurement=True,
        branch_enabling_measurement=True,
    )
    repair = learning_value_profile(
        "set_field_decision",
        params={"decision_status": "open"},
    )
    assert measure["learning_value_weight"] == 1.2
    assert measure["learning_value_tier"] == "core_reasoning"
    assert repair["learning_value_weight"] == 1.2


def test_indispensable_bookkeeping_is_retained_but_downweighted() -> None:
    bind = learning_value_profile("bind_evidence")
    inspect = learning_value_profile("inspect_task_state")
    stop = learning_value_profile("stop")
    assert bind["learning_value_tier"] == "supporting"
    assert inspect["learning_value_tier"] == "minimal_control"
    assert stop["learning_value_weight"] == 0.2
    assert stop["supervision_subtype"] == "terminal_verification"


def test_materialization_does_not_relabel_semantic_claim_creation() -> None:
    profile = supervision_profile(
        "create_candidate_claim",
        full_target_call_materializable=True,
    )
    assert profile["class"] == "evidence_interpretation"
    assert profile["training_weight"] == 1.0
