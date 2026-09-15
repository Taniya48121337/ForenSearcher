import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_belief_update_uptake import audit


def test_existing_pool_belief_update_uptake_audit_is_reproducible() -> None:
    report = audit()
    assert report["training_qualification_changed"] is False
    assert report["overall"] == {
        "expected_updates": 99,
        "passed_updates": 96,
        "pass_rate": 0.969697,
    }
    assert report["by_dataset"]["VidDiffBench"]["failure_counts"] == {
        "new_atom_named": 3
    }
