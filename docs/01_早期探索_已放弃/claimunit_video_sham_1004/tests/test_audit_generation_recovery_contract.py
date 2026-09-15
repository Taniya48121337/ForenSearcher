from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_generation_recovery_contract import audit  # noqa: E402


def test_bounded_checkpoint_recovery_contract() -> None:
    assert audit()["all_passed"] is True
