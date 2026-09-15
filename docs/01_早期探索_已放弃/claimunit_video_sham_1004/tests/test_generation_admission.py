from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generation_admission import (  # noqa: E402
    generation_admission_summary,
    generation_failure_metadata,
)


class Failure(RuntimeError):
    def __init__(self, artifact):
        super().__init__("failed")
        self.failure_artifact = artifact


def test_proposal_exhaustion_taxonomy_preserves_reason_counts() -> None:
    metadata = generation_failure_metadata(
        Failure(
            {
                "schema_version": "evidence_choice_proposal_failure_v1",
                "failure_kind": "bounded_checkpoint_proposal_exhaustion",
                "reason_counts": {"proposal_coverage_failed": 2},
            }
        )
    )
    assert metadata["failure_category"] == "preaction_proposal_quality_exhausted"
    assert metadata["failure_reason_counts"] == {"proposal_coverage_failed": 2}


def test_admission_summary_separates_recovered_and_unrecovered_failures() -> None:
    report = generation_admission_summary(
        [
            {
                "status": "passed",
                "attempts": [
                    {
                        "status": "execution_failed",
                        "failure_category": "preaction_proposal_quality_exhausted",
                        "failure_reason_counts": {"proposal_coverage_failed": 2},
                    },
                    {"status": "passed"},
                ],
            },
            {
                "status": "execution_failed",
                "failure_category": "selected_branch_execution_inadmissible",
                "failure_reason_counts": {
                    "selected_branch_execution_inadmissible": 1
                },
            },
        ]
    )
    assert report["accepted_case_fraction"] == 0.5
    assert report["cases_recovered_by_later_attempt"] == 1
    assert report["proposal_rejection_reason_counts"] == {
        "proposal_coverage_failed": 2,
        "selected_branch_execution_inadmissible": 1,
    }
