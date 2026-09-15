from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from integrate_viddiff_typed_observer import integrate  # noqa: E402


def test_typed_observer_replaces_legacy_candidate_without_exporting_labels(
    tmp_path: Path,
) -> None:
    adaptive = tmp_path / "adaptive.jsonl"
    adaptive.write_text(
        json.dumps(
            {
                "sample_key": "surgery_1",
                "task": {"comparison_statement": "More errors are corrected"},
                "candidate_tools": [
                    {"name": "compare_uniform_timeline"},
                    {"name": "compare_predicate_independent", "legacy": True},
                ],
                "routing_contract": {
                    "initial_candidates": ["compare_uniform_timeline"]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    probe = tmp_path / "probe.json"
    probe.write_text(
        json.dumps(
            {
                "external_service_calls": 0,
                "generator_label_visibility": False,
                "cases": [
                    {
                        "sample_key": "surgery_1",
                        "comparison_statement": "More errors are corrected",
                        "protocol_family": "event_count",
                        "records": {
                            "a": {"interval_s": [0, 2], "observed_units": []},
                            "b": {"interval_s": [1, 3], "observed_units": []},
                        },
                        "measurement": {
                            "preferred_video": "c",
                            "decision_rule": "predicate_metric_margin_insufficient_abstain",
                            "confidence": 0.2,
                            "metric_a": 1,
                            "metric_b": 1,
                            "usable_a": True,
                            "usable_b": True,
                            "reasons_a": [],
                            "reasons_b": [],
                        },
                        "oracle_direction": "a",
                        "direction_correct": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "integrated.jsonl"
    report = integrate(adaptive, [probe], output)
    assert report["augmented_cases"] == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    predicates = [
        item
        for item in row["candidate_tools"]
        if item["name"] == "compare_predicate_independent"
    ]
    assert len(predicates) == 1
    assert predicates[0].get("legacy") is None
    assert predicates[0]["tool_observation"]["preferred_video"] == "c"
    assert row["routing_contract"]["initial_candidates"][0] == (
        "compare_predicate_independent"
    )
    rendered = json.dumps(row).lower()
    assert "oracle_direction" not in rendered
    assert "direction_correct" not in rendered
