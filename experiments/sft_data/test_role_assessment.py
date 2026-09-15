import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from assessment_packet import build_packet
from judge_by_actor import merge_verdicts


def test_each_role_gets_all_its_actual_images_without_other_actor_targets(tmp_path, monkeypatch):
    import assessment_packet
    monkeypatch.setattr(assessment_packet, "ROOT", tmp_path)
    reference = tmp_path / "data/toy/1/reference.json"
    reference.parent.mkdir(parents=True)
    reference.write_text("{}", encoding="utf-8")
    run = tmp_path / "runs/toy"
    run.mkdir(parents=True)
    (run / "run_config.json").write_text(json.dumps({"dataset": "toy", "case": "1"}))
    links = []
    for number, actor, images in [(1, "W1-1", ["E1", "E2"]), (2, "W1-2", ["E3"]),
                                  (3, "W1-1", ["E4", "E2"])]:
        directory = run / "requests" / f"request_{number:04d}"
        directory.mkdir(parents=True)
        (directory / "request.json").write_text(json.dumps({"payload": {"workspace_id": actor}, "image_evidence_ids": images}))
        (directory / "response.json").write_text(json.dumps({"action": "READ_RECORD"}))
        (directory / "transport.json").write_text("{}")
        links.append({"request_id": directory.name})
    (run / "execution_links.json").write_text(json.dumps(links))
    packet, selected = build_packet(run, max_images=None, actor="W1-1")
    assert [step["request"] for step in packet["steps"]] == ["request_0001", "request_0003"]
    assert {eid for eid, _ in selected} == {"E1", "E2", "E4"}
    assert not packet["unattached_image_ids"]
    assert all(step["actor"] == "W1-1" for step in packet["steps"])


def test_role_merging_cannot_omit_or_duplicate_a_target():
    full = {"steps": [{"request": "r1", "actor": "GIA"}, {"request": "r2", "actor": "W1-1"}]}
    first, second = {"steps": full["steps"][:1]}, {"steps": full["steps"][1:]}
    verdict = lambda request: {"roles": [], "target_findings": [{"request_id": request}]}
    merged = merge_verdicts(full, {"GIA": (first, verdict("r1")), "W1-1": (second, verdict("r2"))})
    assert [item["request_id"] for item in merged["target_findings"]] == ["r1", "r2"]
    with pytest.raises(RuntimeError, match="omitted or duplicated"):
        merge_verdicts(full, {"GIA": (first, verdict("r1"))})
    with pytest.raises(RuntimeError):
        merge_verdicts(full, {"GIA": (first, verdict("r1")), "W1-1": (second, verdict("r1"))})
