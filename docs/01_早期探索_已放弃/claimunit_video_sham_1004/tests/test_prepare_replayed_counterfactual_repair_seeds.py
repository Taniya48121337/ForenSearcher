from __future__ import annotations

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_replayed_counterfactual_repair_seeds import build
from prepare_closed_loop_resume_seed import build_resume_seed


CANDIDATES = (
    ROOT
    / "outputs/selection/viddiff_authorized_20260823_pilot5_frozen"
    / "viddiff_best_of_tree_seed_20260821_n5/candidate_measurements.student.jsonl"
)


def test_replay_builds_full_label_blind_repair_seed() -> None:
    episode = (
        ROOT
        / "outputs/student_external_contract_v1"
        / "viddiff_authorized_20260823_surgery117_clean_full_v1/episode.json"
    )
    seeds, manifest = build([episode], CANDIDATES)

    # Migrating the legacy claim-first episode inserts one explicit claim revision
    # before binding the evidence-first adjudication winner.
    assert manifest["clean_actions_replayed"] == 12
    assert manifest["oracle_or_verifier_used"] is False
    assert len(seeds) == 1
    seed = seeds[0]
    assert seed["metadata"]["mutation_operator"] == "evidence_dropout"
    assert seed["metadata"]["seed_contains_oracle_or_verifier"] is False
    state = json.loads(seed["messages"][1]["content"])["current_harness_state"]
    assert state["evidence_assessments"] == {}
    assert state["evidence_adjudications"] == {}


def test_resume_selects_matching_row_from_batch_and_preserves_repair_mode() -> None:
    failed_episode = (
        ROOT
        / "outputs/student_external_contract_v1"
        / "viddiff_authorized_20260823_strict2_repair_rollout_v1"
        / "cases/viddiffbench/viddiff-surgery_117/repair.json"
    )
    batch_seed = (
        ROOT
        / "outputs/student_external_contract_v1"
        / "viddiff_authorized_20260823_strict2_repair_seeds_v2/seeds.jsonl"
    )
    row = build_resume_seed(failed_episode, 4, batch_seed)

    assert ":repair:" in row["metadata"]["row_id"]
    assert row["metadata"]["source_kind"] == "repair"
    assert row["metadata"]["trajectory_mode"] == "repair"
