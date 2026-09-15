"""Run six additional clean/repair episodes covering new capability families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from transformers import AutoTokenizer

from evaluate_student_context_arm import _load_model
from pipeline import ROOT, decode, write_json
from replay_generated_viddiff_checkpoint import _seed_environment
from run_student_closed_loop_canary import (
    DEFAULT_ADAPTER,
    DEFAULT_MODEL,
    DEFAULT_SFT,
    _run_episode,
    _viddiff_item,
)
from shadow_execute_generated_actions import _jsonl, _payload
from vcsl_batch import VCSLFeatureEnvironment, VCSL_TOOL_REGISTRY
from vcsl_probe import DEFAULT_FEATURE_ROOT
from viddiff_batch import VIDDIFF_TOOL_REGISTRY, VidDiffAdaptiveMeasurementEnvironment
from videosham_batch import (
    EDITED_ROOT,
    MULTIFAMILY_TOOL_REGISTRY,
    SOURCE_ROOT,
    MultiFamilyVideoEnvironment,
)


DEFAULT_OUTPUT = (
    ROOT
    / "outputs/student_external_contract_v1"
    / "student_closed_loop_expanded_six_episode_panel.json"
)

VIDEO_4197_CLEAN = "VideoSham:videosham-4197:LLM-TRJ-4197-CLEAN-001:clean:1"
VIDEO_4197_CLEAN_SECOND = "VideoSham:videosham-4197:LLM-TRJ-4197-CLEAN-001:clean:2"
VIDEO_4197_REPAIR = "VideoSham:videosham-4197:LLM-TRJ-4197-REPAIR-001:repair:1"
VCSL_DP_CLEAN = (
    "VCSL:vcsl-a6264ea57d364eaca85e8eef6bf788d7-"
    "c8c2d0a9f1174aa6a63971328c8c586f:LLM-TRJ-VCSL-"
    "A6264EA57D364EACA85E8EEF6BF788D7-C8C2D0A9F1174AA6A63971328C8C586F-"
    "CLEAN-001:clean:1"
)
VCSL_TN_REPAIR = (
    "VCSL:vcsl-06feccaaee94443a8ddfcd59d7938bad-"
    "808d18d803c649e0badaf17caff18f5c:LLM-TRJ-VCSL-"
    "06FECCAAEE94443A8DDFCD59D7938BAD-808D18D803C649E0BADAF17CAFF18F5C-"
    "REPAIR-001:repair:1"
)
VIDDIFF_155_CLEAN = (
    "VidDiffBench:viddiff-surgery_155:LLM-TRJ-VIDDIFF-SURGERY_155-"
    "CLEAN-001:clean:1"
)
VIDDIFF_70_REPAIR = (
    "VidDiffBench:viddiff-surgery_70:LLM-TRJ-VIDDIFF-SURGERY_70-"
    "REPAIR-001:repair:1"
)


def _public_video_identity(row: dict[str, Any]) -> tuple[str, list[float]]:
    state = _payload(row)["current_harness_state"]
    source_id = None
    scope = None
    for atom in (state.get("evidence_store") or {}).values():
        observation = atom.get("observation") or {}
        source_id = observation.get("source_id", source_id)
        scope = observation.get("comparison_scope_s", scope)
    if source_id is None or scope is None:
        raise ValueError("public VideoSham state lacks source identity or scope")
    return str(source_id), list(scope)


def _video_4197_environment(
    rows: dict[str, dict[str, Any]], *, seed_clean_measurement: bool
) -> MultiFamilyVideoEnvironment:
    source_id, scope = _public_video_identity(rows[VIDEO_4197_CLEAN_SECOND])
    source_path = SOURCE_ROOT / "4197_unedited.mp4"
    suspect_path = EDITED_ROOT / "4197_processed.mp4"
    environment = MultiFamilyVideoEnvironment(
        manifest={
            "case_id": "videosham-4197",
            "source_id": source_id,
            "comparison_scope_s": scope,
        },
        source_frames=decode(source_path),
        suspect_frames=decode(suspect_path),
        source_path=source_path,
        suspect_path=suspect_path,
    )
    if seed_clean_measurement:
        environment.execute("inspect_media", {"media_refs": ["source", "suspect"]})
        selected = environment.eligible_tool_names()
        if len(selected) != 1:
            raise ValueError(f"VideoSham triage does not select one tool: {selected}")
        environment.execute(
            selected[0],
            {
                "source_ref": "source",
                "suspect_ref": "suspect",
                "comparison_scope_s": scope,
            },
        )
    return environment


def _vcsl_environment(
    *, query_id: str, reference_id: str, variant: str, seed_clean_measurement: bool
) -> VCSLFeatureEnvironment:
    environment = VCSLFeatureEnvironment(
        case={
            "pair_id": f"vcsl-{query_id}-{reference_id}",
            "query_id": query_id,
            "reference_id": reference_id,
        },
        feature_root=DEFAULT_FEATURE_ROOT,
        routing_variant=variant,
    )
    if seed_clean_measurement:
        environment.execute(
            "inspect_pair_features",
            {"query_ref": "query", "reference_ref": "reference"},
        )
        name = environment.localizer_spec["tool_name"]
        environment.execute(
            name,
            {
                "query_ref": "query",
                "reference_ref": "reference",
                "method": environment.localizer_spec["method"],
            },
        )
    return environment


def _viddiff_repair_environment(seed_row: dict[str, Any]) -> VidDiffAdaptiveMeasurementEnvironment:
    item = _viddiff_item("surgery_70")
    environment = VidDiffAdaptiveMeasurementEnvironment(item)
    _seed_environment(
        environment,
        _payload(seed_row)["current_harness_state"],
        item,
    )
    return environment


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = {
        str(item["metadata"]["row_id"]): item for item in _jsonl(args.sft)
    }
    q_tn = "06feccaaee94443a8ddfcd59d7938bad"
    r_tn = "808d18d803c649e0badaf17caff18f5c"
    q_dp = "a6264ea57d364eaca85e8eef6bf788d7"
    r_dp = "c8c2d0a9f1174aa6a63971328c8c586f"
    specs: list[tuple[str, str, str, Callable[[], Any], list[dict[str, Any]]]] = [
        (
            "videosham-4197-spatial-clean",
            "clean",
            VIDEO_4197_CLEAN,
            lambda: _video_4197_environment(rows, seed_clean_measurement=False),
            MULTIFAMILY_TOOL_REGISTRY,
        ),
        (
            "videosham-4197-decision-reopen-repair",
            "repair",
            VIDEO_4197_REPAIR,
            lambda: _video_4197_environment(rows, seed_clean_measurement=True),
            MULTIFAMILY_TOOL_REGISTRY,
        ),
        (
            "vcsl-dp-clean",
            "clean",
            VCSL_DP_CLEAN,
            lambda: _vcsl_environment(
                query_id=q_dp,
                reference_id=r_dp,
                variant="dp",
                seed_clean_measurement=False,
            ),
            VCSL_TOOL_REGISTRY,
        ),
        (
            "vcsl-tn-decision-reopen-repair",
            "repair",
            VCSL_TN_REPAIR,
            lambda: _vcsl_environment(
                query_id=q_tn,
                reference_id=r_tn,
                variant="tn_stability",
                seed_clean_measurement=True,
            ),
            VCSL_TOOL_REGISTRY,
        ),
        (
            "viddiff-surgery155-conflict-clean",
            "clean",
            VIDDIFF_155_CLEAN,
            lambda: VidDiffAdaptiveMeasurementEnvironment(_viddiff_item("surgery_155")),
            VIDDIFF_TOOL_REGISTRY,
        ),
        (
            "viddiff-surgery70-conflict-repair",
            "repair",
            VIDDIFF_70_REPAIR,
            lambda: _viddiff_repair_environment(rows[VIDDIFF_70_REPAIR]),
            VIDDIFF_TOOL_REGISTRY,
        ),
    ]
    if args.case_name:
        requested = set(args.case_name)
        available = {spec[0] for spec in specs}
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError(f"unknown expanded-panel case names: {unknown}")
        specs = [spec for spec in specs if spec[0] in requested]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model(args.model, args.adapter)
    episodes = []
    for case_name, mode, row_id, environment_builder, registry in specs:
        episode = _run_episode(
            tokenizer=tokenizer,
            model=model,
            seed_row=rows[row_id],
            environment=environment_builder(),
            tool_registry=registry,
            max_turns=args.max_turns,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
        )
        episode["case_name"] = case_name
        episode["mode"] = mode
        episode["seed_row_id"] = row_id
        episodes.append(episode)
        write_json(args.output.parent / f"student_closed_loop_expanded_{case_name}.json", episode)
    return {
        "audit": "student_closed_loop_expanded_capability_panel",
        "episode_mode": "mixed",
        "teacher_actions_used_during_generation": False,
        "summary": {
            "episodes": len(episodes),
            "terminal_successes": sum(item["terminal_success"] for item in episodes),
            "episodes_with_rejection": sum(item["rejection_count"] > 0 for item in episodes),
            "all_terminal_success": all(item["terminal_success"] for item in episodes),
        },
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=Path, default=DEFAULT_SFT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=6144)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--case-name",
        action="append",
        help="Run only the named case; repeat to select multiple cases.",
    )
    args = parser.parse_args()
    result = run(args)
    write_json(args.output, result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    for episode in result["episodes"]:
        print(json.dumps({
            "case_name": episode["case_name"],
            "terminal_success": episode["terminal_success"],
            "rejection_count": episode["rejection_count"],
            "tool_sequence": episode["tool_sequence"],
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
