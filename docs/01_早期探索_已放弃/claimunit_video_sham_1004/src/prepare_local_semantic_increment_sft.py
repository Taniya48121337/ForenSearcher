from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json, write_jsonl
from run_student_closed_loop_canary import (
    POST_ADJUDICATION_SYSTEM,
    PROGRESSIVE_MEASUREMENT_SYSTEM,
    _preexecution_reasoning_problems,
)


ARM = "decision_sufficient_state_v6_executable_affordances"
SOURCE_CURRICULUM = (
    ROOT
    / "outputs/student_external_contract_v1/progressive_curriculum_v12_handles"
    / ARM
    / "train/sft_all.jsonl"
)
OUTPUT = ROOT / "outputs/student_external_contract_v1/local_semantic_increment_v2"
ROUTE_ROW_IDS = {
    "VidDiffBench:viddiff-surgery_108:student-closed-loop:repair:3",
    "VidDiffBench:viddiff-surgery_108:student-closed-loop:repair:4",
    "VidDiffBench:viddiff-surgery_117:student-closed-loop:repair:4",
}
CLEAN_ROUTE_REASONING = {
    "VidDiffBench:viddiff-surgery_108:student-closed-loop:repair:3": (
        "The latest active atom `EA-VIDDIFF-COMPARE_GLOBAL_MOTION_SYMMETRIC-002` "
        "reports `preferred_video=c`, `confidence=0.66`, and `order_consistent=false`; "
        "the other active measurement atoms still declare support for the comparison "
        "fields but disagree on direction. The unresolved observation gap is whether "
        "predicate-relevant local behavior can provide an order-stable view, not missing "
        "claim support. `compare_action_refined_symmetric` is the best next route because "
        "it prospectively checks that local behavior in both A/B orders without reusing "
        "the dropped tombstone."
    ),
    "VidDiffBench:viddiff-surgery_108:student-closed-loop:repair:4": (
        "The latest active atom `EA-VIDDIFF-COMPARE_ACTION_REFINED_SYMMETRIC-002` "
        "reports `preferred_video=a` and `order_consistent=true`, while the active "
        "measurement routes retain field support but still disagree on direction. The "
        "remaining observation gap is independent coarse chronological corroboration, "
        "not an open-field or binding gap. `compare_uniform_timeline` is the required "
        "next route because it inspects uniformly sampled whole-timeline visual coverage "
        "while leaving global motion and A/B order consistency unobserved."
    ),
    "VidDiffBench:viddiff-surgery_117:student-closed-loop:repair:4": (
        "The latest active atom `EA-VIDDIFF-COMPARE_ACTION_REFINED_SYMMETRIC-002` "
        "reports `preferred_video=c`, `confidence=0.49`, and `order_consistent=false`, "
        "whereas the active uniform-timeline view supports A; all active atoms keep their "
        "declared supports_fields despite that disagreement. The unresolved observation "
        "gap is a role-swapped global-motion check, not missing field support. The visible "
        "repair_restoration_candidates therefore makes `compare_global_motion_symmetric` "
        "the required next measurement to test whether a direction survives both A/B "
        "orders without forcing a winner."
    ),
}
POST_SOURCES = (
    (
        "outputs/student_external_contract_v1/viddiff_counterfactual_repair_pair16_84_v1/"
        "cases/viddiffbench/viddiff-surgery_84/repair.json",
        4,
    ),
    (
        "outputs/student_external_contract_v1/viddiff_counterfactual_repair_pilot108_v3/"
        "cases/viddiffbench/viddiff-surgery_108/repair.json",
        6,
    ),
    (
        "outputs/student_external_contract_v1/viddiff_counterfactual_repair_pilot117_v2/"
        "cases/viddiffbench/viddiff-surgery_117/repair.json",
        6,
    ),
)
PROTECTED_CASES = {"viddiff-surgery_16", "viddiff-surgery_26"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _assistant_action(row: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    assistant = next(
        item for item in reversed(row.get("messages") or []) if item.get("role") == "assistant"
    )
    call = (assistant.get("tool_calls") or [])[0]
    function = call.get("function") or {}
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return str(function.get("name") or ""), arguments, str(assistant.get("content") or "")


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    user = next(
        item for item in reversed(row.get("messages") or []) if item.get("role") == "user"
    )
    return json.loads(str(user.get("content") or "{}"))


def _set_system(row: dict[str, Any], content: str) -> None:
    system = next(item for item in row.get("messages") or [] if item.get("role") == "system")
    system["content"] = content


def compile_increment(output: Path = OUTPUT) -> dict[str, Any]:
    source_rows = _jsonl(SOURCE_CURRICULUM)
    route_rows = [
        copy.deepcopy(row)
        for row in source_rows
        if str((row.get("metadata") or {}).get("row_id") or "") in ROUTE_ROW_IDS
    ]
    if len(route_rows) != len(ROUTE_ROW_IDS):
        raise ValueError("missing route contrast rows")
    for row in route_rows:
        _set_system(row, PROGRESSIVE_MEASUREMENT_SYSTEM)
        metadata = row.setdefault("metadata", {})
        row_id = str(metadata.get("row_id") or "")
        assistant = next(
            item for item in reversed(row.get("messages") or [])
            if item.get("role") == "assistant"
        )
        assistant["content"] = CLEAN_ROUTE_REASONING[row_id]
        metadata["local_increment_role"] = (
            "dropout_restoration_support_distinction"
            if row_id.endswith("surgery_117:student-closed-loop:repair:4")
            else "measurement_route_observation_gap"
        )

    post_rows: list[dict[str, Any]] = []
    for source, turn_index in POST_SOURCES:
        episode = json.loads((ROOT / source).read_text(encoding="utf-8"))
        turn = next(
            item for item in (episode.get("episode") or {}).get("turns") or []
            if int(item.get("turn", 0)) == turn_index
        )
        if turn.get("status") != "accepted" or turn.get("generated_tool") != "set_field_decision":
            raise ValueError(f"invalid post-adjudication source: {source}:{turn_index}")
        row = copy.deepcopy(turn.get("sft_datum") or {})
        _set_system(row, POST_ADJUDICATION_SYSTEM)
        case_id = str(episode.get("case_id") or "")
        metadata = row.setdefault("metadata", {})
        metadata.update(
            {
                "row_id": f"VidDiffBench:{case_id}:local-semantic-increment:repair:{turn_index}",
                "case_id": case_id,
                "local_increment_role": "dropout_post_bind_commit",
                "teacher_action_used": False,
                "oracle_or_verifier_used": False,
            }
        )
        post_rows.append(row)

    rows = [*route_rows, *post_rows]
    audits = []
    for row in rows:
        serialized = json.dumps(row, ensure_ascii=False)
        if any(case in serialized for case in PROTECTED_CASES):
            raise ValueError("protected case leaked into local increment")
        name, arguments, reasoning = _assistant_action(row)
        payload = _payload(row)
        state = payload.get("current_harness_state") or {}
        problems = _preexecution_reasoning_problems(
            reasoning,
            name,
            arguments,
            meta_visible=False,
            visible_state=state,
        )
        audits.append(
            {
                "row_id": (row.get("metadata") or {}).get("row_id"),
                "target_tool": name,
                "reasoning_contract_problems": problems,
            }
        )
    if any(item["reasoning_contract_problems"] for item in audits):
        raise ValueError(f"local increment failed current rubric: {audits}")

    train_dir = output / ARM / "train"
    write_jsonl(train_dir / "sft_all.jsonl", rows)
    report = {
        "artifact": output.name,
        "rows": len(rows),
        "measurement_route_rows": len(route_rows),
        "curated_support_binding_distinction_rows": len(CLEAN_ROUTE_REASONING),
        "dropout_post_bind_commit_rows": len(post_rows),
        "protected_cases_excluded": sorted(PROTECTED_CASES),
        "teacher_actions_used": False,
        "oracle_or_verifier_used": False,
        "external_transfer_performed": False,
        "all_reasoning_contracts_clean": True,
        "audits": audits,
    }
    write_json(output / "compile_report.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(compile_increment(), ensure_ascii=False, indent=2))
