"""Re-audit saved closed-loop turns with the current reasoning contract."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pipeline import write_json
from run_student_closed_loop_canary import _preexecution_reasoning_problems


PRIVACY_PATTERNS = {
    "local_windows_path": re.compile(r"[A-Za-z]:\\"),
    "local_posix_path": re.compile(r"(?:^|\s)/(?:home|Users|mnt|tmp)/"),
    "oracle_field": re.compile(r"\boracle(?:_direction|_label|_value)?\b", re.I),
    "verifier_field": re.compile(r"\bverifier(?:_score|_result)?\b", re.I),
    "dataset_gt_field": re.compile(r"\bdifferences_gt\b", re.I),
}


def _visible_state(turn: dict[str, Any]) -> dict[str, Any]:
    datum = turn.get("sft_datum") or {}
    messages = datum.get("messages") or []
    user = next((item for item in messages if item.get("role") == "user"), {})
    try:
        payload = json.loads(str(user.get("content") or "{}"))
    except json.JSONDecodeError:
        return {}
    return payload.get("current_harness_state") or {}


def audit_episode(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    episode = artifact.get("episode") or artifact
    turn_reports: list[dict[str, Any]] = []
    for turn in episode.get("turns") or []:
        reasoning = str(turn.get("reasoning") or "")
        state = _visible_state(turn)
        meta_visible = any(
            (atom.get("observation") or {}).get("measurement_method")
            == "compare_symmetry_residual_abstention"
            for atom in (state.get("evidence_store") or {}).values()
        )
        problems = _preexecution_reasoning_problems(
            reasoning,
            turn.get("generated_tool"),
            turn.get("generated_arguments") or {},
            meta_visible=meta_visible,
            visible_state=state,
        )
        privacy_problems = [
            name for name, pattern in PRIVACY_PATTERNS.items() if pattern.search(reasoning)
        ]
        turn_reports.append(
            {
                "turn": turn.get("turn"),
                "tool": turn.get("generated_tool"),
                "status": turn.get("status"),
                "reasoning_problems": problems,
                "privacy_problems": privacy_problems,
                "passed": not problems
                and not privacy_problems
                and turn.get("status") in {"accepted", "executed"},
            }
        )
    source_quality = artifact.get("quality") or {}
    return {
        "episode_path": str(path),
        "dataset": artifact.get("dataset") or episode.get("dataset"),
        "case_id": artifact.get("case_id") or episode.get("case_id"),
        "mode": artifact.get("mode") or episode.get("mode"),
        "turns": len(turn_reports),
        "terminal_success": bool(episode.get("terminal_success")),
        "source_score": source_quality.get("score"),
        "source_accepted": source_quality.get("accepted_for_high_quality_sft"),
        "semantic_passed": all(item["passed"] for item in turn_reports),
        "turn_reports": turn_reports,
    }


def run(paths: list[Path]) -> dict[str, Any]:
    episodes = [audit_episode(path) for path in paths]
    return {
        "audit": "saved_episode_current_semantic_contract_v1",
        "episodes": len(episodes),
        "turns": sum(item["turns"] for item in episodes),
        "all_passed": all(
            item["terminal_success"]
            and item["source_score"] == 100
            and item["source_accepted"] is True
            and item["semantic_passed"]
            for item in episodes
        ),
        "episode_reports": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, action="append", default=[])
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional panel manifest whose items contain episode_path.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = list(args.episode)
    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        paths.extend(Path(str(item["episode_path"])) for item in manifest.get("items") or [])
    if not paths:
        parser.error("provide at least one --episode or --manifest")
    report = run(paths)
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
