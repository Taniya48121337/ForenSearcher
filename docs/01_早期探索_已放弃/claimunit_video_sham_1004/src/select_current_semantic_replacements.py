"""Find already executed alternatives for episodes that fail the current contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_saved_episode_semantics import audit_episode
from pipeline import write_json
from utqr_v6_hybrid import DEFAULT_RUBRIC


DATASET_DIR = {"VideoSham": "videosham", "VCSL": "vcsl", "VidDiffBench": "viddiffbench"}


def _artifact_hits(path: Path, patterns: list[str]) -> list[str]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    episode = artifact.get("episode") or artifact
    text = "\n".join(
        str(turn.get("reasoning") or "").lower() for turn in episode.get("turns") or []
    )
    return [pattern for pattern in patterns if pattern.lower() in text]


def select(failed_audit_path: Path, search_root: Path) -> dict[str, Any]:
    failed_audit = json.loads(failed_audit_path.read_text(encoding="utf-8"))
    patterns = json.loads(DEFAULT_RUBRIC.read_text(encoding="utf-8")).get(
        "known_process_artifacts", []
    )
    failed = [
        item for item in failed_audit.get("episode_reports") or []
        if item.get("semantic_passed") is not True
    ]
    rows = []
    for item in failed:
        dataset = str(item["dataset"])
        case_id = str(item["case_id"])
        mode = str(item["mode"])
        pattern = f"**/cases/{DATASET_DIR[dataset]}/{case_id}/{mode}.json"
        candidates = []
        for path in search_root.glob(pattern):
            report = audit_episode(path)
            hits = _artifact_hits(path, patterns)
            if (
                report.get("semantic_passed") is True
                and report.get("terminal_success") is True
                and report.get("source_score") == 100
                and report.get("source_accepted") is True
                and not hits
            ):
                artifact = json.loads(path.read_text(encoding="utf-8"))
                episode = artifact.get("episode") or artifact
                candidates.append(
                    {
                        "episode_path": str(path),
                        "turns": report["turns"],
                        "rejections": int(episode.get("rejection_count") or 0),
                        "modified_time": path.stat().st_mtime,
                    }
                )
        candidates.sort(
            key=lambda row: (row["rejections"], row["turns"], -row["modified_time"])
        )
        rows.append(
            {
                "dataset": dataset,
                "case_id": case_id,
                "mode": mode,
                "alternatives": len(candidates),
                "selected": candidates[0] if candidates else None,
            }
        )
    return {
        "artifact": "current_semantic_replacement_selection_v1",
        "failed_episodes": len(rows),
        "episodes_with_existing_clean_replacement": sum(row["selected"] is not None for row in rows),
        "episodes_requiring_new_rollout": sum(row["selected"] is None for row in rows),
        "items": rows,
    }


def apply_selection(manifest: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    selected = {
        (str(item["dataset"]), str(item["case_id"]), str(item["mode"])): item["selected"]
        for item in selection.get("items") or []
        if item.get("selected") is not None
    }
    items = []
    replacements_applied = 0
    for item in manifest.get("items") or []:
        identity = (str(item["dataset"]), str(item["case_id"]), str(item["mode"]))
        replacement = selected.get(identity)
        updated = dict(item)
        if replacement is not None:
            updated["replaced_episode_path"] = updated["episode_path"]
            updated["episode_path"] = replacement["episode_path"]
            replacements_applied += 1
        items.append(updated)
    return {
        "artifact": "current_semantic_panel_manifest_v2_existing_replacements",
        "episodes": len(items),
        "replacements_applied": replacements_applied,
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed-audit", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--updated-manifest", type=Path)
    args = parser.parse_args()
    report = select(args.failed_audit, args.search_root)
    write_json(args.output, report)
    if args.base_manifest is not None and args.updated_manifest is not None:
        manifest = json.loads(args.base_manifest.read_text(encoding="utf-8"))
        write_json(args.updated_manifest, apply_selection(manifest, report))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
