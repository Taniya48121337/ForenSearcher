"""Export student-visible causal-recovery transcripts for independent blind review."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json


DEFAULT_INPUT = (
    ROOT
    / "outputs/student_external_contract_v1/recovery_batch_candidate_v6"
    / "recovery_episodes.jsonl"
)
DEFAULT_RUBRIC = ROOT / "rubric/unified_trajectory_quality_rubric_v9.json"
DEFAULT_OUTPUT = (
    ROOT / "outputs/student_external_contract_v1/recovery_blind_review_packet_v2"
)

WINDOWS_PATH = re.compile(r"(?i)\b[A-Z]:\\[^\s\"']+")
VIDDIFF_URI = re.compile(r"viddiff://[^\s\"']+", re.I)
VIDEOSHAM_CASE = re.compile(r"videosham-\d+", re.I)
VIDDIFF_CASE = re.compile(r"viddiff-surgery_\d+", re.I)
VCSL_CASE = re.compile(r"vcsl-[0-9a-f]{32}-[0-9a-f]{32}", re.I)
PAIR_GUID = re.compile(r"[0-9a-f]{32}-[0-9a-f]{32}", re.I)
TASK_ID = re.compile(r"TASK-[A-Z0-9_-]+", re.I)
EVIDENCE_ID = re.compile(r"\bEA-[A-Z0-9_-]+\b", re.I)
CLAIM_ID = re.compile(r"\b(?:CLAIM|CANDIDATE)-[A-Z0-9_-]+\b", re.I)

PRIVATE_KEYS = {
    "dataset",
    "case_id",
    "trajectory_id",
    "contract_id",
    "intervention",
    "intervention_id",
    "operator",
    "oracle",
    "verifier_result",
    "metrics",
}
LOGICAL_ID_KEYS = {
    "sample_key": "<SAMPLE_ID>",
    "source_id": "<SOURCE_ID>",
    "query_id": "<QUERY_ID>",
    "reference_id": "<REFERENCE_ID>",
}
ASSET_KEYS = {
    "source": "<MEDIA_A>",
    "suspect": "<MEDIA_B>",
    "video_a": "<MEDIA_A>",
    "video_b": "<MEDIA_B>",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class Redactor:
    def __init__(self) -> None:
        self.evidence: dict[str, str] = {}
        self.claims: dict[str, str] = {}

    @staticmethod
    def _mapped(token: str, mapping: dict[str, str], prefix: str) -> str:
        key = token.upper()
        if key not in mapping:
            mapping[key] = f"<{prefix}_{len(mapping) + 1:02d}>"
        return mapping[key]

    def text(self, value: str) -> str:
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                pass
            else:
                return json.dumps(
                    self.value(parsed), ensure_ascii=False, separators=(",", ":")
                )
        value = WINDOWS_PATH.sub("<LOCAL_ASSET>", value)
        value = VIDDIFF_URI.sub("<MEDIA_ASSET>", value)
        value = VCSL_CASE.sub("<CASE_ID>", value)
        value = PAIR_GUID.sub("<CASE_ID>", value)
        value = VIDEOSHAM_CASE.sub("<CASE_ID>", value)
        value = VIDDIFF_CASE.sub("<CASE_ID>", value)
        value = TASK_ID.sub("<TASK_ID>", value)
        value = EVIDENCE_ID.sub(
            lambda match: self._mapped(match.group(0), self.evidence, "EVIDENCE"), value
        )
        value = CLAIM_ID.sub(
            lambda match: self._mapped(match.group(0), self.claims, "CLAIM"), value
        )
        for name in ("VideoSham", "VCSL", "VidDiffBench"):
            value = re.sub(re.escape(name), "<DATASET>", value, flags=re.I)
        return value

    def value(self, value: Any, key: str | None = None) -> Any:
        if key in PRIVATE_KEYS:
            return "<REDACTED>"
        if key in LOGICAL_ID_KEYS:
            return LOGICAL_ID_KEYS[key]
        if key in ASSET_KEYS:
            return ASSET_KEYS[key]
        if isinstance(value, dict):
            return {
                self.text(name): self.value(item, name) for name, item in value.items()
            }
        if isinstance(value, list):
            return [self.value(item, key) for item in value]
        if isinstance(value, str):
            if key and (key == "evidence_id" or key.endswith("evidence_ids")):
                return self._mapped(value, self.evidence, "EVIDENCE")
            if key and key in {"claim_id", "candidate_claim_id"}:
                return self._mapped(value, self.claims, "CLAIM")
            return self.text(value)
        return value


def sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    redactor = Redactor()
    return redactor.value(copy.deepcopy(messages))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_sheet(path: Path, review_ids: list[str], dimensions: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["review_id"] + dimensions + [
        "visible_hard_failure",
        "accept_for_sft",
        "strongest_turn",
        "weakest_turn",
        "required_revision",
        "reviewer_notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for review_id in review_ids:
            writer.writerow({"review_id": review_id})


def export_packets(
    input_path: Path,
    rubric_path: Path,
    output_dir: Path,
    seed: int = 20260824,
    artifact_id: str = "blind_recovery_review_packet_v2",
) -> dict[str, Any]:
    episodes = _read_jsonl(input_path)
    rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
    judge = rubric["blind_judge_contract"]
    dimensions = list(judge["dimensions"])

    packets: list[dict[str, Any]] = []
    private_key: list[dict[str, Any]] = []
    for index, episode in enumerate(episodes, start=1):
        review_id = f"review-{index:03d}"
        packets.append(
            {
                "review_id": review_id,
                "instructions": (
                    "Judge only the visible transcript. Do not infer an oracle answer or "
                    "reward a trajectory merely because it reaches stop."
                ),
                "messages": sanitize_messages(episode["messages"]),
            }
        )
        private_key.append(
            {
                "review_id": review_id,
                "episode_id": episode.get("episode_id"),
                "dataset": episode.get("dataset"),
                "case_id": episode.get("case_id"),
            }
        )

    public_dir = output_dir / "public_share_only"
    private_dir = output_dir / "private_do_not_share_with_judges"
    _write_jsonl(public_dir / "review_packets.jsonl", packets)
    write_json(private_dir / "review_key.json", {"items": private_key})

    order_a = [item["review_id"] for item in packets]
    order_b = order_a.copy()
    random.Random(seed).shuffle(order_a)
    random.Random(seed + 1).shuffle(order_b)
    _write_sheet(public_dir / "reviewer_A.csv", order_a, dimensions)
    _write_sheet(public_dir / "reviewer_B.csv", order_b, dimensions)

    adjudication_fields = [
        "review_id",
        "dimension",
        "reviewer_A_score",
        "reviewer_B_score",
        "adjudicated_score",
        "adjudication_reason",
    ]
    with (private_dir / "adjudication.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        csv.DictWriter(handle, fieldnames=adjudication_fields).writeheader()

    guide = f"""# Causal-recovery trajectory blind review

Review the transcript as if it were a candidate SFT demonstration. Score each
dimension from 1 (unusable) to 5 (excellent). Do not search for the case, infer
the hidden dataset, or consult another judge.

## Dimensions

{chr(10).join(f'- `{name}`' for name in dimensions)}

For every row, also record whether a visible hard failure exists, whether you
would accept the episode for SFT, the strongest and weakest turn, and any
required revision. A visible hard failure includes an ungrounded tool choice,
fabricated evidence, stale-evidence reuse, off-target field mutation, premature
stop, or reasoning that does not explain the next state consequence.

## Independence and adjudication

Reviewer A and Reviewer B must work independently and cannot see the mapping,
private intervention, verifier outcome, metrics, or each other's scores. A
dimension range above one point requires later adjudication. Passing the human
review still does not establish sealed-holdout generalization or student gain.
"""
    (public_dir / "REVIEW_GUIDE.md").write_text(guide, encoding="utf-8")

    manifest = {
        "artifact": artifact_id,
        "episodes": len(packets),
        "minimum_independent_judges": judge["minimum_independent_judges"],
        "dimensions": dimensions,
        "score_range": judge["score_range"],
        "minimum_dimension_mean": judge["minimum_dimension_mean"],
        "maximum_unadjudicated_range": judge["maximum_unadjudicated_range"],
        "public_packet_contains_dataset_name": False,
        "public_packet_contains_case_id": False,
        "public_packet_contains_private_intervention": False,
        "public_packet_contains_verifier_or_metrics": False,
        "human_review_completed": False,
        "share_boundary": "Share only public_share_only/ with judges.",
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--artifact-id", default="blind_recovery_review_packet_v2"
    )
    args = parser.parse_args()
    report = export_packets(
        args.input, args.rubric, args.output_dir, args.seed, args.artifact_id
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
