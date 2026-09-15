"""Attach UTQR-6 anchors and an auditable score form to a blind episode packet."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from pipeline import write_jsonl
from utqr_v6_hybrid import DEFAULT_RUBRIC


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build(packet_path: Path, rubric_path: Path = DEFAULT_RUBRIC) -> list[dict]:
    rubric = _json(rubric_path)
    dimensions = rubric["dimensions"]
    rows = []
    for episode in _jsonl(packet_path):
        blind_id = str(episode.get("blind_id") or "")
        rows.append(
            {
                "blind_id": blind_id,
                "judge_instruction": (
                    "Score only the visible trajectory. Do not infer the dataset, case, hidden answer, "
                    "or benchmark result. A 5 requires the literal score_5 anchor; plausible or merely "
                    "executable work should receive 3. Cite a concrete turn for every dimension."
                ),
                "trajectory": copy.deepcopy(episode),
                "dimensions": copy.deepcopy(dimensions),
                "judgment_form": {
                    "blind_id": blind_id,
                    "judge_id": None,
                    "dimension_scores": {item["id"]: None for item in dimensions},
                    "dimension_evidence": {item["id"]: None for item in dimensions},
                    "hard_fail": None,
                    "hard_fail_reason": None,
                    "keep_for_sft": None,
                    "overall_comment": None,
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind-packet", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build(args.blind_packet, args.rubric)
    write_jsonl(args.output, rows)
    print(json.dumps({"episodes": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
