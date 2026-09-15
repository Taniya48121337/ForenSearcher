"""Order-consistent pairwise verification of unary VidDiff VLM candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiment_viddiff_local_qwen3vl import DEFAULT_MODEL, LocalQwenPredicateScorer
from experiment_viddiff_qwen_unary_ranker import _metrics
from pipeline import ROOT, write_json


DEFAULT_UNARY = (
    ROOT / "outputs/student_external_contract_v1/viddiff_qwen_unary_tip_grasp_v1.json"
)
DEFAULT_ASSETS = (
    ROOT / "outputs/student_external_contract_v1/viddiff_qwen_unary_tip_grasp_assets_v1"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/student_external_contract_v1/viddiff_qwen_tip_grasp_pair_consistency_v1.json"
)


def _mosaic(asset_root: Path, video_path: str) -> Path:
    return asset_root / f"{Path(video_path).stem}_uniform.jpg"


def _mapped_swapped(direction: str) -> str:
    return {"a": "b", "b": "a", "c": "c"}[direction]


def run(
    unary_path: Path,
    asset_root: Path,
    model_path: Path,
) -> dict[str, Any]:
    unary = json.loads(unary_path.read_text(encoding="utf-8"))
    scorer = LocalQwenPredicateScorer(model_path)
    cases = []
    candidates = [row for row in unary["cases"] if row["predicted_direction"] != "c"]
    for index, row in enumerate(candidates, start=1):
        path_a = _mosaic(asset_root, row["videos"][0])
        path_b = _mosaic(asset_root, row["videos"][1])
        forward = scorer.compare_pair(
            mosaic_a=path_a,
            mosaic_b=path_b,
            statement=unary["statement"],
        )
        swapped = scorer.compare_pair(
            mosaic_a=path_b,
            mosaic_b=path_a,
            statement=unary["statement"],
        )
        forward_direction = str(forward["preferred_video"])
        swapped_mapped = _mapped_swapped(str(swapped["preferred_video"]))
        unary_direction = str(row["predicted_direction"])
        consistent = forward_direction == swapped_mapped == unary_direction
        final = unary_direction if consistent else "c"
        cases.append(
            {
                **row,
                "unary_direction": unary_direction,
                "pairwise_forward": forward,
                "pairwise_swapped": swapped,
                "mapped_swapped_direction": swapped_mapped,
                "three_way_consistent": consistent,
                "final_direction": final,
                "direction_correct_after_consistency": final == row["oracle_direction"],
            }
        )
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(candidates)}",
                    "sample_key": row["sample_key"],
                    "unary": unary_direction,
                    "forward": forward_direction,
                    "swapped_mapped": swapped_mapped,
                    "final": final,
                    "oracle": row["oracle_direction"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    verified = {row["sample_key"]: row["final_direction"] for row in cases}
    final_rows = []
    for row in unary["cases"]:
        final_direction = verified.get(row["sample_key"], "c")
        # _metrics consumes scores; encode the already verified class without
        # changing its public interface.
        encoded = {
            **row,
            "score_a": 100.0 if final_direction == "a" else 0.0,
            "score_b": 100.0 if final_direction == "b" else 0.0,
        }
        final_rows.append(encoded)
    return {
        "experiment": "viddiff_qwen_unary_plus_order_consistent_pairwise_v1",
        "statement": unary["statement"],
        "label_boundary": {
            "oracle_used_during_measurement_or_consistency_check": False,
            "oracle_used_after_all_predictions_for_metrics": True,
            "external_service_calls": 0,
        },
        "policy": "emit direction only when unary, forward pairwise, and mapped swapped pairwise agree",
        "summary": _metrics(final_rows, 15.0),
        "target_surgery_123": next(row for row in cases if row["sample_key"] == "surgery_123"),
        "directional_candidates": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unary", type=Path, default=DEFAULT_UNARY)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.unary, args.assets, args.model)
    write_json(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

