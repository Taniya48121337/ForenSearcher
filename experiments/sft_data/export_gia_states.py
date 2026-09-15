"""Export reviewed offline GIA decisions, preserving their non-executed origin."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "src"))
from foren_searcher.local_models import extract_json_object


def main(name, assessment_name="assessment_r2"):
    source = ROOT / "gia_decisions" / name
    assessment = source / assessment_name
    verdict = extract_json_object((assessment / "judgment.json").read_text(encoding="utf-8"))
    rows, excluded = [], []
    for target in verdict["states"]:
        state = source / target["id"]
        if not (target["disposition"] == "accept_candidate" and type(target["target_rating"]) is int
                and 3 <= target["target_rating"] <= 4 and target["critical_unknown"] is False):
            excluded.append(target)
            continue
        directories = sorted((state / "requests").glob("request_*"))
        if len(directories) != 1:
            raise RuntimeError("A multi-request state requires per-request assessment")
        directory = directories[0]
        request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
        transport = json.loads((directory / "transport.json").read_text(encoding="utf-8"))
        output = extract_json_object((directory / "response.json").read_text(encoding="utf-8"))
        accepted = json.loads((state / "decision.json").read_text(encoding="utf-8"))
        if transport["returncode"] != 0 or transport.get("external_tool_items") or output != accepted:
            raise RuntimeError("Offline target is not the original clean actor decision")
        images = [str((directory / "actor" / f"image_{i+1:02d}.png").resolve())
                  for i in range(len(request["image_evidence_ids"]))]
        if any(not Path(image).is_file() for image in images):
            raise FileNotFoundError("Missing actor image")
        reference = json.loads((state / "reference.json").read_text(encoding="utf-8"))
        rows.append({"messages": [
            {"role": "system", "content": request["system_prompt"]},
            {"role": "user", "content": "<image>\n" * len(images) + json.dumps({
                "payload": request["payload"], "image_notes": request["image_notes"]}, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(output, ensure_ascii=False)}], "images": images,
            "metadata": {**reference["provenance"], "role": "GIA", "state": target["id"],
                         "construction": name, "target_rating": target["target_rating"],
                         "loss_scope": "assistant_output_only", "assessment": assessment_name,
                         "review_scope": "automated_candidate_not_independent_human_certification"}})
    destination = assessment / "sft_candidates"
    destination.mkdir(exist_ok=False)
    (destination / "gia.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (destination / "excluded_states.json").write_text(json.dumps(excluded, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"GIA_offline_candidates": len(rows), "excluded_states": len(excluded),
               "source_cases": len({(row["metadata"]["dataset"], row["metadata"]["case"]) for row in rows}),
               "executed_tool_trajectories": 0}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (source / "selected_assessment.json").write_text(json.dumps({"assessment": assessment_name}, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--assessment", default="assessment_r2")
    args = parser.parse_args()
    main(args.name, args.assessment)
