"""Review GIA with actor-visible state plus already acquired report source images."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "src"))
from assessment_packet import build_packet, read_json
from judge_by_actor import merge_verdicts
from judge_run import assess_packet, finalize_assessment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--source-assessment", required=True)
    parser.add_argument("--assessment", required=True)
    parser.add_argument("--rubric", default="rubric_r6.md")
    args = parser.parse_args()
    run = ROOT / "runs" / args.run
    destination = run / args.assessment
    if destination.exists():
        raise ValueError("Inspect an existing assessment before any retry")
    full_packet, all_images = build_packet(run, max_images=None)
    gia_packet, gia_images = build_packet(run, actor="GIA", max_images=None)
    final_steps = [s for s in gia_packet["steps"] if s["output"].get("decision") == "FINISH"]
    if len(final_steps) != 1:
        raise ValueError("Select a completed investigation with one final GIA output")
    final = final_steps[0]
    cited = set(re.findall(r"\bE\d+\b", final["output"]["answer"]))
    first_seen = {}
    for step in full_packet["steps"]:
        for eid in step["actor_input"]["image_evidence_ids"]:
            first_seen.setdefault(eid, step["request"])
    selected = dict(gia_images)
    for eid, path in all_images:
        if eid in cited:
            if first_seen[eid] > final["request"]:
                raise ValueError("A source image appeared only after the assessed output")
            selected.setdefault(eid, path)
    supplied = set(selected) - {eid for eid, _ in gia_images}
    gia_packet["attached_images_in_order"] = list(selected)
    gia_packet["reference_images_scope"] = {
        "purpose": "Verify source facts behind existing HI reports used by GIA. These additional images were NOT attached to GIA merely because the assessor sees them.",
        "additional_source_image_ids": sorted(supplied, key=lambda x: int(x[1:])),
        "first_actual_actor_attachment": {eid: first_seen[eid] for eid in supplied},
        "actor_inputs_and_outputs": "Unchanged. Use each step's actor_input for visibility and planning judgments.",
        "source_assessment": args.source_assessment,
    }
    print(json.dumps({"run": args.run, "gia_actual_unique_images": len(gia_images),
                      "additional_source_images": len(supplied), "assessor_images": len(selected)}), flush=True)
    assess_packet(gia_packet, list(selected.items()), destination / "GIA", args.rubric)
    parts = {"GIA": (gia_packet, read_json(destination / "GIA/judgment.json"))}
    actors = sorted({step["actor"] for step in full_packet["steps"]} - {"GIA"})
    for actor in actors:
        source = run / args.source_assessment / actor
        parts[actor] = (read_json(source / "packet.json"), read_json(source / "judgment.json"))
    combined = merge_verdicts(full_packet, parts)
    combined["assessment_scope"] = "Existing complete HI reviews reused; GIA independently reassessed with unchanged actual inputs and additional already acquired source images. Additional assessor images are not actor-visible evidence. No independent human certification."
    for name, value in [("packet.json", full_packet), ("judgment.json", combined),
                        ("review_sources.json", {"HI": args.source_assessment, "GIA": "GIA"})]:
        (destination / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "rubric.md").write_text((ROOT / args.rubric).read_text(encoding="utf-8"), encoding="utf-8")
    finalize_assessment(destination)
    subprocess.run([sys.executable, str(ROOT / "export_sft.py"), args.run,
                    "--assessment", args.assessment], check=True)
    (run / "selected_assessment.json").write_text(json.dumps({
        "assessment": args.assessment, "rubric_file": args.rubric,
        "reason": "GIA_review_includes_acquired_source_images_for_report_fact_checking; HI_reviews_reused"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
