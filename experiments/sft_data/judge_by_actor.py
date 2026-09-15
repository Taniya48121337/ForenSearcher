"""Review each actual actor with all its attached images, then combine target judgments."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "src"))
from foren_searcher.local_models import extract_json_object
from assessment_packet import build_packet
from judge_run import finalize_assessment, validate_targets


def merge_verdicts(packet, parts):
    combined = {"assessment_scope": "Separate independent CLI sessions by actor; all images actually attached to that actor are supplied. No independent human certification.",
                "roles": [], "target_findings": [], "construction_revisions": [], "rubric_revisions": []}
    for actor, (part_packet, verdict) in parts.items():
        validate_targets(part_packet, verdict)
        if any(step["actor"] != actor for step in part_packet["steps"]):
            raise ValueError("A role packet contains another actor's target")
        for key in ("roles", "target_findings"):
            combined[key].extend(verdict[key])
        for key in ("construction_revisions", "rubric_revisions"):
            combined[key].extend(f"{actor}: {item}" for item in verdict.get(key, []))
    combined["target_findings"].sort(key=lambda item: item["request_id"])
    validate_targets(packet, combined)
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--assessment", default="assessment_r2_by_actor")
    parser.add_argument("--rubric", default="rubric_r2.md")
    parser.add_argument("--workers", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    run = ROOT / "runs" / args.run
    destination = run / args.assessment
    destination.mkdir(parents=True, exist_ok=True)
    packet, _ = build_packet(run, max_images=None)
    actors = sorted({step["actor"] for step in packet["steps"]})
    def review_actor(actor):
        if not re.fullmatch(r"GIA|W[0-9]+-[0-9]+", actor):
            raise ValueError(f"Unexpected actor id: {actor}")
        folder = destination / actor
        if not (folder / "scores.json").exists():
            if folder.exists():
                raise RuntimeError(f"Unfinished existing assessment; inspect before resuming: {folder}")
            print(json.dumps({"run": args.run, "actor": actor, "status": "assessment_started"}), flush=True)
            with (destination / (actor + ".log")).open("w", encoding="utf-8") as log:
                subprocess.run([sys.executable, str(ROOT / "judge_run.py"), args.run, "--assessment",
                                args.assessment + "/" + actor, "--actor", actor, "--max-images", "0", "--rubric", args.rubric],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        part_packet = json.loads((folder / "packet.json").read_text(encoding="utf-8"))
        verdict = extract_json_object((folder / "judgment.json").read_text(encoding="utf-8"))
        print(json.dumps({"run": args.run, "actor": actor, "status": "assessment_finished"}), flush=True)
        return actor, (part_packet, verdict)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        parts = dict(executor.map(review_actor, actors))
    combined = merge_verdicts(packet, parts)
    (destination / "packet.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "judgment.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "rubric.md").write_text((ROOT / args.rubric).read_text(encoding="utf-8"), encoding="utf-8")
    finalize_assessment(destination)
    if not (destination / "sft_candidates/summary.json").exists():
        subprocess.run([sys.executable, str(ROOT / "export_sft.py"), args.run, "--assessment", args.assessment], check=True)
    (run / "selected_assessment.json").write_text(json.dumps({"assessment": args.assessment,
        "reason": "role-specific_review_with_all_actual_actor_images",
        "rubric_file": args.rubric}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run": args.run, "actors": len(actors), "assessment": args.assessment, "status": "exported"}), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
