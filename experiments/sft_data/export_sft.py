"""Export separately assessed GIA and HI targets with the actual input images."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "src"))
from foren_searcher.local_models import extract_json_object


def eligible(target, link, transport):
    return (target.get("disposition") == "accept_candidate"
            and type(target.get("target_rating")) is int and 3 <= target["target_rating"] <= 4
            and target.get("critical_unknown") is False
            and link.get("status") in {"executed", "report_committed", "gia_decision_accepted"}
            and link.get("raw_equals_accepted") is True
            and transport.get("returncode") == 0 and not transport.get("external_tool_items"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--assessment", default="assessment_r2")
    args = parser.parse_args()
    run = ROOT / "runs" / args.run
    assessment = run / args.assessment
    verdict = extract_json_object((assessment / "judgment.json").read_text(encoding="utf-8"))
    links = {row["request_id"]: row for row in json.loads((run / "execution_links.json").read_text(encoding="utf-8"))}
    packet = json.loads((assessment / "packet.json").read_text(encoding="utf-8"))
    outputs, rejected = {"HI": [], "GIA": []}, []
    for target in verdict["target_findings"]:
        request_id = target["request_id"]
        if request_id not in links:
            incomplete = {item["request_id"]: item for item in packet.get("execution_summary", {}).get("incomplete_requests", [])}
            if request_id not in incomplete:
                raise RuntimeError("Assessment refers to an unknown request")
            rejected.append({"request_id": request_id, "assessment": target,
                             "execution_status": incomplete[request_id]["reason"]})
            continue
        directory = run / "requests" / request_id
        request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
        transport = json.loads((directory / "transport.json").read_text(encoding="utf-8"))
        link = links[request_id]
        if not eligible(target, link, transport):
            rejected.append({"request_id": request_id, "assessment": target, "execution_status": link["status"]})
            continue
        role = "HI" if request["payload"].get("workspace_id") else "GIA"
        images = [str((directory / "actor" / f"image_{i + 1:02d}.png").resolve())
                  for i in range(len(request["image_evidence_ids"]))]
        if any(not Path(path).is_file() for path in images):
            raise FileNotFoundError("A teacher image is missing")
        content = "".join("<image>\n" for _ in images)
        content += json.dumps({"payload": request["payload"], "image_notes": request["image_notes"]}, ensure_ascii=False)
        outputs[role].append({
            "messages": [{"role": "system", "content": request["system_prompt"]},
                         {"role": "user", "content": content},
                         {"role": "assistant", "content": json.dumps(link["raw_output"], ensure_ascii=False)}],
            "images": images,
            "metadata": {"role": role, "dataset": packet["dataset"], "case": packet["case"],
                         "run": args.run, "workspace": request["payload"].get("workspace_id"),
                         "request": request_id, "image_evidence_ids": request["image_evidence_ids"],
                         "assessment": args.assessment, "target_rating": target["target_rating"],
                         "input_source": "actual_runtime_request", "target_source": "actual_cli_output",
                         "review_scope": "automated_candidate_not_independent_human_certification",
                         "loss_scope": "assistant_output_only"}})
    destination = assessment / "sft_candidates"
    destination.mkdir(exist_ok=False)
    for role, rows in outputs.items():
        (destination / (role.lower() + ".jsonl")).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (destination / "excluded_targets.json").write_text(json.dumps(rejected, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"run": args.run, "counts": {role: len(rows) for role, rows in outputs.items()},
               "excluded": len(rejected), "source_cases": 1,
               "note": "Candidate rows share one case; they are not independent case counts. Mask system/user tokens in training."}
    (destination / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
