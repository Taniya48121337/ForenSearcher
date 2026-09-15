"""Regenerate one HI terminal report from its recorded input; do not reuse its old suffix."""
from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "src"))
from cli_generator import CliGenerator
from assessment_packet import build_packet
from judge_run import assess_packet, finalize_assessment
from foren_searcher.local_models import _validate_hi_output, extract_json_object


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class RecordedInputGenerator(CliGenerator):
    def __init__(self, source, output, instruction_file=None):
        self.source = source
        self.recorded_request = read(source / "request.json")
        instruction_path = Path(instruction_file) if instruction_file else ROOT / "actor_report_r4.txt"
        super().__init__(store=None, output_dir=output,
                         instruction=instruction_path.read_text(encoding="utf-8"))
        self.instruction += "\n当前样本只要求基于已有状态生成一份局部报告。输出报告，不提出新的工具动作；不要为满足报告形式而补写未取得的证据。"

    def _image_message(self, view):
        content = [{"type": "text", "text": note} for note in self.recorded_request["image_notes"]]
        for i, _ in enumerate(self.recorded_request["image_evidence_ids"]):
            data = (self.source / "actor" / f"image_{i+1:02d}.png").read_bytes()
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode("ascii")}})
        return {"role": "user", "content": content}, list(self.recorded_request["image_evidence_ids"])


async def main(args):
    source_run = ROOT / "runs" / args.run
    source = source_run / "requests" / args.request
    request = read(source / "request.json")
    original_output = extract_json_object((source / "response.json").read_text(encoding="utf-8"))
    if "findings" not in original_output or not request["payload"].get("workspace_id"):
        raise ValueError("This operation is restricted to an existing HI terminal report")
    destination = ROOT / "hi_terminal_repairs" / args.name
    if args.finalize_existing:
        export_candidate(destination, args.assessment)
        return
    if args.reassess_existing:
        output = read(destination / "decision.json")
        provenance = read(destination / "provenance.json")
        packet, images = report_packet(source_run, args.request, destination, output, provenance)
        assess_packet(packet, images, destination / args.assessment, rubric_file=args.rubric)
        export_candidate(destination, args.assessment)
        return
    destination.mkdir(parents=True, exist_ok=False)
    generator = RecordedInputGenerator(source, destination / "requests", args.instruction_file)
    payload = copy.deepcopy(request["payload"])
    output = await generator.generate_json(system_prompt=request["system_prompt"], payload=payload)
    if "findings" not in output or "action" in output:
        raise ValueError("A terminal repair cannot execute a new tool action")
    _validate_hi_output(output, workspace_id=payload["workspace_id"], model_view=payload["model_view"],
                        actions=tuple(payload["available_actions"]))
    (destination / "decision.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    config = read(source_run / "run_config.json")
    provenance = {"dataset": config["dataset"], "case": str(config["case"]), "source_run": args.run,
                  "source_request": args.request, "usage_split": config.get("usage", "development"),
                  "construction": args.name, "request": "request_0001", "role": "HI",
                  "input_source": "recorded_terminal_state_with_generic_report_instruction",
                  "target_source": "actual_cli_output", "is_executed_tool_trajectory": False,
                  "new_tool_actions": 0, "reused_downstream_decisions": 0,
                  "local_output_validation": "passed"}
    (destination / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    packet, images = report_packet(source_run, args.request, destination, output, provenance)
    assess_packet(packet, images, destination / args.assessment, rubric_file=args.rubric)
    export_candidate(destination, args.assessment)


def report_packet(source_run, source_request, destination, output, provenance):
    request = read(source_run / "requests" / source_request / "request.json")
    actor = request["payload"]["workspace_id"]
    packet, images = build_packet(source_run, max_images=None, actor=actor)
    prefix = [step for step in packet["steps"] if step["request"] < source_request]
    allowed_images = set(request["image_evidence_ids"])
    for step in prefix:
        allowed_images.update(step["actor_input"]["image_evidence_ids"])
    images = [(eid, path) for eid, path in images if eid in allowed_images]
    new_request_dir = destination / "requests/request_0001"
    new_request = read(new_request_dir / "request.json")
    transport = read(new_request_dir / "transport.json")
    packet.update(steps=[{"request": "terminal_request_0001", "actor": actor,
                          "actor_input": new_request, "output": output, "transport": transport,
                          "execution_link": {"status": "offline_terminal_replay", "new_tool_actions": 0,
                                             "local_output_validation": "passed"}}],
                  recorded_prefix=prefix, evaluation_mode="terminal_replay", provenance=provenance,
                  attached_images_in_order=[eid for eid, _ in images], unattached_image_ids=[])
    for key in ("result", "workspace_outcomes", "all_workspaces_reported", "runtime_returned_result", "execution_summary"):
        packet.pop(key, None)
    return packet, images


def export_candidate(destination, assessment_name="assessment_r2"):
    assessment = destination / assessment_name
    finalize_assessment(assessment)
    packet = read(assessment / "packet.json")
    provenance = read(destination / "provenance.json")
    output = read(destination / "decision.json")
    new_request_dir = destination / "requests/request_0001"
    new_request = read(new_request_dir / "request.json")
    transport = read(new_request_dir / "transport.json")
    raw = extract_json_object((new_request_dir / "response.json").read_text(encoding="utf-8"))
    if transport["returncode"] != 0 or transport.get("external_tool_items") or raw != output:
        raise ValueError("The exported report is not the clean original teacher output")
    verdict = extract_json_object((assessment / "judgment.json").read_text(encoding="utf-8"))
    target_id = packet["steps"][0]["request"]
    target = next(item for item in verdict["target_findings"] if item["request_id"] == target_id)
    accepted = (target["disposition"] == "accept_candidate" and type(target["target_rating"]) is int
                and 3 <= target["target_rating"] <= 4 and target["critical_unknown"] is False)
    rows = []
    if accepted:
        attached = [str((new_request_dir / "actor" / f"image_{i+1:02d}.png").resolve())
                    for i, _ in enumerate(new_request["image_evidence_ids"])]
        rows.append({"messages": [{"role": "system", "content": new_request["system_prompt"]},
                                  {"role": "user", "content": "<image>\n" * len(attached) + json.dumps({
                                      "payload": new_request["payload"], "image_notes": new_request["image_notes"]}, ensure_ascii=False)},
                                  {"role": "assistant", "content": json.dumps(output, ensure_ascii=False)}],
                     "images": attached, "metadata": {**provenance, "image_evidence_ids": new_request["image_evidence_ids"],
                         "target_rating": target["target_rating"], "assessment": assessment_name,
                         "loss_scope": "assistant_output_only", "review_scope": "automated_candidate_not_independent_human_certification"}})
    candidate_dir = assessment / "sft_candidates"
    candidate_dir.mkdir()
    (candidate_dir / "hi.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {"HI_terminal_candidates": len(rows), "source_cases": 1, "new_tool_actions": 0,
               "reused_downstream_decisions": 0, "target_assessment": target}
    (candidate_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "selected_assessment.json").write_text(json.dumps({"assessment": assessment_name}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"name": destination.name, "candidate": accepted, "status": output["status"]}), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("request")
    parser.add_argument("--name", required=True)
    parser.add_argument("--finalize-existing", action="store_true")
    parser.add_argument("--reassess-existing", action="store_true")
    parser.add_argument("--assessment", default="assessment_r2")
    parser.add_argument("--rubric", default="rubric_r2.md")
    parser.add_argument("--instruction-file")
    asyncio.run(main(parser.parse_args()))
