"""Assess offline GIA states separately from executed tool trajectories."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "src"))
from foren_searcher.local_models import extract_json_object
from assessment_packet import pack_repeated


def main(args):
    source = ROOT / "gia_decisions" / args.name
    folder = source / args.assessment
    folder.mkdir(exist_ok=False)
    states, images = [], {}
    for state in sorted(source.glob("state_*")):
        requests = []
        for directory in sorted((state / "requests").glob("request_*")):
            request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
            response = directory / "response.json"
            if not response.exists():
                continue
            requests.append({"id": directory.name, "input": request,
                             "output": extract_json_object(response.read_text(encoding="utf-8"))})
            for i, eid in enumerate(request["image_evidence_ids"]):
                images.setdefault(eid, directory / "actor" / f"image_{i + 1:02d}.png")
        states.append({"id": state.name, "requests": requests,
                       "accepted_decision": json.loads((state / "decision.json").read_text(encoding="utf-8")),
                       "reference": json.loads((state / "reference.json").read_text(encoding="utf-8"))})
    packet = {"states": states, "attached_images": list(images),
              "scope": "offline_next_decisions; target_actions_unexecuted; prior_memory_reads_if_any_are_recorded_in_input_history_and_provenance"}
    (folder / "packet.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    rubric = (ROOT / args.rubric).read_text(encoding="utf-8")
    (folder / "rubric.md").write_text(rubric, encoding="utf-8")
    prompt = ("独立评估这些GIA离线决策样本。只用给定材料与附件，不调用工具或读文件。"
              "actor输入不含reference。reference_expectation是待核查的标注建议，不是必须照抄的唯一行动；"
              "结合真实任务和输入自行判断必要补查、停止或其他合理路径。不得仅因决定名称符合建议就给高分。"
              "观察是否还夹带不必要的调查任务。当前目标决定未执行，因此execution维度不适用；不要声称真实联动成功。"
              "若输入memory_history包含已执行的读取及原文返回，可作为当前决定的真实依据；它不代表当前新提议也已执行。"
              "按rubric为每个state给出JSON：{states:[{id,dimensions:[{id,status,rating,reason}],"
              "target_rating:0到4,disposition:accept_candidate或repair_reexecute或context_only或exclude,"
              "critical_unknown:boolean,reason:string}],construction_revisions:[string],rubric_revisions:[string]}。"
              "dimensions使用rubric中的七个英文id，status为scored、unknown或not_applicable。"
              "对象{$object:oN}指objects.oN的内容，需要递归解析。\nRUBRIC:\n" + rubric +
              "\nPACKET:\n" + json.dumps(pack_repeated(packet), ensure_ascii=False))
    (folder / "prompt.txt").write_text(prompt, encoding="utf-8")
    command = [shutil.which("codex"), "exec", "--ephemeral", "--sandbox", "read-only", "--skip-git-repo-check",
               "--json", "--color", "never", "-C", str(folder), "-o", str(folder / "judgment.json")]
    for path in images.values():
        command.extend(["--image", str(path)])
    command.extend(["--", "-"])
    completed = subprocess.run(command, input=prompt, encoding="utf-8", errors="replace", capture_output=True, timeout=900)
    (folder / "cli_events.jsonl").write_text(completed.stdout, encoding="utf-8")
    (folder / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError("Offline GIA assessment failed")
    for line in completed.stdout.splitlines():
        event = json.loads(line)
        if event.get("item", {}).get("type") in {"command_execution", "mcp_tool_call", "web_search", "file_change"}:
            raise RuntimeError("Assessor used an out-of-packet tool")
    judgment = extract_json_object((folder / "judgment.json").read_text(encoding="utf-8"))
    if len(judgment["states"]) != len(states) or {x["id"] for x in judgment["states"]} != {x["id"] for x in states}:
        raise RuntimeError("Missing or duplicate state assessments")
    print(json.dumps(judgment, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="videosham_1028_r1")
    parser.add_argument("--assessment", default="assessment_r2")
    parser.add_argument("--rubric", default="rubric_r2.md")
    main(parser.parse_args())
