"""Construct GIA decision states from verified evidence in a real investigation."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "src"))
from cli_generator import CliGenerator
from foren_searcher import GlobalStore
from foren_searcher.local_models import TransformersGiaPolicy


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_states():
    source = ROOT / "runs/videosham_1028_r1"
    first = load(source / "requests/request_0001/request.json")
    final = load(source / "requests/request_0023/request.json")
    final_view = final["payload"]["global_view"]
    visual_report = final_view["latest_reports"][0]
    evidence = {item["evidence_id"]: item for item in final_view["current_materials"]}
    supporting = {item["evidence_id"]: item for item in visual_report["supporting_records"]}
    image_cards = {item["evidence_id"]: item for item in visual_report["image_inputs"]}
    common_question = "判断给定视频 A 与 B 是否完全相同；若存在能够确认的差异，说明其对画面含义的影响。"
    variants = [
        ("state_01", "candidate", common_question,
         "现有证据仅给出算法对应候选。须核实实际内容，不能直接认定完全相同或编造标题差异；读取已有帧或下发核验任务均可。"),
        ("state_02", "visual", common_question,
         "已有真实文字差异足以否定完全相同并解释主题提示变化。可停止，不必先证明所有声音或像素差异；不能扩大到修改方向和实际谈话。"),
        ("state_03", "visual", "核查给定视频 A 与 B 是否存在言语内容替换，及其是否改变画面含义。",
         "同样的视觉报告不足以回答言语替换。应针对声音内容取证，或在有明确能力限制时给出无法判定；不能由标题反推音轨。"),
    ]
    states = []
    for identifier, kind, question, expectation in variants:
        view = copy.deepcopy(first["payload"]["global_view"])
        view["task"] = question
        view["original_materials"] = [item for item in view["original_materials"] if item.get("record_type") == "evidence"]
        view["history_index"] = []
        view["runtime_constraints"] = {"round_index": 2, "can_create_round": True, "turn_index": 2,
                                       "max_turns": 8, "remaining_turns": 6}
        if kind == "candidate":
            ids = ["E9"]
            statement = "视觉对齐工具给出两侧0–4.84秒的对应候选，锚点处为同秒对应。"
            limitations = ["该报告仅转述工具候选，尚未核验对应帧中的实际文字或其他内容。"]
            task = "取得视频 A、B 的视觉对应候选，并报告工具实际给出的范围与限制。"
        else:
            ids = ["E11", "E15"]
            statement = ("在各自1.6秒的对应画面中，A左下角蓝色标题框显示单行白色文字“lorem ipsum”；"
                         "B相应位置显示两行白色文字“A conversation”“about CRPS”。"
                         "B提供关于CRPS的谈话这一主题提示，A未提供该主题信息。")
            limitations = ["判断限于该组成对帧；标题不能证明实际谈话内容、人物诊断或版本制作先后。"]
            task = "核对两侧1.6秒画面的标题内容及文字提示差异。"
        finding = {"record_type": "finding", "finding_id": "F1", "workspace_id": "W1-1",
                   "statement": statement, "evidence_ids": ids, "limitations": limitations}
        view["current_materials"] = [copy.deepcopy(evidence[eid]) for eid in ids]
        view["latest_reports"] = [{"task": task, "materials": ["E1", "E2"], "run_status": "reported",
            "report": {"record_type": "report", "task_id": "W1-1", "status": "COMPLETED",
                       "finding_ids": ["F1"], "selected_evidence_ids": ids, "findings": [finding]},
            "supporting_records": [copy.deepcopy(supporting[eid]) for eid in ids if eid in supporting],
            "image_inputs": [copy.deepcopy(image_cards[eid]) for eid in ids if eid in image_cards]}]
        view["image_inputs"] = [copy.deepcopy(image_cards[eid]) for eid in ids if eid in image_cards]
        view["image_inputs_omitted"] = []
        view["image_input_errors"] = []
        states.append({"id": identifier, "view": view, "actions": first["payload"]["available_actions"],
                       "reference_expectation": expectation,
                       "provenance": {"dataset": "VideoSham", "case": "1028",
                                      "source_run": "videosham_1028_r1", "source_evidence_ids": ids,
                                      "input_source": "offline_state_from_actual_evidence",
                                      "report_source": "curator_summary_checked_against_actual_evidence",
                                      "is_executed_tool_trajectory": False}})
    return source, states


async def main(args):
    source, states = make_states()
    if args.state:
        states = [state for state in states if state["id"] == args.state]
        if not states:
            raise ValueError("Unknown state")
    instruction = (ROOT / args.instruction_file).read_text(encoding="utf-8") if args.instruction_file else ""
    output = ROOT / "gia_decisions" / args.name
    output.mkdir(parents=True, exist_ok=False)
    store = GlobalStore(source / "store.sqlite3", source / "artifacts")
    for state in states:
        folder = output / state["id"]
        folder.mkdir()
        (folder / "reference.json").write_text(json.dumps({key: value for key, value in state.items()
            if key not in {"view", "actions"}}, ensure_ascii=False, indent=2), encoding="utf-8")
        generator = CliGenerator(store=store, output_dir=folder / "requests", instruction=instruction)
        policy = TransformersGiaPolicy(generator)
        decision = await policy.decide(model_view=state["view"], history=(), actions=tuple(state["actions"]))
        (folder / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"state": state["id"], "decision": decision, "executed": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="videosham_1028_r1")
    parser.add_argument("--state")
    parser.add_argument("--instruction-file")
    asyncio.run(main(parser.parse_args()))
