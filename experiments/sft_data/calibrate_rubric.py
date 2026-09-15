"""Measure false acceptance and rejection on evidence-backed diagnostic examples."""
from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "src"))
from foren_searcher.local_models import extract_json_object


def examples():
    context = {"source_case": "VideoSham/1028", "observed_times_s": [0.0], "duration_s": 4.84,
               "images": {"E5": "视频A的0秒帧，附件1", "E7": "视频B的0秒帧，附件2"},
               "visible_evidence_ids": ["E5", "E7"],
               "inspection_scope": "只提供这组成对帧，没有其他画面或声音检查。"}
    items = []

    def add(identifier, role, task, candidate, valid, explanation, extra=None):
        state = copy.deepcopy(context)
        state.update(extra or {})
        items.append({"id": identifier, "role": role, "task": task, "state": state,
                      "candidate": candidate, "expected_admissible": valid,
                      "reference_reason": explanation,
                      "origin": "diagnostic_edit_of_actual_evidence_not_executed_trajectory"})

    correct = "A左下角蓝框写着lorem ipsum；B相应位置写着A conversation about CRPS。B提供关于CRPS谈话的主题提示，A没有这项文字提示。[E5][E7]"
    add("d01", "HI", "比较给定0秒帧的标题文字和主题提示。", correct, True, "两个标题可由实际像素直接核对；只回答所接局部问题。")
    add("d02", "HI", "比较给定0秒帧的标题文字和主题提示。",
        "A写着A conversation about CRPS，B写着lorem ipsum，因此A的文字明确提供CRPS主题。[E5][E7]", False, "将两个视频的标题对应关系颠倒。")
    add("d03", "HI", "根据已提供帧报告能够确认的视频差异及检查范围。",
        correct + "这一判断仅限0秒；未检查的时刻和声音保持未知。", True, "准确局部结论与完整性限制相容。")
    add("d04", "HI", "根据已提供帧报告能够确认的视频差异及检查范围。",
        correct + "已逐帧核验整个4.84秒视频，除标题外所有像素和音轨完全一致。", False, "单组成对帧不能支持全片逐帧或音频结论。")
    action_context = {"available_actions": [{"name": "READ_RECORD", "arguments": {"record_id": "visible E id"}}],
                      "current_focus_candidates": ["E5", "E7"], "history": [], "images_not_currently_attached": True}
    add("d05", "HI", "在尚未同时附图时核对两侧0秒帧的标题差异。",
        {"action": "READ_RECORD", "arguments": {"record_id": "E5"}, "focus_evidence_ids": ["E5", "E7"]},
        True, "首次安排同时看两侧已有帧是合理动作，不应按重复读取一律拒绝。", action_context)
    add("d06", "HI", "在尚未同时附图时核对两侧0秒帧的标题差异。",
        {"action": "READ_RECORD", "arguments": {"record_id": "E5"}, "focus_evidence_ids": ["E5", "E200"]},
        False, "E200既不存在于当前证据也不在焦点可选集合。", action_context)
    report_context = {"verified_local_report": correct, "can_create_round": True}
    add("d07", "GIA", "判断两个视频是否完全相同，说明能够确认的不同。",
        {"decision": "FINISH", "answer": "两者并非完全相同：0秒标题文字不同。" + correct + "未据此判断其他时刻、音频或制作先后。"},
        True, "一个已核实的真实差异足以否定完全相同，不必穷尽所有差异。", report_context)
    add("d08", "GIA", "判断两个视频是否完全相同，说明能够确认的不同。",
        {"decision": "INVESTIGATE", "tasks": [{"task": "必须先查出原作者的姓名及精确拍摄地址，才能回答两个视频是否完全相同。", "materials": ["E5", "E7"]}]},
        False, "已确认的标题差异已经回答相同性；追加作者地址既非必要也不能由所给局部帧保证执行。", report_context)
    add("d09", "GIA", "判断两个视频的言语内容是否被替换。",
        {"decision": "INVESTIGATE", "tasks": [{"task": "核对两份视频实际言语内容是否不同；没有可辨认声音时说明限制，不能用画面标题代替言语证据。", "materials": ["E1", "E2"]}]},
        True, "针对仍缺少的声音信息补查是合理分工。", {**report_context, "visible_evidence_ids": ["E1", "E2", "E5", "E7"], "original_materials": ["E1:视频A", "E2:视频B"]})
    add("d10", "GIA", "判断两个视频的言语内容是否被替换。",
        {"decision": "FINISH", "answer": "已确认言语被替换：A说lorem ipsum，B说A conversation about CRPS。[E5][E7]"},
        False, "把画面文字当作实际言语。", report_context)
    qv = {"source_case": "QVHighlights/0", "observed_times_s": [49.0], "duration_s": 150,
          "images": {"E24": "视频49秒帧，附件3"}, "visible_evidence_ids": ["E24"],
          "inspection_scope": "只核对49秒帧；它显示女子在车内举起打开的食品盒。"}
    add("d11", "HI", "找出一个符合描述的时刻：Blonde woman holds up food in her car. 不要求完整区间。",
        "49秒可以看到女子在车内举起打开的食品盒；盒内能看见带绿色配料的食物。[E24] 这不是完整事件起止时间。",
        True, "该任务只要求一个时刻，不能以未输出完整标准区间为由拒绝。", qv)
    add("d12", "HI", "找出一个符合描述的时刻：Blonde woman holds up food in her car. 不要求完整区间。",
        "49秒是女子在餐馆厨房制作食物的画面，没有车内场景或举起食品盒。[E24]",
        False, "场景与动作描述同所给像素直接矛盾。", qv)
    random.Random(20260912).shuffle(items)
    image_paths = [
        ROOT / "runs/videosham_1028_r1/requests/request_0003/actor/image_01.png",
        ROOT / "runs/videosham_1028_r1/requests/request_0003/actor/image_02.png",
        ROOT / "runs/qvhighlights_0_r1/requests/request_0004/actor/image_06.png",
    ]
    return items, image_paths


def main(args):
    folder = ROOT / "rubric_calibration" / args.name
    folder.mkdir(parents=True, exist_ok=False)
    if args.suite == "visual_task":
        from rubric_visual_task_diagnostics import examples as visual_task_examples
        items, images = visual_task_examples()
    elif args.suite == "fact_state":
        from rubric_fact_state_diagnostics import examples as fact_state_examples
        items, images = fact_state_examples()
    elif args.suite == "binding":
        from rubric_binding_diagnostics import examples as binding_examples
        items, images = binding_examples()
    elif args.suite == "source":
        from rubric_source_diagnostics import examples as source_examples
        items, images = source_examples()
    elif args.suite == "scope":
        from rubric_scope_diagnostics import examples as scope_examples
        items, images = scope_examples()
    elif args.suite == "text":
        from rubric_text_diagnostics import examples as text_examples
        items, images = text_examples()
    else:
        items, images = examples()
    (folder / "reference_labels.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    packet = [{k: v for k, v in item.items() if k not in {"expected_admissible", "expected_critical_unknown", "reference_reason"}} for item in items]
    rubric = (ROOT / args.rubric).read_text(encoding="utf-8")
    prompt = ("你是离线数据评审员。只使用给定信息和附件，不调用任何工具或读取文件。"
              "这些是评分诊断样本，不是已执行的轨迹。诊断的candidate可能正确或错误，分别独立评估。"
              "评审能看到全部附件，但假设演员只看到每项state规定的信息；不要把评审附件扩大成演员可见状态。"
              "依照rubric判断每条candidate是否适合作为其对应task和state的正向SFT目标。"
              "不要因为句子谨慎而忽略实质错误，也不要要求任务未要求的额外事实。"
              "返回JSON：{items:[{id,admissible:boolean,score:0到100,critical_unknown:boolean,reason:string}]}。"
              "逐条评分，不省略。不要猜测正负例比例。\nRUBRIC:\n" + rubric + "\nITEMS:\n" + json.dumps(packet, ensure_ascii=False))
    (folder / "prompt.txt").write_text(prompt, encoding="utf-8")
    command = [shutil.which("codex"), "exec", "--ephemeral", "--sandbox", "read-only", "--skip-git-repo-check",
               "--json", "--color", "never", "-C", str(folder), "-o", str(folder / "judgment.json")]
    for path in images:
        command.extend(["--image", str(path)])
    command.extend(["--", "-"])
    completed = subprocess.run(command, input=prompt, encoding="utf-8", errors="replace", capture_output=True, timeout=900)
    (folder / "cli_events.jsonl").write_text(completed.stdout, encoding="utf-8")
    (folder / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError("Calibration CLI failed; retained actual outputs")
    for line in completed.stdout.splitlines():
        event = json.loads(line)
        if event.get("item", {}).get("type") in {"command_execution", "mcp_tool_call", "web_search", "file_change"}:
            raise RuntimeError("Calibration used an out-of-packet tool")
    judgment = extract_json_object((folder / "judgment.json").read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in judgment["items"]}
    if len(by_id) != len(items) or len(judgment["items"]) != len(items) or set(by_id) != {item["id"] for item in items}:
        raise RuntimeError("Missing or duplicate diagnostic judgments")
    false_accept, false_reject, unknown = [], [], []
    for item in items:
        result = by_id[item["id"]]
        if result["critical_unknown"]:
            unknown.append(item["id"])
        admitted = result["admissible"] is True and result["critical_unknown"] is False
        if admitted and not item["expected_admissible"]:
            false_accept.append(item["id"])
        if not admitted and item["expected_admissible"]:
            false_reject.append(item["id"])
    metrics = {"total": len(items), "positive": sum(x["expected_admissible"] for x in items),
               "negative": sum(not x["expected_admissible"] for x in items), "false_accept_ids": false_accept,
               "false_reject_ids": false_reject, "critical_unknown_ids": unknown,
               "suite": args.suite,
               "scope": "development_diagnostics_not_generalization_evidence"}
    if any("expected_critical_unknown" in item for item in items):
        metrics["critical_unknown_mismatch_ids"] = [item["id"] for item in items
            if by_id[item["id"]]["critical_unknown"] != item["expected_critical_unknown"]]
        metrics["reference_unjudgeable_ids"] = [item["id"] for item in items if item["expected_critical_unknown"]]
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--rubric", default="rubric_r2.md")
    parser.add_argument("--suite", choices=["visual", "text", "scope", "source", "binding", "fact_state", "visual_task"], default="visual")
    main(parser.parse_args())
