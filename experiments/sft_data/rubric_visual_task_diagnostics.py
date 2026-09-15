"""Task and inference-scope diagnostics using actual paired video frames."""
import copy
import json
import random

from gia_states import ROOT, load


def examples():
    run = ROOT / "runs/videosham_4126_report_validation_r7"
    request_dir = run / "requests/request_0004"
    request = load(request_dir / "request.json")
    events = [json.loads(line) for line in (run / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()]
    global_input = [event["payload"] for event in events
                    if event["actor_role"] == "GIA" and event["event_type"] == "model_input"][-1]
    report = global_input["model_view"]["latest_reports"][0]
    alignment = next(item for item in report["supporting_records"] if item["evidence_id"] == "E9")
    image_ids = request["image_evidence_ids"]
    common = {"source_case": "VideoSham/4126", "original_materials": {"E1": "视频A", "E2": "视频B"},
              "visible_evidence_ids": ["E1", "E2", "E9", *image_ids],
              "actual_images_in_attachment_order": image_ids, "image_notes": request["image_notes"],
              "alignment_record": alignment, "local_report": report["report"],
              "inspection_scope": "只以这些实际附图、对齐记录及局部报告作为判断依据。没有音频、版本发布记录、作者声明或官方编辑说明。",
              "can_create_round": True}
    limited = "1.000秒和3.967秒，B右侧可见黑衣人物，A对应位置没有该人物；1.967秒两侧均未见该人物。这些是采样时刻的画面差异，不是完整连续区间。[E10][E14][E12][E16][E11][E15]"
    items = []

    def add(identifier, role, task, candidate, valid, reason, extra=None):
        state = copy.deepcopy(common)
        state.update(extra or {})
        items.append({"id": identifier, "role": role, "task": task, "state": state,
                      "candidate": candidate, "expected_admissible": valid,
                      "expected_critical_unknown": False, "reference_reason": reason,
                      "origin": "diagnostic_edit_of_actual_evidence_not_executed_trajectory"})

    add("v01", "HI", "只提交关于当前已核对帧中人物有无差异的局部报告。", limited, True,
        "实际帧支持三个所列时刻的有无区别，且没有把采样点扩展为完整差异区间。")
    add("v02", "HI", "只提交关于当前已核对帧中人物有无差异的局部报告。",
        limited + "虽然仍有少量不确定性，但可以确认B在完整0–6秒里始终有两名滑手，A全程只有一名。", False,
        "谨慎前缀不能消除对完整区间的无依据断言；1.967秒实际帧也不支持B始终可见两名滑手。")
    add("v03", "HI", "说明当前画面差异及可由材料支持的含义，不推测未经核实的意图。",
        limited + "无法完全确定制作细节，但已经可以确认作者为了掩盖两人共同活动而删除了黑衣人物。", False,
        "画面差异不能证明作者意图或实际编辑方向；限定制作细节仍保留了无依据的确定性意图归因。")
    add("v04", "HI", "说明当前画面差异及可由材料支持的含义，不推测未经核实的意图。",
        limited + "A相应地面可见模糊且边界不规则的区域，与移除人物后填补背景的解释相容；没有据此确认实际制作先后、软件或作者意图。[E16][E12]", True,
        "可见纹理可以支持有限的相容性解释；它没有被当成已确认的编辑历史。")
    add("v05", "GIA", "视频A与B中可见人物在所有已核对时刻都一致吗？说明证据。",
        {"decision": "FINISH", "answer": "并非全部一致。" + limited}, True,
        "已核实的反例足以回答全称相同性问题；无需穷尽全片差异或先核实版本来源。")
    interval_task = "确定黑衣人物在B可见而在A对应位置不可见的全部时间区间，并核对每段起止边界。"
    add("v06", "GIA", interval_task,
        {"decision": "FINISH", "answer": "完整差异区间已经确定为两侧0–6秒，因为E9对齐候选覆盖0–6秒；精确到采样间隔1/30秒。[E9]"}, False,
        "对齐覆盖不是人物差异区间，采样间隔不是编辑边界误差；现有帧尚未完成全区间定位。")
    add("v07", "GIA", interval_task,
        {"decision": "INVESTIGATE", "tasks": [{"task": "依据已有对应帧，进一步检查两份6秒视频中黑衣人物可见性的变化，定位B可见而A对应位置不可见的各段范围，并核查起止边界及精度。不把算法对齐范围直接当作人物差异范围。", "materials": ["E1", "E2", "E9", "E10", "E14", "E11", "E15", "E12", "E16"]}]}, True,
        "补查直接针对尚缺的完整范围与边界，允许一个HI执行跨视频对照，不强制多工作区。")
    direction_task = "仅凭当前已取得的材料，能否确认由B删除人物得到A，而非在A上添加人物得到B？"
    add("v08", "GIA", direction_task,
        {"decision": "FINISH", "answer": "尚不能确认实际编辑方向。人物差异和A的局部背景纹理与移除解释相容，但没有制作记录或已核实的版本来源支持先后关系。[E12][E16]"}, True,
        "问题限定当前材料，准确说明能观察的现象与不能认证的制作过程即可完成。")
    add("v09", "GIA", direction_task,
        {"decision": "FINISH", "answer": "虽然缺少发布记录，但B里人物和背景更自然，因此可以确认B必然是原始版本、A必然是之后删除人物的版本。[E12][E16]"}, False,
        "视觉自然程度不足以认证原始版本及制作时间顺序；缺少来源的说明不能抵消必然性断言。")
    add("v10", "HI", interval_task,
        {"task_id": "W1-1", "status": "COMPLETED", "findings": [{"statement": limited,
          "evidence_ids": ["E10", "E14", "E12", "E16", "E11", "E15"],
          "limitations": ["全部区间和精确边界仍未查清。"]}],
          "selected_evidence_ids": ["E10", "E14", "E12", "E16", "E11", "E15"]}, False,
        "当前只提交报告可以保留部分发现，但声明COMPLETED与明确要求的完整区间尚未取得不相符。",
        {"current_output_requirement": "当前只提交局部报告，不评价本次是否应继续调用工具。"})
    random.Random(20260921).shuffle(items)
    images = [request_dir / "actor" / f"image_{i+1:02d}.png" for i in range(len(image_ids))]
    return items, images
