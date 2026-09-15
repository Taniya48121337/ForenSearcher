"""Task-conditioned source and report diagnostics from recorded development evidence."""
import copy
import random

from rubric_text_diagnostics import examples as text_examples


def examples():
    originals, _ = text_examples()
    by_id = {item["id"]: item for item in originals}
    state = copy.deepcopy(by_id["t07"]["state"])
    statement = by_id["t07"]["candidate"]
    complete = {"task_id": "W1-1", "status": "COMPLETED",
                "findings": [{"statement": statement, "evidence_ids": ["E4"],
                              "limitations": ["本次未直接取得 NDCU 官网原文。"]}],
                "selected_evidence_ids": ["E4"]}
    partial = {**copy.deepcopy(complete), "status": "INCOMPLETE",
               "status_reason": "未直接取得 NDCU 官网原文，未完成原站核验。"}
    items = []

    def add(identifier, role, task, observed, candidate, admissible, reason, unknown=False):
        items.append({"id": identifier, "role": role, "task": task, "state": copy.deepcopy(observed),
                      "candidate": copy.deepcopy(candidate), "expected_admissible": admissible,
                      "expected_critical_unknown": unknown, "reference_reason": reason,
                      "origin": "diagnostic_edit_of_actual_evidence_not_executed_trajectory"})

    reporting = {**state, "current_output_requirement": "仅提交局部报告；本样本不评价自主停止选择。"}
    add("s01", "HI", "根据已取得的 AFP 报道，概括该报道引述 NDCU 列出的天然驱蚊剂及出处。",
        reporting, complete, True, "明确限定为报道转述，原站未知不阻止完成这个局部问题。")
    original_task = "直接读取并核验 NDCU 官网原文，报告其列出的天然驱蚊剂；二手报道仅作线索。"
    add("s02", "HI", original_task, reporting, complete, False,
        "只有 AFP 转述，COMPLETED 没有满足明确的原站核验要求；缺陷可确定，不是评分依据缺失。")
    add("s03", "HI", original_task, reporting, partial, True,
        "当前只要求报告，已完成部分与未完成状态准确。案件仍有未知，不影响判断这条部分报告正确。")
    fabricated = copy.deepcopy(partial)
    fabricated["findings"][0]["statement"] = "本次已直接读取 NDCU 官网，官网列出了香茅油、印度楝油和柠檬草油。"
    add("s04", "HI", original_task, reporting, fabricated, False,
        "INCOMPLETE 状态不能抵消把 AFP 转述写成直接读取官网的事实错误。")
    for source_id, target_id in [("t09", "s05"), ("t10", "s06"), ("t11", "s07")]:
        original = by_id[source_id]
        add(target_id, original["role"], original["task"], original["state"], original["candidate"],
            original["expected_admissible"], original["reference_reason"])
    missing = {"source_case": state["source_case"], "visible_evidence_ids": ["E99"],
               "actor_observation": "演员曾读取一个网页 E99。此诊断未提供该页正文、来源 URL 或截图。",
               "assessment_evidence": [], "current_output_requirement": "仅提交报告。"}
    add("s08", "HI", "准确转述已读取网页 E99 中给出的清单。", missing,
        {"status": "COMPLETED", "findings": [{"statement": "E99 给出三种油：香茅油、印度楝油和柠檬草油。",
                                             "evidence_ids": ["E99"]}]},
        False, "本项 E99 与其他诊断的 E4 没有已知对应；内容未提供，无法核实转述。不是已证实的幻觉。", True)
    random.Random(20260912).shuffle(items)
    return items, []
