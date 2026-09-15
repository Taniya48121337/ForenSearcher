"""Construct task-conditioned GIA states using reports from real text investigations."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys

from gia_states import ROOT, load
from cli_generator import CliGenerator
from foren_searcher import GlobalStore
from foren_searcher.local_models import TransformersGiaPolicy


CASES = {
    "averitec_2694": {"run": "averitec_2694_q8_source_validation", "dataset": "AVeriTeC", "case": "2694", "evidence": ["E2", "E3", "E4", "E5"],
        "tasks": [
            ("依据当前已取得的材料，说明 Sol Plaatje University 与 University of Mpumalanga 的成立、首批招生和新建教学设施交付是否是同一个时间点，并给出能够支持的时间线。",
             "报告及原文区分2013年成立、2014年招生和2016学年设施交付，成立与全部建设完成不能互换。可以先读必要原文，然后完成有限时间线，不要求穷尽后续校园扩建。"),
            ("查清 Sol Plaatje University 与 University of Mpumalanga 各自首期校园建设项目最终的实际总支出，并核对各自最终支出与批准预算是否一致。说明项目阶段、统计口径与出处。",
             "已有正文提供合同金额、部分费用、约数和预算内的转述，也给出结项报告链接；这些数字不自动等于两个学校各自同口径的最终总支出与批准预算。应沿已知报告取得或读取所需依据，或安排范围明确的调查；既有材料足以回答的部分应保留，不强制采用某个固定分工。"),
            ("仅依据当前取得的材料，NEC 项目介绍中的 R1.5 billion 和 R1.6 billion，以及 Sol Plaatje University 页面中的约 R1.4 billion，是否是在给同一个项目、同一口径的总支出报出互相矛盾的数字？解释各金额的对象与证据能够支持的范围。",
             "E4的R1.5 billion指Wits自己的35个项目；R1.6 billion是两所新大学相关NEC合同金额；E3约R1.4 billion指SPU在2013–2018年的首期资本支出。不能把三者当作同一个量直接判矛盾，也不能无依据把它们相加、相减或断言已核清最终账目。问题限定当前材料，通常无需外部调查，可先读取必要既有正文。")
        ]},
    "averimatec_423": {"run": "averimatec_423_q0_validation_r3", "dataset": "AVerImaTeC", "case": "423", "evidence": ["E5", "E1"],
        "tasks": [
            ("根据当前已取得的科学论文和机构说明，解释相对海平面、绝对海平面与全球平均海平面变化的区别，并说明来源范围。",
             "已有报告与所引正文能支持这些概念；可先读取尚未展开的必要原文。问题不要求测量实际加速度，不因缺少趋势序列而强制追加调查。"),
            ("核查截至2022年9月3日“海平面上升速率没有增加”这一声明，区分全球趋势与局部海平面现象，依据当时可用的观测或科学评估回答。",
             "概念定义没有给出速率随时间变化的观测。应针对全球平均海平面上升速率的变化查找日期与统计口径可核对的资料；网页日期未知不能假定它在截止日期前存在。"),
            ("仅依据当前已取得的材料，能否判断截至2022年9月3日全球海平面上升速率是否增加？解释证据所能回答和不能回答的问题。",
             "当前已展示的正文主要定义海平面变化与参考系，未展示用于判断加速度的时间序列。可准确限定为已展示部分而说明不足，也可先读取已有论文尚未展示的正文；不得预判未读内容，不虚构数据或替任务新增外部搜索。"),
            ("只从当前材料给出的概念出发，某一海岸的相对海平面下降，是否足以单独推翻全球平均海平面上升？说明推理依据，不核查某个具体海岸的实测数据。",
             "局部相对量受到陆地运动影响，全球平均是空间加权量。一个局部相对下降不能单独否定全球平均上升；无需先测出全球实际趋势才能回答这一逻辑问题。")
        ]},
    "averitec_2799": {"run": "averitec_2799_q0_r1", "dataset": "AVeriTeC", "case": "2799", "evidence": ["E6"],
        "tasks": [
            ("California Professional Firefighters 如何公开回应“加州山火是森林管理不善造成”的指责？说明回应要点。",
             "已取得的组织声明可以支持回应要点；如所需原文未完整显示，可先读取，再给出局部事实答案。无需先查清具体火灾起因。"),
            ("确定2018年 Camp Fire 的起火原因，并指出支持该原因的调查依据。",
             "工会对笼统森林管理指责的回应不等于Camp Fire起火原因调查结论。应检索相关调查依据，不能仅据该声明完成原因归属。"),
            ("只依据当前已取得的材料，判断能否确定2018年 Camp Fire 的具体起火原因，并解释依据范围。",
             "可以结束并明确现有材料不足以确定具体起火原因；任务限定只依据当前材料，不要求另开外部调查。")
        ]},
    "averimatec_63": {"run": "averimatec_63_q0_r1", "dataset": "AVerImaTeC", "case": "63", "evidence": ["E4"],
        "tasks": [
            ("根据已取得的 AFP 2023年6月20日报道，NDCU 被引述列出了哪些天然驱蚊剂？说明信息出处。",
             "可依据该报道归纳其转述的清单；须保持AFP转述的来源身份，不声称已直接打开官网或提供独立效果证明。"),
            ("核查截至2023年5月25日，NDCU 官网当时是否已经列出香茅油、印度楝油和柠檬草油这三类驱蚊剂。",
             "6月20日报道不能确定5月25日的网页状态。需要日期可核对的历史来源，或在确实不可取得时说明不能判定。"),
            ("仅依据目前取得的材料，能否证明2023年5月25日的 NDCU 官网已经包含这份天然驱蚊剂清单？解释时间依据。",
             "可以结束并说明目前不能证明该历史网页状态；不得把6月20日转述提前成5月25日直接证据。")
        ]}
}


def make_states(key):
    spec = CASES[key]
    source = ROOT / "runs" / spec["run"]
    config = json.loads((source / "run_config.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (source / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()]
    last = [event["payload"] for event in events if event["actor_role"] == "GIA" and event["event_type"] == "model_input"][-1]
    allowed = set(spec["evidence"])
    template = copy.deepcopy(last["model_view"])
    template["original_materials"] = []
    template["current_materials"] = [item for item in template["current_materials"] if item.get("evidence_id") in allowed]
    template["history_index"] = []
    template.pop("visual_observation_summary", None)
    template.pop("visual_observation_summary_truncated", None)
    template["image_inputs"] = []
    for report in template["latest_reports"]:
        report["materials"] = [item for item in report["materials"] if item in allowed]
        report["image_inputs"] = []
        report.pop("visual_observation_summary", None)
        report["supporting_records"] = [item for item in report["supporting_records"] if item["evidence_id"] in allowed]
    template["runtime_constraints"] = {"round_index": 2, "can_create_round": True, "turn_index": 1,
                                       "max_turns": 8, "remaining_turns": 7}
    states = []
    for i, (task, expectation) in enumerate(spec["tasks"], 1):
        view = copy.deepcopy(template)
        view["task"] = task
        states.append({"id": f"state_{i:02d}", "view": view, "actions": last["actions"],
            "reference_expectation": expectation,
            "provenance": {"dataset": spec["dataset"], "case": spec["case"], "source_run": spec["run"],
                           "usage_split": config.get("usage", "development"),
                           "source_evidence_ids": spec["evidence"], "input_source": "offline_state_from_actual_evidence",
                           "report_source": "actual_hi_report_retained_without_rewriting",
                           "state_changes": "new_question_and_restricted_visible_records",
                           "is_executed_tool_trajectory": False}})
    return source, states


async def main(args):
    source, states = make_states(args.case)
    destination = ROOT / "gia_decisions" / args.name
    destination.mkdir(parents=True, exist_ok=False)
    store = GlobalStore(source / "store.sqlite3", source / "artifacts")
    instruction = (ROOT / "actor_gia_r2.txt").read_text(encoding="utf-8")
    for state in states:
        folder = destination / state["id"]
        folder.mkdir()
        (folder / "reference.json").write_text(json.dumps({k: v for k, v in state.items() if k not in {"view", "actions"}},
            ensure_ascii=False, indent=2), encoding="utf-8")
        generator = CliGenerator(store=store, output_dir=folder / "requests", instruction=instruction)
        decision = await TransformersGiaPolicy(generator).decide(model_view=state["view"], history=(), actions=tuple(state["actions"]))
        (folder / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"state": state["id"], "decision": decision, "executed": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=list(CASES), required=True)
    parser.add_argument("--name", required=True)
    asyncio.run(main(parser.parse_args()))
