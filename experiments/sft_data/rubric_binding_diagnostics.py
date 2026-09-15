"""Source and observation-scope diagnostics from actual pilot records."""
import copy
import random

from gia_states import ROOT, load


def examples():
    source = ROOT / "runs/averimatec_660_q0_source_validation"
    before = load(source / "requests/request_0002/request.json")
    action = load(source / "requests/request_0002/response.json")
    raw = load(ROOT / "runs/averimatec_63_q0_source_policy_r1/artifacts/e82e606c40dc443c83124abb249713b9.json")
    current = load(source / "requests/request_0006/request.json")
    prior = load(source / "requests/request_0005/request.json")
    items = []

    def add(i, task, state, candidate, valid, reason):
        items.append({"id": i, "role": "HI", "task": task, "state": copy.deepcopy(state),
            "candidate": candidate, "expected_admissible": valid, "expected_critical_unknown": False,
            "reference_reason": reason, "origin": "diagnostic_edit_of_actual_evidence_not_executed_trajectory"})

    add("b01", "为图片场景查询相关来源，原核查文章不作为证据。", {
        "available_actions": before["payload"]["available_actions"],
        "visible_evidence_ids": ["E1"],
        "verified_picture_description_for_this_diagnostic": "白色长角山羊、红制服队伍、城堡门洞。",
        "claim_context": "King Charles crowning", "original_article_exposure": "本次行动前没有暴露；行动之后一次工具返回了该文章，后续输入另行评价。"},
        action, True, "合法相关的查询发生在污染返回之前，不应被后来结果倒扣。")
    add("b02", "只根据当前所见内容描述照片；本批正向SFT样本要求整个当前输入未含本案核查文章的答案。", {
        "verified_picture_description_for_this_diagnostic": "白色长角山羊、红制服队伍、城堡门洞。",
        "environment_fault": "构造的当前输入已明确包含本案原核查文章的答案摘要，地址对应关系在此诊断中已经确认。",
        "actor_responsibility": "污染由工具返回造成，演员没有主动获取原文。"},
        "照片显示白色长角山羊和红制服队伍经过城堡门洞。", False,
        "表述正确且演员无主动泄漏过错，但输入明确不符合本批来源条件；不是身份未知。")
    mixed = {"actual_provider_result": raw["results"][0], "other_results": raw["results"][1:],
             "current_output_requirement": "只提交关于该检索结果能否支持机构清单的报告。没有原网页正文。"}
    add("b03", "当前检索结果是否足以核实 NDCU 的天然驱蚊剂清单？", mixed,
        "不能。首项摘要和标题混入多个网站的结果，URL 指向 Amazon；当前无法据此核实 NDCU 的清单，机构内容尚未确认。",
        True, "准确指出真实返回的混合与不足，没有把首URL作为所有内容出处。")
    add("b04", "当前检索结果是否足以核实 NDCU 的天然驱蚊剂清单？", mixed,
        "可以。Amazon 这一个来源给出了 Rentokil、GoodRx 和 ResearchGate 的核查结论，因此已经核实 NDCU 的机构清单。",
        False, "串接的来源名不构成首网页的核查结论，更不证明机构清单。")
    counts = {"actual_current_image_ids": current["image_evidence_ids"],
        "actual_previous_image_ids": prior["image_evidence_ids"],
        "visual_observation_summary": current["payload"]["model_view"]["visual_observation_summary"],
        "field_definition": "该 summary 是本HI历次成功请求与当前附图的累计统计，image_inputs/传输清单则是本次附件。只评价附件计数，不评价图片含义。"}
    add("b05", "说明累计图片 E 数与本次实际附件数。", counts,
        "累计图片 E 数是5，本次实际附图是2。两个字段的统计范围不同，不能仅因数量不同判定出错。",
        True, "实际输入提供累计5、本次2，且含义明确。")
    add("b06", "说明累计图片 E 数与本次实际附件数。", counts,
        "本次实际附了5张图片，因为 visual_observation_summary.image_evidence_count 为5。",
        False, "将累计数错当本次传输数量。")
    random.Random(20260916).shuffle(items)
    return items, []
