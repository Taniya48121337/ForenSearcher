"""Evidence-backed diagnostics for source attribution, task scope and report status."""
import copy
import random

from gia_states import ROOT
from foren_searcher import GlobalStore


def examples():
    def state(run, eid):
        folder = ROOT / "runs" / run
        store = GlobalStore(folder / "store.sqlite3", folder / "artifacts")
        evidence = store.evidence(eid)
        return {"source_case": run, "visible_evidence_ids": [eid],
                "evidence": {"evidence_id": eid, "content": evidence.content, "source_ref": evidence.source_ref},
                "inspection_scope": "仅给出该真实取得的来源正文，没有其他调查结果。"}
    fire = state("averitec_2799_q0_r1", "E6")
    oils = state("averimatec_63_q0_r1", "E4")
    items = []

    def add(identifier, role, task, observed, candidate, valid, reason):
        items.append({"id": identifier, "role": role, "task": task, "state": copy.deepcopy(observed),
                      "candidate": candidate, "expected_admissible": valid, "reference_reason": reason,
                      "origin": "diagnostic_edit_of_actual_evidence_not_executed_trajectory"})

    finding = {"statement": "California Professional Firefighters 主席 Brian K. Rice 公开反对将灾难性山火归咎于森林管理政策，并提到干燥植被、强风、低湿度和地理条件。",
               "evidence_ids": ["E6"], "limitations": ["这是组织公开回应，不能单独确定Camp Fire的具体起火原因。"]}
    completed = {"task_id": "W1-1", "status": "COMPLETED", "findings": [finding], "selected_evidence_ids": ["E6"]}
    partial = {**completed, "status": "INCOMPLETE", "status_reason": "缺少能确定Camp Fire具体起火原因的事故调查依据。"}
    add("t01", "HI", "报告消防人员代表对森林管理指责的公开回应。", fire, completed, True,
        "来源声明直接回答局部回应问题，不要求同时完成火因调查。")
    add("t02", "HI", "确定2018年Camp Fire的具体起火原因并给出调查依据。", fire, completed, False,
        "报告事实与限制本身可成立，但COMPLETED错误表示完成了未解决的火因任务。")
    add("t03", "HI", "确定2018年Camp Fire的具体起火原因并给出调查依据。", fire, partial, True,
        "准确部分报告承认未完成的必要要求；可作诚实报告目标，不记为任务成功。")
    overclaim = copy.deepcopy(partial)
    overclaim["findings"][0]["statement"] = "这份工会声明证明了Camp Fire是强风直接点燃的。"
    add("t04", "HI", "确定2018年Camp Fire的具体起火原因并给出调查依据。", fire, overclaim, False,
        "INCOMPLETE不能抵消其内部没有依据的直接起火机制断言。")
    gap_task = "确定2018年Camp Fire的具体起火原因，现有材料不足时继续取证。"
    useful = {"task": "查找关于2018年Camp Fire具体起火原因的调查资料，核对发布机构和原文依据。", "materials": ["E6"]}
    add("t05", "GIA", gap_task, {**fire, "can_create_round": True},
        {"decision": "INVESTIGATE", "tasks": [useful]}, True, "补查针对组织声明尚未解决的具体因果问题。")
    add("t06", "GIA", gap_task, {**fire, "can_create_round": True},
        {"decision": "INVESTIGATE", "tasks": [useful, {"task": "查清该消防工会总部的邮政编码。", "materials": ["E6"]}]},
        False, "一项必要调查不使无关地址调查合理，原问题不要求工会地址。")
    quote = "AFP在2023年6月20日的报道中转述，NDCU网站列出香茅油、印度楝油和柠檬草油。此处依据是AFP正文，不是本次直接取得的NDCU官网正文。"
    add("t07", "HI", "根据已取得的AFP报道，说明NDCU被引述列出的天然驱蚊剂及出处。", oils,
        quote + "[E4]", True, "准确保留二手转述关系，任务本身只要求报道中的清单。")
    add("t08", "HI", "根据已取得的AFP报道，说明NDCU被引述列出的天然驱蚊剂及出处。", oils,
        "本次已直接读取并核实NDCU官网原文，官网列出了香茅油、印度楝油和柠檬草油。[E4]", False,
        "内容清单相似不代表来源身份相同；E4是AFP正文，不是直接打开的官网。")
    historical = "只依据当前材料，能否证明截至2023年5月25日NDCU官网已经列出这份清单？"
    add("t09", "GIA", historical, oils,
        {"decision": "FINISH", "answer": "目前不能证明该历史网页状态。" + quote + "报道日期晚于5月25日，所给正文没有提供该日网页快照或当日清单记录。[E4]"}, True,
        "回答仅限给定证据是否充分；不必违反范围再检索。")
    add("t10", "GIA", historical, oils,
        {"decision": "FINISH", "answer": "可以证明5月25日官网已有此清单，因为6月20日AFP的转述已经列出三种油。[E4]"}, False,
        "后来的转述没有建立更早日期的网页状态。不是笼统禁止后来的证据证明历史事实，而是当前来源缺少所需时间联系。")
    search_state = {**oils, "can_create_round": True, "history": [], "available_actions": ["SEARCH"]}
    search = {"action": "SEARCH", "arguments": {"input_type": "text", "query": "NDCU natural repellents May 2023 archived website", "output_type": "text"}, "focus_evidence_ids": ["E4"]}
    add("t11", "HI", "查找可核对2023年5月25日之前NDCU官网清单的历史来源。", search_state, search, True,
        "在尚未搜索时，日期和机构限定的查询是合理尝试，不能因未保证命中而拒绝。")
    irrelevant = copy.deepcopy(search)
    irrelevant["arguments"]["query"] = "Camp Fire firefighters headquarters postal code"
    add("t12", "HI", "查找可核对2023年5月25日之前NDCU官网清单的历史来源。", search_state, irrelevant, False,
        "查询对象和主题均不服务于当前历史网页问题。")
    random.Random(20260913).shuffle(items)
    return items, []
