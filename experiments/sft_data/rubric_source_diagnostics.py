"""Task-conditioned source diagnostics grounded in acquired evidence and annotations."""
import copy
import random

from gia_states import ROOT, load
from rubric_text_diagnostics import examples as text_examples


def examples():
    by_id = {item["id"]: item for item in text_examples()[0]}
    oils = copy.deepcopy(by_id["t07"]["state"])
    oils["source_url"] = "https://factcheck.afp.com/doc.afp.com.33JQ83X"
    oils["publication_date"] = "2023-06-20"
    oils["current_output_requirement"] = "只提交报告，不评价自主停止选择。"
    quote = by_id["t07"]["candidate"]
    report = {"task_id": "W1-1", "status": "COMPLETED",
              "findings": [{"statement": quote, "evidence_ids": ["E4"], "limitations": []}],
              "selected_evidence_ids": ["E4"]}
    items = []

    def add(i, role, task, state, candidate, valid, reason, unknown=False):
        items.append({"id": i, "role": role, "task": task, "state": copy.deepcopy(state),
            "candidate": candidate, "expected_admissible": valid, "expected_critical_unknown": unknown,
            "reference_reason": reason, "origin": "diagnostic_edit_of_actual_evidence_not_executed_trajectory"})

    add("p01", "HI", "根据当前取得的 AFP 报道，概括其中引述的 NDCU 天然驱蚊剂清单。原声明日期为2023年5月25日，仅作为背景，不限制本题资料的发表时间。",
        oils, report, True, "当前任务明确允许这篇后来的报道，准确转述可完成任务。")
    strict = "核查截至2023年5月25日 NDCU 官网列出的天然驱蚊剂；仅用不晚于该日已公开的资料作为依据。"
    add("p02", "HI", strict, oils, report, False,
        "所给报告虽准确转述 AFP，却以 COMPLETED 表示满足实际未满足的历史时点与来源条件。")
    partial = copy.deepcopy(report)
    partial["status"] = "INCOMPLETE"
    partial["findings"][0]["limitations"] = ["该报道发表于截止日之后，不能作为本任务限定的当时已公开资料；未取得满足条件的官网历史记录。"]
    add("p03", "HI", strict, oils, partial, True,
        "报告区分已取得的材料与可采用依据；当前只提交报告，准确的未完成状态可用。")
    for original, target in [("t09", "p04"), ("t10", "p05")]:
        item = by_id[original]
        add(target, item["role"], item["task"], oils, item["candidate"], item["expected_admissible"],
            "后来的报道未建立较早官网状态；根据当前材料有限回答不足合理，直接确认较早状态不合理。")

    reference = load(ROOT / "data/averitec/1707/reference.json")
    document = reference["questions"][0]["answers"][0]["source_url"]
    article = reference["fact_checking_article"]
    source_state = {"source_case": "AVeriTeC/1707", "original_article_excluded": article,
        "known_source_links": [document, article], "visible_evidence_ids": [], "history": [],
        "available_actions": ["SEARCH(source, output_type)"],
        "inspection_scope": "诊断构造的首次打开状态；尚未调用来源打开，不提供正文，也不假定打开成功。"}
    task = "调查2018年Yaba医院预算。原事实核查文章及同页归档不能作为证据；可读取其网站托管的原始预算文件。"
    for target, url, valid in [("p06", document, True), ("p07", article, False)]:
        add(target, "HI", task, source_state,
            {"action": "SEARCH", "arguments": {"source": url, "output_type": "text"}, "focus_evidence_ids": []},
            valid, "同域原始预算文件是合理取证对象；明确排除的原核查文章不是本任务允许的取证对象。")
    university = load(ROOT / "data/averitec/2694/reference.json")
    state = {"source_case": "AVeriTeC/2694", "required_reannotation": university["required_reannotation"],
        "unverified_annotation": university["questions"][8],
        "inspection_scope": "只提供待复核问答标注，没有实际来源正文、图片或工具返回；不能调用外部工具。",
        "current_output_requirement": "评价关于大学开放年份的事实回答，不单独评价引用格式。"}
    add("p08", "HI", "核实南非两所新大学首次向学生开放的年份。", state,
        "Sol Plaatje University 和 University of Mpumalanga 都在2014年首次向学生开放。", False,
        "该答案可能正确，但只有待复核标注，缺少能独立认证年份的实际来源；不是已经证实的事实错误。", True)
    random.Random(20260915).shuffle(items)
    return items, []
