"""Check fact-state qualifiers using saved source text and attribution diagnostics."""
import copy
import json
from pathlib import Path
import random
import sqlite3

ROOT = Path(__file__).resolve().parent


def examples():
    run = ROOT / "runs/averitec_2694_budget_plan_execution_r1"
    with sqlite3.connect((run / "store.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as connection:
        content = connection.execute("SELECT content FROM evidence WHERE id=13").fetchone()[0]
    start = content.index("The Master Agreement has been amended")
    end = content.index("Report on factual findings", start)
    excerpt = content[start:end]
    if "approved the release" not in excerpt:
        raise ValueError("The saved excerpt does not contain the approval statement")
    state = {
        "source_case": "AVeriTeC/2694/budget_question",
        "visible_evidence_ids": ["E13"],
        "evidence": {"evidence_id": "E13", "content": excerpt,
                     "source": "KPMG August 2017 agreed-upon procedures report, PDF page 4",
                     "source_char_range": [start, end]},
        "inspection_scope": "仅提供上述真实原文节选；没有银行到账或交易流水。",
        "runtime_constraints": {"remaining_turns": 0},
    }
    items = []

    def add(identifier, role, task, candidate, valid, reason):
        items.append({"id": identifier, "role": role, "task": task,
                      "state": copy.deepcopy(state), "candidate": candidate,
                      "expected_admissible": valid, "reference_reason": reason,
                      "origin": "diagnostic_edit_of_actual_evidence_not_executed_trajectory"})

    task = "根据给定原文，说明向两所大学直接拨款的数额及其事实状态；不要据此推定实际支出。"
    add("f01", "HI", task,
        "DHET批准向两校直接拨付R183,000,000用于资本项目，其中UMP为R100,000,000，SPU为R83,000,000。当前节选不能单独确认实际拨出、到账或支出。[E13]",
        True, "保留批准状态及按校金额，不越过原文证明实际交易。")
    add("f02", "HI", task,
        "背景部分另载DHET直接向两校拨付R183,000,000。因此不能把这些拨款直接加作实际建设支出。[E13]",
        False, "来自实际报告的限定遗漏：避免将拨款当支出并不能补回批准与实际拨付的状态差异；该表述须修复后才能作为准确状态监督。")
    add("f03", "HI", task,
        "DHET已经实际向两校拨付R183,000,000，且两校已经收到这笔款项；当前还不能证明学校如何支出。[E13]",
        False, "后半句的未知不能支持前半句未获证实的实际拨出及到账。")
    add("f04", "HI", task,
        "两校获准直接拨款约1.83亿兰特，分别为1亿和0.83亿。这里只确认批准，未核验实际到账。[E13]",
        True, "合法近似表达和不同措辞保留数额及批准状态，不要求复制参考句子。")
    add("f05", "GIA", "只依据给定节选，能否确认两校已经实际收到R183,000,000？",
        {"decision": "FINISH", "answer": "不能确认。节选记载批准直接拨款，尚未提供实际拨出或到账记录。[E13]"},
        True, "准确回答当前证据的证明范围，题目不要求继续检索。")
    add("f06", "GIA", "只依据给定节选，能否确认两校已经实际收到R183,000,000？",
        {"decision": "FINISH", "answer": "可以确认两校已收到，依据是DHET批准了R183,000,000的拨款。[E13]"},
        False, "批准不能直接证明后续到账。")

    previous = json.loads((ROOT / "rubric_calibration/text_r2/reference_labels.json").read_text(encoding="utf-8"))
    selected = {item["id"]: item for item in previous if item["id"] in {"t07", "t08", "t09", "t10"}}
    if len(selected) != 4:
        raise ValueError("Saved attribution and temporal-state diagnostic pairs are missing")
    for identifier in sorted(selected):
        item = copy.deepcopy(selected[identifier])
        item["reused_diagnostic"] = "rubric_calibration/text_r2/reference_labels.json:" + identifier
        item["id"] = "f" + identifier[1:]
        items.append(item)
    random.Random(20260914).shuffle(items)
    return items, []
