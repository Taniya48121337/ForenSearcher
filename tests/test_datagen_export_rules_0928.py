"""09-28 导出规则：核查站点沾染标记、记忆动作/重复动作降权、报告状态权重、rationale 剥离。"""

from __future__ import annotations

import json

from foren_searcher.datagen.export import adjust_action_weights, fact_check_exposure, report_weight
from foren_searcher.datagen.verify.verdict import Verdict
from foren_searcher.local_models import strip_rationale


def _row(run, ws, turn, action, weight, gain=1):
    return {"messages": [], "images": [], "metadata": {"role": "hi", "unit": "step", "bucket": "accepted", "run": run,
                                                       "workspace": ws, "turn": turn, "action": action,
                                                       "sample_weight": weight, "clue_gain": gain}}


def test_memory_action_without_gain_is_halved_and_long_repeats_keep_only_ends() -> None:
    rows = [_row("r", "W1-1", 1, "READ_RECORD", 1.0, gain=0), _row("r", "W1-1", 2, "READ_RECORD", 2.0, gain=1)]
    rows += [_row("r", "W1-2", t, "INSPECT", 2.0) for t in range(1, 6)]           # 5 连 INSPECT
    rows += [_row("r", "W1-3", t, "INSPECT", 2.0) for t in range(1, 4)]           # 3 连：不动
    out = adjust_action_weights(rows)
    w = {(m["metadata"]["workspace"], m["metadata"]["turn"]): m["metadata"]["sample_weight"] for m in out}
    assert w[("W1-1", 1)] == 0.5 and w[("W1-1", 2)] == 2.0
    assert [w[("W1-2", t)] for t in range(1, 6)] == [2.0, 1.0, 1.0, 1.0, 2.0]
    assert all(w[("W1-3", t)] == 2.0 for t in range(1, 4))
    assert out[0]["metadata"]["weight_adjustments"] == ["memory_action_gain0"]


def test_fact_check_exposure_distinguishes_context_from_target() -> None:
    context = [{"role": "user", "content": json.dumps({
        "candidate_sources": ["R3 page: — X | Snopes.com — https://www.snopes.com/fact-check/x/ — ..."],
        "items": ["[R3] https://www.snopes.com/fact-check/x/; page=ready(E11), document=not_requested",
                  "[R4] https://nydailynews.com/y; page=ready(E12)"]})}]
    read_fc = fact_check_exposure(context + [{"role": "assistant", "content": "x"}],
                                  {"action": "READ_RECORD", "arguments": {"record_id": "E11"}, "focus_evidence_ids": []})
    assert read_fc == {"in_context": ["snopes.com"], "in_target": ["snopes.com"]}
    read_ok = fact_check_exposure(context + [{"role": "assistant", "content": "x"}],
                                  {"action": "READ_RECORD", "arguments": {"record_id": "E12"}, "focus_evidence_ids": []})
    assert read_ok == {"in_context": ["snopes.com"], "in_target": []}
    open_fc = fact_check_exposure([{"role": "user", "content": "{}"}, {"role": "assistant", "content": "x"}],
                                  {"action": "SEARCH", "arguments": {"source": "https://www.politifact.com/a", "output_type": "text"}})
    assert open_fc["in_target"] == ["politifact.com"] and open_fc["in_context"] == []
    report = fact_check_exposure(context + [{"role": "assistant", "content": "x"}],
                                 {"task_id": "W1-1", "status": "COMPLETED", "findings": [{"statement": "s", "evidence_ids": ["E11"], "limitations": []}],
                                  "selected_evidence_ids": ["E11"]})
    assert report["in_target"] == ["snopes.com"]


class _D:
    def __init__(self, output):
        self.output = output


def _verdict(errors):
    v = Verdict.__new__(Verdict)
    v.errors = errors
    return v


def test_report_weight_prefers_completed_and_specific_incomplete() -> None:
    assert report_weight(_D({"status": "COMPLETED"}), _verdict([])) == 3.0
    assert report_weight(_D({"status": "COMPLETED"}), _verdict([1])) == 2.0
    assert report_weight(_D({"status": "INCOMPLETE", "status_reason": "还需要更多资料"}), _verdict([])) == 1.0
    assert report_weight(_D({"status": "INCOMPLETE", "status_reason": "缺少原始视频的上传页面：E7 只是转载，未能取得 YouTube 原始记录的发布日期"}), _verdict([])) == 2.0


def test_strip_rationale_only_removes_that_key() -> None:
    assert strip_rationale({"decision": "FINISH", "answer": "a", "rationale": "r"}) == {"decision": "FINISH", "answer": "a"}
    assert strip_rationale({"action": "SEARCH"}) == {"action": "SEARCH"}
