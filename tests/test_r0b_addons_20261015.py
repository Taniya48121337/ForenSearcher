"""10-15 对话 5''（r0b 前的追加 10 项）离线测试：理由与动作对象一致性、S3 退选不沿用理由、S4 程序校验、failed 页模板句、
检索类未读命中页 / 参考图并排的前置条件与第 2 轮、GIA 不追报告已写明匹配的缺口、hidden_context 词形、溯源类"未确定"归 held。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "experiments/sft_data"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from foren_searcher.datagen import walker as w  # noqa: E402
from foren_searcher.datagen.export import _provenance_unverified_vs_false, _token_in, hidden_knowledge  # noqa: E402
from foren_searcher.datagen.errors import ErrorType, Severity  # noqa: E402


# ---------------------------------------------------------------------------- 追加 1：理由与动作对象

def test_rationale_target_mismatch_and_equivalence():
    from oracle_generator import rationale_target_check

    bad = {"action": "READ_RECORD", "arguments": {"record_id": "E14"}, "focus_evidence_ids": [],
           "rationale": "Need to read the ASR transcript (E11) to compare the claim."}
    assert rationale_target_check(bad, "")["status"] == "mismatch"
    # 检索结果里 R5 的页面是 E12：读 E12 时理由说 R5 也算对
    ok = {"action": "READ_RECORD", "arguments": {"record_id": "E12"}, "rationale": "R5 (Reuters) may date the event."}
    shown = '"[R5] https://x.example/a; page=ready(E12), document=not_requested"'
    assert rationale_target_check(ok, shown)["status"] == "ok"
    # INSPECT 的对象是整段素材，理由引比对结果说明为什么取帧是正常的：不判
    inspect = {"action": "INSPECT", "arguments": {"targets": ["E2"], "request": {"operation": "get_frames"}},
               "rationale": "The compare_media result E26 lists spans; extract frames of B."}
    assert rationale_target_check(inspect, "")["status"] == "skip"
    dates = {"action": "SEARCH", "arguments": {"source": "R5", "output_type": "text", "include": ["dates"]},
             "rationale": "Need to verify R2 (Reuters) publication date."}
    assert rationale_target_check(dates, "")["status"] == "mismatch"


def test_s3_substitution_drops_s3_rationale():
    report = w.Decision("report", "E9_report", {"task_id": "W1-1", "status": "INCOMPLETE"})
    slot = w.Decision("next_slot", "E4_next_step", {}, slot={"name": "S3_next", "choices": ["read", "dates"],
                      "unread": [{"evidence_id": "E4"}], "undated": [{"candidate": "R5"}], "offsets": {},
                      "report_decision": report.to_dict()})
    got = w.resolve_slot_choice(slot, {"next": "read", "evidence_id": "E6", "rationale": "E6 is the earliest dated source."})
    assert got.output["arguments"]["record_id"] == "E4" and got.slot.get("substituted")
    assert "E6" not in got.output["rationale"]
    got = w.resolve_slot_choice(slot, {"next": "dates", "candidate": "R2", "rationale": "Check R2's date."})
    assert got.output["arguments"]["source"] == "R5" and got.slot.get("substituted") and "R2" not in got.output["rationale"]
    got = w.resolve_slot_choice(slot, {"next": "read", "evidence_id": "E4", "rationale": "Read E4, the earliest copy."})
    assert not got.slot.get("substituted") and got.output["rationale"] == "Read E4, the earliest copy."


# ---------------------------------------------------------------------------- 追加 3：S4 程序校验

def test_s4_contract_problems():
    from oracle_generator import s4_contract_problems

    kinds = {"E1": {"acquired_by": None, "image": True}, "E7": {"acquired_by": "SEARCH/text", "image": True},
             "E21": {"acquired_by": "INSPECT/compare_media", "image": False},
             "E30": {"acquired_by": "INSPECT/get_frames", "image": True}, "E31": {"acquired_by": "INSPECT/get_frames", "image": True}}
    report = {"findings": [
        {"statement": "The picture shows a railway settlement.", "evidence_ids": ["E1"],
         "limitations": ["The reference photographs are not available for a feature-by-feature comparison."]},
        {"statement": "A 2.0-4.0 s matches B 2.0-4.0 s.", "evidence_ids": ["E30", "E31"], "limitations": []},
        {"statement": "The frames show the same scene. Its original location could not be determined.", "evidence_ids": ["E30"]},
        {"statement": "The claim could not be verified.", "evidence_ids": ["E1"]},
    ]}
    problems = s4_contract_problems(report, kinds)
    assert any("finding 1 limitation" in p and "E7" in p for p in problems)
    assert any("finding 2 cites only sampled frames" in p and "E21" in p for p in problems)
    assert any("finding 3 statement contains" in p for p in problems)
    assert not any(p.startswith("finding 4") for p in problems)          # 对声明的判词不算


# ---------------------------------------------------------------------------- 追加 4：failed 页模板句

def _search_payload(items: list[str]) -> dict:
    return {"workspace_id": "W1-1", "available_actions": [{"name": "SEARCH"}],
            "model_view": {"task": "t", "task_id": "W1-1", "history": [{"action": "SEARCH", "request": {"input_type": "image"}}],
                           "latest_observation": {"action": "SEARCH", "items": items, "details": {"results": []}}}}


def test_failed_page_limitation():
    items = ["[R1] https://a.example/x; page=failed, document=not_requested", "[R2] https://b.example/y; page=ready(E4), document=x",
             "[R3] https://c.example/z; page=failed, document=not_requested"]
    state = w.HiState(_search_payload(items), {})
    assert w._failed_page_limitation(state) == ["2 of 3 candidate pages failed to load and were not examined; an earlier or "
                                                "other copy may exist among them."]
    state = w.HiState(_search_payload(items[1:2] + [items[0], items[1].replace("R2", "R4").replace("b.example", "d.example")]), {})
    assert w._failed_page_limitation(state)                              # 1 / 3：正好 1/3 也算
    state = w.HiState(_search_payload([items[1]]), {})
    assert w._failed_page_limitation(state) == []


# ---------------------------------------------------------------------------- 追加 2：未读命中页、GIA 可修缺口

def test_must_read_and_open_gap():
    pages = [{"evidence_id": "E12", "candidate": "R3", "url": "https://n.example/kheer-bhawani-mela-2017",
              "title": "Kashmiri Pandits visit Kheer Bhawani temple for mela", "snippet": None},
             {"evidence_id": "E13", "candidate": "R4", "url": "https://n.example/visual-origins", "title": "Find image sources"}]
    claim = "Mehbooba Mufti performed puja at Kheer Bhawani temple after the abrogation of Article 370 in Kashmir"
    must = w._must_read(pages, set(), claim)
    assert [p["evidence_id"] for p in must] == ["E12"]
    assert w._must_read(pages, {"E12"}, claim) == []
    early = w._must_read(pages, set(), "something else entirely here", dated={"R4": "2016-06-12"}, claim_date="2019-08-10")
    assert [p["evidence_id"] for p in early] == ["E13"]
    clause = w._open_gap_clause(must)
    assert clause.startswith("candidate pages that match the claim were not examined: https://n.example/kheer")
    gap = w._search_gap("dated copies were found but none was read; " + clause)
    assert gap and gap["kind"] == "unread_pages" and gap["urls"] == ["https://n.example/kheer-bhawani-mela-2017"]


def test_gia_skips_gap_resolved_in_findings():
    reported = [{"report": {"task_id": "W1-1", "status": "INCOMPLETE",
                            "status_reason": "audio of Material A and Material B 2.0-4.8 s: speech not transcribed",
                            "findings": [{"statement": "The audio of A and B matches over 8 of 8 sampled windows.",
                                          "evidence_ids": ["E24"]}]}}]
    gaps = w.classify_gaps(reported, {"A": 4.84, "B": 4.84})
    assert gaps and gaps[0]["kind"] == "resolved_in_findings" and not gaps[0]["fixable"]
    reported[0]["report"]["findings"][0]["statement"] = "The audio of A and B does not match."
    assert w.classify_gaps(reported, {"A": 4.84, "B": 4.84})[0]["kind"] != "resolved_in_findings"


def test_visual_only_task():
    assert w._visual_only("Determine every time interval during which the person in black is visible in B but not in A.")
    assert not w._visual_only("Compare A with B, including whether the speech or sound differs.")


# ---------------------------------------------------------------------------- 追加 6：词形

def test_token_word_forms():
    assert _token_in("Russian", "the title says Russia Beyond")
    assert _token_in("July", "(Last Update on Jul 8, 2026)")
    assert not _token_in("Kolkata", "dakshin kalkata")
    msgs = [{"role": "user", "content": "results E18-E20 West Bengal Lockdown"}, {"role": "assistant", "content": ""}]
    out = {"action": "SEARCH", "arguments": {}, "rationale": "Reading the results suggests a possible Kolkata origin. Running OCR next."}
    assert hidden_knowledge(msgs, {}, out) == ["Kolkata"]


# ---------------------------------------------------------------------------- 追加 9：溯源类"未确定" vs GT=false

def test_provenance_unverified_vs_false():
    err = SimpleNamespace(severity=Severity.MAJOR, type=ErrorType.OMITTED_REQUIRED_FACT, extra={"check": "label_expected"})
    unit = SimpleNamespace(errors=[err])
    case = SimpleNamespace(task_kind="provenance", checks=[SimpleNamespace(kind="label_expected", params={"label": "false"})])
    dec = SimpleNamespace(role="GIA", kind="answer", output={"answer": "Where the photo was first published could not be determined [E9]."})
    assert _provenance_unverified_vs_false(dec, unit, case)
    dec2 = SimpleNamespace(role="GIA", kind="answer", output={"answer": "The photo is from 2016 [E9]."})
    assert not _provenance_unverified_vs_false(dec2, unit, case)
    other = SimpleNamespace(severity=Severity.CRITICAL, type=ErrorType.UNSUPPORTED_ASSERTION, extra={})
    assert not _provenance_unverified_vs_false(dec, SimpleNamespace(errors=[err, other]), case)
    assert not _provenance_unverified_vs_false(dec, unit, SimpleNamespace(task_kind="claim_check", checks=case.checks))
