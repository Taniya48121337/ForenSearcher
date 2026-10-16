"""10-16 r0g 收尾：核验后处理把充分性层判 completed_without_finishing 的报告改为 INCOMPLETE（datagen.status_rewrite）。"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from foren_searcher.datagen import status_rewrite as sr  # noqa: E402

REPORT = {"task_id": "W1-1", "status": "COMPLETED", "findings": [{"statement": "The Diplomat page is dated 2019-11-05.",
                                                                    "evidence_ids": ["E4"], "limitations": []}]}


def test_targets_only_severe_cwf():
    rows = [{"unit": "report", "subject": {"run": "r", "workspace_id": "W1-1"},
             "errors": [{"type": "completed_without_finishing", "severity": "critical", "detail": "x"}], "sufficiency": {}},
            {"unit": "report", "subject": {"run": "r", "workspace_id": "W2-1"},
             "errors": [{"type": "completed_without_finishing", "severity": "minor"}]},
            {"unit": "answer", "subject": {"run": "r"}, "errors": [{"type": "completed_without_finishing", "severity": "critical"}]}]
    assert [t["workspace"] for t in sr.targets(rows)] == ["W1-1"]


def test_reason_drops_gt_items_and_maps_chinese_keys():
    visible = json.dumps({"view": "The Diplomat page; candidate pages", "report": REPORT})
    suff = {"missing": ["No visual comparison confirming that the Diplomat image is the same as the supplied image.",
                        "The earliest source (Getty, 2013-12-12) was not found."]}
    got = sr.build_reason(suff, [], visible)
    assert got["reason"] == "No visual comparison confirming that the Diplomat image is the same as the supplied image"
    assert got["dropped"][0]["tokens"] == ["Getty", "2013"]
    # missing 全是中文：取未覆盖要求对应的案卷 text_en
    zh = {"missing": ["对应片段在 A、B 两侧成对的起止时间区间。"],
          "requirements": [{"key": "对应片段在 A、B 两侧的时间区间", "covered": False}]}
    reqs = [{"text": "A 与 B 中内容对应（相同或经变换）的片段，以及两侧各自的时间范围", "text_en": "Segments of A and B that correspond"},
            {"text": "未能确定的内容", "text_en": "What could not be determined"}]
    got = sr.build_reason(zh, [], "{}", reqs)
    assert got["reason"] == "not established: segments of A and B that correspond" and got["source"] == "requirements"


def _make_run(tmp: Path) -> Path:
    run = tmp / "x_r0g_1"
    for rid, payload in (("request_0001", {"workspace_id": "W1-1", "model_view": {"task": "t"}}),
                         ("request_0002", {"global_view": {"latest_reports": [{"report": {**{k: v for k, v in REPORT.items()
                                                                                              if k != "findings"},
                                                                                           "finding_ids": ["F1"]}}]}})):
        (run / "requests" / rid).mkdir(parents=True)
        (run / "requests" / rid / "request.json").write_text(json.dumps({"payload": payload, "rendered_payload": payload}),
                                                             encoding="utf-8")
    (run / "requests" / "request_0001" / "response.json").write_text(json.dumps(REPORT), encoding="utf-8")
    links = [{"request_id": "request_0001", "raw_output": REPORT}, {"request_id": "request_0002", "raw_output": {"decision": "FINISH"}}]
    (run / "execution_links.json").write_text(json.dumps(links), encoding="utf-8")
    (run / "trajectory.jsonl").write_text(json.dumps({"payload": REPORT}) + "\n", encoding="utf-8")
    (run / "run_config.json").write_text("{}", encoding="utf-8")
    con = sqlite3.connect(run / "store.sqlite3")
    con.execute("CREATE TABLE reports (task_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL)")
    con.execute("CREATE TABLE rounds (round_index INTEGER PRIMARY KEY, gia_input_json TEXT, gia_decision_json TEXT)")
    con.execute("CREATE TABLE trajectory_events (id INTEGER PRIMARY KEY, payload_json TEXT)")
    con.execute("INSERT INTO reports VALUES ('W1-1', ?)", (json.dumps(REPORT),))
    con.commit()
    con.close()
    return run


def test_rewrite_everywhere_and_finish_marks(tmp_path, monkeypatch):
    run = _make_run(tmp_path)
    monkeypatch.setattr(sr, "walker_gaps", lambda *a: {"fixable": True, "unread_hit_pages": [{"evidence_id": "E6"}]})
    rows = [{"unit": "report", "subject": {"run": run.name, "workspace_id": "W1-1"},
             "errors": [{"type": "completed_without_finishing", "severity": "critical", "detail": "d"}],
             "sufficiency": {"missing": ["The first publication date was not established."]}}]
    done = sr.rewrite_round(rows, tmp_path)
    assert len(done) == 1 and done[0]["finish_after"] == ["request_0002"]
    reason = "The first publication date was not established"
    links = json.loads((run / "execution_links.json").read_text(encoding="utf-8"))
    assert links[0]["raw_output"]["status"] == "INCOMPLETE" and links[0]["raw_output"]["status_reason"] == reason
    gia = json.loads((run / "requests" / "request_0002" / "request.json").read_text(encoding="utf-8"))
    for key in ("payload", "rendered_payload"):
        assert gia[key]["global_view"]["latest_reports"][0]["report"]["status"] == "INCOMPLETE"
    con = sqlite3.connect(run / "store.sqlite3")
    assert json.loads(con.execute("SELECT payload_json FROM reports").fetchone()[0])["status"] == "INCOMPLETE"
    con.close()
    assert (run / "execution_links.json.pre_status_rewrite").is_file()
    assert sr.finish_after_marks(run) == {"request_0002": {"workspace": "W1-1", "report_request": "request_0001"}}
    assert sr.rewritten_requests(run) == {"request_0001": "W1-1"}
    assert sr.rewrite_round(rows, tmp_path) == []          # 改过的不再改


def test_gate_b6_lists_verifier_rewrite_separately():
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    import export_gate

    def row(case: str, status: str, rewritten: bool = False) -> dict:
        meta = {"unit": "report", "bucket": "accepted", "case_id": case, "run": case, "incomplete_reason_checks": [],
                "status_rewritten": rewritten}
        return {"messages": [{"role": "assistant", "content": json.dumps({"status": status, "findings": [{"statement": "x"}]})}],
                "metadata": meta}

    rows = [row("fmnv:a", "COMPLETED"), row("fmnv:b", "COMPLETED"), row("fmnv:c", "COMPLETED"), row("vcsl:d", "INCOMPLETE"),
            row("averimatec:e1", "INCOMPLETE", True), row("averimatec:e2", "INCOMPLETE", True)]
    got = export_gate.check_b6(rows, set())
    assert got["status"] == "通过"
    assert got["value"]["incomplete_rate_accepted"] == 0.25
    assert got["value"]["verifier_rewrite_accepted_answerable"] == 2
    assert got["value"]["incomplete_rate_accepted_with_rewrite"] == 0.5
    assert got["value"]["by_family_accepted_answerable"]["E"] == {"rated": "0/0", "verifier_rewrite": 2, "with_rewrite": "2/2"}
