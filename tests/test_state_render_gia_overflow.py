"""GIA 视图最后一级退化（报告陈述收缩 / 旧报告折叠）与 export 的 RenderError 兜底。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from foren_searcher.state_render import TIERS, TRUNCATION_MARK, RenderError, payload_chars, render_payload

ROOT = Path(__file__).resolve().parents[1]
BUDGET = TIERS["compact_16k"].budget_chars  # type: ignore[union-attr]


def _interval_text(seed: int, n: int) -> str:
    parts = [f"[{seed + i * 0.033333:.6f},{seed + (i + 1) * 0.033333:.6f})秒" for i in range(n)]
    return "材料B可见黑衣人物而A对应位置未见，已确认的差异区间为" + "、".join(parts) + "。"


def _report(ws: str, first: int, n_findings: int, statement_len: int) -> dict[str, Any]:
    findings = []
    for k in range(n_findings):
        fid = f"F{first + k}"
        text = _interval_text(first + k, 60)[:statement_len]
        findings.append({"record_type": "finding", "finding_id": fid, "workspace_id": ws, "statement": text,
                         "evidence_ids": [f"E{first + k}", f"E{first + k + 100}"],
                         "limitations": ["时间按约1/30秒的帧呈现时长定位，" + "两个区间之间的连续性尚未证实。" * 20]})
    return {
        "task": f"核实 {ws} 负责的时间段内的差异边界。",
        "materials": ["E1", "E2"],
        "run_status": "reported",
        "report": {"record_type": "report", "task_id": ws, "status": "INCOMPLETE",
                   "finding_ids": [f["finding_id"] for f in findings], "selected_evidence_ids": ["E1"],
                   "status_reason": "尚不足以确定全部差异段。" * 30, "findings": findings},
        "supporting_records": [{"record_type": "evidence", "evidence_id": f"E{first + k}", "modalities": ["image"],
                                "label": f"视频E1在{k}.000秒的帧", "content": "帧元数据" * 50} for k in range(4)],
    }


def _payload(n_findings: int, statement_len: int, task: str = "找出材料A与B的全部差异区间。") -> dict[str, Any]:
    return {
        "available_actions": [],
        "memory_history": [],
        "global_view": {
            "task": task,
            "current_materials": [{"record_type": "evidence", "evidence_id": f"E{i}", "label": f"帧{i}"} for i in range(1, 30)],
            "latest_reports": [_report("W3-1", 100, n_findings, statement_len),
                               _report("W3-2", 200, n_findings, statement_len),
                               _report("W3-3", 300, n_findings, statement_len)],
            "runtime_constraints": {},
        },
    }


def _statements(view: dict[str, Any], i: int) -> list[str]:
    return [f["statement"] for f in view["latest_reports"][i]["report"]["findings"]]


def test_long_reports_shrink_statements_within_budget():
    payload = _payload(n_findings=8, statement_len=1500)
    out = render_payload(payload, "compact_16k")
    view = out["global_view"]
    assert payload_chars(out) <= BUDGET
    assert view["context_truncated"] is True
    info = view["reports_truncated"]
    assert info["statement_chars"] in (600, 400, 250, 150)
    # 陈述被截到该级别（外加截断标记），且带可读标记
    for i in range(3):
        for s in _statements(view, i):
            assert len(s) <= info["statement_chars"] + len(TRUNCATION_MARK) + 8
            assert "已截断" in s
    # 最新一份报告没有被折叠
    assert "W3-3" not in info["collapsed_reports"]
    assert all(len(s) > 80 + len(TRUNCATION_MARK) + 8 for s in _statements(view, 2))
    # 原 payload 不被改动
    assert len(payload["global_view"]["latest_reports"][0]["report"]["findings"][0]["statement"]) == 1500


def test_many_findings_collapse_earlier_reports_keep_latest():
    payload = _payload(n_findings=22, statement_len=1500)
    out = render_payload(payload, "compact_16k")
    view = out["global_view"]
    assert payload_chars(out) <= BUDGET
    info = view["reports_truncated"]
    assert info["statement_chars"] == 150
    assert info["collapsed_reports"] and info["collapsed_reports"][0] == "W3-1"
    assert "W3-3" not in info["collapsed_reports"]
    collapsed = view["latest_reports"][0]
    finding = collapsed["report"]["findings"][0]
    assert set(finding) == {"finding_id", "statement", "evidence_ids"}
    assert finding["statement"].startswith(_interval_text(100, 60)[:80]) and len(finding["statement"]) < 80 + len(TRUNCATION_MARK) + 8
    assert all(isinstance(r, str) and r.startswith("E") for r in collapsed["supporting_records"])
    # 最新一份：完整结构，陈述是 150 级而不是 80 字折叠
    latest = view["latest_reports"][2]["report"]["findings"]
    assert all("limitations" in f and f["statement"].startswith(_interval_text(300 + i, 60)[:150]) for i, f in enumerate(latest))


def test_small_payload_not_marked():
    out = render_payload(_payload(n_findings=2, statement_len=120), "compact_16k")
    assert "reports_truncated" not in out["global_view"]
    assert "context_truncated" not in out["global_view"]


def test_task_alone_over_budget_still_raises():
    payload = _payload(n_findings=2, statement_len=120, task="总任务" * (BUDGET // 2))
    with pytest.raises(RenderError) as info:
        render_payload(payload, "compact_16k")
    assert info.value.budget == BUDGET and info.value.chars is not None and info.value.chars > BUDGET


def test_legacy_untouched():
    payload = _payload(n_findings=8, statement_len=1500)
    assert render_payload(payload, "legacy") == payload


# ------------------------------------------------------------------ export 兜底

RUN = ROOT / "experiments" / "sft_data" / "runs" / "true_4124903_r1"
VERDICTS = ROOT / "experiments" / "sft_data" / "verdicts_r0_judged.jsonl"
CASES = ROOT / "experiments" / "sft_data" / "cases"


@pytest.mark.skipif(not (RUN / "store.sqlite3").is_file() or not VERDICTS.is_file(), reason="试点 run / 判定文件未随仓库入库")
def test_export_skips_render_overflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from foren_searcher.datagen import export
    from foren_searcher.datagen.cases import load_cases
    from foren_searcher.datagen.runlog import load_run
    from foren_searcher.datagen.specs import Spec
    from foren_searcher.datagen.verify.verdict import read_verdicts

    real = export.build_messages

    def fake(system: str, decision: Any, renderer: str) -> Any:
        if decision.role == "GIA":
            raise RenderError("超预算", chars=99_999, budget=11_000)
        return real(system, decision, renderer)

    monkeypatch.setattr(export, "build_messages", fake)
    log = load_run(RUN)
    overflow: list[dict[str, Any]] = []
    try:
        case = load_cases(CASES).get(log.case_id)
        reports = [v for v in read_verdicts(VERDICTS) if v.subject.get("run") == log.name]
        specs = {"hi": Spec.load(ROOT / "specs" / "hi@0.1.0"), "gia": Spec.load(ROOT / "specs" / "gia@0.1.0")}
        out = tmp_path / "export"
        rows, verdicts = export.export_run(log, case, reports, specs, renderer="compact_8k", out_dir=out, image_root=out / "images",
                                           render_overflow=overflow)
        n_gia = sum(1 for d in log.decisions if d.role == "GIA")
    finally:
        log.close()
    assert rows and all(r["metadata"]["role"] == "hi" for r in rows)
    assert len(overflow) == n_gia > 0
    assert set(overflow[0]) >= {"run", "request", "role", "kind", "chars", "budget"}
    assert overflow[0]["role"] == "gia" and overflow[0]["chars"] == 99_999 and overflow[0]["budget"] == 11_000
    manifest = export.write_outputs(rows, verdicts, out_dir=out, specs=specs, renderer="compact_8k", verdicts_source=str(VERDICTS),
                                    cases_dir=str(CASES), render_overflow=overflow)
    assert manifest["render_overflow"] == overflow
    assert manifest["counts"]["gia_accepted"] == manifest["counts"]["gia_held"] == manifest["counts"]["gia_rejected"] == 0
    assert sum(manifest["counts"].values()) == len(rows)
