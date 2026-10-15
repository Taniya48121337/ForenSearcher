"""出错点重答（bridge.py）：假教师 + 假核验模型，在真实 r0e run（true_20361105_r0e_1，答案因标签与无据断言被拒）上端到端。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from foren_searcher.datagen import bridge as B
from foren_searcher.datagen.cases import load_cases
from foren_searcher.datagen.specs import Spec
from foren_searcher.datagen.verify.verdict import read_verdicts

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "sft_data" / "runs" / "true_20361105_r0e_1"
FIX = Path(__file__).parent / "fixtures" / "bridge_20261005"
CASES = ROOT / "experiments" / "sft_data" / "cases"
needs_run = pytest.mark.skipif(not (RUN / "store.sqlite3").is_file() or not (RUN / "requests").is_dir(), reason="r0e run 未随仓库入库")

GOOD = {"decision": "FINISH", "rationale": "转写与两篇报道一致，结论限定在片段范围内。",
        "answer": "部分属实。视频转写显示她先说在网上嘲弄国会议员的人应该被起诉。[E3] 两篇报道把这段话归于 Frederica Wilson。[E4][E8]"}


class FakeJudge:
    """pair：文本来源支持句首（引语取自来源原文）；图片来源支持；汇总：全覆盖；标签：答案含"部分属实"算命中。"""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def judge(self, *, system_prompt: str, payload: dict[str, Any], images: list, schema: dict | None = None) -> dict[str, Any]:
        props = set((schema or {}).get("properties") or {})
        self.calls.append(",".join(sorted(props)))
        if "supported_parts" in props:
            content = str((payload.get("source") or {}).get("content") or "")
            part = str(payload.get("statement") or "")[:20]
            parts = [{"part": part, "quote": content[:40]}] if content.strip() and not images else [{"part": part, "quote": "画面"}]
            return {"supported_parts": parts, "contradicted_parts": [], "source_reports_claim": False, "statement_asserts_fact": True,
                    "scope_overreach": False, "notes": "fake"}
        if "unsupported_parts" in props:
            return {"unsupported_parts": [], "grounded": True, "notes": "fake"}
        if "status" in props and "quote" in props:          # 标签语义判定
            answer = str(payload.get("answer") or "")
            return {"status": "contradicts" if "核心说法属实" in answer else "matches", "quote": answer[:6], "notes": ""}
        if "requirements" in props:
            return {"requirements": [{"id": r["id"], "covered": True, "by": [], "note": ""} for r in payload.get("requirements") or []]}
        raise AssertionError(f"没想到的调用：{sorted(props)}")


def runner_returning(output: dict[str, Any] | None, calls: list[dict[str, Any]]):
    def run(request: dict[str, Any], images: list[Path], directory: Path):
        calls.append({"request": request, "images": images, "dir": directory})
        if output is None:
            return None, {"returncode": 1}
        return json.dumps(output, ensure_ascii=False), {"returncode": 0}
    return run


def _units() -> list[dict[str, Any]]:
    return [json.loads(line) for line in (FIX / "verdicts_units.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def test_select_targets_only_negative_answers_with_existing_runs(tmp_path: Path) -> None:
    rows = [{"unit": "answer", "disposition": "negative", "subject": {"run": "a", "request": "request_0009", "case_id": "x"},
             "errors": [{"type": "unsupported_assertion", "severity": "critical"}, {"type": "x", "severity": "minor"}]},
            {"unit": "answer", "disposition": "positive", "subject": {"run": "a", "request": "request_0010"}},
            {"unit": "plan", "disposition": "negative", "subject": {"run": "a", "request": "request_0001"}},
            {"unit": "answer", "disposition": "negative", "subject": {"run": "gone", "request": "request_0003"}}]
    (tmp_path / "a" / "requests").mkdir(parents=True)
    targets = B.select_targets(rows, tmp_path)
    assert [(t.run, t.request, t.errors) for t in targets] == [("a", "request_0009", ["unsupported_assertion"])]


@needs_run
def test_bridge_end_to_end_on_real_run(tmp_path: Path) -> None:
    targets = B.select_targets(_units(), RUN.parent, runs={RUN.name})
    assert [t.request for t in targets] == ["request_0008"]
    calls: list[dict[str, Any]] = []
    spec = Spec.load(ROOT / "specs" / "gia@0.6.0")
    case = load_cases(CASES).get("true:20361105")
    out = asyncio.run(B.bridge_run(RUN, targets, case, read_verdicts(FIX / "verdicts_reports.jsonl"), FakeJudge(),
                                   runner_returning(GOOD, calls), spec, out_dir=tmp_path, attempts=2, renderer="compact_16k",
                                   judge_name="fake", say=lambda _: None))
    # 教师看到同一 system / 附图，payload 是按导出档位渲染后的学生那份（10-05：教师输入 = 学生输入）
    original = json.loads((RUN / "requests" / "request_0008" / "request.json").read_text(encoding="utf-8"))
    assert len(calls) == 1 and len(calls[0]["images"]) == 3
    shown = calls[0]["request"]
    assert shown["system_prompt"] == original["system_prompt"] and shown["image_evidence_ids"] == original["image_evidence_ids"]
    assert shown["payload"] != original["payload"]
    assert shown["payload"]["global_view"]["runtime_constraints"]["state_renderer"] == "compact_16k"
    expected, info = B.teacher_request(original, "compact_16k")
    assert shown == expected and info and info["renderer"] == "compact_16k" and info["rendered_chars"] < info["raw_chars"]
    view = json.loads((tmp_path / "requests" / RUN.name / "request_0008_b1" / "teacher_view.json").read_text(encoding="utf-8"))
    assert view["state_render"]["renderer"] == "compact_16k"
    (rec,) = out
    assert rec["status"] == "ok" and rec["disposition"] == "positive", rec.get("errors")
    row, pair = rec["row"], rec["pair"]
    assert row["metadata"]["origin"] == "bridge" and row["metadata"]["bucket"] == "accepted" and row["metadata"]["spec"] == "gia@0.6.0"
    assert json.loads(row["messages"][-1]["content"])["answer"] == GOOD["answer"]
    assert pair["messages"] == row["messages"][:-1] and json.loads(pair["chosen"]) == GOOD
    assert json.loads(pair["rejected"])["answer"].startswith("核心说法属实")
    assert all((tmp_path / p).is_file() for p in row["images"])
    text = B.write_outputs(out, tmp_path, round_tag="t")
    assert "重答后通过：1 个" in text
    assert len((tmp_path / "gia_preference_pairs.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    # 再跑一次：已有 response.json，不再调教师
    again: list[dict[str, Any]] = []
    asyncio.run(B.bridge_run(RUN, targets, case, read_verdicts(FIX / "verdicts_reports.jsonl"), FakeJudge(),
                             runner_returning(GOOD, again), spec, out_dir=tmp_path, attempts=2, renderer="compact_16k",
                             judge_name="fake", say=lambda _: None))
    assert again == []


@needs_run
def test_bridge_records_failures_without_data(tmp_path: Path) -> None:
    targets = B.select_targets(_units(), RUN.parent, runs={RUN.name})
    spec = Spec.load(ROOT / "specs" / "gia@0.6.0")
    case = load_cases(CASES).get("true:20361105")
    reports = read_verdicts(FIX / "verdicts_reports.jsonl")
    # 教师改成继续调查：不是 FINISH，两次都记 not_finish
    calls: list[dict[str, Any]] = []
    out = asyncio.run(B.bridge_run(RUN, targets, case, reports, FakeJudge(), runner_returning({"decision": "INVESTIGATE", "tasks": []}, calls),
                                   spec, out_dir=tmp_path / "a", attempts=2, renderer="compact_16k", judge_name="fake", say=lambda _: None))
    assert [r["status"] for r in out] == ["not_finish", "not_finish"] and not any(r.get("row") for r in out)
    # 重答仍与案卷标签相反：判 negative，不出偏好对
    bad = {**GOOD, "answer": "核心说法属实。[E3]"}
    out = asyncio.run(B.bridge_run(RUN, targets, case, reports, FakeJudge(), runner_returning(bad, []), spec, out_dir=tmp_path / "b",
                                   attempts=1, renderer="compact_16k", judge_name="fake", say=lambda _: None))
    assert out[0]["disposition"] == "negative" and out[0]["pair"] is None and out[0]["row"]["metadata"]["bucket"] == "rejected"
    # 教师调用失败
    out = asyncio.run(B.bridge_run(RUN, targets, case, reports, FakeJudge(), runner_returning(None, []), spec, out_dir=tmp_path / "c",
                                   attempts=1, renderer="compact_16k", judge_name="fake", say=lambda _: None))
    assert out[0]["status"] == "teacher_failed"
