"""step / plan / answer 单元、规则校准、hindsight、export 的测试（含对 true_4124903_r1 的真实 run 冒烟）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foren_searcher.datagen.cases import Case, RequiredEvidence, Requirement, load_cases
from foren_searcher.datagen.errors import ErrorType, Severity, error
from foren_searcher.datagen.verify.calibrate import recalibrate
from foren_searcher.datagen.verify.hindsight import locate_quote
from foren_searcher.datagen.verify.verdict import Verdict, tainted

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "sft_data" / "runs" / "true_4124903_r1"
VERDICTS = ROOT / "experiments" / "sft_data" / "verdicts_r0_judged.jsonl"
CASES = ROOT / "experiments" / "sft_data" / "cases"
needs_run = pytest.mark.skipif(not (RUN / "store.sqlite3").is_file() or not VERDICTS.is_file(), reason="试点 run / 判定文件未随仓库入库")


# ------------------------------------------------------------------ 案卷扩展

def test_required_evidence_matchers_and_requirements():
    req = RequiredEvidence(key="tweets", description="警方推文", url_pattern="*prefpolice/status/7537*", supports=("REQ1",))
    assert req.matches(source_url="https://x.com/prefpolice/status/753719036245073920", acquired_by=None)
    assert req.matches(source_url="https://twitter.com/prefpolice/status/753708496382812160", acquired_by=None)
    assert not req.matches(source_url="https://x.com/someone/status/1", acquired_by=None)
    frames = RequiredEvidence(key="frames", description="帧", acquired_by="INSPECT/get_frames", supports=("REQ2",))
    assert frames.matches(source_url=None, acquired_by="INSPECT/get_frames") and not frames.matches(source_url=None, acquired_by="SEARCH/open")
    yt = RequiredEvidence(key="yt", description="y", url_pattern="youtube.com/watch[?]v=abc*")
    assert yt.matches(source_url="https://www.youtube.com/watch?v=abc", acquired_by=None)
    case = Case(case_id="true:1", dataset="true", source_id="1", split="dev", task_kind="claim_check", task_text="t", family_key="true:1",
                requirements=(Requirement(id="REQ1", text="a"), Requirement(id="REQ2", text="b", must_cite=False)),
                required_evidence=(req, frames))
    assert case.must_cite_requirements == ("REQ1",)
    with pytest.raises(Exception):
        Case(case_id="true:1", dataset="true", source_id="1", split="dev", task_kind="claim_check", task_text="t", family_key="true:1",
             required_evidence=(RequiredEvidence(key="x", description="d", supports=("REQ9",)),))


def test_case_file_round_trip(tmp_path: Path):
    src = CASES / "true_4124903.json"
    if not src.is_file():
        pytest.skip("案卷未入库")
    case = Case.load(src)
    assert case.case_id == "true:4124903" and case.must_cite_requirements == ("REQ1", "REQ2")
    case.save(tmp_path / "c.json")
    again = Case.load(tmp_path / "c.json")
    assert again.to_dict() == case.to_dict()
    assert "true:4124903" in load_cases(CASES)


# ------------------------------------------------------------------ 传播与校准

def test_minor_errors_do_not_taint():
    minor = error(ErrorType.SCOPE_OVERREACH, "x", severity=Severity.MINOR, extra={"finding": "F5"})
    major = error(ErrorType.SCOPE_OVERREACH, "x", extra={"finding": "F3"})
    critical = error(ErrorType.UNSUPPORTED_ASSERTION, "x", extra={"finding": "F7"})
    assert tainted([minor]) == [] and tainted([major]) == ["F3"] and tainted([minor, critical]) == ["F7"]


def _verdict(errors, judgments, coverage) -> Verdict:
    return Verdict(subject={"workspace_id": "W"}, errors=errors, judgments=judgments, coverage=coverage, disposition="negative")


def test_calibration_r1_multi_evidence_scope_dismissed():
    scope = error(ErrorType.SCOPE_OVERREACH, "由局部推整体", severity=Severity.MINOR, span="材料 A 与场景相符", evidence_ids=["E8"], method="model", extra={"finding": "F3"})
    judgments = [{"finding_id": "F3", "evidence_id": e, "verdict": "supported", "contradicted": []} for e in ("E3", "E7", "E8")]
    coverage = [{"finding_id": "F3", "grounded": True, "unsupported_parts": []}]
    new = recalibrate(_verdict([scope], judgments, coverage))
    assert new.errors == [] and new.disposition == "positive"
    assert new.extra["calibration"]["applied"][0]["rule"] == "R1_multi_evidence_scope"
    # 只引 2 条 E：不豁免
    new2 = recalibrate(_verdict([scope], judgments[:2], coverage))
    assert [e.type for e in new2.errors] == [ErrorType.SCOPE_OVERREACH]


def test_calibration_r2_relational_clause_downgrade():
    unsupported = error(ErrorType.UNSUPPORTED_ASSERTION, "所引 E 均未支持", span="这与报道描述相容", evidence_ids=["E3"], method="model", extra={"finding": "F7"})
    judgments = [{"finding_id": "F7", "evidence_id": "E3", "verdict": "supported", "is_image": True},
                 {"finding_id": "F5", "evidence_id": "E24", "verdict": "supported", "is_image": False}]
    new = recalibrate(_verdict([unsupported], judgments, []))
    assert [e.type for e in new.errors] == [ErrorType.INCOMPLETE_CITATION_SET]
    assert new.errors[0].severity is Severity.MAJOR and new.disposition == "positive" and new.tainted_findings == []
    # 长句 / 无关系词：不降级
    long = error(ErrorType.UNSUPPORTED_ASSERTION, "x", span="视频拍摄于 2016 年 7 月 14 日晚间的巴黎耶拿桥附近并由目击者上传", method="model", extra={"finding": "F7"})
    assert [e.type for e in recalibrate(_verdict([long], judgments, [])).errors] == [ErrorType.UNSUPPORTED_ASSERTION]


def test_locate_quote_normalizes_whitespace_and_quotes():
    content = "[21] AFP geolocated the graveyard using Google Street View imagery\n\n[22] An AFP journalist visited"
    assert 3 <= locate_quote(content, "AFP geolocated the graveyard  using Google Street View") <= 6   # 去空白后按比例换算，近似坐标
    assert locate_quote(content, "nothing like this") is None


# ------------------------------------------------------------------ 真实 run：单元核验 + 导出

@needs_run
def test_units_and_export_on_true_4124903(tmp_path: Path):
    from foren_searcher.datagen.export import export_run, write_outputs
    from foren_searcher.datagen.runlog import load_run
    from foren_searcher.datagen.specs import Spec
    from foren_searcher.datagen.verify.verdict import read_verdicts

    log = load_run(RUN)
    try:
        case = load_cases(CASES).get(log.case_id)
        reports = [v for v in read_verdicts(VERDICTS) if v.subject.get("run") == log.name]
        assert len(reports) == 2 and log.gia_is_model()
        specs = {"hi": Spec.load(ROOT / "specs" / "hi@0.1.0"), "gia": Spec.load(ROOT / "specs" / "gia@0.1.0")}
        out = tmp_path / "export"
        rows, verdicts = export_run(log, case, reports, specs, renderer="compact_8k", out_dir=out, image_root=out / "images")
    finally:
        log.close()
    by = {(r["metadata"]["role"], r["metadata"]["unit"], r["metadata"]["workspace"], r["metadata"]["turn"]): r["metadata"] for r in rows}
    # 校准后：W1-1 仍 negative（completed_without_finishing），W1-2 转 positive（关系句降级）
    reports_after = {v.subject["workspace_id"]: v for v in verdicts if v.unit == "report"}
    assert reports_after["W1-1"].disposition == "negative" and reports_after["W1-2"].disposition == "positive"
    assert reports_after["W1-1"].tainted_findings == [] and reports_after["W1-2"].tainted_findings == []
    # 步级：抽帧步收益最高、封顶权重 3；负工作区里 gain=0 的步 held
    assert by[("hi", "step", "W1-1", 1)]["clue_gain"] >= 6 and by[("hi", "step", "W1-1", 1)]["sample_weight"] == 3.0
    assert by[("hi", "step", "W1-1", 6)]["bucket"] == "held" and by[("hi", "step", "W1-1", 11)]["bucket"] == "held"
    assert by[("hi", "step", "W1-2", 2)]["bucket"] == "accepted" and by[("hi", "step", "W1-2", 2)]["sample_weight"] == 1.0
    assert by[("hi", "report", "W1-1", 12)]["bucket"] == "rejected"
    # 09-28 导出规则：W1-2 报告引用了 Snopes 的 E（教师自己读 / 引核查站点）→ accepted 降为 held
    assert by[("hi", "report", "W1-2", 8)]["bucket"] == "held"
    assert by[("hi", "report", "W1-2", 8)]["bucket_reason"].startswith("fact_check_target:snopes.com")
    # GIA：派工 positive（只有 overlapping minor），FINISH positive 且纠正了 HI 的过度 COMPLETED
    plan = by[("gia", "plan", None, 0)]
    answer = by[("gia", "answer", None, 0)]
    assert plan["bucket"] == "accepted" and plan["error_types"] == ["overlapping_sibling_tasks"] and set(plan["round_yield"]) == {"REQ1", "REQ2", "REQ3"}
    assert answer["bucket"] == "accepted" and answer["tainted_input"] is False and answer["recovered_from_tainted"] is True and answer["sample_weight"] == 2.0
    # 行形态：最后一条是 assistant 紧凑 JSON；图片占位符数 == images 数；system 来自规格
    for row in rows:
        assert row["messages"][-1]["role"] == "assistant" and json.loads(row["messages"][-1]["content"])
        placeholders = sum(1 for m in row["messages"] for p in (m["content"] if isinstance(m["content"], list) else []) if p.get("type") == "image")
        assert placeholders == len(row["images"]) and all((out / p).is_file() for p in row["images"])
        assert row["metadata"]["system_source"] == "spec_student_prompt" and row["metadata"]["spec_hash"].startswith("sha256:")
    manifest = write_outputs(rows, verdicts, out_dir=out, specs=specs, renderer="compact_8k", verdicts_source=str(VERDICTS), cases_dir=str(CASES))
    # 09-28 导出规则：原 17 行 accepted 里有 3 行（W1-1 t3、W1-2 t5 两步 + W1-2 报告）是教师自己打开 / 读 / 引 Snopes，降为 held
    assert manifest["counts"]["hi_accepted"] == 14 and manifest["counts"]["gia_accepted"] == 2
    assert manifest["counts"]["hi_held"] == 5 and manifest["counts"]["hi_rejected"] == 1
    assert (out / "verdicts_units.jsonl").is_file() and (out / "manifest.json").is_file()
