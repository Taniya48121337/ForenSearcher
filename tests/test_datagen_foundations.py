"""datagen 地基：错误清单、案例文件、证据扰动器。全部不需要模型。"""

import json
from pathlib import Path

import pytest

from foren_searcher.contracts import Finding
from foren_searcher.datagen import cases as cases_mod
from foren_searcher.datagen import perturb
from foren_searcher.datagen.errors import SPECS, ErrorRecord, ErrorType, Level, Role, Severity, error, tally, types_for
from foren_searcher.store import GlobalStore


# ------------------------------------------------------------------ errors

def test_every_error_type_has_a_spec_and_round_trips():
    assert set(SPECS) == set(ErrorType)
    record = error(ErrorType.WRONG_CITATION, "E4 讲的是另一场比赛", span="视频拍摄于 2022 年", evidence_ids=["E4"])
    assert record.severity is Severity.CRITICAL
    assert record.spec.level is Level.FINDING and record.spec.propagates
    assert ErrorRecord.from_dict(record.to_dict()) == record


def test_error_record_rejects_bad_input():
    with pytest.raises(ValueError):
        error(ErrorType.EMPTY_REPORT, "   ")
    with pytest.raises(ValueError):
        error(ErrorType.EMPTY_REPORT, "x", method="guess")


def test_tally_has_all_keys_and_role_filters_work():
    counts = tally([error(ErrorType.EMPTY_REPORT, "a"), error(ErrorType.EMPTY_REPORT, "b")])
    assert counts["empty_report"] == 2 and len(counts) == len(ErrorType)
    gia_planning = types_for(Role.GIA, level=Level.PLANNING)
    assert ErrorType.PREMATURE_FINISH in gia_planning and ErrorType.WRONG_TOOL not in gia_planning
    assert all(SPECS[t].exec_checkable for t in types_for(Role.HI, exec_checkable=True))


# ------------------------------------------------------------------ cases

def make_case(**overrides):
    base = dict(
        case_id="videosham:4126", dataset="videosham", source_id="4126", split="heldout", task_kind="video_diff",
        task_text="比较视频 A 与候选视频 B 的差异", family_key="videosham:src_4126",
        media=(cases_mod.Media("V0", "inputs/A.mp4", "video/mp4"), cases_mod.Media("V1", "inputs/B.mp4", "video/mp4")),
        required_evidence=(cases_mod.RequiredEvidence("edit_interval", "被编辑的时间区间及帧对比"),),
        checks=(cases_mod.Check("interval_iou", {"start": 12.0, "end": 15.5, "min_iou": 0.5}),
                cases_mod.Check("evidence_key_required", {"key": "edit_interval"})),
    )
    base.update(overrides)
    return cases_mod.Case(**base)


def test_case_round_trip_and_heldout_downgrade(tmp_path: Path):
    case = make_case()
    path = case.save(tmp_path / "case.json")
    loaded = cases_mod.Case.load(path)
    assert loaded == case and loaded.effective_split == "dev_eval"
    used = loaded.mark_used_for_revision("hi@0.1.1")
    assert used.effective_split == "dev_train" and used.used_for_revision == ("hi@0.1.1",)
    assert json.loads(path.read_text(encoding="utf-8"))["effective_split"] == "dev_eval"


@pytest.mark.parametrize("overrides, message", [
    ({"dataset": "VideoSham", "case_id": "VideoSham:4126"}, "小写"),
    ({"case_id": "videosham:9999"}, "必须等于"),
    ({"checks": (cases_mod.Check("evidence_key_required", {"key": "nope"}),)}, "不存在的 required_evidence"),
    ({"split": "train"}, "split"),
])
def test_case_validation_messages(overrides, message):
    with pytest.raises(cases_mod.CaseError, match=message):
        make_case(**overrides)


def test_check_requires_its_parameters():
    with pytest.raises(cases_mod.CaseError, match="缺少参数"):
        cases_mod.Check("interval_iou", {"start": 1})
    with pytest.raises(cases_mod.CaseError, match="未知"):
        cases_mod.Check("magic", {})


def test_split_leakage_detects_shared_family():
    dev = make_case(case_id="videosham:1", source_id="1", split="dev")
    held = make_case(case_id="videosham:2", source_id="2", split="heldout")
    assert cases_mod.check_split_leakage([dev, held])  # 同一个 family_key
    other = make_case(case_id="videosham:3", source_id="3", split="heldout", family_key="videosham:src_3")
    assert not cases_mod.check_split_leakage([dev, other])


# ------------------------------------------------------------------ perturb

def seed_store(root: Path) -> tuple[GlobalStore, dict[str, str]]:
    store = GlobalStore(root / "store.sqlite3", root / "artifacts")
    material = store.record_evidence(content="用户上传的视频", modalities=("video",), source_ref="V0",
                                     locator="file", acquired_by="HARNESS/ingest", discovered_by="HARNESS")
    page = store.record_evidence(content="某新闻页面显示视频发布于 2022-04-18。", source_ref="R1", locator="paragraph:3",
                                 acquired_by="SEARCH/open", discovered_by="W1-1")
    frame = store.record_evidence(content="从 V0 第 12 秒抽出的帧", modalities=("image",), derived_from=(material.evidence_id,),
                                  locator="t=12.0", acquired_by="INSPECT/get_frames", discovered_by="W1-1")
    ocr = store.record_evidence(content="帧内文字：KTVU", derived_from=(frame.evidence_id,), locator="bbox",
                                acquired_by="INSPECT/ocr", discovered_by="W1-1")
    return store, {"material": material.evidence_id, "page": page.evidence_id, "frame": frame.evidence_id, "ocr": ocr.evidence_id}


def test_snapshot_copies_store_and_leaves_original_untouched(tmp_path: Path):
    store, ids = seed_store(tmp_path / "orig")
    db, art = perturb.snapshot(store.db_path, store.artifact_root, tmp_path / "copy")
    copy = GlobalStore(db, art)
    assert [e.evidence_id for e in copy.evidence_records()] == [e.evidence_id for e in store.evidence_records()]
    with pytest.raises(perturb.PerturbationError, match="非空"):
        perturb.snapshot(store.db_path, store.artifact_root, tmp_path / "copy")


def test_mask_cascades_to_derived_evidence_and_records_expectation(tmp_path: Path):
    store, ids = seed_store(tmp_path / "orig")
    db, art = perturb.snapshot(store.db_path, store.artifact_root, tmp_path / "copy")
    record = perturb.mask(db, art, [ids["frame"]])
    assert record.target_ids == (ids["frame"],) and record.cascaded_ids == (ids["ocr"],)
    copy = GlobalStore(db, art)
    assert copy.evidence(ids["ocr"]).content == perturb.MASK_NOTICE
    assert copy.evidence(ids["page"]).content.startswith("某新闻页面")
    assert store.evidence(ids["ocr"]).content == "帧内文字：KTVU"  # 原库不动
    [expectation] = record.expected
    assert expectation.kind == "finding_must_not_cite" and set(expectation.params["evidence_ids"]) == {ids["frame"], ids["ocr"]}
    assert expectation.if_violated is ErrorType.UNSUPPORTED_ASSERTION
    assert copy.healthcheck().status == "ok"


def test_mask_refuses_initial_material_and_cited_evidence(tmp_path: Path):
    store, ids = seed_store(tmp_path / "orig")
    db, art = perturb.snapshot(store.db_path, store.artifact_root, tmp_path / "copy")
    with pytest.raises(perturb.PerturbationError, match="原始材料"):
        perturb.mask(db, art, [ids["material"]])
    copy = GlobalStore(db, art)
    copy.add_finding(Finding(finding_id="F1", workspace_id="W1-1", statement="视频 2022 年已存在", evidence_ids=(ids["page"],)))
    with pytest.raises(perturb.PerturbationError, match="已被 Finding 引用"):
        perturb.mask(db, art, [ids["page"]])
    perturb.mask(db, art, [ids["page"]], allow_cited=True)


def test_replace_and_inject_expectations(tmp_path: Path):
    store, ids = seed_store(tmp_path / "orig")
    db, art = perturb.snapshot(store.db_path, store.artifact_root, tmp_path / "copy")
    replaced = perturb.replace(db, art, ids["page"], "某新闻页面显示视频发布于 2026-09-01。")
    assert replaced.expected[0].kind == "answer_should_change"
    injected = perturb.inject(db, art, [{"content": "另一座桥梁在 2019 年维修的报道", "source_ref": "R9"}],
                              protected_ids=[ids["page"]])
    assert len(injected.target_ids) == 1 and injected.target_ids[0].startswith("E")
    kinds = {e.kind for e in injected.expected}
    assert kinds == {"finding_must_not_cite", "answer_unchanged", "citations_unchanged"}
    assert GlobalStore(db, art).healthcheck().status == "ok"
    with pytest.raises(perturb.PerturbationError):
        perturb.inject(db, art, [])


def test_evaluate_turns_outcomes_into_error_types(tmp_path: Path):
    store, ids = seed_store(tmp_path / "orig")
    db, art = perturb.snapshot(store.db_path, store.artifact_root, tmp_path / "copy")
    record = perturb.mask(db, art, [ids["page"]])
    bad_report = {"status": "COMPLETED", "findings": [{"statement": "2022 年已存在", "evidence_ids": [ids["page"]]}]}
    good_report = {"status": "INCOMPLETE", "findings": []}
    [bad] = perturb.evaluate(record, report=bad_report)
    [good] = perturb.evaluate(record, report=good_report)
    assert bad["ok"] is False and bad["error_type"] == "unsupported_assertion"
    assert good["ok"] is True and good["error_type"] is None
    [pending] = perturb.evaluate(record)
    assert pending["ok"] is None
