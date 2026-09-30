"""案卷 checks 按要求（REQ）收窄：Check.requirement 序列化、exec_checks 按工作区范围跳过、answer 单元全查。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from foren_searcher.datagen.cases import Case, CaseError, Check, Requirement
from foren_searcher.datagen.errors import ErrorType
from foren_searcher.datagen.runlog import Decision
from foren_searcher.datagen.verify.context import DictLookup
from foren_searcher.datagen.verify.exec_checks import exec_checks
from foren_searcher.datagen.verify.hindsight import UsefulEvidence
from foren_searcher.datagen.verify.units import check_answer

ROOT = Path(__file__).resolve().parents[1]


def make_case(checks: tuple[Check, ...]) -> Case:
    return Case(case_id="true:1", dataset="true", source_id="1", split="dev_train", task_kind="claim_check",
                task_text="核查铁塔火情", family_key="true:1",
                requirements=(Requirement(id="REQ1", text="官方对火情原因的记载"), Requirement(id="REQ2", text="视频画面")),
                checks=checks)


ENTITY = Check(kind="entity_in_evidence", params={"text": "耶拿桥"}, requirement="REQ1")


def lookup() -> DictLookup:
    return DictLookup({"E1": "警方通报：耶拿桥附近一辆烟花卡车起火。", "E2": "视频帧：画面中有烟雾。"})


def entity_errors(errors: list[Any]) -> list[Any]:
    return [e for e in errors if e.extra.get("check") == "entity_in_evidence"]


# (a) 序列化兼容
def test_check_requirement_serialization_roundtrip_and_legacy():
    case = make_case((ENTITY, Check(kind="date_in_range", params={"min": "2016-07-13"})))
    data = case.to_dict()
    assert data["checks"][0]["requirement"] == "REQ1"
    assert "requirement" not in data["checks"][1]
    assert Case.from_dict(data) == case
    legacy = dict(data, checks=[{"kind": "entity_in_evidence", "params": {"text": "耶拿桥"}, "note": ""}])
    assert Case.from_dict(legacy).checks[0].requirement is None
    with pytest.raises(CaseError):
        make_case((Check(kind="entity_in_evidence", params={"text": "x"}, requirement="REQ9"),))
    with pytest.raises(CaseError):
        Check(kind="entity_in_evidence", params={"text": "x"}, requirement="bad")


def test_real_case_file_scopes_entity_to_req1():
    case = Case.load(ROOT / "experiments/sft_data/cases/true_4124903.json")
    entity = [c for c in case.checks if c.kind == "entity_in_evidence"]
    assert entity and entity[0].requirement == "REQ1"


# (b) exec_checks 按工作区范围
def test_exec_checks_skip_entity_outside_workspace_scope():
    case = make_case((ENTITY,))
    video_report = {"status": "COMPLETED", "findings": [{"statement": "画面中只有烟雾，看不出袭击。", "evidence_ids": ["E2"]}]}
    assert entity_errors(exec_checks(video_report, lookup(), case=case, requirement_ids={"REQ2"})) == []
    for ids in ({"REQ1"}, None):
        errs = entity_errors(exec_checks(video_report, lookup(), case=case, requirement_ids=ids))
        assert [e.type for e in errs] == [ErrorType.OMITTED_REQUIRED_FACT]
    police_report = {"status": "COMPLETED", "findings": [{"statement": "起火点在耶拿桥附近的烟花卡车。", "evidence_ids": ["E1"]}]}
    assert entity_errors(exec_checks(police_report, lookup(), case=case, requirement_ids={"REQ1"})) == []


def test_unscoped_check_still_runs_everywhere():
    case = make_case((Check(kind="entity_in_evidence", params={"text": "耶拿桥"}),))
    video_report = {"status": "COMPLETED", "findings": [{"statement": "画面中只有烟雾。", "evidence_ids": ["E2"]}]}
    assert len(entity_errors(exec_checks(video_report, lookup(), case=case, requirement_ids={"REQ2"}))) == 1


# (c) answer 单元：实体两种判定
def answer_verdict(answer: str) -> Any:
    decision = Decision(request_id="request_0009", role="GIA", actor_id="GIA", turn=1, round_index=1, kind="answer",
                        output={"answer": answer}, status="gia_decision_accepted", system_prompt="", payload={},
                        image_evidence_ids=[], image_notes=[], image_paths=[], visible_ids={"E1", "E2"})
    log: Any = SimpleNamespace(lookup=lookup())
    return check_answer(decision, log, make_case((ENTITY,)), UsefulEvidence(), [], subject={})


def test_answer_unit_checks_entity_regardless_of_requirement():
    missing = entity_errors(answer_verdict("声明不实：视频只显示烟雾（E2）。").errors)
    assert [e.type for e in missing] == [ErrorType.OMITTED_REQUIRED_FACT]
    unsupported = entity_errors(answer_verdict("声明不实：火情在耶拿桥（E2）。").errors)
    assert [e.type for e in unsupported] == [ErrorType.UNSUPPORTED_ASSERTION]
    assert unsupported[0].extra["confidence"] == "flag"
    assert entity_errors(answer_verdict("声明不实：火情在耶拿桥（E1）。").errors) == []


def test_entity_aliases_match_cited_text_only():
    alias = Check(kind="entity_in_evidence", params={"text": "耶拿桥", "aliases": ["pont d'Iéna"]}, requirement="REQ1")
    look = DictLookup({"E1": "Un camion d'artifices a pris feu sur le pont d'Iéna."})
    ok = {"status": "COMPLETED", "findings": [{"statement": "起火点在耶拿桥。", "evidence_ids": ["E1"]}]}
    assert entity_errors(exec_checks(ok, look, case=make_case((alias,)), requirement_ids={"REQ1"})) == []
    only_alias = {"status": "COMPLETED", "findings": [{"statement": "Fire on the pont d'Iéna.", "evidence_ids": ["E1"]}]}
    assert [e.type for e in entity_errors(exec_checks(only_alias, look, case=make_case((alias,))))] == [ErrorType.OMITTED_REQUIRED_FACT]
