"""数据来源 D（离线换题）：变体校验、状态构造、导出门槛、runlog 的来源标记。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from foren_searcher.datagen.export import offline_gate
from foren_searcher.datagen.offline_gia import VariantError, build_state, check_source, load_variants

ROOT = Path(__file__).resolve().parents[1]


def _write(directory: Path, name: str, variants: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps({"source_run": name, "source_request": "request_0009",
                                                        "variants": variants}, ensure_ascii=False), encoding="utf-8")


def test_each_group_needs_both_finish_and_investigate(tmp_path: Path) -> None:
    _write(tmp_path, "r1", [{"id": "a", "task": "t1", "expected_decision": "FINISH"},
                            {"id": "b", "task": "t2", "expected_decision": "investigate"}])
    variants = load_variants(tmp_path)
    assert [(v.run_name, v.expected_decision) for v in variants] == [("r1__D_a", "FINISH"), ("r1__D_b", "INVESTIGATE")]
    _write(tmp_path, "r2", [{"id": "a", "task": "t", "expected_decision": "FINISH"}])
    with pytest.raises(VariantError, match="对照"):
        load_variants(tmp_path)


@pytest.mark.parametrize("bad, message", [
    ([{"id": "a", "task": "", "expected_decision": "FINISH"}], "缺 id 或 task"),
    ([{"id": "a", "task": "t", "expected_decision": "WAIT"}], "只能是"),
    ([{"id": "a", "task": "t", "expected_decision": "FINISH"}, {"id": "a", "task": "u", "expected_decision": "INVESTIGATE"}], "重复"),
])
def test_variant_validation(tmp_path: Path, bad: list[dict], message: str) -> None:
    _write(tmp_path, "r", bad)
    with pytest.raises(VariantError, match=message):
        load_variants(tmp_path)


def test_repository_variant_files_are_valid() -> None:
    variants = load_variants(ROOT / "experiments" / "sft_data" / "offline_gia")
    assert variants and {v.expected_decision for v in variants} == {"FINISH", "INVESTIGATE"}
    assert all(v.rationale for v in variants)


def _request(can_create: bool = True) -> dict:
    view = {"task": "旧任务", "original_materials": [
        {"record_type": "material", "reference_id": "C0", "content": json.dumps({"task": "旧任务"}, ensure_ascii=False)},
        {"record_type": "evidence", "evidence_id": "E1"}],
        "latest_reports": [{"run_status": "reported", "report": {"status": "COMPLETED"}}],
        "runtime_constraints": {"can_create_round": can_create, "round_index": 2}, "model_reasoning": "x"}
    return {"payload": {"global_view": view, "available_actions": [{"name": "READ_RECORD"}]}}


def test_build_state_swaps_task_only() -> None:
    source = _request()
    view, actions = build_state(source, "新任务")
    assert view["task"] == "新任务"
    assert json.loads(view["original_materials"][0]["content"]) == {"task": "新任务"}
    assert view["latest_reports"] == source["payload"]["global_view"]["latest_reports"]      # 报告原样
    assert "model_reasoning" not in view and actions == [{"name": "READ_RECORD"}]
    assert source["payload"]["global_view"]["task"] == "旧任务"                               # 源请求不被改


def test_investigate_needs_a_state_that_can_open_rounds() -> None:
    view, _ = build_state(_request(can_create=False), "t")
    with pytest.raises(VariantError, match="不能开新轮"):
        check_source(SimpleNamespace(expected_decision="INVESTIGATE", run_name="r__D_a"), view)
    check_source(SimpleNamespace(expected_decision="FINISH", run_name="r__D_a"), view)


def _decision(kind: str, role: str = "GIA") -> SimpleNamespace:
    return SimpleNamespace(role=role, kind=kind)


def test_offline_gate() -> None:
    log = SimpleNamespace(config={"expected_decision": "INVESTIGATE"})
    assert offline_gate(log, _decision("plan"), "accepted", 1.0, None) == ("accepted", 1.0, None)
    assert offline_gate(log, _decision("plan"), "rejected", 0.0, "plan_errors:x") == ("rejected", 0.0, "plan_errors:x")
    assert offline_gate(log, _decision("answer"), "accepted", 1.0, None)[:2] == ("rejected", 0.0)
    assert offline_gate(log, _decision("memory"), "held", 0.0, "m")[2] == "offline_memory_action"


def test_runlog_origin_and_case_ids() -> None:
    from foren_searcher.datagen.runlog import RunLog

    def log(config: dict) -> RunLog:
        return RunLog(run_dir=Path("x"), config=config, events=[], decisions=[], store=None, lookup=None, _scratch=Path("."))

    a = log({"dataset": "TRUE", "case": "4124903"})
    d = log({"dataset": "true", "case": "4124903", "origin": "D"})
    assert (a.origin, a.case_id, a.family_case_id) == ("A", "true:4124903", "true:4124903")
    assert (d.origin, d.case_id, d.family_case_id) == ("D", None, "true:4124903")


def test_status_of_detects_revised_variant(tmp_path):
    """变体题面改过后，旧 run 判为 stale，retire 挪进 _stale/ 不删。"""

    import json as _json

    from foren_searcher.datagen.offline_gia import Variant, retire, status_of

    v = Variant("src_run", "request_0001", "v1", "新题面", "FINISH", "", "x.json")
    out = tmp_path / v.run_name
    out.mkdir()
    (out / "execution_links.json").write_text("{}", encoding="utf-8")
    (out / "run_config.json").write_text(_json.dumps({"task": "旧题面", "expected_decision": "FINISH"}), encoding="utf-8")
    assert status_of(out) == "done"
    assert status_of(out, v) == "stale"
    moved = retire(out)
    assert not out.exists() and moved.parent.name == "_stale" and (moved / "run_config.json").is_file()
    (tmp_path / v.run_name).mkdir()
    (tmp_path / v.run_name / "execution_links.json").write_text("{}", encoding="utf-8")
    (tmp_path / v.run_name / "run_config.json").write_text(_json.dumps({"task": "新题面", "expected_decision": "FINISH"}), encoding="utf-8")
    assert status_of(tmp_path / v.run_name, v) == "done"
