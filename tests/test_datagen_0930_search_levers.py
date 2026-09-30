"""09-30 动作分布的几项改动：hi@0.4.0 / gia@0.5.0 规格、--harness-override、search_excerpt_chars、导出按数据集压份额、
compare_rounds 的声明类取证指标。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from foren_searcher.datagen.export import apply_dataset_share_caps, parse_share_caps
from foren_searcher.datagen.harness_params import HarnessParamError, build_harness_settings, parse_overrides
from foren_searcher.datagen.specs import Spec, validate_bump
from foren_searcher.hi.workspace import best_excerpt

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "specs"


def test_new_specs_bump_cleanly_and_carry_search_guidance():
    hi0, hi1 = Spec.load(SPECS / "hi@0.3.0"), Spec.load(SPECS / "hi@0.4.0")
    gia0, gia1 = Spec.load(SPECS / "gia@0.4.0"), Spec.load(SPECS / "gia@0.5.0")
    assert validate_bump(hi0, hi1) == [] and validate_bump(gia0, gia1) == []
    assert "不要在查询里加 fact check" in hi1.teacher_instruction and "一手来源" in hi1.teacher_instruction
    assert "rationale" in gia1.teacher_instruction
    assert '"selected_evidence_ids":["所有引用的 E 编号"],"rationale"' in hi1.student_prompt
    assert '"rationale":"一句话说明为什么这样决定"' in gia1.student_prompt


def test_run_cases_defaults_point_at_latest_specs():
    from foren_searcher.datagen import run_cases

    assert run_cases.DEFAULT_SPECS == ("hi@0.4.0", "gia@0.5.0")


def test_overrides_parse_validate_and_take_precedence():
    assert parse_overrides(["image_input_limit=16", "search_excerpt_chars=600"]) == {"image_input_limit": 16, "search_excerpt_chars": 600}
    with pytest.raises(HarnessParamError):
        parse_overrides(["image_input_limit=999"])
    with pytest.raises(HarnessParamError):
        parse_overrides(["no_such_key=1"])
    with pytest.raises(HarnessParamError):
        parse_overrides(["image_input_limit"])
    config = SimpleNamespace(runtime=__import__("foren_searcher.config", fromlist=["load_config"]).load_config().runtime)
    hi, gia = Spec.load(SPECS / "hi@0.4.0"), Spec.load(SPECS / "gia@0.5.0")
    settings = build_harness_settings(config, hi, gia, overrides={"image_input_limit": 16, "search_excerpt_chars": 600})
    assert settings.inspect_policy.maximum_visual_qa_images == 16
    assert settings.memory_policy.search_excerpt_chars == 600
    assert settings.applied["image_input_limit"]["source"] == "override"
    plain = build_harness_settings(config, hi, gia)
    assert plain.memory_policy.search_excerpt_chars == 0


def test_best_excerpt_picks_matching_window():
    text = ("Museum intro.\nIn 2000 Virginia legislators asked Minnesota to return the flag. Governor Jesse Ventura refused.\n"
            "Weather and sports.")
    got = best_excerpt(text, "Minnesota refused Virginia flag", 200)
    assert got.startswith("In 2000") and "Ventura refused" in got and "Museum" not in got
    assert best_excerpt("x" * 300, "", 50).endswith("…") and len(best_excerpt("x" * 300, "", 50)) == 51
    assert best_excerpt("", "q", 50) == ""


def _row(dataset: str, weight: float, role: str = "hi", bucket: str = "accepted") -> dict:
    return {"metadata": {"role": role, "bucket": bucket, "dataset": dataset, "sample_weight": weight}}


def test_dataset_share_cap_scales_weights_only():
    rows = [_row("videosham", 3.0) for _ in range(6)] + [_row("true", 3.0) for _ in range(4)] + [_row("videosham", 3.0, bucket="held")]
    report = apply_dataset_share_caps(rows, parse_share_caps(["videosham=0.34"]))
    acc = [r["metadata"] for r in rows if r["metadata"]["bucket"] == "accepted"]
    vs = sum(m["sample_weight"] for m in acc if m["dataset"] == "videosham")
    total = sum(m["sample_weight"] for m in acc)
    assert abs(vs / total - 0.34) < 1e-3 and len(acc) == 10
    assert report["hi"]["videosham"]["share_before"] == 0.6
    assert rows[-1]["metadata"]["sample_weight"] == 3.0          # held 行不动
    assert all("dataset_share_cap" in m.get("weight_adjustments", []) for m in acc if m["dataset"] == "videosham")
    only = [_row("videosham", 1.0)]
    assert apply_dataset_share_caps(only, {"videosham": 0.34}) == {}      # 只有一个数据集：无从压缩
    with pytest.raises(SystemExit):
        parse_share_caps(["videosham=1.5"])


def _compare_rounds():
    spec = importlib.util.spec_from_file_location("compare_rounds", ROOT / "scripts" / "compare_rounds.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_claim_source_metrics_counts_queries_domains_and_fact_checks(tmp_path):
    cr = _compare_rounds()
    run = tmp_path / "true_1_r0c_1"
    run.mkdir()
    (run / "run_config.json").write_text(json.dumps({"dataset": "true"}), encoding="utf-8")
    events = [
        {"actor_role": "HI", "event_type": "model_output", "payload": {"output": {"action": "SEARCH", "arguments": {"query": "flag Virginia fact check"}}}},
        {"actor_role": "HI", "event_type": "observation", "payload": {"observation": {"action": "SEARCH", "items": [
            "[R1] https://www.mnhs.org/mnopedia/flag; page=ready(E2), document=not_requested",
            "[R2] https://www.minnpost.com/fact-briefs/2023/07/x; page=ready(E3), document=not_requested",
            "[R3] https://en.wikipedia.org/wiki/x; page=failed, document=not_requested"]}}},
        {"actor_role": "HI", "event_type": "model_output", "payload": {"output": {"action": "READ_RECORD", "arguments": {"record_id": "E2"}}}},
        {"actor_role": "HI", "event_type": "model_output", "payload": {"output": {"action": "SEARCH", "arguments": {"source": "https://twincities.com/x"}}}},
        {"actor_role": "HI", "event_type": "observation", "payload": {"observation": {"action": "SEARCH", "items": [
            "[R4] https://www.twincities.com/x; page=ready(E6), document=not_requested"]}}},
        {"actor_role": "HI", "event_type": "workspace_report", "payload": {"report": {"findings": [{"evidence_ids": ["E2", "E6", "E3"]}]}}},
    ]
    (run / "trajectory.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    m = cr.claim_source_metrics(run)
    assert m["queries"] == 1 and m["fact_check_queries"] == 1 and m["direct_opens"] == 1 and m["pages_opened"] == 3
    assert m["domains_opened"] == 2 and m["domains_read"] == 1 and m["domains_cited"] == 2
    assert m["fact_check_pages_read_or_cited"] == 1
    video = tmp_path / "videosham_1_r0c_1"
    video.mkdir()
    (video / "run_config.json").write_text(json.dumps({"dataset": "videosham"}), encoding="utf-8")
    assert cr.claim_source_metrics(video) is None
    agg = cr.claim_aggregate([m])
    assert agg["claim_runs"] == 1 and agg["claim_fact_check_query_share"] == 1.0


@pytest.mark.parametrize("chars", [0, 120])
def test_execute_search_appends_excerpt_only_when_enabled(chars):
    import asyncio

    from foren_searcher.contracts import WorkspaceTask
    from foren_searcher.memory import MemoryPolicy
    from foren_searcher.search.contracts import MaterialResult, SearchResponse, SearchResult
    from foren_searcher.store import GlobalStore
    from foren_searcher.workspace import WorkspaceHarness

    store = GlobalStore()

    class FakeSearch:
        async def execute(self, call, *, workspace_id):
            cid, _ = store.get_or_create_candidate(identity_key="url:https://example.test/a", source_url="https://example.test/a",
                                                  candidate_type="page")
            ev = store.record_evidence(content="Museum intro.\nIn 2000 Virginia asked Minnesota to return the flag. Ventura refused.",
                                       label="page", source_ref=cid, locator="paragraph:1", acquired_by="SEARCH/open",
                                       discovered_by=workspace_id)
            return SearchResponse("text", "target_reached",
                                  (SearchResult(cid, "https://example.test/a", "page", None, (MaterialResult("page", "ready", (ev.card(),)), MaterialResult("document", "absent"))),))

    harness = WorkspaceHarness(task=WorkspaceTask("W1-1", "核查弗吉尼亚州是否多次要求归还战旗", ()), store=store,
                               search_service=FakeSearch(), memory_policy=MemoryPolicy(search_excerpt_chars=chars))  # type: ignore[arg-type]
    asyncio.run(harness.execute_search({"input_type": "text", "query": "Virginia Minnesota flag refused", "output_type": "text"}))
    items = harness.memory.latest_observation["items"]
    excerpts = [x for x in items if "摘录" in x]
    if chars:
        assert len(excerpts) == 1 and "Ventura refused" in excerpts[0] and "Museum" not in excerpts[0]
    else:
        assert excerpts == []
