"""10-14 合并目录恢复 DETECT v2.4 的输出写法（合并后掉分的根因，见 claude/合并后掉分根因_DETECT输出写法_20261014.md）。

1. compare_media / video_structure 的摘要行：结论与"该怎么写"紧挨着数据（换物、换脸、约 10 s 的生成片段）；
2. 观察里对照图按 E 编号列出，结尾一句 Next；对照图记为已看过；
3. 报告引用不合规、重试后仍不合规时由运行时修补，报告不再整份作废；
4. video_structure 给每个约 10 s 的镜头单独一张三帧图。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from foren_searcher.policies import EVAL_GUARDS, runtime_guard_scope


def test_compare_lines_carry_detect_attribution_and_instructions():
    from foren_searcher.inspect import source_compare

    import re

    text = re.sub(r'"\s*\n\s*(f?)"', "", open(source_compare.__file__, encoding="utf-8").read())    # join split literals
    assert "an object there was replaced, added or altered (typical of AI video editing); name the object in B" in text
    assert "most likely an AI-generated clip made to fit this scene; say what it shows" in text
    assert "shorter mismatch(es) under 3 s" in text


def _harness(tmp_path):
    from foren_searcher.hi.workspace import WorkspaceHarness, WorkspaceTask
    from foren_searcher.store import GlobalStore

    store = GlobalStore(tmp_path / "s.sqlite3", tmp_path / "art")
    harness = WorkspaceHarness(task=WorkspaceTask("W1-1", "t", ()), store=store, inspect_service=None)
    harness.seen = []
    harness._mark_content_observed = harness.seen.append
    return harness


def test_observation_names_images_by_evidence_id_and_ends_with_next(tmp_path):
    harness = _harness(tmp_path)
    images = (SimpleNamespace(evidence_id="E44", kind="region", index=0, caption="A 308.0-320.3 s: region in red differs"),
              SimpleNamespace(evidence_id="E45", kind="segment", index=0, caption="added footage 1"))
    result = SimpleNamespace(summary_lines=(
        "B matches A: 97% of A's half-second samples are found in B.",
        "Changed region: A 308.0-320.3 s ... name the object in B and what A shows instead. "
        "Visual check of comparison image 1: B: Refrigerator; A: Vending machine"), images=images,
        report={"source_match": True})
    lines = harness._report_observation_lines(SimpleNamespace(operation="compare_media", result=result))
    assert "Visual check of E44: B: Refrigerator; A: Vending machine" in lines[1]
    assert lines[2] == "Image E44: A 308.0-320.3 s: region in red differs" and lines[3].startswith("Image E45")
    assert lines[-1].startswith("Next: for each listed span, region and person")
    assert harness.seen == ["E44", "E45"]
    no_match = SimpleNamespace(summary_lines=("B is probably not the source of A",), images=(), report={"source_match": False})
    assert not any(x.startswith("Next") for x in harness._report_observation_lines(
        SimpleNamespace(operation="compare_media", result=no_match)))


def _view(observed=("E3", "E44", "E48"), visible=("E1", "E3", "E44", "E48", "E559")):
    return {"evidence_view": [{"evidence_id": x} for x in visible], "content_observed_evidence_ids": list(observed)}


def test_salvage_repairs_citations_instead_of_dropping_the_report(monkeypatch):
    from foren_searcher import local_models as L

    monkeypatch.setattr(L, "_workspace_evidence_ids", lambda view: {x["evidence_id"] for x in view["evidence_view"]})
    monkeypatch.setattr(L, "_hi_report_source_ids", lambda ids, view: set())
    report = {"task_id": "W1-1", "status": "COMPLETED", "selected_evidence_ids": ["E48"], "findings": [
        {"statement": "At A 308 s the refrigerator in B is a vending machine in A (E44, E559).", "evidence_ids": ["E48"],
         "limitations": []},
        {"statement": "The page E34 lists the original.", "evidence_ids": ["E34"], "limitations": []}]}
    error = ValueError("Finding text references Evidence missing from its evidence_ids: E44, E559. Declare ...")
    with runtime_guard_scope(EVAL_GUARDS):
        fixed = L._salvage_report_citations(report, _view(), error)
    assert [f["evidence_ids"] for f in fixed["findings"]] == [["E48", "E44"]]          # E559 unseen → removed from text
    assert "E559" not in fixed["findings"][0]["statement"] and "(E44" in fixed["findings"][0]["statement"]
    assert fixed["selected_evidence_ids"] == ["E48", "E44"]
    assert L._salvage_report_citations(report, _view(), error) is None                 # guard off → unchanged behaviour
    with runtime_guard_scope(EVAL_GUARDS):
        assert L._salvage_report_citations(report, _view(), ValueError("HI action is not currently available")) is None


def test_hi_policy_salvages_on_the_last_attempt(monkeypatch):
    from foren_searcher import local_models as L

    calls = []
    bad = {"task_id": "W1-1", "status": "COMPLETED", "selected_evidence_ids": ["E48"], "findings": [
        {"statement": "Vending machine replaced the refrigerator (E559).", "evidence_ids": ["E48"], "limitations": []}]}

    class Gen:
        async def generate_json(self, *, system_prompt, payload):
            calls.append(payload.get("validation_error"))
            return dict(bad)

    def validate(output, *, workspace_id, model_view, actions):
        for f in output["findings"]:
            if "E559" in f["statement"]:
                raise ValueError("Finding text references Evidence missing from its evidence_ids: E559. Declare ...")

    monkeypatch.setattr(L, "_validate_hi_output", validate)
    monkeypatch.setattr(L, "_normalize_hi_output", lambda output, workspace_id, view: output)
    monkeypatch.setattr(L, "_hi_turn_prompt", lambda view, actions: "prompt")
    monkeypatch.setattr(L, "_workspace_evidence_ids", lambda view: {x["evidence_id"] for x in view["evidence_view"]})
    monkeypatch.setattr(L, "_hi_report_source_ids", lambda ids, view: set())
    policy = L.TransformersHiPolicy(Gen())
    with runtime_guard_scope(EVAL_GUARDS) as state:
        out = asyncio.run(policy.decide(workspace_id="W1-1", model_view={"direct_visual_input": True, **_view()}, actions=()))
    assert len(calls) == 2 and "E559" not in out["findings"][0]["statement"]
    assert "report_citations_salvaged" in state.contract_flags


def test_video_structure_gives_each_ten_second_shot_its_own_image(tmp_path):
    from foren_searcher.inspect.video_structure import analyse

    fps, w, h = 10, 160, 90
    path = tmp_path / "v.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(0)
    for colour, seconds in (((200, 40, 40), 3), ((40, 200, 40), 10), ((40, 40, 200), 3)):     # cut, 10 s shot, cut
        base = np.zeros((h, w, 3), np.uint8)
        base[:] = colour
        for t in range(seconds * fps):
            frame = base.copy()
            cv2.circle(frame, (10 + (t * 3) % (w - 20), h // 2), 8, (int(rng.integers(0, 255)), 128, 128), -1)
            out.write(frame)
    out.release()
    report, images = analyse(path)
    flagged = report["generator_length_shots"]
    assert flagged, report["summary_lines"]
    assert any("Check whether its content belongs with the surrounding footage" in x for x in report["summary_lines"])
    assert images[0]["kind"] == "shot" and "beginning, middle, end" in images[0]["caption"]
