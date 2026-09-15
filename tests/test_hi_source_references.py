from copy import deepcopy

import pytest

from foren_searcher.local_models import _normalize_hi_output, _validate_hi_output


def report(statement="视频 E1 的这帧显示红色方块（E2）。", references=None):
    return {"task_id": "W1-1", "status": "COMPLETED", "findings": [
        {"statement": statement, "evidence_ids": references if references is not None else ["E1", "E2"],
         "limitations": ["只确认已观察范围。"]}], "selected_evidence_ids": ["E1", "E2"]}


def view(modality="video", operation="get_frames"):
    return {"task_id": "W1-1", "materials": {"E1": {"evidence_id": "E1", "modalities": [modality],
            "derived_from": []}}, "evidence_view": [
        {"evidence_id": "E2", "derived_from": ["E1"], "acquired_by": "INSPECT/" + operation},
        {"evidence_id": "E3", "derived_from": [], "modalities": ["image"]}],
        "content_observed_evidence_ids": ["E2"]}


def normalize_and_validate(raw, model_view):
    normalized = _normalize_hi_output(raw, "W1-1", model_view)
    _validate_hi_output(normalized, workspace_id="W1-1", model_view=model_view, actions=())
    return normalized


@pytest.mark.parametrize("modality,operation", [("video", "get_frames"), ("audio", "asr"), ("text", "pdf_pages")])
@pytest.mark.parametrize("references", [["E1", "E2"], ["E2"]])
def test_observed_derived_content_supports_a_source_mention(modality, operation, references):
    raw = report("来源 E1 的已观察内容见 E2。", references)
    original = deepcopy(raw)
    result = normalize_and_validate(raw, view(modality, operation))
    assert result["findings"][0]["evidence_ids"] == ["E2"]
    assert result["findings"][0]["statement"] == raw["findings"][0]["statement"]
    assert raw == original


@pytest.mark.parametrize("references,statement", [
    (["E1"], "原视频中始终存在红色方块。"),
    (["E2", "E3"], "两帧一致。"),
    (["E2"], "E3 与 E2 一致。"),
    (["E2"], "E999 与 E2 一致。"),
])
def test_source_mentions_cannot_supply_unobserved_or_unknown_content(references, statement):
    with pytest.raises(ValueError):
        normalize_and_validate(report(statement, references), view())


def test_provenance_is_required_for_each_finding_separately():
    raw = report()
    raw["findings"].append({"statement": "来源 E1 的第二项结论。", "evidence_ids": ["E3"], "limitations": []})
    model_view = view()
    model_view["content_observed_evidence_ids"].append("E3")
    with pytest.raises(ValueError, match="missing from its evidence_ids"):
        normalize_and_validate(raw, model_view)


def test_transitive_provenance_does_not_mark_parents_as_observed():
    model_view = view()
    model_view["evidence_view"].append({"evidence_id": "E4", "derived_from": ["E2"]})
    model_view["content_observed_evidence_ids"] = ["E4"]
    result = normalize_and_validate(report("E1 中 E2 的裁剪内容见 E4。", ["E1", "E2", "E4"]), model_view)
    assert result["findings"][0]["evidence_ids"] == ["E4"]
    assert model_view["content_observed_evidence_ids"] == ["E4"]


def test_provenance_in_untrusted_body_is_not_used():
    model_view = view()
    model_view["evidence_view"][0].pop("derived_from")
    model_view["latest_observation"] = {"content": {"evidence_id": "E2", "derived_from": ["E1"]}}
    with pytest.raises(ValueError):
        normalize_and_validate(report(), model_view)


def test_observed_original_remains_content_evidence():
    model_view = view()
    model_view["content_observed_evidence_ids"].append("E1")
    result = normalize_and_validate(report(), model_view)
    assert result["findings"][0]["evidence_ids"] == ["E1", "E2"]
