"""INSPECT 请求与响应契约测试。

每个用例向解析器提供一个边界输入，确认强类型参数、目标数量和 Evidence 引用闭包。
"""

from __future__ import annotations

import math

import pytest

from foren_searcher.inspect.contracts import (
    AlignMediaResult,
    AlignmentAnchor,
    AlignmentMatch,
    AlignmentSideCoverage,
    AudioQaResult,
    InspectFeedback,
    InspectResponse,
    InspectValidationError,
    LocatedFrame,
    parse_inspect_call,
)
from foren_searcher.inspect.policy import InspectPolicy


@pytest.mark.parametrize('time_a,time_b,expected', [(6.0, 0.0, -6.0), (0.0, 6.0, 6.0), (2.3, 2.4, .1)])
def test_alignment_anchor_offsets_are_distinct_from_estimated_range_edges(time_a, time_b, expected):
    coverage = AlignmentSideCoverage((0, 30), 'frames', 300, ())
    result = AlignMediaResult('visual', coverage, coverage, (
        AlignmentMatch((5.9, 24.0), (0.0, 18.0), .99, (
            AlignmentAnchor(time_a, time_b, 'E1', 'E2'),)),))
    value = result.to_dict()
    anchor = value['matches'][0]['anchors'][0]
    assert anchor['time_b_minus_a_s'] == expected
    assert anchor['time_a_s'] == time_a and anchor['time_b_s'] == time_b
    assert result.referenced_evidence_ids() == ('E1', 'E2')
    assert value['matches'][0]['range_a_s'] == [5.9, 24.0]


VALID = {
    "get_frames": (['E1'], {}),
    "get_pages": (['E1'], {"page_range": [1, 2], "count": 2}),
    "crop": (['E1'], {"box": [0, 0, 1, 1]}),
    "locate_regions": (['E1'], {"query": "红色车辆"}),
    "ocr": (['E1'], {"range_s": None}),
    "asr": (['E1'], {"language": "zh"}),
    "audio_qa": (['E1'], {"question": "有没有警报声？", "audio_track": 0}),
    "visual_qa": (['E1', 'E2'], {"question": "有何差异？"}),
    "locate_frames": (['E1'], {"query": {"type": "text_image", "evidence_id": "E2", "text": "夜间桥梁"}}),
    "align_media": (['E1', 'E2'], {"mode": "visual"}),
}


@pytest.mark.parametrize(
    "field,value",
    (
        ("execution_timeout_s", 0),
        ("acquisition_timeout_s", math.inf),
        ("backend_workers", 0),
        ("maximum_frames", True),
        ("maximum_image_pixels", -1),
        ("maximum_download_bytes", 0),
    ),
)
def test_inspect_policy_rejects_disabled_or_unbounded_limits(
    field: str, value: object
) -> None:
    """INSPECT 的资源边界不能被零值、无穷值或布尔值关闭。"""

    with pytest.raises(ValueError, match=field):
        InspectPolicy(**{field: value})


@pytest.mark.parametrize(("operation", "case"), VALID.items())
def test_all_ten_operations_parse_to_typed_calls(operation, case):
    """验证 all ten operations parse to typed calls 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    targets, params = case
    parsed = parse_inspect_call({"targets": targets, "request": {"operation": operation, "params": params}})
    assert parsed.request.operation == operation
    assert parsed.to_dict()["targets"] == targets


@pytest.mark.parametrize("value", ["E0", "E01", "R1", "https://example.test/x", "x.png", 1])
def test_targets_reject_noncanonical_evidence_ids(value):
    """验证 targets reject noncanonical evidence ids 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with pytest.raises(InspectValidationError):
        parse_inspect_call({"targets": [value], "request": {"operation": "ocr", "params": {}}})


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_counts_are_positive_integers_not_booleans(value):
    """验证 counts are positive integers not booleans 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with pytest.raises(InspectValidationError):
        parse_inspect_call({"targets": ["E1"], "request": {"operation": "get_frames", "params": {"count": value}}})


@pytest.mark.parametrize("value", [math.nan, math.inf, -0.1])
def test_times_are_finite_and_nonnegative(value):
    """验证 times are finite and nonnegative 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with pytest.raises(InspectValidationError):
        parse_inspect_call({"targets": ["E1"], "request": {"operation": "get_frames", "params": {"times_s": [value]}}})


def test_mutually_exclusive_fields_and_unknown_fields_fail():
    """验证 mutually exclusive fields and unknown fields fail 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    for params in ({"times_s": [1], "count": 2}, {"range_s": [0,1], "surprise": 3}):
        with pytest.raises(InspectValidationError):
            parse_inspect_call({"targets": ["E1"], "request": {"operation": "get_frames", "params": params}})


def test_visual_qa_rejects_duplicate_inputs_but_align_allows_them():
    """验证 visual qa rejects duplicate inputs but align allows them 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with pytest.raises(InspectValidationError):
        parse_inspect_call({"targets": ["E1", "E1"], "request": {"operation": "visual_qa", "params": {"question": "比较"}}})
    assert parse_inspect_call({"targets": ["E1", "E1"], "request": {"operation": "align_media", "params": {"mode": "visual"}}})


def test_feedback_serialization_keeps_fixed_nullable_fields():
    """验证 feedback serialization keeps fixed nullable fields 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    value = InspectFeedback("warning", "decode", "COUNT_REDUCED", "only one distinct frame").to_dict()
    assert value == {
        "severity": "warning",
        "stage": "decode",
        "code": "COUNT_REDUCED",
        "message": "only one distinct frame",
        "target_evidence_id": None,
        "locator": None,
        "retryable": False,
    }


def test_count_reduced_is_warning_only_and_ready_cannot_hide_errors():
    """验证 count reduced is warning only and ready cannot hide errors 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with pytest.raises(ValueError, match="COUNT_REDUCED"):
        InspectFeedback("error", "decode", "COUNT_REDUCED", "wrong severity")
    with pytest.raises(ValueError, match="ready"):
        InspectResponse(
            "audio_qa",
            "ready",
            AudioQaResult((0.0, 1.0), 0, "answer", ()),
            None,
            (),
            (InspectFeedback("error", "process", "PROCESS_FAILED", "hidden"),),
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -0.1, 1.1])
def test_located_frame_score_must_be_finite_probability(bad):
    """验证 located frame score must be finite probability 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    with pytest.raises(ValueError, match="score"):
        LocatedFrame("E1", 0.0, bad)
