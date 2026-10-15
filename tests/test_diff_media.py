"""diff_media：逐帧像素差异的帧级片段、时间映射、配准、区域、图片对、参数校验、工具目录与规格开关。

测试视频在用例内生成（MJPG，同一帧内容逐字节相同），因此无差异帧的差值为 0，片段边界可精确断言。
"""

from __future__ import annotations

import copy
import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from foren_searcher.inspect.contracts import (
    DEFAULT_OPERATIONS, DiffMediaParams, InspectValidationError, OPT_IN_OPERATIONS, inspect_action_schema,
    parse_inspect_call,
)
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.inspect.policy import InspectPolicy
from foren_searcher.inspect.tool_cache import ToolCache, apply_tool_cache
from foren_searcher.store import GlobalStore

ROOT = Path(__file__).resolve().parents[1]
FPS = 10.0
SIZE = (320, 180)
ENABLED = DEFAULT_OPERATIONS | {"diff_media"}


def background(seed: int = 7) -> np.ndarray:
    """平滑随机纹理：既有足够 SIFT 特征，又不会因插值产生大片高频差异。"""

    rng = np.random.default_rng(seed)
    return cv2.resize(rng.integers(0, 255, (12, 20, 3), dtype=np.uint8), SIZE, interpolation=cv2.INTER_CUBIC)


def write_video(path: Path, frames: list[np.ndarray], fps: float = FPS) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def with_box(frame: np.ndarray, box: tuple[int, int, int, int], color=(20, 20, 20)) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, box[:2], box[2:], color, -1)
    return out


def store_with(tmp_path: Path, frames_a: list[np.ndarray], frames_b: list[np.ndarray]) -> tuple[GlobalStore, list[str]]:
    paths = [tmp_path / "a.avi", tmp_path / "b.avi"]
    write_video(paths[0], frames_a)
    write_video(paths[1], frames_b)
    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    ids = [store.commit_ingested_evidence(source_ref=f"I{i}", media_type="video/x-msvideo", modalities=("video",),
                                          label=p.name, original_filename=p.name, path=p).evidence_id
           for i, p in enumerate(paths)]
    return store, ids


async def run_diff(store: GlobalStore, ids: list[str], params: dict | None = None, **service_kwargs):
    async with create_inspect_service(store=store, enabled_operations=ENABLED, **service_kwargs) as service:
        return await service.execute({"targets": ids, "request": {"operation": "diff_media", "params": params or {}}},
                                     workspace_id="W1-1")


def spans(response) -> list[tuple[int, int]]:
    return [(s.start_frame_a, s.end_frame_a) for s in response.result.segments]


@pytest.mark.asyncio
async def test_segments_are_frame_exact_and_composites_carry_times(tmp_path: Path) -> None:
    base = background()
    total = 90
    box = (230, 40, 290, 150)
    frames_b = [with_box(base, box) if 6 <= i <= 40 or i >= 69 else base for i in range(total)]
    store, ids = store_with(tmp_path, [base] * total, frames_b)
    response = await run_diff(store, ids)
    assert response.status == "ready", response.feedback
    assert spans(response) == [(6, 41), (69, total)]
    first, second = response.result.segments
    assert (first.start_s, first.end_s, first.end_reason) == (pytest.approx(0.6), pytest.approx(4.1), "no_difference")
    assert (first.first_diff_frame_a, first.last_diff_frame_a, first.differing_frames) == (6, 40, 35)
    assert (second.start_s, second.end_s, second.end_reason) == (pytest.approx(6.9), pytest.approx(9.0), "range_end")
    assert first.start_b_s == pytest.approx(0.6) and first.end_b_s == pytest.approx(4.1)
    # 差异框在矩形附近（归一化到 A 画面）。
    x1, y1, x2, y2 = first.union_bbox
    assert x1 == pytest.approx(230 / 320, abs=.03) and x2 == pytest.approx(291 / 320, abs=.03)
    assert y1 == pytest.approx(40 / 180, abs=.03) and y2 == pytest.approx(151 / 180, abs=.03)
    value = response.result.to_dict()
    assert value["differs"] is True and value["media_kind"] == "video"
    assert value["coverage"]["compared_frames"] == total and value["coverage"]["stride_frames"] == 1
    assert value["coverage"]["frame_index_basis"] == {"a": "decode_order", "b": "decode_order"}
    timeline = value["timeline"]
    assert timeline["aggregation"] == "per_compared_frame" and len(timeline["values"]) == total
    assert timeline["values"][5] == 0 and timeline["values"][6] > 0 and timeline["values"][41] == 0
    assert value["registration"]["applied"] is False and value["mapping"]["source"] == "default_identity"
    assert value["segment_definition"] and any("哪一侧被编辑" in text for text in value["limitations"])
    assert "0.600–4.100" in value["summary"]
    json.dumps(value, ensure_ascii=False)
    # 每段首个 / 峰值 / 最后差异帧各一张对照图（首帧与峰值可能重合时去重）。
    images = response.result.images
    assert 4 <= len(images) <= 6 and {i.kind for i in images} >= {"first", "last"}
    for image in images:
        evidence = store.evidence(image.evidence_id)
        assert evidence.derived_from == tuple(ids) and evidence.acquired_by == "INSPECT/diff_media"
        assert f"{image.time_a_s:.3f}秒" in evidence.label and f"第{image.frame_a}帧" in evidence.label
        locator = json.loads(evidence.locator)
        assert locator["time_a_s"] == pytest.approx(image.time_a_s) and locator["frame_b"] == image.frame_b
        # 矩形宽 0.19 < 0.25 画面宽：加第二行放大局部，三栏高 180 + 分隔 6 + 放大行 180。
        assert locator["zoom"]["factor"] > 1 and "第二行" in evidence.label and "放大" in evidence.label
        window = locator["zoom"]["window"]
        assert window[0] <= locator["bbox"][0] and window[2] >= locator["bbox"][2]
        with Image.open(store.artifact_path(evidence.artifact_ref)) as picture:
            assert picture.height == 180 + 6 + 180 and picture.width > 3 * 320
    assert {i.frame_a for i in images if i.kind == "first"} == {6, 69}
    assert set(first.image_evidence_ids) <= {i.evidence_id for i in images}
    saved = json.loads(store.evidence(response.result_evidence_id).content)
    assert saved["segments"][0]["end_frame_a"] == 41 and saved["backend"] == "OpenCV per-frame pixel difference"


@pytest.mark.asyncio
async def test_mapping_offset_aligns_delayed_b_and_reports_unavailable_tail(tmp_path: Path) -> None:
    base = background(11)
    total, delay = 60, 5

    def scene(i: int) -> np.ndarray:
        frame = base.copy()
        x = 10 + 4 * i
        cv2.rectangle(frame, (x, 120), (x + 18, 150), (250, 250, 250), -1)
        return frame

    frames_a = [scene(i) for i in range(total)]
    # B 比 A 晚 delay 帧开始，且 A 帧 20–30 对应的 B 帧多一个黑块；B 与 A 等长，A 末尾 delay 帧映射到 B 之外。
    frames_b = [scene(max(0, i - delay)) for i in range(total)]
    frames_b = [with_box(f, (30, 20, 90, 70)) if 20 <= i - delay <= 30 else f for i, f in enumerate(frames_b)]
    store, ids = store_with(tmp_path, frames_a, frames_b)
    identity = await run_diff(store, ids)
    assert identity.status == "ready" and spans(identity) == [(1, total)]
    mapped = await run_diff(store, ids, {"mapping": {"offset_s": delay / FPS, "scale": 1}})
    assert mapped.status == "ready", mapped.feedback
    assert spans(mapped) == [(20, 31)]
    segment = mapped.result.segments[0]
    assert segment.start_b_s == pytest.approx((20 + delay) / FPS) and segment.first_diff_frame_b == 20 + delay
    coverage = mapped.result.coverage
    assert coverage["b_unavailable_frames"] == delay
    assert coverage["b_unavailable_ranges_a_s"] == [[pytest.approx((total - delay) / FPS), pytest.approx(total / FPS)]]
    assert mapped.result.timeline["values"][-1] is None
    assert mapped.result.mapping["source"] == "params" and mapped.result.mapping["offset_s"] == pytest.approx(.5)
    # range_a_s 只比较 A 的一部分；片段末尾按范围结束。
    ranged = await run_diff(store, ids, {"mapping": {"offset_s": delay / FPS}, "range_a_s": [2.5, 4.0]})
    assert spans(ranged) == [(25, 31)] and ranged.result.coverage["a_frames_in_range"] == 15
    outside = await run_diff(store, ids, {"mapping": {"offset_s": 100}})
    assert outside.status == "failed" and outside.feedback[0].code == "OUT_OF_RANGE"


@pytest.mark.asyncio
async def test_registration_aligns_rescaled_and_shifted_b(tmp_path: Path) -> None:
    base = background(3)
    total = 40
    matrix = np.array([[1.05, 0, -0.025 * SIZE[0] + 5], [0, 1.05, -0.025 * SIZE[1] + 3]], dtype=np.float64)
    shifted = cv2.warpAffine(base, matrix, SIZE, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    frames_b = [with_box(shifted, (120, 50, 180, 120)) if 10 <= i <= 20 else shifted for i in range(total)]
    store, ids = store_with(tmp_path, [base] * total, frames_b)
    registered = await run_diff(store, ids)
    assert registered.status == "ready", registered.feedback
    info = registered.result.registration
    assert info["applied"] is True and info["mode"] == "homography" and info["inliers"] >= 15
    assert info["homography_b_to_a_normalized"][0][0] == pytest.approx(1 / 1.05, abs=.02)
    assert spans(registered) == [(10, 21)]
    unregistered = await run_diff(store, ids, {"register": False})
    assert unregistered.result.registration["applied"] is False
    assert spans(unregistered) == [(0, total)]


@pytest.mark.asyncio
async def test_different_sizes_compare_after_resize_without_false_differences(tmp_path: Path) -> None:
    base = background(5)
    total = 30
    larger = cv2.resize(base, (480, 270), interpolation=cv2.INTER_CUBIC)
    frames_b = [with_box(larger, (60, 60, 150, 180)) if 12 <= i <= 17 else larger for i in range(total)]
    store, ids = store_with(tmp_path, [base] * total, frames_b)
    response = await run_diff(store, ids)
    assert response.status == "ready", response.feedback
    info = response.result.registration
    assert info["mode"] == "resize" and info["b_size"] == [480, 270] and info["a_size"] == [320, 180]
    assert spans(response) == [(12, 18)]
    x1, y1, x2, y2 = response.result.segments[0].union_bbox
    assert x1 == pytest.approx(60 / 480, abs=.03) and y2 == pytest.approx(181 / 270, abs=.03)


@pytest.mark.asyncio
async def test_region_restricts_comparison(tmp_path: Path) -> None:
    base = background(9)
    total = 40
    frames_b = []
    for i in range(total):
        frame = base
        if 5 <= i <= 15:
            frame = with_box(frame, (20, 40, 80, 120))
        if 20 <= i <= 30:
            frame = with_box(frame, (230, 40, 290, 120))
        frames_b.append(frame)
    store, ids = store_with(tmp_path, [base] * total, frames_b)
    everything = await run_diff(store, ids)
    assert spans(everything) == [(5, 16), (20, 31)]
    right = await run_diff(store, ids, {"region": {"x": 0.5, "y": 0, "w": 0.5, "h": 1}, "min_area": 0.001})
    assert right.status == "ready" and spans(right) == [(20, 31)]
    assert right.result.region == (0.5, 0.0, 0.5, 1.0) and right.result.min_area == .001
    assert right.result.segments[0].union_bbox[0] >= .5
    assert right.result.coverage["compared_pixels"] == 160 * 180


def _png(image: np.ndarray) -> bytes:
    ok, data = cv2.imencode(".png", image)
    assert ok
    return data.tobytes()


@pytest.mark.asyncio
async def test_image_pair_is_a_single_comparison(tmp_path: Path) -> None:
    base = background(13)
    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    ids = [store.commit_ingested_evidence(source_ref=f"I{i}", media_type="image/png", modalities=("image",),
                                          label=f"{i}.png", original_filename=f"{i}.png", data=_png(image)).evidence_id
           for i, image in enumerate((base, with_box(base, (100, 30, 160, 90)), base.copy()))]
    changed = await run_diff(store, ids[:2])
    assert changed.status == "ready", changed.feedback
    result = changed.result
    assert result.media_kind == "image" and result.differs is True and result.segments == ()
    assert result.timeline is None and result.mapping is None
    bbox = result.image_comparison["bbox"]
    assert bbox[0] == pytest.approx(100 / 320, abs=.03) and bbox[3] == pytest.approx(91 / 180, abs=.03)
    assert len(result.images) == 1 and result.images[0].kind == "single"
    assert store.evidence(result.images[0].evidence_id).derived_from == tuple(ids[:2])
    same = await run_diff(store, [ids[0], ids[2]])
    assert same.status == "ready" and same.result.differs is False and same.result.image_comparison["changed_area"] == 0
    rejected = await run_diff(store, ids[:2], {"mapping": {"offset_s": 1}})
    assert rejected.status == "failed" and rejected.feedback[0].code == "INVALID_INPUT"


@pytest.mark.asyncio
async def test_mixed_media_and_frame_budget(tmp_path: Path) -> None:
    base = background(17)
    total = 90
    frames_b = [with_box(base, (40, 40, 100, 100)) if 30 <= i <= 59 else base for i in range(total)]
    store, ids = store_with(tmp_path, [base] * total, frames_b)
    picture = store.commit_ingested_evidence(source_ref="P0", media_type="image/png", modalities=("image",),
                                             label="p.png", original_filename="p.png", data=_png(base)).evidence_id
    mixed = await run_diff(store, [ids[0], picture])
    assert mixed.status == "failed" and mixed.feedback[0].code == "UNSUPPORTED_MEDIA"
    reduced = await run_diff(store, ids, policy=InspectPolicy(maximum_frames=30))
    assert reduced.status == "ready"
    assert [f.code for f in reduced.feedback] == ["COUNT_REDUCED"] and reduced.feedback[0].severity == "warning"
    assert reduced.result.coverage["stride_frames"] == 3 and reduced.result.coverage["processed_frames"] == 30
    assert spans(reduced) == [(30, 60)]
    assert any("每 3 帧比较 1 帧" in text for text in reduced.result.limitations)
    small = await run_diff(store, ids, policy=InspectPolicy(maximum_visual_qa_images=2))
    assert len(small.result.images) == 2


@pytest.mark.parametrize("targets,params", [
    (["E1"], {}),
    (["E1", "E2", "E3"], {}),
    (["E1", "E2"], {"surprise": 1}),
    (["E1", "E2"], {"mapping": {"offset_s": 1, "scale": 0}}),
    (["E1", "E2"], {"mapping": {"scale": -1}}),
    (["E1", "E2"], {"mapping": {"offset_s": float("nan")}}),
    (["E1", "E2"], {"mapping": {"offset_s": True}}),
    (["E1", "E2"], {"mapping": {"offset_s": 1, "formula": "time_b_s = scale * time_a_s + offset_s"}}),
    (["E1", "E2"], {"range_a_s": [2, 1]}),
    (["E1", "E2"], {"region": {"x": 0.6, "y": 0, "w": 0.5, "h": 1}}),
    (["E1", "E2"], {"region": {"x": 0, "y": 0, "w": 0, "h": 1}}),
    (["E1", "E2"], {"region": {"x": 0, "y": 0, "w": 1}}),
    (["E1", "E2"], {"min_area": 0}),
    (["E1", "E2"], {"min_area": 1}),
    (["E1", "E2"], {"min_area": True}),
    (["E1", "E2"], {"register": "yes"}),
    (["E1", "E2"], []),
])
def test_invalid_params_are_rejected(targets, params) -> None:
    with pytest.raises(InspectValidationError) as error:
        parse_inspect_call({"targets": targets, "request": {"operation": "diff_media", "params": params}})
    assert error.value.operation == "diff_media"


def test_valid_params_parse_to_typed_call() -> None:
    call = parse_inspect_call({"targets": ["E1", "E2"], "request": {"operation": "diff_media", "params": {
        "mapping": {"offset_s": -1.25, "scale": 0.5}, "range_a_s": [0, 3], "region": {"x": 0, "y": 0.5, "w": 1, "h": 0.5},
        "min_area": 0.002, "register": False}}})
    params = call.request.params
    assert isinstance(params, DiffMediaParams)
    assert (params.mapping.offset_s, params.mapping.scale, params.range_a_s, params.register) == (-1.25, .5, (0, 3), False)
    assert call.to_dict()["request"]["params"]["region"] == {"x": 0.0, "y": 0.5, "w": 1.0, "h": 0.5}
    assert parse_inspect_call({"targets": ["E1", "E2"], "request": {"operation": "diff_media", "params": {}}})


def test_catalog_describes_diff_media_and_default_runtime_hides_it() -> None:
    from foren_searcher.local_models import _inspect_model_catalog

    schema = inspect_action_schema()
    branch = next(b for b in schema["oneOf"] if b["properties"]["request"]["properties"]["operation"]["const"] == "diff_media")
    assert branch["properties"]["targets"]["minItems"] == branch["properties"]["targets"]["maxItems"] == 2
    assert set(branch["properties"]["request"]["properties"]["params"]["properties"]) == {
        "mapping", "range_a_s", "region", "min_area", "register"}
    catalog = _inspect_model_catalog({"name": "INSPECT", "schema": copy.deepcopy(schema)})["operation_catalog"]
    description = catalog["diff_media"]["description"]
    assert "align_media" in description and "time_mapping" in description and "不要靠逐张取帧目测" in description
    # 10-12：合并目录的 compare_media / video_structure / locate 也是显式开启；旧默认目录不变
    assert OPT_IN_OPERATIONS == {"diff_media", "compare_media", "video_structure", "locate"}
    assert "diff_media" not in DEFAULT_OPERATIONS and not (DEFAULT_OPERATIONS & OPT_IN_OPERATIONS)


@pytest.mark.asyncio
async def test_operation_is_opt_in_for_service_and_workspace(tmp_path: Path) -> None:
    from foren_searcher.hi.workspace import WorkspaceHarness

    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    default = create_inspect_service(store=store)
    assert default.enabled_operations == DEFAULT_OPERATIONS
    blocked = await default.execute({"targets": ["E1", "E2"], "request": {"operation": "diff_media", "params": {}}},
                                    workspace_id="W1-1")
    assert blocked.status == "failed" and "not enabled" in blocked.feedback[0].message
    await default.aclose()

    def visible(service) -> set[str]:
        fake = SimpleNamespace(search_service=None, inspect_service=service, _custom_actions={})
        actions = WorkspaceHarness.eligible_actions(fake)  # type: ignore[arg-type]
        inspect = next(a for a in actions if a["name"] == "INSPECT")
        return {b["properties"]["request"]["properties"]["operation"]["const"] for b in inspect["schema"]["oneOf"]}

    policy = InspectPolicy()
    assert "diff_media" not in visible(SimpleNamespace(policy=policy, enabled_operations=None))
    assert "diff_media" not in visible(SimpleNamespace(policy=policy, enabled_operations=DEFAULT_OPERATIONS))
    assert "diff_media" in visible(SimpleNamespace(policy=policy, enabled_operations=ENABLED))


def test_spec_harness_param_enables_diff_media_only_for_new_spec() -> None:
    from foren_searcher.config import Config, RuntimeConfig
    from foren_searcher.datagen import run_cases
    from foren_searcher.datagen.harness_params import HARNESS_PARAM_MAP, HarnessParamError, build_harness_settings, parse_overrides
    from foren_searcher.datagen.specs import DEFAULT_HARNESS_PARAMS, Spec, validate_bump

    old, new = Spec.load(ROOT / "specs" / "hi@0.4.1"), Spec.load(ROOT / "specs" / "hi@0.4.2")
    assert validate_bump(old, new) == []
    assert new.student_prompt == old.student_prompt and new.harness_params["inspect_diff_media"] is True
    assert "inspect_diff_media" not in old.harness_params and "diff_media" in new.teacher_instruction
    assert set(HARNESS_PARAM_MAP) == set(DEFAULT_HARNESS_PARAMS)
    gia = SimpleNamespace(spec_id="gia", name="gia@9.9.9", harness_params={
        key: value for key, value in old.harness_params.items() if key != "max_workspace_turns"})
    config = Config(runtime=RuntimeConfig())
    base_ops = ("get_frames", "align_media")
    before = build_harness_settings(config, old, gia)
    assert before.inspect_operations.diff_media is False
    assert run_cases.with_spec_operations(base_ops, {}, before)[0] == base_ops
    after = build_harness_settings(config, new, gia)
    assert after.inspect_operations.diff_media is True and after.applied["inspect_diff_media"]["source"] == "hi@0.4.2"
    operations, detail = run_cases.with_spec_operations(base_ops, {"mode": "auto"}, after)
    assert operations == (*base_ops, "diff_media") and detail["spec_ops"]["diff_media"]["enabled"] is True
    recorded = build_harness_settings(config, new, gia, mode="record")
    assert run_cases.with_spec_operations(base_ops, {}, recorded)[0] == base_ops
    assert parse_overrides(["inspect_diff_media=true"]) == {"inspect_diff_media": True}
    with pytest.raises(HarnessParamError):
        build_harness_settings(config, replace(new, harness_params={**new.harness_params, "inspect_diff_media": 1}), gia)


class CountingBackend:
    """包装真实本地后端，只统计 diff_media 的实际计算次数。"""

    def __new__(cls):
        from foren_searcher.inspect.backends.local import LocalInspectBackend

        class _Counting(LocalInspectBackend):
            calls = 0

            async def diff_media(self, media, params, policy):
                type(self).calls += 1
                return await super().diff_media(media, params, policy)

        return _Counting()


@pytest.mark.asyncio
async def test_tool_cache_reuses_diff_media_with_new_evidence_ids(tmp_path: Path) -> None:
    base = background(21)
    frames_b = [with_box(base, (50, 50, 110, 110)) if 4 <= i <= 9 else base for i in range(20)]
    root = tmp_path / "cache"
    results = []
    for name in ("first", "second"):
        directory = tmp_path / name
        directory.mkdir()
        store, ids = store_with(directory, [base] * 20, frames_b)
        backend = CountingBackend()
        service = create_inspect_service(store=store, backend=backend, enabled_operations=ENABLED)
        apply_tool_cache(service, ToolCache(root, "on"))
        response = await service.execute({"targets": ids, "request": {"operation": "diff_media", "params": {}}},
                                         workspace_id="W1-1")
        await service.aclose()
        assert response.status == "ready"
        results.append((store, ids, response, type(backend).calls))
    (_, _, first, first_calls), (store, ids, second, second_calls) = results
    assert first_calls == 1 and second_calls == 0
    assert spans(second) == spans(first) == [(4, 10)]
    for image in second.result.images:
        assert store.evidence(image.evidence_id).derived_from == tuple(ids)
    details = store.inspect_execution(second.execution_id)["details"]
    assert details["tool_cache"]["status"] == "hit"


def test_zoom_row_only_for_small_difference_boxes() -> None:
    from foren_searcher.inspect import media_diff

    assert media_diff.zoom_window(None) is None
    assert media_diff.zoom_window((0.1, 0.1, 0.6, 0.7)) is None                     # 宽高都 ≥ 25%：不放大
    window = media_diff.zoom_window((0.97, 0.95, 1.0, 1.0))                          # 右下角几个像素
    assert window is not None and window[2] == pytest.approx(1.0) and window[3] == pytest.approx(1.0)
    assert window[2] - window[0] == pytest.approx(0.12) and window[3] - window[1] == pytest.approx(0.12)
    tall = media_diff.zoom_window((0.4, 0.0, 0.45, 1.0))                             # 只有宽度小于 25% 也放大
    assert tall is not None and tall[1] == 0 and tall[3] == 1 and tall[0] <= 0.4 and tall[2] >= 0.45

    base = background(23)
    registration = media_diff.estimate_registration([(base, base)], SIZE, SIZE, False)
    mask = np.zeros((SIZE[1], SIZE[0]), dtype=bool)
    mask[170:178, 310:318] = True
    changed = with_box(base, (310, 170, 317, 177))
    image, zoom = media_diff.composite(base, changed, registration, mask, (310 / 320, 170 / 180, 318 / 320, 178 / 180),
                                       None, ("A", "B", "diff"))
    assert zoom is not None and zoom["factor"] == pytest.approx(180 / round(0.12 * 180), abs=.1)
    assert image.shape[0] == 180 + 6 + 180
    zoom_row = image[186:]
    assert (zoom_row[..., 2] > 200).any() and ((zoom_row[..., 2] > 200) & (zoom_row[..., 1] < 50)).any()   # 红色描边
    big = np.zeros_like(mask)
    big[20:160, 20:300] = True
    image, zoom = media_diff.composite(base, changed, registration, big, (20 / 320, 20 / 180, 300 / 320, 160 / 180),
                                       None, ("A", "B", "diff"))
    assert zoom is None and image.shape[0] == 180
