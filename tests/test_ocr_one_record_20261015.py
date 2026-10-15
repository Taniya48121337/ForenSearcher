"""10-15（清单 A7）：一次 OCR 调用 = 一个结果 E（kind=ocr）。

以前视频 OCR 把每个采样帧存成一个图片 E（fmnv_a_1Kk 一次 519 个），视图里的帧卡片被挤掉，walker 选的 E7 被拒；
观察里带全部逐帧结果（31 万字）被截成 preview。现在：
- 视频 OCR 不产生帧 E，只有一个结果 E；行指向输入视频、用 time_s 定位；
- 观察只列出现最多的若干条文字和总数，details.result 是汇总；
- 结果 E 以汇总开头，逐框保留文字、坐标、置信度，READ_RECORD 逐页读完能拿到全部；
- 图片 OCR（walker M3c / E1 / L1）结果形态不变：行仍指向输入图片 E。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from foren_searcher.inspect.contracts import OcrResult, ocr_summary_line, ocr_text_summary
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.inspect.policy import InspectPolicy
from foren_searcher.store import GlobalStore
from foren_searcher.store.records import MemoryPolicy, RecordAccess

from inspect_helpers import ingest, make_video


class _CaptionOcr:
    """每帧返回同一条字幕和一个随帧变化的数字；第 3 帧起多一条台标。"""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, source):
        self.calls += 1
        out = [([[10, 70], [150, 70], [150, 90], [10, 90]], "Los Angeles Wildfires Latest", 0.99),
               ([[2, 2], [20, 2], [20, 12], [2, 12]], str(self.calls), 0.9)]
        if self.calls >= 3:
            out.append(([[120, 2], [158, 2], [158, 14], [120, 14]], "Global NEWS", 0.97))
        return out, None


def _video(tmp_path: Path, store: GlobalStore):
    path = tmp_path / "input.avi"
    make_video(path, frames=40, fps=10.0)
    return store.commit_ingested_evidence(source_ref="V0", media_type="video/x-msvideo", modalities=("video",),
                                          label="video", original_filename=path.name, path=path)


def _read_all(store: GlobalStore, record_id: str, max_chars: int = 800) -> tuple[str, int]:
    access = RecordAccess(store=store, role="HI", actor_id="W1-1", round_index=1,
                          policy=MemoryPolicy(max_read_chars=max_chars))
    parts, offset, pages = [], 0, 0
    while offset is not None:
        page = access.read(record_id, offset=offset)
        parts.append(page["content"])
        offset = page["next_offset"]
        pages += 1
    return "".join(parts), pages


def test_video_ocr_makes_one_evidence_with_every_box(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", SimpleNamespace(RapidOCR=_CaptionOcr))
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    video = _video(tmp_path, store)
    before = {e.evidence_id for e in store.evidence_records()}
    service = create_inspect_service(store=store, policy=InspectPolicy(ocr_base_fps=2.0))
    try:
        response = asyncio.run(service.execute({"targets": [video.evidence_id], "request": {"operation": "ocr", "params": {}}},
                                               workspace_id="W1-1"))
    finally:
        asyncio.run(service.aclose())
    assert response.status == "ready", response.to_dict()
    created = [e for e in store.evidence_records() if e.evidence_id not in before]
    # 只有媒体画像（入库后第一次 INSPECT 的副产物）和一个 OCR 结果 E；没有帧 E
    assert [e.acquired_by for e in created] == ["INSPECT/media_profile", "INSPECT/ocr"]
    record = store.evidence(response.result_evidence_id)
    assert record.modalities == ("text",)
    assert record.label.startswith(f"OCR of {video.evidence_id}: ") and "Los Angeles Wildfires Latest" in record.label
    assert " text boxes, " in record.label and " distinct texts" in record.label
    assert response.result.referenced_evidence_ids() == ()
    assert all(row.image_evidence_id == video.evidence_id and row.time_s is not None for row in response.result.results)
    data = json.loads(record.content)
    assert list(data)[:4] == ["kind", "operation", "source_evidence_ids", "summary"]
    assert data["kind"] == "ocr" and data["source_evidence_ids"] == [video.evidence_id]
    rows = data["results"]
    assert [r["time_s"] for r in rows] == sorted(r["time_s"] for r in rows)
    boxes = sum(len(r["items"]) for r in rows)
    assert boxes == data["summary"]["text_boxes"] == sum(len(r.items) for r in response.result.results)
    first = rows[0]["items"][0]
    assert set(first) == {"text", "box", "confidence"} and len(first["box"]) == 4
    # READ_RECORD 分页读完 = 记录全文（每个框都在）
    text, pages = _read_all(store, record.evidence_id)
    assert pages > 1 and text == record.content


def test_image_ocr_rows_still_point_at_the_input_image(tmp_path: Path, monkeypatch):
    from inspect_helpers import image_bytes

    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", SimpleNamespace(RapidOCR=_CaptionOcr))
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    image = ingest(store, image_bytes(), source="I0", filename="input.png", media_type="image/png", modality="image")
    service = create_inspect_service(store=store, policy=InspectPolicy())
    try:
        response = asyncio.run(service.execute({"targets": [image.evidence_id], "request": {"operation": "ocr", "params": {}}},
                                               workspace_id="W1-1"))
    finally:
        asyncio.run(service.aclose())
    assert response.status == "ready"
    assert response.result.referenced_evidence_ids() == (image.evidence_id,)
    data = json.loads(store.evidence(response.result_evidence_id).content)
    assert data["results"][0]["image_evidence_id"] == image.evidence_id and "time_s" not in data["results"][0]
    assert [t["text"] for t in data["summary"]["texts"]] == ["Los Angeles Wildfires Latest", "1"]


def test_workspace_observation_lists_texts_and_total_not_every_frame(tmp_path: Path, monkeypatch):
    from foren_searcher.hi.workspace import WorkspaceHarness, WorkspaceTask

    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", SimpleNamespace(RapidOCR=_CaptionOcr))
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    video = _video(tmp_path, store)
    service = create_inspect_service(store=store, policy=InspectPolicy(ocr_base_fps=2.0))
    harness = WorkspaceHarness(task=WorkspaceTask("W1-1", "read the captions", (video.evidence_id,)), store=store,
                               inspect_service=service)
    try:
        response = asyncio.run(harness.execute_inspect({"targets": [video.evidence_id],
                                                        "request": {"operation": "ocr", "params": {}}}))
    finally:
        asyncio.run(service.aclose())
    observation = harness.memory.latest_observation
    rid = response.result_evidence_id
    assert observation["items"][0].startswith(f"INSPECT ocr returned ready; Result Evidence: {rid};")
    line = next(x for x in observation["items"] if x.startswith(f"OCR of {video.evidence_id}:"))
    assert '"Los Angeles Wildfires Latest"' in line and '"Global NEWS"' in line and f"READ_RECORD {rid}" in line
    result = observation["details"]["result"]
    assert set(result) == {"summary", "coverage"} and "results" not in result
    assert result["summary"]["text_boxes"] > len(result["summary"]["texts"])
    assert len(json.dumps(observation)) < 6000
    assert rid in harness.memory.content_observed_evidence_ids
    harness.refresh_evidence_view()
    assert [c for c in harness.memory.evidence_view if "image" in (c.get("modalities") or [])] == []


def test_summary_keeps_the_most_frequent_texts_in_time_order():
    rows = [{"time_s": t, "items": [{"text": "Caption", "box": [0, 0, 1, 1], "confidence": 0.9}]
             + ([{"text": f"noise{t}", "box": [0, 0, 1, 1], "confidence": 0.5}] if t else [])} for t in range(20)]
    rows.append({"time_s": 25.0, "items": [{"text": "Joanne  Low", "box": [0, 0, 1, 1], "confidence": 0.98}]})
    rows.append({"time_s": 26.0, "items": [{"text": "joanne low", "box": [0, 0, 1, 1], "confidence": 0.99}]})
    summary = ocr_text_summary({"results": rows}, limit=3)
    assert summary["text_boxes"] == 41 and summary["distinct_texts"] == 21 and summary["frames"] == 22
    assert [t["text"] for t in summary["texts"]] == ["Caption", "noise1", "Joanne  Low"]
    assert summary["texts"][2] == {"text": "Joanne  Low", "frames": 2, "max_confidence": 0.99, "first_s": 25.0, "last_s": 26.0}
    line = ocr_summary_line(summary, "E9", "E1")
    assert line.startswith("OCR of E1: 41 text boxes, 21 distinct texts in 22 of 22 frames. Texts (most frequent): ")
    assert "18 more distinct texts" in line and line.endswith("READ_RECORD E9 lists every text box with its time, position and confidence.")
    empty = ocr_summary_line(ocr_text_summary({"results": [{"image_evidence_id": "E2", "time_s": None, "items": []}]}), "E3", "E2")
    assert empty == ("OCR of E2: 0 text boxes, 0 distinct texts in 0 of 1 image; no text was recognised "
                     "(OCR can miss text, so this does not show there is none).")


def test_state_render_digest_of_an_ocr_step_keeps_the_texts():
    from foren_searcher.state_render import TIERS, _digest_observation, _latest_observation

    summary = ocr_text_summary({"results": [{"time_s": float(t), "items": [{"text": "Los Angeles Wildfires Latest", "box": [0, 0, 1, 1],
                                                                            "confidence": 0.99}]} for t in range(300)]})
    observation = {"action": "INSPECT", "items": ["INSPECT ocr returned ready; Result Evidence: E12; ...",
                                                  ocr_summary_line(summary, "E12", "E1")],
                   "details": {"operation": "ocr", "status": "ready", "result_evidence_id": "E12",
                               "result": {"summary": summary, "coverage": {"range_s": [0, 300]}}}}
    tier = TIERS["compact_16k"]
    digest = _digest_observation(observation, tier)
    assert digest["details"]["result_evidence_id"] == "E12"
    assert digest["details"]["ocr_texts"] == ["Los Angeles Wildfires Latest (0.0–299.0 s)"]
    assert digest["details"]["ocr_counts"] == {"text_boxes": 300, "distinct_texts": 1, "frames": 300}
    latest = _latest_observation(observation, tier)
    # 最新一步：文字在观察行里，details 只留计数和覆盖
    assert '"Los Angeles Wildfires Latest"' in latest["items"][1]
    assert latest["details"]["result"]["summary"]["text_boxes"] == 300 and "texts" not in latest["details"]["result"]["summary"]


def test_old_ocr_results_with_frame_handles_still_map_to_the_video():
    """旧缓存条目（帧带 handle、artifacts 里有帧）命中时：帧不存 E，行指向输入视频。"""

    from foren_searcher.inspect.backends.base import BackendOutcome
    from foren_searcher.inspect.service import InspectService

    outcome = BackendOutcome({"coverage": {"range_s": [0, 1], "sampling": {"base_fps": 2.0, "base_frames": 2, "scene_frames": 0,
                                                                           "refinement_frames": 0, "processed_frames": 2},
                                           "failed_ranges": []},
                              "results": [{"image_handle": "ocr-frame:0", "image_evidence_id": None, "time_s": 0.25,
                                           "items": [{"text": "A", "box": [0, 0, 1, 1], "confidence": 0.9}]},
                                          {"image_handle": "ocr-frame:1", "image_evidence_id": None, "time_s": 0.75, "items": []}]},
                             (), (), {})
    result = InspectService._public_result("ocr", outcome, {}, ("E1",))
    assert isinstance(result, OcrResult)
    assert [(r.image_evidence_id, r.time_s) for r in result.results] == [("E1", 0.25), ("E1", 0.75)]


def test_ocr_label_singular_counts():
    from types import SimpleNamespace

    from foren_searcher.inspect.service import ocr_label

    one = SimpleNamespace(text_summary=lambda limit=4: {"text_boxes": 1, "distinct_texts": 1, "texts": [{"text": "3IR2TT"}]})
    none = SimpleNamespace(text_summary=lambda limit=4: {"text_boxes": 0, "distinct_texts": 0, "texts": []})
    assert ocr_label(one, ("E1",)) == 'OCR of E1: 1 text box, 1 distinct text ("3IR2TT")'
    assert ocr_label(none, ("E2",)) == "OCR of E2: 0 text boxes, 0 distinct texts; no text recognised"
