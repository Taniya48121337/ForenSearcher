"""InspectService 编排和生命周期测试。

用假解析器/后端精确控制结果，验证校验时序、handle→E 映射、部分成功、审计与关闭语义。
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
import threading
from pathlib import Path

import pytest

from foren_searcher.inspect.backends.base import BackendFailure, BackendOutcome, MediaProfileData, PendingArtifact
from foren_searcher.inspect.backends.local import LocalInspectBackend
from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.inspect.policy import InspectPolicy
from foren_searcher.store import GlobalStore

from inspect_helpers import image_bytes, ingest


class PartialFrameBackend(LocalInspectBackend):
    """测试辅助类 PartialFrameBackend；用可控状态替代真实外部依赖。"""

    async def probe(self, path, media_kind):
        """模拟 probe 接口，返回由当前用例预设的可控结果。"""

        return MediaProfileData("video", width=32, height=24, duration_s=4, fps=10, frame_count=40, backend="fake-frame", version="1")

    async def get_frames(self, media, params, policy):
        """测试辅助函数 get_frames；为用例提供可重复的输入或可控行为。"""

        artifact = PendingArtifact("frame:0", image_bytes(size=(32,24)), "image/png", ".png", "video_frame", media[0].evidence_id, "frame", {"time_s":1.004}, "INSPECT/get_frames")
        issue = BackendFailure("DECODE_FAILED", "second frame failed", stage="decode", retryable=True, locator={"requested_time_s":2.0})
        return BackendOutcome({"frames":[{"handle":"frame:0","time_s":1.004,"width":32,"height":24}]},(artifact,),(issue,),{"backend":"fake-frame"})


class SlowBackend(LocalInspectBackend):
    """测试辅助类 SlowBackend；用可控状态替代真实外部依赖。"""

    async def crop(self, media, params, policy):
        """测试辅助函数 crop；为用例提供可重复的输入或可控行为。"""

        await asyncio.sleep(.2)
        return await super().crop(media, params, policy)


class BlockingBackend(LocalInspectBackend):
    """测试辅助类 BlockingBackend；用可控状态替代真实外部依赖。"""

    async def crop(self, media, params, policy):
        """测试辅助函数 crop；为用例提供可重复的输入或可控行为。"""

        time.sleep(.2)
        return await super().crop(media, params, policy)


class CountingBlockingBackend(LocalInspectBackend):
    """测试辅助类 CountingBlockingBackend；用可控状态替代真实外部依赖。"""

    def __init__(self):
        """保存测试替身需要的预设输入、状态和调用记录。"""

        super().__init__()
        self.active = 0
        self.maximum_active = 0
        self.lock = threading.Lock()

    async def crop(self, media, params, policy):
        """测试辅助函数 crop；为用例提供可重复的输入或可控行为。"""

        with self.lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            time.sleep(.05)
            return await super().crop(media, params, policy)
        finally:
            with self.lock:
                self.active -= 1


def execution_count(store: GlobalStore) -> int:
    """测试辅助函数 execution_count；为用例提供可重复的输入或可控行为。"""

    with sqlite3.connect(store.db_path) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM inspect_executions").fetchone()[0])


@pytest.mark.asyncio
async def test_partial_result_commits_only_real_artifact_and_closes_references(tmp_path: Path):
    """验证 partial result commits only real artifact and closes references 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore(tmp_path/"case.sqlite3",tmp_path/"artifacts")
    video = ingest(store,b"fixture",source="V0",filename="input.mp4",media_type="video/mp4",modality="video")
    service = create_inspect_service(store=store,backend=PartialFrameBackend())
    response = await service.execute({"targets":[video.evidence_id],"request":{"operation":"get_frames","params":{"times_s":[1,2]}}},workspace_id="W1-1")
    assert response.status == "partial" and len(response.result.frames) == 1
    assert response.result.frames[0].frame_duration_s is None
    assert response.to_dict()['result']['frames'][0]['frame_duration_s'] is None
    assert response.feedback[0].code == "DECODE_FAILED"
    assert response.result.frames[0].evidence_id in {x.evidence_id for x in response.evidence}
    record = store.inspect_execution(1)
    assert record["status"] == "partial" and record["finished_at"]


@pytest.mark.asyncio
async def test_static_invalid_call_creates_no_execution_but_valid_timeout_does(tmp_path: Path):
    """验证 static invalid call creates no execution but valid timeout does 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore(tmp_path/"case.sqlite3",tmp_path/"artifacts")
    image = ingest(store,image_bytes(),source="I0",filename="input.png",media_type="image/png",modality="image")
    service = create_inspect_service(store=store,backend=SlowBackend(),policy=InspectPolicy(execution_timeout_s=.02))
    invalid = await service.execute({"targets":["E01"],"request":{"operation":"crop","params":{"box":[0,0,1,1]}}},workspace_id="W1-1")
    assert invalid.status == "failed" and execution_count(store) == 0
    timed = await service.execute({"targets":[image.evidence_id],"request":{"operation":"crop","params":{"box":[0,0,1,1]}}},workspace_id="W1-1")
    assert timed.status == "failed" and timed.feedback[0].code == "TIMEOUT"
    assert execution_count(store) == 1 and store.inspect_execution(1)["finished_at"]


@pytest.mark.asyncio
async def test_blocking_backend_is_isolated_so_timeout_still_works(tmp_path: Path):
    """验证 blocking backend is isolated so timeout still works 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore(tmp_path/"case.sqlite3",tmp_path/"artifacts")
    image = ingest(store,image_bytes(),source="I0",filename="input.png",media_type="image/png",modality="image")
    service = create_inspect_service(store=store,backend=BlockingBackend(),policy=InspectPolicy(execution_timeout_s=.03))
    started = time.perf_counter()
    response = await service.execute({"targets":[image.evidence_id],"request":{"operation":"crop","params":{"box":[0,0,1,1]}}},workspace_id="W1-1")
    elapsed = time.perf_counter() - started
    assert response.status == "failed" and response.feedback[0].code == "TIMEOUT"
    assert elapsed < .15
    evidence_after_timeout = tuple(
        item.evidence_id for item in store.evidence_records()
    )
    await service.aclose()
    # 不可中断的同步处理器可以在专用线程中完成，但它只返回内存结果；
    # 超时后的 Service 协程已经退出，因此迟到的 Artifact 不得进入共享 Store。
    assert tuple(item.evidence_id for item in store.evidence_records()) == evidence_after_timeout
    assert store.inspect_execution(1)["status"] == "failed"


@pytest.mark.asyncio
async def test_backend_executor_is_bounded_and_close_waits_for_owned_work(tmp_path: Path):
    """验证 backend executor is bounded and close waits for owned work 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore(tmp_path/"case.sqlite3",tmp_path/"artifacts")
    image = ingest(store,image_bytes(),source="I0",filename="input.png",media_type="image/png",modality="image")
    backend = CountingBlockingBackend()
    service = create_inspect_service(store=store,backend=backend,policy=InspectPolicy(backend_workers=2, execution_timeout_s=2))
    responses = await asyncio.gather(*(service.execute({"targets":[image.evidence_id],
        "request":{"operation":"crop","params":{"box":[0,0,1-index/20,1]}}},
        workspace_id=f"W1-{index}") for index in range(6)))
    assert all(response.status == "ready" for response in responses)
    assert backend.maximum_active == 2
    await service.aclose()
    assert backend.active == 0
