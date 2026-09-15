"""固定 INSPECT 操作分发器；它从不自行读取 Store 状态。"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping

from .backends.base import BackendOutcome, PreparedMedia
from .contracts import InspectCall, InspectOperation
from .policy import InspectPolicy
from .processors.base import InspectProcessor


class InspectDispatcher:
    """把后端操作隔离到专用线程，保持主 asyncio 循环可响应。"""

    def __init__(self, probe_backend: Any, processors: Mapping[InspectOperation, InspectProcessor], *, maximum_workers: int = 4) -> None:
        """创建有界线程池，并检查十个 operation 都有明确 Processor。"""

        self.probe_backend = probe_backend
        self.processors = dict(processors)
        self._executor = ThreadPoolExecutor(max_workers=max(1, maximum_workers), thread_name_prefix="foren-inspect")
        self._closed = False
        self._local_model_lock: Any | None = None
        expected: tuple[InspectOperation, ...] = (
            "get_frames", "get_pages", "crop", "locate_regions", "ocr", "asr",
            "audio_qa", "visual_qa", "locate_frames", "align_media",
        )
        missing = [name for name in expected if name not in self.processors]
        if missing:
            raise ValueError(f"dispatcher is missing INSPECT processors: {missing}")

    def coordinate_local_models(self, lock: Any) -> None:
        """调查启动前接入本地代理的模型锁；外部后端保持原有调度。"""
        from .backends.local import LocalInspectBackend
        if isinstance(self.probe_backend, LocalInspectBackend):
            self._local_model_lock = lock
            with lock:
                self.probe_backend.release_gpu_models()

    async def _isolated(self, method: Any, *args: Any, gpu_operation: bool = False) -> Any:
        """在工作线程自己的事件循环中运行一个后端异步方法。"""

        if self._closed:
            raise RuntimeError("INSPECT dispatcher is closed")
        cancelled = threading.Event()
        def invoke() -> Any:
            """线程入口：建立并关闭该线程独有的事件循环。"""

            if gpu_operation and self._local_model_lock is not None:
                with self._local_model_lock:
                    if cancelled.is_set():
                        raise asyncio.CancelledError
                    try:
                        return asyncio.run(method(*args))
                    finally:
                        self.probe_backend.release_gpu_models()
            return asyncio.run(method(*args))
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._executor, invoke)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def aclose(self) -> None:
        """幂等关闭线程池，并取消尚未启动的排队任务。"""

        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)

    async def probe(self, path: Any, media_kind: str) -> Any:
        """在线程隔离边界内探测媒体属性。"""

        return await self._isolated(self.probe_backend.probe, path, media_kind)

    async def execute(self, call: InspectCall, media: tuple[PreparedMedia, ...], *, policy: InspectPolicy, query_media: PreparedMedia | None = None) -> BackendOutcome:
        """按 operation 选后端方法；LOCATE_FRAMES 额外传查询图片。"""

        processor = self.processors[call.request.operation]
        return await self._isolated(
            processor.process,
            media,
            call.request.params,
            policy,
            query_media,
            gpu_operation=self._local_model_lock is not None and self.probe_backend.uses_gpu_model(call.request.operation),
        )
