"""共享本地模型的 HI 批量队列；推理锁由队列持有，独立于请求生命周期。"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from contextvars import Context, copy_context
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Callable


@dataclass(eq=False)
class GenerationJob:
    system_prompt: str
    payload: dict[str, Any]
    stop_event: threading.Event
    queued_at: float
    on_start: Callable[[], None]
    result: asyncio.Future[Any]
    stopped: asyncio.Future[None]
    loop: asyncio.AbstractEventLoop
    context: Context = field(default_factory=copy_context)
    started: bool = False
    in_generation: bool = False
    cancel_ack_sent: bool = False
    on_generated: Callable[[Any], None] | None = None

    def publish(self, result: Any) -> None:
        """终态结果可先返回；同批其他请求仍由队列持有推理锁。"""
        def complete() -> None:
            if not self.result.done():
                if isinstance(result, BaseException):
                    self.result.set_exception(result)
                else:
                    self.result.set_result(result)
            if not self.stopped.done():
                self.stopped.set_result(None)
        self.loop.call_soon_threadsafe(complete)

    def acknowledge_cancel(self) -> None:
        """工作线程确认取消后，请求可退出；本批推理锁仍由队列持有。"""
        if self.cancel_ack_sent:
            return
        self.cancel_ack_sent = True
        def acknowledge() -> None:
            if not self.stopped.done():
                self.stopped.set_result(None)
        self.loop.call_soon_threadsafe(acknowledge)


class LocalGenerationQueue:
    def __init__(self, gate: asyncio.Lock, ensure_open: Callable[[], None],
                 generate: Callable[[list[GenerationJob]], list[Any]], max_batch_size: int) -> None:
        self.gate = gate
        self.ensure_open = ensure_open
        self.generate = generate
        self.max_batch_size = max_batch_size
        self.pending: list[GenerationJob] = []
        self.runner: asyncio.Task[None] | None = None
        self._collection_seconds = 0.0
        self._changed = asyncio.Event()

    def stop_collecting(self) -> None:
        """唤醒收批等待；调用方取得模型锁后再次清除上一批可能更新的窗口。"""
        self._collection_seconds = 0.0
        self._changed.set()

    async def _collect(self) -> None:
        while self.pending and len(self.pending) < self.max_batch_size:
            remaining = self.pending[0].queued_at + self._collection_seconds - time.perf_counter()
            if remaining <= 0:
                return
            self._changed.clear()
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=remaining)
            except TimeoutError:
                return

    async def submit(self, *, system_prompt: str, payload: dict[str, Any],
                     stop_event: threading.Event, queued_at: float, on_start: Callable[[], None]) -> Any:
        loop = asyncio.get_running_loop()
        job = GenerationJob(system_prompt, payload, stop_event, queued_at, on_start,
                            loop.create_future(), loop.create_future(), loop)
        self.pending.append(job)
        self._changed.set()
        if self.runner is None:
            self.runner = asyncio.create_task(self._run())
        try:
            return await asyncio.shield(job.result)
        except asyncio.CancelledError:
            stop_event.set()
            job.result.cancel()
            if not job.started:
                if job in self.pending:
                    self.pending.remove(job)
                self._changed.set()
            elif not job.in_generation:
                # 仍在准备输入或等待单行回退；工作线程会在 generate 前检查 stop_event。
                if not job.stopped.done():
                    job.stopped.set_result(None)
            else:
                while not job.stopped.done():
                    try:
                        await asyncio.shield(job.stopped)
                    except asyncio.CancelledError:
                        continue
            raise

    async def _run(self) -> None:
        try:
            while self.pending:
                async with self.gate:
                    await self._collect()
                    jobs: list[GenerationJob] = []
                    dispatch_stopped = threading.Event()
                    loop = asyncio.get_running_loop()
                    batch_started: float
                    def generate_on_worker() -> list[Any]:
                        nonlocal batch_started
                        ready: Future[None] = Future()
                        def select() -> None:
                            try:
                                # 工作线程实际可用时才取队列，包含线程等待期间到达的 HI。
                                jobs.extend(self.pending[:self.max_batch_size])
                                del self.pending[:len(jobs)]
                                self.ensure_open()
                                for job in jobs:
                                    job.started = True
                                    job.on_start()
                                    if dispatch_stopped.is_set():
                                        job.stop_event.set()
                                ready.set_result(None)
                            except BaseException as exc:
                                ready.set_exception(exc)
                        loop.call_soon_threadsafe(select)
                        ready.result()
                        batch_started = time.perf_counter()
                        return self.generate(jobs) if jobs else []
                    try:
                        batch_started = time.perf_counter()
                        worker = asyncio.create_task(asyncio.to_thread(generate_on_worker))
                        try:
                            results = await asyncio.shield(worker)
                        except asyncio.CancelledError:
                            dispatch_stopped.set()
                            for job in jobs:
                                job.stop_event.set()
                            # 即使服务关闭时多次取消，也不让另一次 generate 提前进入。
                            while not worker.done():
                                try:
                                    await asyncio.shield(worker)
                                except asyncio.CancelledError:
                                    continue
                                except BaseException:
                                    break
                            raise
                        # 只在刚完成多 HI 推理后短暂收批；单条或失败请求恢复即时调度。
                        self._collection_seconds = (
                            min(2.0, (time.perf_counter() - batch_started) * .05)
                            if len(jobs) > 1 and all(not job.stop_event.is_set() for job in jobs)
                            and all(not isinstance(result, BaseException) for result in results) else 0.0
                        )
                        for job, result in zip(jobs, results, strict=True):
                            if not job.result.done():
                                if isinstance(result, BaseException):
                                    job.result.set_exception(result)
                                else:
                                    job.result.set_result(result)
                    except BaseException as exc:
                        self._collection_seconds = 0.0
                        for job in jobs:
                            if not job.result.done():
                                job.result.set_exception(exc)
                    finally:
                        for job in jobs:
                            if not job.stopped.done():
                                job.stopped.set_result(None)
        finally:
            self.runner = None
