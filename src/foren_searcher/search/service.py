"""SEARCH 的主编排服务：预算、缓存复用、来源打开和统一发布。

``SearchService.execute`` 是 Workspace 唯一面对的 SEARCH 入口。它先验证请求并创建
审计记录；查询请求经过 Provider 发现 R 候选，直接打开请求则复用或创建一个 R；
随后并发打开候选，将每份材料提交为 E，最后返回固定形状的 SearchResponse。
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from PIL import Image

from ..store.resolver import EvidenceResolutionError, EvidenceResolver
from ..store import GlobalStore
from .contracts import (
    CandidateBatch, CandidateSource, MaterialResult, OpenSourceRequest,
    OpenedSource, OutputType, SearchError, SearchFailure, SearchQueryRequest,
    SearchRequest, SearchResponse, SearchResult, parse_search_request,
)
from .identity import candidate_identity, normalize_url
from .openers.dispatcher import SourceDispatcher
from .policy import SearchPolicy
from .providers.base import SearchProvider


# 只有瞬时网络类错误值得在同一次动作预算内重试。
_RETRYABLE = {"NETWORK_ERROR", "TIMEOUT", "PROVIDER_ERROR", "RATE_LIMITED"}
_LOGGER = logging.getLogger(__name__)


class SearchService:
    """模型可调用的唯一 SEARCH 实现。"""

    def __init__(self, *, store: GlobalStore, provider: SearchProvider, opener: SourceDispatcher,
                 policy: SearchPolicy | None = None, close_callback: Any | None = None,
                 allow_image_uploads: bool = True, resolver: EvidenceResolver | None = None) -> None:
        """注入存储、发现 Provider、打开器及资源关闭回调。"""

        self.store, self.provider, self.opener = store, provider, opener
        self.policy = policy or SearchPolicy()
        self._close_callback = close_callback
        self._allow_image_uploads = allow_image_uploads
        self.resolver = resolver or EvidenceResolver(store)
        self._closed = False
        # 关闭服务时要等正在执行的调用收尾，防止先关网络客户端。
        self._active_tasks: set[asyncio.Task[Any]] = set()
        self._close_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> SearchService:
        """进入异步上下文，并拒绝复用已关闭服务。"""

        if self._closed:
            raise RuntimeError("SEARCH service is closed")
        return self

    async def __aexit__(self, *_args: object) -> None:
        """离开异步上下文时可靠关闭服务。"""

        await self.aclose()

    async def aclose(self) -> None:
        """幂等关闭；shield 防止等待者取消时连关闭任务也被取消。"""

        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        """等待活跃 SEARCH 调用结束，然后释放底层资源。"""

        current = asyncio.current_task()
        pending = tuple(task for task in self._active_tasks if task is not current and not task.done())
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if self._close_callback is not None:
            await self._close_callback()

    async def execute(self, call: SearchRequest | dict[str, Any], *, workspace_id: str) -> SearchResponse:
        """跟踪当前任务并执行一次完整 SEARCH。"""

        if self._closed:
            raise RuntimeError("SEARCH service is closed")
        task = asyncio.current_task()
        if task is not None:
            self._active_tasks.add(task)
        try:
            return await self._execute(call, workspace_id=workspace_id)
        finally:
            if task is not None:
                self._active_tasks.discard(task)

    async def _execute(self, call: SearchRequest | dict[str, Any], *, workspace_id: str) -> SearchResponse:
        """处理请求验证、主流程、异常收敛和审计终结。"""

        try:
            request = parse_search_request(call) if isinstance(call, dict) else call
            if not isinstance(request, (SearchQueryRequest, OpenSourceRequest)):
                raise ValueError("unsupported SEARCH request")
        except (TypeError, ValueError) as exc:
            # 请求尚未建审计记录；仍返回一个契约合法的 error 响应。
            output = call.get("output_type") if isinstance(call, dict) else "text"
            output = output if isinstance(output, str) and output in {"text", "image", "video"} else "text"
            return SearchResponse(output, "error", (), SearchError("validate", "INVALID_INPUT", str(exc)))  # type: ignore[arg-type]

        try:
            execution_id = self.store.create_search_execution(request.to_dict(), workspace_id)
        except Exception as exc:
            _LOGGER.error("SEARCH execution could not be created", extra={"search_exception_type": type(exc).__name__})
            return SearchResponse(request.output_type, "error", (),
                SearchError("save", "SAVE_FAILED", f"SEARCH execution could not be started ({type(exc).__name__})"))
        # monotonic 不受系统时钟调整影响，适合预算和耗时测量。
        started = time.monotonic()
        _LOGGER.info("SEARCH execution started", extra={"search_execution_id": execution_id})
        results: list[SearchResult] = []
        provider_requests = 0
        successes = 0
        top_error: SearchError | None = None
        stop_reason = "candidates_exhausted"
        cancelled: asyncio.CancelledError | None = None
        try:
            if isinstance(request, OpenSourceRequest):
                # 直接打开只处理一个来源，因此使用单来源总超时。
                candidate, candidate_id = self._known_candidate(request, execution_id)
                try:
                    result, success, _links = await asyncio.wait_for(
                        self._open_candidate(candidate, candidate_id, request, workspace_id, execution_id),
                        timeout=self.policy.single_source_total_s,
                    )
                except asyncio.TimeoutError:
                    timeout_error = SearchError("open", "TIMEOUT", "source opening reached its time limit")
                    result = self._failed_result(candidate, candidate_id, request.output_type, timeout_error, request)
                    success = False
                    top_error = timeout_error
                results.append(result)
                successes = int(success)
                stop_reason = "target_reached" if success else "timeout" if top_error else "candidates_exhausted"
            else:
                # 查询路径负责 Provider 分页、并发打开和目标数量控制。
                results, successes, provider_requests, stop_reason = await self._execute_query(
                    request, execution_id, workspace_id, started)
                if stop_reason == "timeout":
                    top_error = SearchError("open", "TIMEOUT", "SEARCH reached its total time limit; committed materials were retained")
        except asyncio.TimeoutError:
            stop_reason = "timeout"
            top_error = SearchError("open", "TIMEOUT", "SEARCH reached its total time limit; committed materials were retained")
        except asyncio.CancelledError as exc:
            # 先完成审计记录，再在函数末尾重新抛出取消。
            cancelled = exc
            stop_reason = "error"
            top_error = SearchError("open", "TIMEOUT", "SEARCH execution was cancelled; committed materials were retained")
        except SearchFailure as exc:
            stop_reason = "error"
            top_error = exc.error
        except Exception as exc:
            _LOGGER.error("SEARCH execution failed unexpectedly", extra={
                "search_execution_id": execution_id, "search_exception_type": type(exc).__name__,
            })
            stop_reason = "error"
            top_error = SearchError("save", "SAVE_FAILED", f"SEARCH could not publish its records ({type(exc).__name__})")
        # 到这里所有内部异常都被收敛成稳定 SearchResponse。
        response = SearchResponse(request.output_type, stop_reason, tuple(results), top_error)  # type: ignore[arg-type]
        elapsed_ms = int((time.monotonic() - started) * 1000)
        try:
            # 从 Store 重新统计比信任局部计数更适合异常/取消路径。
            provider_requests = self.store.provider_call_count(execution_id)
            candidate_count = self.store.execution_candidate_count(execution_id) if isinstance(request, SearchQueryRequest) else len(results)
            self.store.finish_search_execution(execution_id, stop_reason=stop_reason, error=top_error.to_dict() if top_error else None,
                provider_request_count=provider_requests, candidate_count=candidate_count,
                successful_source_count=successes, elapsed_ms=elapsed_ms)
        except Exception as exc:
            _LOGGER.error("SEARCH execution record could not be finalized", extra={
                "search_execution_id": execution_id, "search_exception_type": type(exc).__name__,
            })
            if cancelled is None:
                response = SearchResponse(request.output_type, "error", tuple(results),
                    SearchError("save", "SAVE_FAILED", f"SEARCH execution record could not be finalized ({type(exc).__name__})"))
        _LOGGER.info("SEARCH execution finished", extra={
            "search_execution_id": execution_id,
            "search_stop_reason": stop_reason,
            "search_result_count": len(results),
            "search_successful_source_count": successes,
            "search_elapsed_ms": elapsed_ms,
        })
        if cancelled is not None:
            # 调用方仍需观察到 CancelledError，不能被普通响应吞掉。
            raise cancelled
        return response

    async def _execute_query(
        self, request: SearchQueryRequest, execution_id: int, workspace_id: str, started: float,
    ) -> tuple[list[SearchResult], int, int, str]:
        """执行查询型 SEARCH，返回结果、成功数、请求数和停止原因。"""

        output: list[SearchResult] = []
        successes = 0
        successful_candidates: set[str] = set()
        provider_requests = 0
        offset = 0
        seen: set[str] = set()
        candidates: list[tuple[CandidateSource, str]] = []
        image_temp: tempfile.TemporaryDirectory[str] | None = None
        image_path: Path | None = None
        try:
            if request.input_type in {"image", "text_image"}:
                # 上传外部 Provider 前先检查部署策略并生成受控 JPEG 副本。
                if not self._allow_image_uploads:
                    raise SearchFailure(SearchError("upload", "ACCESS_RESTRICTED", "this deployment does not allow sending investigation images to a search provider"))
                evidence_id = request.query if isinstance(request.query, str) else request.query["evidence_id"]
                image_temp, image_path, adaptation = self._prepare_query_image(evidence_id)
                upload_artifact = self.store.save_artifact(image_path.read_bytes(), media_type="image/jpeg", suffix=".jpg", role="search_query_upload")
                self.store.record_search_image_upload(execution_id, evidence_id, upload_artifact, adaptation)
            while provider_requests < self.policy.maximum_provider_requests and len(candidates) < self.policy.maximum_candidates:
                # 每轮 Provider 请求获得下一批候选，直到命中任一预算。
                try:
                    batch, used, raw_ref = await self._provider_batch(
                        request, offset, self.policy.initial_candidates, image_path,
                        execution_id, provider_requests, started,
                    )
                except SearchFailure as exc:
                    if not output or exc.error.code == "SAVE_FAILED":
                        raise
                    reason = "timeout" if time.monotonic() - started >= self.policy.search_total_s else "candidates_exhausted"
                    return output, successes, provider_requests, reason
                provider_requests += used
                for candidate in batch.candidates:
                    # Candidate 先全局去重并持久化，再登记本次发现轨迹。
                    identity = candidate_identity(candidate)
                    candidate_id, _created = self.store.get_or_create_candidate(
                        identity_key=identity, source_url=candidate.source_url,
                        search_title=candidate.search_title, search_snippet=candidate.search_snippet,
                        provider=candidate.provider or batch.provider, engine=candidate.engine or batch.engine,
                        provider_rank=candidate.provider_rank, discovered_execution_id=execution_id,
                        raw_response_ref=raw_ref, candidate_type=candidate.candidate_type,
                        image_url=candidate.image_url, thumbnail_url=candidate.thumbnail_url,
                        platform=candidate.platform, platform_video_id=candidate.platform_video_id,
                        extra=candidate.extra)
                    self.store.record_candidate_discovery(
                        execution_id, candidate_id,
                        provider=candidate.provider or batch.provider,
                        engine=candidate.engine or batch.engine,
                        provider_rank=candidate.provider_rank,
                        raw_response_ref=raw_ref,
                    )
                    if identity in seen:
                        continue
                    seen.add(identity)
                    candidates.append((candidate, candidate_id))
                    if len(candidates) >= self.policy.maximum_candidates:
                        break
                while len(output) < len(candidates):
                    candidate, candidate_id = candidates[len(output)]
                    if successes >= self.policy.target_sources:
                        # 仍把剩余候选显式返回为 not_requested，响应形状可审计。
                        output.append(self._not_requested(candidate, candidate_id, request.output_type, request))
                        continue
                    remaining = self.policy.search_total_s - (time.monotonic() - started)
                    if remaining <= 0:
                        output.extend(self._not_requested(item, item_id, request.output_type, request)
                                      for item, item_id in candidates[len(output):])
                        return output, successes, provider_requests, "timeout"
                    follows_video_links = request.input_type in {"image", "text_image"} and request.output_type == "video"
                    if follows_video_links:
                        # 图片找视频需要从匹配页面逐个抽取一跳视频链接，保持顺序。
                        opened = [await self._open_candidate_with_timeout(
                            candidate, candidate_id, request, workspace_id, execution_id, remaining)]
                    else:
                        # 普通候选可并发打开，但不超过剩余目标和并发预算。
                        needed = self.policy.target_sources - successes
                        width = min(max(1, self.policy.source_concurrency), needed, len(candidates) - len(output))
                        selected = candidates[len(output):len(output) + width]
                        opened = list(await asyncio.gather(*(self._open_candidate_with_timeout(
                            item, item_id, request, workspace_id, execution_id, remaining)
                            for item, item_id in selected)))
                    for (current, current_id), (result, success, links) in zip(
                        candidates[len(output):len(output) + len(opened)], opened, strict=True,
                    ):
                        output.append(result)
                        # 不同候选 URL 可能跳转到同一 R，不能重复占用来源目标。
                        if success:
                            successful_candidates.add(result.candidate_id)
                        successes = len(successful_candidates)
                        if not follows_video_links or success:
                            continue
                        for link in links:
                            # 仅跟随识别为视频的一跳链接，且仍受 maximum_candidates 约束。
                            if _looks_like_video_url(link) and len(candidates) < self.policy.maximum_candidates:
                                follow = CandidateSource(link, provider="one-hop", engine="page-link",
                                    provider_rank=len(candidates) + 1, candidate_type="video")
                                identity = candidate_identity(follow)
                                if identity not in seen:
                                    seen.add(identity)
                                    follow_id, _ = self.store.get_or_create_candidate(identity_key=identity, source_url=link,
                                        provider="one-hop", engine="page-link", provider_rank=len(candidates) + 1,
                                        discovered_execution_id=execution_id, candidate_type="video", extra={"from_candidate": candidate_id})
                                    self.store.record_candidate_discovery(
                                        execution_id, follow_id, provider="one-hop", engine="page-link",
                                        provider_rank=len(candidates) + 1, raw_response_ref=None,
                                    )
                                    candidates.append((follow, follow_id))
                if successes >= self.policy.target_sources:
                    return output, successes, provider_requests, "target_reached"
                if not batch.candidates or provider_requests >= self.policy.maximum_provider_requests:
                    break
                offset += self.policy.initial_candidates
            return output, successes, provider_requests, "target_reached" if successes >= self.policy.target_sources else "candidates_exhausted"
        finally:
            # 查询图片副本只用于本次 Provider 调用，结束后无条件清理。
            if image_temp is not None:
                image_temp.cleanup()

    async def _open_candidate_with_timeout(
        self, candidate: CandidateSource, candidate_id: str, request: SearchRequest,
        workspace_id: str, execution_id: int, remaining_s: float,
    ) -> tuple[SearchResult, bool, tuple[str, ...]]:
        """用单来源预算和全局剩余预算的较小值打开候选。"""

        try:
            return await asyncio.wait_for(
                self._open_candidate(candidate, candidate_id, request, workspace_id, execution_id),
                timeout=min(self.policy.single_source_total_s, remaining_s),
            )
        except asyncio.TimeoutError:
            return (
                self._failed_result(candidate, candidate_id, request.output_type,
                    SearchError("open", "TIMEOUT", "source opening reached its time limit"), request),
                False,
                (),
            )

    async def _provider_batch(
        self, request: SearchQueryRequest, offset: int, limit: int, image_path: Path | None,
        execution_id: int, requests_used: int, started: float,
    ) -> tuple[CandidateBatch, int, str]:
        """在请求预算内调用 Provider、记录每次尝试并保存原始响应。"""

        used = 0
        last: SearchFailure | None = None
        while requests_used + used < self.policy.maximum_provider_requests:
            attempt = requests_used + used + 1
            before = time.monotonic()
            try:
                # Provider 超时同时受单次上限和整个 SEARCH 剩余时间约束。
                remaining = self.policy.search_total_s - (time.monotonic() - started)
                if remaining <= 0:
                    raise asyncio.TimeoutError
                batch = await asyncio.wait_for(self.provider.search(request, offset=offset, limit=limit, image_path=image_path), timeout=min(self.policy.provider_timeout_s, remaining))
                used += 1
                raw_ref = self._save_provider_response(batch)
                # 只有原始响应已经可靠保存后，这次调用才记作 success。
                self.store.record_provider_call(execution_id, provider=batch.provider, engine=batch.engine, attempt=attempt,
                    elapsed_ms=int((time.monotonic() - before) * 1000), outcome="success", raw_response_ref=raw_ref,
                    request_metadata={"offset": offset, "limit": limit, **batch.request_metadata})
                return batch, used, raw_ref
            except asyncio.CancelledError:
                # 尽力记录取消，但审计失败不能取代原本的任务取消语义。
                try:
                    self.store.record_provider_call(
                        execution_id, provider=type(self.provider).__name__, engine="unknown",
                        attempt=attempt, elapsed_ms=int((time.monotonic() - before) * 1000),
                        outcome="cancelled",
                        error=SearchError("search", "TIMEOUT", "search provider call was cancelled").to_dict(),
                        request_metadata={"offset": offset, "limit": limit},
                    )
                except Exception as exc:
                    _LOGGER.error(
                        "SEARCH provider cancellation record could not be saved",
                        extra={"search_execution_id": execution_id, "search_exception_type": type(exc).__name__},
                    )
                raise
            except asyncio.TimeoutError:
                failure = SearchFailure(SearchError("search", "TIMEOUT", "search provider timed out"))
            except SearchFailure as exc:
                failure = exc
            used += 1
            last = failure
            self.store.record_provider_call(execution_id, provider=type(self.provider).__name__, engine="unknown", attempt=attempt,
                elapsed_ms=int((time.monotonic() - before) * 1000), outcome="error", error=failure.error.to_dict(),
                request_metadata={"offset": offset, "limit": limit})
            if failure.error.code not in _RETRYABLE or requests_used + used >= self.policy.maximum_provider_requests:
                break
            if failure.error.code == "RATE_LIMITED":
                # 仅当 Provider 给出等待时间且全局预算足够时才 sleep。
                wait = failure.retry_after_s
                remaining = self.policy.search_total_s - (time.monotonic() - started)
                if wait is None or wait > remaining:
                    break
                await asyncio.sleep(wait)
        assert last is not None
        raise last

    def _save_provider_response(self, batch: CandidateBatch) -> str:
        """把 Provider 原始 JSON 保存成 A 引用，支持事后复核。"""

        try:
            payload = json.dumps(batch.raw_response, ensure_ascii=False, default=str).encode("utf-8")
            return self.store.save_artifact(payload, media_type="application/json", suffix=".json", role="provider_response")
        except Exception as exc:
            raise SearchFailure(SearchError("save", "SAVE_FAILED", "raw provider response could not be saved")) from exc

    def _known_candidate(self, request: OpenSourceRequest, execution_id: int) -> tuple[CandidateSource, str]:
        """解析直接打开请求中的已有 R，或为 URL 建立 Candidate。"""

        if request.source.startswith("R"):
            try:
                record = self.store.candidate(request.source)
            except KeyError as exc:
                raise SearchFailure(SearchError("validate", "INVALID_INPUT", "Candidate Source does not exist")) from exc
            return self._candidate_from_record(record), request.source
        # 直接 URL 没有 Provider 类型提示，只能由输出目标和 .pdf 后缀保守推断。
        candidate_type: Literal["image", "video", "document", "page"] = "image" if request.output_type == "image" else "video" if request.output_type == "video" else "document" if urlsplit(request.source).path.lower().endswith(".pdf") else "page"
        candidate = CandidateSource(request.source, provider="direct", engine="direct", candidate_type=candidate_type)
        candidate_id, _ = self.store.get_or_create_candidate(identity_key=candidate_identity(candidate),
            source_url=request.source, provider="direct", engine="direct", provider_rank=1,
            discovered_execution_id=execution_id, candidate_type=candidate_type,
            image_url=request.source if candidate_type == "image" else None, extra={})
        if candidate_type == "image":
            candidate = replace(candidate, image_url=request.source)
        return candidate, candidate_id

    async def _open_candidate(
        self, candidate: CandidateSource, candidate_id: str, request: SearchRequest,
        workspace_id: str, execution_id: int,
    ) -> tuple[SearchResult, bool, tuple[str, ...]]:
        """复用已完成材料；尚未取得结果的请求独立打开并提交 Evidence。"""

        cached_rows = self.store.candidate_evidence(candidate_id)
        cached: dict[str, MaterialResult] = {}
        for kind, status, evidence in cached_rows:
            # 同一种材料可能有多条 Evidence（例如不同语言字幕），合并卡片。
            existing = cached.get(kind)
            cards = (() if existing is None else existing.evidence) + (evidence.card(),)
            cached[kind] = MaterialResult(kind, status, cards, None if status == "ready" else SearchError("extract", "EXTRACT_FAILED", "previous acquisition was partial"))  # type: ignore[arg-type]
        if self._cached_satisfies(cached, request.output_type, request, candidate):
            # 缓存已经满足此次具体请求时，不再碰网络。
            self._record_source_attempt(
                execution_id, candidate_id, attempt=0, elapsed_ms=0, outcome="reused",
            )
            result = self._assemble_result(candidate, candidate_id, cached, None, request)
            return result, True, ()
        candidate_to_open = candidate
        if request.output_type == "image" and cached:
            # 告诉 ImageSourceOpener 哪些材料已缓存，让它只补缺项。
            ready_kinds = tuple(kind for kind, material in cached.items() if material.status in {"ready", "partial"})
            candidate_to_open = replace(candidate, extra={**candidate.extra, "_search_cached_materials": ready_kinds})
        opened: OpenedSource | None = None
        last_error: SearchFailure | None = None
        for attempt in range(2):
            # 单个来源最多尝试两次，且只重试瞬时错误。
            before = time.monotonic()
            try:
                opened = await self.opener.open(candidate_to_open, request)
            except asyncio.CancelledError:
                self._record_source_attempt(
                    execution_id, candidate_id, attempt=attempt + 1,
                    elapsed_ms=int((time.monotonic() - before) * 1000), outcome="cancelled",
                    error=SearchError("open", "TIMEOUT", "source opening was cancelled").to_dict(),
                )
                raise
            except SearchFailure as exc:
                last_error = exc
                self._record_source_attempt(
                    execution_id, candidate_id, attempt=attempt + 1,
                    elapsed_ms=int((time.monotonic() - before) * 1000), outcome="error",
                    error=exc.error.to_dict(),
                )
                if exc.error.code not in _RETRYABLE or attempt:
                    break
                if exc.error.code == "RATE_LIMITED":
                    if exc.retry_after_s is None:
                        break
                    await asyncio.sleep(exc.retry_after_s)
            except Exception as exc:
                # 打开器异常仅影响本来源；不能让并发取得的其他材料消失于响应。
                error = (SearchError("fetch", "AUTH_ERROR", "source TLS certificate could not be verified")
                         if isinstance(exc, ssl.SSLCertVerificationError)
                         else SearchError("fetch", "NETWORK_ERROR", "source TLS connection failed")
                         if isinstance(exc, ssl.SSLError)
                         else SearchError("open", "EXTRACT_FAILED", f"source opener failed ({type(exc).__name__})"))
                last_error = SearchFailure(error)
                _LOGGER.error("SEARCH source opener failed unexpectedly", extra={
                    "search_execution_id": execution_id, "search_candidate_id": candidate_id,
                    "search_exception_type": type(exc).__name__,
                })
                self._record_source_attempt(
                    execution_id, candidate_id, attempt=attempt + 1,
                    elapsed_ms=int((time.monotonic() - before) * 1000), outcome="error",
                    error=error.to_dict(),
                )
                break
            else:
                self._record_source_attempt(
                    execution_id, candidate_id, attempt=attempt + 1,
                    elapsed_ms=int((time.monotonic() - before) * 1000), outcome="success",
                )
                break
        if opened is None:
            assert last_error is not None
            return self._failed_result(candidate, candidate_id, request.output_type, last_error.error, request), False, ()
        if candidate.candidate_type == "image":
            # 图片身份同时依赖图片 URL 与来源页，跳转后需重新计算。
            final_identity = candidate_identity(replace(candidate, source_url=opened.final_url))
        elif candidate.platform and candidate.platform_video_id:
            final_identity = candidate_identity(candidate)
        else:
            final_identity = f"url:{normalize_url(opened.final_url)}"
        canonical_id = self.store.update_candidate_final_url(candidate_id, opened.final_url, final_identity)
        if canonical_id != candidate_id:
            # 最终 URL 可能揭示它与已有候选相同，切换到规范 R 并重载缓存。
            candidate_id = canonical_id
            cached = {}
            for kind, status, evidence in self.store.candidate_evidence(candidate_id):
                existing = cached.get(kind)
                cards = (() if existing is None else existing.evidence) + (evidence.card(),)
                cached[kind] = MaterialResult(kind, status, cards, None if status == "ready" else SearchError("extract", "EXTRACT_FAILED", "previous acquisition was partial"))  # type: ignore[arg-type]
        combined = dict(cached)
        for material in opened.materials:
            # 与缓存内容相同的材料无需再次写入 Artifact/Evidence。
            if (material.kind in cached and cached[material.kind].status in {"ready", "partial"}
                    and self._opened_material_is_cached(cached[material.kind], material)):
                continue
            if material.status in {"ready", "partial"}:
                try:
                    # 文件、E 和 R→E 由 Store 在一个受保护提交序列中保存。
                    evidence = self.store.commit_material_evidence(candidate_id=candidate_id,
                        material_kind=material.kind, material_status=material.status,
                        artifact_bytes=material.artifact_bytes, media_type=material.media_type,
                        suffix=material.suffix, modalities=material.modalities, label=material.label,
                        content=material.content, locator=material.locator, acquired_by=material.acquired_by,
                        discovered_by=workspace_id, production_method=material.production_method)
                except Exception as exc:
                    raise SearchFailure(SearchError("save", "SAVE_FAILED", "source material could not be committed")) from exc
                previous = combined.get(material.kind)
                cards = (() if previous is None else previous.evidence) + (evidence.card(),)
                combined[material.kind] = MaterialResult(material.kind, material.status, cards, material.error)
            else:
                combined[material.kind] = MaterialResult(material.kind, material.status, (), material.error)
        result = self._assemble_result(candidate, candidate_id, combined, opened, request)
        success = self._is_success(combined, request.output_type, opened, request)
        return result, success, opened.discovered_links

    def _record_source_attempt(
        self, execution_id: int, candidate_id: str, *, attempt: int,
        elapsed_ms: int, outcome: str, error: dict[str, Any] | None = None,
    ) -> None:
        """尽力写来源打开审计；审计失败只记录日志，不覆盖业务结果。"""

        try:
            self.store.record_source_open_attempt(
                execution_id, candidate_id, attempt=attempt, elapsed_ms=elapsed_ms,
                outcome=outcome, error=error,
            )
        except Exception as exc:
            _LOGGER.error(
                "SEARCH source attempt record could not be saved",
                extra={"search_execution_id": execution_id, "search_candidate_id": candidate_id,
                       "search_exception_type": type(exc).__name__},
            )

    def _assemble_result(self, candidate: CandidateSource, candidate_id: str, materials: dict[str, MaterialResult], opened: OpenedSource | None, request: SearchRequest) -> SearchResult:
        """按输出类型固定顺序补齐材料槽位，并组装 SearchResult。"""

        expected = _expected_kinds(request)
        ordered = tuple(materials.get(kind, MaterialResult(kind, "not_requested")) for kind in expected)
        download_check = None
        video_material = None
        if request.output_type == "video":
            download_check = opened.download_check if opened and opened.download_check else {
                **self._cached_download_check(materials),
            }
            video_material = opened.video_material if opened and opened.video_material else self._cached_video_material(materials)
        return SearchResult(candidate_id, candidate.source_url, candidate.search_title, candidate.search_snippet, ordered,
            download_check, video_material)

    def _failed_result(self, candidate: CandidateSource, candidate_id: str, output_type: OutputType, error: SearchError, request: SearchRequest) -> SearchResult:
        """为完全未打开的候选构造形状稳定的失败结果。"""

        target = "image" if output_type == "image" else "document" if output_type == "text" and candidate.candidate_type == "document" else "page"
        materials = tuple(MaterialResult(kind, "failed", (), error) if kind == target else MaterialResult(kind, "not_requested") for kind in _expected_kinds(request))
        return SearchResult(candidate_id, candidate.source_url, candidate.search_title, candidate.search_snippet, materials,
            {"status": "blocked" if error.code in {"ACCESS_RESTRICTED", "AUTH_ERROR"} else "unknown",
             "reason": error.message} if output_type == "video" else None,
            {"status": "not_requested", "evidence_id": None, "error": None} if output_type == "video" else None)

    def _not_requested(self, candidate: CandidateSource, candidate_id: str, output_type: OutputType, request: SearchRequest) -> SearchResult:
        """为因目标已满足而未打开的候选构造显式占位结果。"""

        return SearchResult(candidate_id, candidate.source_url, candidate.search_title, candidate.search_snippet,
            tuple(MaterialResult(kind, "not_requested") for kind in _expected_kinds(request)),
            {"status": "not_checked", "reason": None} if output_type == "video" else None,
            {"status": "not_requested", "evidence_id": None, "error": None} if output_type == "video" else None)

    def _is_success(
        self, materials: dict[str, MaterialResult], output_type: OutputType,
        opened: OpenedSource | None, request: SearchRequest,
    ) -> bool:
        """依据输出类型判断这次来源打开是否计入 target_sources。"""

        if output_type == "text":
            # 仅元数据不能算作获得了正文证据。
            if opened is not None:
                return any(item.kind in {"page", "document"} and item.status in {"ready", "partial"}
                           and item.production_method != "source_metadata" for item in opened.materials)
            return any(
                material.status in {"ready", "partial"}
                and any(self.store.evidence(card.evidence_id).production_method != "source_metadata" for card in material.evidence)
                for kind in ("page", "document") if (material := materials.get(kind)) is not None)
        if output_type == "image":
            image_ready = bool(materials.get("image") and materials["image"].status in {"ready", "partial"})
            if isinstance(request, OpenSourceRequest):
                return image_ready
            page_ready = bool(materials.get("page") and materials["page"].status in {"ready", "partial"})
            return image_ready or page_ready
        # 视频搜索目前以能确认具体视频及其来源元数据为成功标准。
        return bool(opened and opened.video_material is not None and materials.get("page") and materials["page"].status in {"ready", "partial"})

    def _cached_satisfies(
        self, materials: dict[str, MaterialResult], output_type: OutputType,
        request: SearchRequest, candidate: CandidateSource,
    ) -> bool:
        """判断已有 Candidate Evidence 是否完整满足当前请求细节。"""

        if output_type == "text":
            return any(
                material.status in {"ready", "partial"}
                and any(self.store.evidence(card.evidence_id).production_method != "source_metadata" for card in material.evidence)
                for kind in ("page", "document") if (material := materials.get(kind)) is not None)
        if output_type == "image":
            image_ready = bool(materials.get("image") and materials["image"].status in {"ready", "partial"})
            page_ready = bool(materials.get("page") and materials["page"].status in {"ready", "partial"})
            if isinstance(request, OpenSourceRequest):
                # 直接打开图片时，如来源页与原图不同，两者都应已取得。
                page_expected = bool(candidate.image_url and candidate.source_url != candidate.image_url)
                return image_ready and (page_ready or not page_expected)
            return image_ready or page_ready
        page = materials.get("page")
        if not page or page.status not in {"ready", "partial"} or not any("yt-dlp" in card.acquired_by for card in page.evidence):
            return False
        if isinstance(request, OpenSourceRequest) and request.include == ("subtitles",):
            # 请求了特定字幕时，任意旧字幕并不足够，语言也必须匹配。
            subtitle = materials.get("subtitle")
            return bool(subtitle and subtitle.status in {"ready", "partial"}
                        and request.subtitle_language is not None
                        and self._cached_subtitle_has_language(subtitle, request.subtitle_language))
        return True

    @staticmethod
    def _cached_video_material(materials: dict[str, MaterialResult]) -> dict[str, Any]:
        """从缓存材料恢复视频文件的公开状态字段。"""

        video = materials.get("video")
        if video and video.status in {"ready", "partial"} and video.evidence:
            return {"status": "ready", "evidence_id": video.evidence[0].evidence_id, "error": None}
        return {"status": "not_requested", "evidence_id": None, "error": None}

    def _cached_download_check(self, materials: dict[str, MaterialResult]) -> dict[str, Any]:
        """从缓存的 yt-dlp 原始 JSON 恢复 formats_found 状态。"""

        page = materials.get("page")
        if page:
            for card in page.evidence:
                if "yt-dlp" not in card.acquired_by:
                    continue
                try:
                    # 原始 JSON Artifact 是下载格式判断的权威来源。
                    evidence = self.store.evidence(card.evidence_id)
                    if evidence.artifact_ref is None:
                        continue
                    raw = json.loads(self.store.artifact_path(evidence.artifact_ref).read_text(encoding="utf-8"))
                except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if isinstance(raw, dict) and isinstance(raw.get("formats"), list):
                    return {"status": "formats_found", "reason": None}
        return {"status": "not_checked", "reason": None}

    def _opened_material_is_cached(self, cached: MaterialResult, opened: Any) -> bool:
        """普通材料按 kind 去重；字幕还需比较首行语言元数据。"""

        if opened.kind != "subtitle":
            return True
        opened_header = self._subtitle_header(opened.content)
        if opened_header is None:
            return False
        for card in cached.evidence:
            try:
                header = self._subtitle_header(self.store.evidence(card.evidence_id).content)
            except KeyError:
                continue
            if header == opened_header:
                return True
        return False

    def _cached_subtitle_has_language(self, material: MaterialResult, language: str) -> bool:
        """检查缓存字幕中是否存在指定语言。"""

        for card in material.evidence:
            try:
                header = self._subtitle_header(self.store.evidence(card.evidence_id).content)
            except KeyError:
                continue
            if header is not None and header.get("language") == language:
                return True
        return False

    @staticmethod
    def _subtitle_header(content: str) -> dict[str, Any] | None:
        """容错解析字幕 Evidence 第一行的 JSON 元数据。"""

        try:
            value = json.loads(content.splitlines()[0])
        except (IndexError, TypeError, ValueError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _prepare_query_image(self, evidence_id: str) -> tuple[tempfile.TemporaryDirectory[str], Path, dict[str, Any]]:
        """把图片 Evidence 转为 Provider 友好的不超过 500 KiB JPEG。"""

        try:
            resolved = self.resolver.resolve(evidence_id)
            evidence, source = resolved.evidence, resolved.path
        except EvidenceResolutionError as exc:
            raise SearchFailure(SearchError("validate", "INVALID_INPUT", "image Evidence does not exist")) from exc
        if resolved.media_kind != "image" or not evidence.artifact_ref:
            raise SearchFailure(SearchError("validate", "INVALID_INPUT", "Evidence does not resolve to one stored image"))
        temp = tempfile.TemporaryDirectory(prefix="foren-lens-")
        target = Path(temp.name) / "query.jpg"
        try:
            with Image.open(source) as image:
                # 先完整解码，再记录原图信息，避免上传损坏或伪装文件。
                image.load()
                original = {"format": image.format, "width": image.width, "height": image.height,
                            "byte_size": source.stat().st_size, "artifact_ref": evidence.artifact_ref}
                converted = image.convert("RGB")
                converted.thumbnail((2048, 2048))
                # 逐级降低质量和尺寸，尽量保留细节同时满足上传上限。
                for quality in (90, 80, 70, 60, 50):
                    converted.save(target, "JPEG", quality=quality, optimize=True)
                    if target.stat().st_size <= 500 * 1024:
                        break
                    converted.thumbnail((max(320, int(converted.width * 0.8)), max(320, int(converted.height * 0.8))))
                if target.stat().st_size > 500 * 1024:
                    raise ValueError("adapted image is larger than 500 KB")
        except Exception as exc:
            temp.cleanup()
            raise SearchFailure(SearchError("validate", "UNSUPPORTED_FORMAT", "image Evidence cannot be adapted for provider upload")) from exc
        adaptation = {"source_evidence_id": evidence_id, "original": original,
                      "transmitted": {"format": "JPEG", "width": converted.width,
                                      "height": converted.height, "byte_size": target.stat().st_size}}
        return temp, target, adaptation

    @staticmethod
    def _candidate_from_record(record: dict[str, Any]) -> CandidateSource:
        """把 Store 中的 Candidate 字典恢复为强类型 CandidateSource。"""

        return CandidateSource(record["source_url"], record.get("search_title"), record.get("search_snippet"),
            record.get("provider") or "direct", record.get("engine") or "direct", record.get("provider_rank") or 1,
            record.get("candidate_type") or "unknown", record.get("image_url"), record.get("thumbnail_url"),
            record.get("platform"), record.get("platform_video_id"), record.get("extra") or {})


def _expected_kinds(request: SearchRequest) -> tuple[str, ...]:
    """返回每种输出响应必须包含的固定材料槽位顺序。"""

    if request.output_type == "image":
        return ("image", "page", "thumbnail")
    if request.output_type == "video":
        return ("page", "cover", "subtitle")
    return ("page", "document")


def _looks_like_video_url(url: str) -> bool:
    """识别页面中可作为一跳候选的常见视频链接。"""

    lowered = url.lower()
    return any(value in lowered for value in ("youtube.com/watch", "youtu.be/", "vimeo.com/", "bilibili.com/video/", "tiktok.com/"))
