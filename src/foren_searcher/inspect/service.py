"""INSPECT 的编排、持久化与公开响应规范化。

Service 负责把 E 引用解析成本地媒体、执行 probe 和真实范围校验，再把强类型调用交给
Dispatcher/Backend。后端只返回临时 handle；Service 提交每个派生 Artifact 后把 handle
替换成正式 E ID，最后检查响应引用闭包并写入执行审计。
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import replace
from typing import Any, Literal, cast

from ..contracts import Evidence, EvidenceCard
from ..store.resolver import EvidenceResolutionError, EvidenceResolver, ResolvedEvidence
from ..store import ArtifactMetadata, GlobalStore
from .backends.base import BackendFailure, BackendOutcome, MediaProfileData, PendingArtifact, PreparedMedia
from .contracts import (
    AlignMediaParams, AlignMediaResult, AlignmentAnchor, AlignmentMatch,
    AlignmentSideCoverage, AsrResult, AsrSegment, AudioQaParams, AudioQaResult,
    CropResult, FrameResult, GetFramesParams, GetFramesResult, GetPagesParams,
    GetPagesResult, ImageFrameQuery, InspectCall, InspectFeedback, InspectOperation,
    InspectResponse, InspectResult, InspectValidationError, LocateFramesCoverage,
    LocateFramesParams, LocateFramesResult, LocateRegionsResult, LocatedFrame,
    OcrImageResult, OcrItem, OcrParams, OcrResult, OcrSampling, OcrVideoCoverage, PageResult,
    RegionResult, TextImageFrameQuery, TimeCoverage, VisualQaParams, VisualQaResult,
    invalid_response, parse_inspect_call,
)
from .dispatcher import InspectDispatcher
from .policy import InspectPolicy
from .acquisition import AcquiredMedia


def _json(value: Any) -> str:
    """生成确定、紧凑且保留中文的 JSON 文本。"""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class InspectService:
    """Workspace 可调用的唯一 INSPECT 门面。"""

    def __init__(self, *, store: GlobalStore, resolver: EvidenceResolver, dispatcher: InspectDispatcher, policy: InspectPolicy, acquirer: Any | None = None, owned_resources: tuple[Any, ...] = (), enabled_operations: frozenset[str] | None = None) -> None:
        """注入 Store、E→文件解析器、操作分发器和可选媒体获取器。"""

        self.store, self.resolver, self.dispatcher, self.policy = store, resolver, dispatcher, policy
        self.acquirer = acquirer
        self.enabled_operations = enabled_operations
        self._owned_resources, self._closed = owned_resources, False
        self._completed_requests: dict[str, InspectResponse] = {}
        self._media_profiles: dict[tuple[ArtifactMetadata, str], MediaProfileData] = {}

    async def __aenter__(self) -> "InspectService":
        """进入异步上下文。"""

        return self

    async def __aexit__(self, *_: object) -> None:
        """离开上下文时释放本服务拥有的资源。"""

        await self.aclose()

    async def aclose(self) -> None:
        """幂等关闭所有 owned_resources；外部注入资源不越权关闭。"""

        if self._closed:
            return
        self._closed = True
        self._completed_requests.clear()
        self._media_profiles.clear()
        for resource in self._owned_resources:
            close = getattr(resource, "aclose", None)
            if close:
                try:
                    await close()
                except Exception:
                    # 关闭是尽力而为；继续释放后续资源。
                    pass

    @staticmethod
    def _feedback(failure: BackendFailure | EvidenceResolutionError, target: str | None = None) -> InspectFeedback:
        """把解析器或后端异常统一映射为公开 InspectFeedback。"""

        stage = getattr(failure, "stage", "resolve")
        return InspectFeedback(getattr(failure, "severity", "error"), stage, failure.code, str(failure), target, getattr(failure, "locator", None), getattr(failure, "retryable", False))  # type: ignore[arg-type]

    async def execute(self, value: InspectCall | dict[str, Any], *, workspace_id: str) -> InspectResponse:
        """验证并执行一次 INSPECT，保证执行审计最终得到终态。"""

        if self._closed:
            raise RuntimeError("INSPECT service is closed")
        try:
            call = value if isinstance(value, InspectCall) else parse_inspect_call(value)
        except InspectValidationError as exc:
            return invalid_response(exc.operation, str(exc))
        if self.enabled_operations is not None and call.request.operation not in self.enabled_operations:
            return invalid_response(call.request.operation, "This INSPECT operation is not enabled for the current runtime")
        execution_id = self.store.create_inspect_execution(call.to_dict(), workspace_id)
        details: dict[str, Any] = {"policy": self.policy.to_dict(), "requested_target_ids": list(call.targets)}
        try:
            # 一个总超时覆盖解析、获取、probe、处理和保存全过程。
            async with asyncio.timeout(self.policy.execution_timeout_s):
                response = await self._execute_validated(call, workspace_id, details)
        except TimeoutError:
            response = InspectResponse(call.request.operation, "failed", None, None, (), (InspectFeedback("error", "process", "TIMEOUT", "INSPECT execution timed out", retryable=True),))
        except asyncio.CancelledError:
            # 取消也先写审计记录，然后保持 asyncio 的取消语义。
            response = InspectResponse(call.request.operation, "failed", None, None, (), (InspectFeedback("error", "process", "TIMEOUT", "INSPECT execution was cancelled", retryable=True),))
            self.store.finish_inspect_execution(execution_id, status=response.status, result_evidence_id=None, feedback=[x.to_dict() for x in response.feedback], details=details)
            raise
        except Exception as exc:
            response = InspectResponse(call.request.operation, "failed", None, None, (), (InspectFeedback("error", "process", "PROCESS_FAILED", f"unexpected INSPECT failure: {type(exc).__name__}"),))
            details["exception_type"] = type(exc).__name__
        self.store.finish_inspect_execution(execution_id, status=response.status, result_evidence_id=response.result_evidence_id, feedback=[x.to_dict() for x in response.feedback], details=details)
        return replace(response, execution_id=execution_id)

    async def _execute_validated(self, call: InspectCall, workspace_id: str, details: dict[str, Any]) -> InspectResponse:
        """解析目标、建立画像、校验真实媒体、执行后端并发布结果。"""

        operation = call.request.operation; cards: list[EvidenceCard] = []; resolved: list[ResolvedEvidence] = []
        for target in call.targets:
            # 每个目标先加入原始 EvidenceCard，再解析文件或按需获取视频。
            try:
                evidence = self.store.evidence(target)
                cards.append(evidence.card())
                item, acquired = await self._resolve_or_acquire(evidence, operation, workspace_id, details)
                resolved.append(item)
                if acquired is not None:
                    cards.append(acquired.card())
            except EvidenceResolutionError as exc:
                return InspectResponse(operation, "failed", None, None, tuple(cards), (self._feedback(exc, target),))
            except BackendFailure as exc:
                return InspectResponse(operation, "failed", None, None, tuple(cards), (self._feedback(exc, target),))
            except KeyError:
                failure = EvidenceResolutionError("NOT_FOUND", f"Evidence {target} does not exist")
                return InspectResponse(operation, "failed", None, None, tuple(cards), (self._feedback(failure, target),))
        query_resolved: ResolvedEvidence | None = None
        params = call.request.params
        if isinstance(params, LocateFramesParams) and isinstance(params.query, (ImageFrameQuery, TextImageFrameQuery)):
            # 图片查询 E 不是 operation target，但同样必须解析并进入响应闭包。
            try:
                query_resolved = self.resolver.resolve(params.query.evidence_id); cards.append(query_resolved.evidence.card())
            except EvidenceResolutionError as exc:
                return InspectResponse(operation, "failed", None, None, tuple(cards), (self._feedback(exc, params.query.evidence_id),))
        prepared: list[PreparedMedia] = []; profile_cards: list[EvidenceCard] = []
        for item in resolved + ([query_resolved] if query_resolved else []):
            assert item is not None
            if item.artifact.byte_size > self.policy.maximum_file_bytes:
                return InspectResponse(
                    operation, "failed", None, None, tuple(cards + profile_cards),
                    (InspectFeedback("error", "validate", "RESOURCE_LIMIT", "media file exceeds configured byte limit", item.evidence.evidence_id),),
                )
            try:
                # resolve 每次校验文件完整性后，才可复用当前服务已探测的属性。
                profile_key = (item.artifact, item.media_kind)
                profile = self._media_profiles.get(profile_key) if self.store.verify_artifacts_on_read else None
                if profile is None:
                    profile = await self.dispatcher.probe(item.path, item.media_kind)
                    if self.store.verify_artifacts_on_read and not self._closed:
                        if len(self._media_profiles) >= 128:
                            self._media_profiles.pop(next(iter(self._media_profiles)))
                        self._media_profiles[profile_key] = profile
            except BackendFailure as exc:
                return InspectResponse(operation, "failed", None, None, tuple(cards + profile_cards), (self._feedback(exc, item.evidence.evidence_id),))
            if profile.duration_s is not None and profile.duration_s > self.policy.maximum_duration_s:
                return InspectResponse(operation, "failed", None, None, tuple(cards + profile_cards), (InspectFeedback("error", "probe", "RESOURCE_LIMIT", "media duration exceeds configured limit", item.evidence.evidence_id),))
            if profile.width and profile.height and profile.width * profile.height > self.policy.maximum_image_pixels and item.media_kind == "image":
                return InspectResponse(operation, "failed", None, None, tuple(cards + profile_cards), (InspectFeedback("error", "probe", "RESOURCE_LIMIT", "image exceeds configured pixel limit", item.evidence.evidence_id),))
            try:
                source_locator = json.loads(item.evidence.locator)
            except (TypeError, ValueError):
                source_locator = item.evidence.locator
            source_context = _json({"derived_from": list(item.evidence.derived_from),
                                    "locator": source_locator})
            prepared.append(PreparedMedia(item.evidence.evidence_id, item.path, item.media_kind, profile, source_context))
            if item.evidence.acquired_by == "INSPECT/acquire" or not item.evidence.acquired_by.startswith("INSPECT/"):
                # 对外部来源媒体保存可复用画像；INSPECT 自己派生的帧不重复画像。
                locator = _json({"probe_backend": profile.backend, "probe_version": profile.version, "profile_version": 1})
                matches = self.store.find_derived_evidence(derived_from=(item.evidence.evidence_id,), acquired_by="INSPECT/media_profile", locator=locator)
                profile_evidence = matches[0] if matches else self.store.commit_derived_evidence(
                    derived_from=(item.evidence.evidence_id,), modalities=("text",),
                    label=self._profile_label(profile), content=_json(profile.to_dict()), locator=locator,
                    acquired_by="INSPECT/media_profile", discovered_by=workspace_id,
                    production_method="media_probe",
                )
                profile_cards.append(profile_evidence.card())
        main_prepared = tuple(prepared[:len(resolved)]); query_prepared = prepared[-1] if query_resolved else None
        validation = self._validate_real(operation, params, main_prepared, query_prepared)
        backend_call = call
        selection_feedback: list[InspectFeedback] = []
        if validation is not None and validation.code == "OUT_OF_RANGE":
            valid_params: GetFramesParams | GetPagesParams | None = None
            profile = main_prepared[0].profile
            # 显式批量选择逐项隔离越界项，原始请求量仍受资源上限约束。
            if isinstance(params, GetFramesParams) and params.times_s and len(params.times_s) <= self.policy.maximum_frames:
                times = tuple(value for value in params.times_s if value < (profile.duration_s or 0))
                if times and len(times) < len(params.times_s):
                    valid_params = replace(params, times_s=times)
            elif isinstance(params, GetPagesParams) and params.pages and len(params.pages) <= self.policy.maximum_pages:
                pages = tuple(value for value in params.pages if value <= (profile.page_count or 0))
                if pages and len(pages) < len(params.pages):
                    valid_params = replace(params, pages=pages)
            if valid_params is not None:
                remaining_validation = self._validate_real(operation, valid_params, main_prepared, query_prepared)
                if remaining_validation is None:
                    backend_call = replace(call, request=replace(call.request, params=valid_params))
                    selection_feedback.append(validation)
                    details["effective_request"] = backend_call.to_dict()
                    validation = None
        if validation:
            return InspectResponse(operation, "failed", None, None, tuple(self._dedupe(cards + profile_cards)), (validation,))
        details["effective_media_evidence_ids"] = [x.evidence_id for x in main_prepared]
        key = _json(call.to_dict())
        cached = self._completed_requests.get(key)
        if cached is not None:
            for card in cached.evidence:
                evidence = self.store.evidence(card.evidence_id)
                if evidence.artifact_ref is not None:
                    self.store.artifact_path(evidence.artifact_ref)
            details.update({"reused":True, "created_evidence_ids":[],
                            "reused_evidence_ids":[card.evidence_id for card in cached.evidence]})
            return cached
        try:
            # 只有全部解析与真实校验通过，才让后端执行昂贵操作。
            outcome = await self.dispatcher.execute(backend_call, main_prepared, policy=self.policy, query_media=query_prepared)
        except BackendFailure as exc:
            return InspectResponse(operation, "failed", None, None, tuple(self._dedupe(cards + profile_cards)), (*selection_feedback, self._feedback(exc, call.targets[0])))
        feedback = selection_feedback + [self._feedback(issue, call.targets[0]) for issue in outcome.issues]
        handle_map: dict[str, str] = {}; artifact_cards: list[EvidenceCard] = []
        created_ids: list[str] = []; reused_ids: list[str] = []
        for artifact in outcome.artifacts:
            # handle 只是后端内存引用，提交后映射为正式 E ID。
            if artifact.handle in handle_map:
                feedback.append(InspectFeedback("error", "save", "SAVE_FAILED", f"duplicate internal Artifact handle: {artifact.handle}")); continue
            try:
                reused = self._reusable_media_image(artifact)
                if reused is not None:
                    evidence = reused
                    reused_ids.append(evidence.evidence_id)
                else:
                    evidence = self.store.commit_derived_evidence(
                        derived_from=(artifact.parent_evidence_id,), modalities=("image",), label=artifact.label,
                        content=_json(dict(artifact.locator)), locator=_json(dict(artifact.locator)), acquired_by=artifact.acquired_by,
                        discovered_by=workspace_id, production_method=artifact.production_method,
                        artifact_bytes=artifact.data, media_type=artifact.media_type, suffix=artifact.suffix, role=artifact.role,
                    )
                    created_ids.append(evidence.evidence_id)
                handle_map[artifact.handle] = evidence.evidence_id; artifact_cards.append(evidence.card())
            except Exception as exc:
                feedback.append(InspectFeedback("error", "save", "SAVE_FAILED", f"Artifact save failed: {type(exc).__name__}"))
        result = self._public_result(operation, outcome, handle_map)
        # 纯媒体操作的结果本身就是新图片 E，无需再保存一条结构化文本 E。
        pure_media = operation in {"get_frames", "get_pages", "crop"}
        usable = self._usable(operation, result)
        result_evidence: Evidence | None = None
        if usable and result is not None and not pure_media:
            # 分析型操作另存结构化结果 E，方便 Findings 直接引用结论数据。
            analysis_context = {key: outcome.details[key] for key in
                ("backend", "model", "provider", "sampling_interval_s", "boundary_uncertainty_s", "speed_ratio_range", "vad_filter", "speech_duration_s", "limitations")
                if key in outcome.details}
            derived = tuple(x.evidence_id for x in main_prepared) + ((query_prepared.evidence_id,) if query_prepared else ())
            try:
                result_evidence = self.store.commit_derived_evidence(
                    derived_from=derived, modalities=("text",), label=f"INSPECT {operation} 结果",
                    content=_json({**analysis_context, **result.to_dict()}), locator=_json(self._result_locator(result)),
                    acquired_by=f"INSPECT/{operation}", discovered_by=workspace_id,
                    production_method="structured_analysis",
                )
            except Exception as exc:
                feedback.append(InspectFeedback("error", "save", "SAVE_FAILED", f"result save failed: {type(exc).__name__}")); result = None; usable = False
        if not usable:
            # 后端没产出可用对象时，即使只有 warning 也要补一个明确 error。
            if not feedback: feedback.append(InspectFeedback("error", "process", "PROCESS_FAILED", "operation produced no usable result"))
            return InspectResponse(operation, "failed", None, None, tuple(self._dedupe(cards + profile_cards + artifact_cards)), tuple(feedback))
        if result_evidence:
            artifact_cards.append(result_evidence.card())
            created_ids.append(result_evidence.evidence_id)
        status = cast(Literal["ready", "partial", "failed"], "partial" if any(x.severity == "error" for x in feedback) else "ready")
        details.update(dict(outcome.details)); details["created_evidence_ids"] = created_ids
        if reused_ids:
            details["reused_evidence_ids"] = list(dict.fromkeys(reused_ids))
        response = InspectResponse(operation, status, result, result_evidence.evidence_id if result_evidence else None, tuple(self._dedupe(cards + profile_cards + artifact_cards)), tuple(feedback))
        if status == "ready":
            if len(self._completed_requests) >= 128:
                self._completed_requests.pop(next(iter(self._completed_requests)))
            self._completed_requests[key] = response
        return response

    def _reusable_media_image(self, artifact: PendingArtifact) -> Evidence | None:
        """同一来源、定位和图片字节的完整帧或文档页保留已有 E 身份。"""

        frame_producers = ("INSPECT/get_frames", "INSPECT/align_media", "INSPECT/locate_frames")
        frame = artifact.acquired_by in frame_producers and artifact.role == "video_frame"
        page = artifact.acquired_by == "INSPECT/get_pages" and artifact.role == "document_page"
        if not frame and not page:
            return None
        ignored = {"requested_time_s"} if frame else set()
        locator = {k: v for k, v in artifact.locator.items() if k not in ignored}
        producers = ((artifact.acquired_by, *(producer for producer in frame_producers if producer != artifact.acquired_by))
                     if frame else (artifact.acquired_by,))
        candidates = [e for producer in producers for e in self.store.find_derived_evidence(
            derived_from=(artifact.parent_evidence_id,), acquired_by=producer,
        )]
        for evidence in candidates:
            if (evidence.artifact_ref is None or evidence.modalities != ("image",)
                    or evidence.production_method != artifact.production_method
                    or evidence.content != evidence.locator):
                continue
            try:
                saved_locator = json.loads(evidence.locator)
            except (ValueError, TypeError):
                continue
            if not isinstance(saved_locator, dict) or {k: v for k, v in saved_locator.items() if k not in ignored} != locator:
                continue
            metadata = self.store.artifact(evidence.artifact_ref)
            if metadata.role != artifact.role or metadata.media_type != artifact.media_type:
                continue
            path = self.store.artifact_path(evidence.artifact_ref)
            if metadata.byte_size == len(artifact.data) and path.read_bytes() == artifact.data:
                return evidence
        return None

    async def _resolve_or_acquire(self, evidence: Evidence, operation: InspectOperation, workspace_id: str, details: dict[str, Any]) -> tuple[ResolvedEvidence, Evidence | None]:
        """优先解析现有 Artifact；合格视频页面可受控获取真正媒体。"""

        direct: ResolvedEvidence | None = None
        try:
            direct = self.resolver.resolve(evidence.evidence_id)
        except EvidenceResolutionError as exc:
            if exc.code != "MATERIAL_UNAVAILABLE":
                raise
        video_capable = operation in {"get_frames", "ocr", "asr", "audio_qa", "locate_frames", "align_media"}
        # 已有可识别媒体直接使用；unknown 只有视频能力操作才尝试 acquisition。
        if direct is not None and (direct.media_kind != "unknown" or not video_capable):
            return direct, None
        candidate_id = evidence.source_ref
        if not video_capable or not candidate_id or not candidate_id.startswith("R"):
            if direct is not None:
                return direct, None
            raise EvidenceResolutionError("MATERIAL_UNAVAILABLE", f"Evidence {evidence.evidence_id} has no media Artifact")
        try:
            candidate = self.store.candidate(candidate_id)
            linked = self.store.candidate_evidence(candidate_id)
        except KeyError as exc:
            raise EvidenceResolutionError("MATERIAL_UNAVAILABLE", "source Candidate is unavailable") from exc
        is_page_trigger = any(kind == "page" and status in {"ready", "partial"} and item.evidence_id == evidence.evidence_id for kind, status, item in linked)
        # 只有 SEARCH 确认的视频 Candidate 来源页 E 才能触发下载。
        if candidate.get("candidate_type") != "video" or not is_page_trigger:
            if direct is not None:
                return direct, None
            raise EvidenceResolutionError("MATERIAL_UNAVAILABLE", "Evidence is not an eligible video source page")
        for kind, status, cached in reversed(linked):
            # 优先复用最近一次成功获取的视频 Evidence，保证幂等。
            if kind != "video" or status != "ready":
                continue
            try:
                cached_media = self.resolver.resolve(cached.evidence_id)
            except EvidenceResolutionError:
                continue
            if cached_media.media_kind == "video":
                details.setdefault("acquisition", []).append({"trigger_evidence_id": evidence.evidence_id, "candidate_id": candidate_id, "result_evidence_id": cached.evidence_id, "reused": True})
                return cached_media, cached
        if self.acquirer is None:
            raise BackendFailure("MATERIAL_UNAVAILABLE", "no controlled media acquirer is configured", stage="acquire")
        acquisition_url = self._candidate_media_url(candidate, direct)
        destination = self.store.new_owned_temp_path(".media")
        try:
            acquired: AcquiredMedia = await self.acquirer.acquire(acquisition_url, destination=destination, policy=self.policy)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        content = _json({"original_filename": acquired.label, "media_type": acquired.media_type,
                         "byte_size": acquired.path.stat().st_size,
                         "acquisition": {key:value for key,value in acquired.details.items() if key != "playlists"}})
        locator = _json({"scope": acquired.details.get("scope", "complete_media"),
                         "source_url": acquired.details.get("final_url") or acquired.details.get("requested_url")})
        committed = self.store.commit_candidate_file_evidence(
            # Store 接管 owned_temp_path；成功后临时文件不再属于 Service。
            candidate_id=candidate_id, owned_temp_path=acquired.path, media_type=acquired.media_type,
            suffix=acquired.suffix, label=acquired.label, content=content, locator=locator,
            discovered_by=workspace_id,
        )
        resolved = self.resolver.resolve(committed.evidence_id)
        if resolved.media_kind != "video":
            raise BackendFailure("UNSUPPORTED_MEDIA", "acquired source is not a decodable video", stage="probe")
        details.setdefault("acquisition", []).append({"trigger_evidence_id": evidence.evidence_id, "candidate_id": candidate_id, "result_evidence_id": committed.evidence_id, "reused": False, **acquired.details})
        return resolved, committed

    @staticmethod
    def _candidate_media_url(candidate: dict[str, Any], trigger: ResolvedEvidence | None) -> str:
        """从已保存且允许的平台元数据中选择完整文件或 HLS 音视频来源。"""
        if trigger is not None and trigger.artifact.media_type in {"application/json", "text/json"}:
            try:
                if trigger.artifact.byte_size > 32 * 1024 * 1024:
                    raise ValueError("metadata too large")
                raw = json.loads(trigger.path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
                raw = None
            if isinstance(raw, dict):
                extractor = str(raw.get("extractor_key") or raw.get("extractor") or "").casefold()
                platform = str(candidate.get("platform") or "").casefold()
                supported = extractor in {"youtube", "youtube:tab"} and platform in {"", "youtube"}
                supported = supported or extractor == "html5mediaembed"
                if supported:
                    formats = raw.get("formats")
                    if isinstance(formats, list):
                        eligible = [item for item in formats if isinstance(item, dict)
                            # 只允许 HTTPS、同时含音视频且容器在白名单中的直链。
                            and isinstance(item.get("url"), str) and item["url"].startswith("https://")
                            and item.get("vcodec") not in {None, "none"} and item.get("acodec") not in {None, "none"}
                            and item.get("protocol") in {None, "https"} and item.get("ext") in {"mp4", "webm"}]
                        if eligible:
                            # 选择分辨率优先、码率次优的渐进式格式。
                            selected = max(eligible, key=lambda item: (float(item.get("height") or 0), float(item.get("tbr") or 0)))
                            return str(selected["url"])
                        hls = [item for item in formats if isinstance(item, dict)
                            and isinstance(item.get("url"), str) and item["url"].startswith("https://")
                            and item.get("vcodec") not in {None, "none"} and item.get("acodec") not in {None, "none"}
                            and item.get("protocol") in {"m3u8", "m3u8_native"} and not item.get("has_drm")]
                        if hls:
                            selected = max(hls, key=lambda item: (float(item.get("height") or 0), float(item.get("tbr") or 0)))
                            manifest = selected.get("manifest_url")
                            return manifest if isinstance(manifest, str) and manifest.startswith("https://") else str(selected["url"])
                    raise BackendFailure("MATERIAL_UNAVAILABLE", "saved platform metadata has no supported audio/video file or HLS format", stage="acquire", retryable=True)
                if extractor:
                    raise BackendFailure("MATERIAL_UNAVAILABLE", "saved platform extractor is not enabled for controlled acquisition", stage="acquire")
        source_url = candidate.get("final_url") or candidate.get("source_url")
        if not isinstance(source_url, str):
            raise BackendFailure("MATERIAL_UNAVAILABLE", "Candidate has no acquisition URL", stage="acquire")
        return source_url

    @staticmethod
    def _dedupe(cards: list[EvidenceCard]) -> list[EvidenceCard]:
        """按 E ID 保序去重响应 EvidenceCard。"""

        seen: set[str] = set(); result = []
        for card in cards:
            if card.evidence_id not in seen: seen.add(card.evidence_id); result.append(card)
        return result

    @staticmethod
    def _profile_label(profile: Any) -> str:
        """把媒体画像格式化成人和模型都容易扫读的中文标签。"""

        parts = [f"媒体属性：{profile.media_kind}"]
        if profile.duration_s is not None: parts.append(f"{profile.duration_s:.3f}秒")
        if profile.width and profile.height: parts.append(f"{profile.width}×{profile.height}")
        if profile.fps: parts.append(f"{profile.fps:.3f}fps")
        if profile.page_count: parts.append(f"{profile.page_count}页")
        parts.append(f"{profile.audio_tracks}条音轨")
        return "，".join(parts) + "；本记录未核验日期、地点或作者等来源元数据，不能据此判断这些信息不存在。"

    def _validate_real(self, operation: InspectOperation, params: Any, media: tuple[PreparedMedia, ...], query_media: PreparedMedia | None) -> InspectFeedback | None:
        """使用 probe 后的真实属性校验媒体类型、范围、数量和模型预算。"""

        kinds = tuple(x.media_kind for x in media)
        # 首先校验每种操作允许的媒体组合。
        expected = {"get_frames":{"video"},"get_pages":{"document"},"crop":{"image"},"locate_regions":{"image"},"ocr":{"image","video"},"asr":{"audio","video"},"audio_qa":{"audio","video"},"visual_qa":{"image"},"locate_frames":{"video"}}.get(operation)
        if expected is not None:
            invalid = next((item for item in media if item.media_kind not in expected), None)
            if invalid is not None:
                message = f"{operation} requires {', '.join(sorted(expected))}; {invalid.evidence_id} is {invalid.media_kind}."
                if operation == "visual_qa" and invalid.media_kind in {"video", "document"}:
                    preparation = "get_frames" if invalid.media_kind == "video" else "get_pages"
                    message += f" Use {preparation} to obtain image Evidence, then pass the returned image IDs to visual_qa."
                return InspectFeedback("error", "resolve", "UNSUPPORTED_MEDIA", message, invalid.evidence_id)
        if operation == "align_media":
            mode = params.mode
            if (mode == "visual" and kinds != ("video","video")) or (mode == "audio" and any(x not in {"audio","video"} for x in kinds)):
                return InspectFeedback("error", "resolve", "UNSUPPORTED_MEDIA", f"align_media {mode} does not support {kinds}", media[0].evidence_id)
        if query_media and query_media.media_kind != "image": return InspectFeedback("error", "resolve", "UNSUPPORTED_MEDIA", "locate_frames query Evidence is not an image", query_media.evidence_id)
        range_fields = []
        # 再校验所有时间范围没有超过对应媒体时长。
        for name in ("range_s","range_a_s","range_b_s"):
            value = getattr(params,name,None)
            if value is not None: range_fields.append((name,value))
        for name, value in range_fields:
            index = 1 if name == "range_b_s" else 0; duration = media[index].profile.duration_s
            if duration is None or value[1] > duration: return InspectFeedback("error", "validate", "OUT_OF_RANGE", f"{name} exceeds media duration", media[index].evidence_id, {"range_s":list(value)})
        if isinstance(params, GetFramesParams) and params.times_s:
            duration = media[0].profile.duration_s or 0
            invalid_times = [value for value in params.times_s if value >= duration]
            if invalid_times:
                return InspectFeedback("error", "validate", "OUT_OF_RANGE",
                    f"times_s must satisfy 0 <= time < {duration} seconds; the endpoint is excluded. "
                    f"Out-of-range values: {invalid_times}", media[0].evidence_id,
                    {"times_s": invalid_times, "duration_s": duration})
        if isinstance(params, GetFramesParams):
            # 计划量使用显式 times 数量、count 或默认值，逐级择一。
            planned = len(params.times_s) if params.times_s else params.count or self.policy.default_frame_count
            if planned > self.policy.maximum_frames: return InspectFeedback("error", "validate", "RESOURCE_LIMIT", "frame request exceeds configured limit", media[0].evidence_id)
        if isinstance(params, GetPagesParams):
            page_count = media[0].profile.page_count or 0
            if params.pages and any(x > page_count for x in params.pages):
                return InspectFeedback("error", "validate", "OUT_OF_RANGE", "pages contains an out-of-range page", media[0].evidence_id,
                    {"pages": [value for value in params.pages if value > page_count], "page_count": page_count})
            if params.page_range and params.page_range[1] > page_count: return InspectFeedback("error", "validate", "OUT_OF_RANGE", "page_range exceeds document", media[0].evidence_id)
            planned = len(params.pages) if params.pages else params.count or self.policy.default_page_count
            if planned > self.policy.maximum_pages: return InspectFeedback("error", "validate", "RESOURCE_LIMIT", "page request exceeds configured limit", media[0].evidence_id)
        if isinstance(params, AudioQaParams) and params.audio_track is not None and params.audio_track >= media[0].profile.audio_tracks:
            return InspectFeedback("error", "validate", "NO_AUDIO", "requested audio track does not exist", media[0].evidence_id, {"audio_track":params.audio_track})
        if isinstance(params, AudioQaParams):
            # 音频模型上下文有独立时长上限，避免超大推理。
            start, end = params.range_s or (0.0, media[0].profile.duration_s or 0.0)
            if end - start > self.policy.maximum_audio_qa_duration_s:
                return InspectFeedback("error", "validate", "RESOURCE_LIMIT", "audio_qa range exceeds the configured model duration limit", media[0].evidence_id, {"range_s": [start, end]})
        if isinstance(params, VisualQaParams) and len(media) > self.policy.maximum_visual_qa_images:
            return InspectFeedback("error", "validate", "RESOURCE_LIMIT", "visual_qa input count exceeds the configured model limit", media[0].evidence_id, {"image_count": len(media), "maximum_images": self.policy.maximum_visual_qa_images})
        if operation in {"asr","audio_qa"} and media[0].profile.audio_tracks == 0: return InspectFeedback("error", "probe", "NO_AUDIO", "media has no usable audio track", media[0].evidence_id)
        if operation == "align_media" and params.mode == "audio" and any(x.profile.audio_tracks == 0 for x in media): return InspectFeedback("error", "probe", "NO_AUDIO", "one alignment input has no usable audio track", media[0].evidence_id)
        return None

    @staticmethod
    def _public_result(operation: InspectOperation, outcome: BackendOutcome, handles: dict[str,str]) -> InspectResult | None:
        """把后端松散字典和 handle 转为公开强类型 Result 与正式 E ID。"""

        raw = outcome.result
        if operation == "get_frames": return GetFramesResult(tuple(FrameResult(handles[x["handle"]],float(x["time_s"]),int(x["width"]),int(x["height"]), x.get("frame_duration_s")) for x in raw["frames"] if x["handle"] in handles))
        if operation == "get_pages": return GetPagesResult(tuple(PageResult(handles[x["handle"]],int(x["page_number"]),int(x["width"]),int(x["height"])) for x in raw["pages"] if x["handle"] in handles))
        if operation == "crop":
            return CropResult(handles[raw["handle"]],int(raw["width"]),int(raw["height"]),tuple(raw["box_px"])) if raw["handle"] in handles else None
        if operation == "locate_regions":
            regions=tuple(RegionResult(handles[x["handle"]],tuple(x["box"]),tuple(x["box_px"]),x["matched_text"],float(x["score"])) for x in raw["regions"] if x["handle"] in handles)
            return LocateRegionsResult(raw["image_evidence_id"],raw["query"],regions)
        if operation == "ocr":
            # 图片 OCR 直接引用原 E；视频 OCR 的 image_handle 需映射到帧 E。
            coverage_raw=raw["coverage"]; ocr_coverage=None
            if coverage_raw: ocr_coverage=OcrVideoCoverage(tuple(coverage_raw["range_s"]),OcrSampling(**coverage_raw["sampling"]),tuple(tuple(x) for x in coverage_raw["failed_ranges"]))
            rows=[]
            for x in raw["results"]:
                evidence_id=x.get("image_evidence_id") or handles.get(x.get("image_handle"));
                if evidence_id: rows.append(OcrImageResult(evidence_id,x["time_s"],tuple(OcrItem(i["text"],tuple(i["box"]),i["confidence"]) for i in x["items"])))
            return OcrResult(ocr_coverage,tuple(rows))
        if operation == "asr":
            cov=raw["coverage"]; return AsrResult(TimeCoverage(tuple(cov["range_s"]),tuple(tuple(x) for x in cov["processed_ranges"]),tuple(tuple(x) for x in cov["failed_ranges"])),raw["language"],int(raw["audio_track"]),tuple(AsrSegment(float(x["start_s"]),float(x["end_s"]),x["text"]) for x in raw["segments"]))
        if operation == "audio_qa": return AudioQaResult(tuple(raw["range_s"]),int(raw["audio_track"]),raw["answer"],tuple(raw["limitations"]))
        if operation == "visual_qa": return VisualQaResult(tuple(raw["image_evidence_ids"]),raw["answer"],tuple(raw["limitations"]))
        if operation == "locate_frames":
            cov=raw["coverage"]; locate_coverage=LocateFramesCoverage(tuple(cov["range_s"]),int(cov["coarse_frames"]),tuple(tuple(x) for x in cov["refined_ranges"]),tuple(tuple(x) for x in cov["failed_ranges"])); return LocateFramesResult(locate_coverage,tuple(LocatedFrame(handles[x["handle"]],float(x["time_s"]),float(x["score"])) for x in raw["candidates"] if x["handle"] in handles))
        cov=raw["coverage"]; a,b=cov["a"],cov["b"]
        # 剩余分支是 align_media，分别重建 A/B 覆盖与锚点。
        side_a=AlignmentSideCoverage(tuple(a["range_s"]),a["processed"]["unit"],a["processed"]["value"],tuple(tuple(x) for x in a["failed_ranges"])); side_b=AlignmentSideCoverage(tuple(b["range_s"]),b["processed"]["unit"],b["processed"]["value"],tuple(tuple(x) for x in b["failed_ranges"]))
        matches=[]
        for match in raw["matches"]:
            anchors=tuple(AlignmentAnchor(float(x["time_a_s"]),float(x["time_b_s"]),handles.get(x.get("image_a_handle")),handles.get(x.get("image_b_handle"))) for x in match["anchors"])
            matches.append(AlignmentMatch(tuple(match["range_a_s"]),tuple(match["range_b_s"]),float(match["score"]),anchors))
        return AlignMediaResult(raw["mode"],side_a,side_b,tuple(matches))

    @staticmethod
    def _usable(operation: InspectOperation, result: InspectResult | None) -> bool:
        """判断结果是否含该操作最少需要的有效内容。"""

        if result is None:
            return False
        if isinstance(result, GetFramesResult):
            return bool(result.frames)
        if isinstance(result, GetPagesResult):
            return bool(result.pages)
        if isinstance(result, CropResult):
            return True
        if isinstance(result, OcrResult):
            return bool(result.results)
        if isinstance(result, LocateFramesResult):
            return result.coverage.coarse_frames > 0
        if isinstance(result, AlignMediaResult):
            return result.coverage_a.value > 0 and result.coverage_b.value > 0
        return True

    @staticmethod
    def _result_locator(result: InspectResult) -> dict[str,Any]:
        """从结构化结果提取足以定位其分析范围的小型元数据。"""

        value = result.to_dict()
        return {key:value[key] for key in ("coverage","range_s","image_evidence_id","image_evidence_ids") if key in value}
