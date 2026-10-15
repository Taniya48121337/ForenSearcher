"""INSPECT 的编排、持久化与公开响应规范化。

Service 负责把 E 引用解析成本地媒体、执行 probe 和真实范围校验，再把强类型调用交给
Dispatcher/Backend。后端只返回临时 handle；Service 提交每个派生 Artifact 后把 handle
替换成正式 E ID，最后检查响应引用闭包并写入执行审计。
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

from ..platform_video import is_video_candidate
from ..contracts import Evidence, EvidenceCard
from ..i18n import english, tr
from ..state_render import corresponding_spans
from ..store.resolver import EvidenceResolutionError, EvidenceResolver, ResolvedEvidence
from ..store import ArtifactMetadata, GlobalStore
from .backends.base import BackendFailure, BackendOutcome, MediaProfileData, PendingArtifact, PreparedMedia
from .contracts import (
    AlignMediaParams, AlignMediaResult, AlignmentAnchor, AlignmentMatch,
    AlignmentSideCoverage, AsrResult, AsrSegment, AudioQaParams, AudioQaResult,
    CompareMediaParams, CompareMediaResult, CropParams, CropResult, DEFAULT_OPERATIONS, DiffImage, DiffMediaResult, DiffSegment,
    InspectRequest, LocateParams, LocateRegionsParams, ReportImage, SoundSegment, TextFrameQuery, VideoStructureParams,
    VideoStructureResult, is_merged_catalog,
    FrameResult, GetFramesParams, GetFramesResult, GetPagesParams,
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

# 10-12：要声音的操作不用低清纯画面兜底，也不复用兜底下的无声文件
SOUND_OPERATIONS = frozenset({"asr", "audio_qa"})
LOW_RES_FALLBACK_CODES = frozenset({"ACCESS_RESTRICTED", "MATERIAL_UNAVAILABLE", "RESOURCE_LIMIT", "TIMEOUT"})


def _video_only(evidence: Evidence) -> bool:
    try:
        acquisition = json.loads(evidence.content).get("acquisition") or {}
    except (TypeError, ValueError, AttributeError):
        return False
    return bool(isinstance(acquisition, dict) and acquisition.get("video_only_height"))

MEDIA_PROFILE_BOILERPLATE_EN = ("; this record does not verify date, place, author or other source metadata "
                                "and cannot show that such information is absent.")


def _json(value: Any) -> str:
    """生成确定、紧凑且保留中文的 JSON 文本。"""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


# 10-14 合并掉分的根因：compare_media / video_structure 的结果 E 过去存整份测量表（约 10 KB、键按字母排，
# 对齐段 / 镜头表 / 静止段在前，summary_lines 在第 8400 字之后），HI 用 READ_RECORD 反复读第一页，答案照着
# 1.5 s 的 reversed / sped_up 段和静止帧写（evidM27h video_154 读 E31 六次、209 读 E12 五次）。DETECT 的结果 E
# 约 2 KB、结论在前。这里改回"记录 = 结论"：只存结论行、附图 E、限制；测量表仍在本步观察的 result 里（walker 等用）。
CONCLUSION_RECORD_OPERATIONS = frozenset({"compare_media", "video_structure"})
CONCLUSION_RECORD_NOTE = ("These lines are the conclusions of the measurement; the raw tables behind them (alignment runs, "
                          "shot list, frame statistics) are not part of this record. Spans under 3 s that align at another speed "
                          "or backwards are alignment noise around cuts.")


def conclusion_record(operation: str, result: Any, context: dict[str, Any]) -> str:
    """DETECT 式结果记录：operation → summary_lines（对照图写成 E 编号）→ corresponding_spans（compare_media，≥ 3 s）→ images
    → limitations → note，保持这个顺序。"""

    images = [{"evidence_id": x.evidence_id, "caption": x.caption} for x in getattr(result, "images", ()) or ()]
    ids = [x["evidence_id"] for x in images]

    def name(match: "re.Match[str]") -> str:
        k = int(match.group(1))
        return ids[k - 1] if 0 < k <= len(ids) and ids[k - 1] else match.group(0)

    lines = [re.sub(r"comparison image (\d+)", name, str(x)) for x in getattr(result, "summary_lines", ()) or ()]
    report = getattr(result, "report", None) or {}
    record: dict[str, Any] = {"operation": operation, "summary_lines": lines}
    spans = corresponding_spans(report.get("correspondences")) if operation == "compare_media" else []
    if spans:
        record["corresponding_spans"] = spans
    record["images"] = images
    if isinstance(report.get("coverage"), dict) and operation == "video_structure":
        record["coverage"] = {k: report["coverage"][k] for k in ("range_s",) if k in report["coverage"]}
    limitations = report.get("limitations") or context.get("limitations")
    if limitations:
        record["limitations"] = list(limitations)[:4]
    record["note"] = CONCLUSION_RECORD_NOTE
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def ocr_record(result: Any, context: dict[str, Any], sources: tuple[str, ...]) -> str:
    """10-15（A7）：一次 OCR 调用的结果 E（kind=ocr）。

    以前视频 OCR 把每个采样帧都存成图片 E（videosham / fmnv 一次 500 多个），视图里的其他卡片被挤掉
    （wk8 fmnv_a_1Kk 选的 E7 掉出视图被拒）；结果 E 本身 22 万字、键按字母排，读第一页只看到采样统计。
    现在只有这一个 E：开头是汇总（不同文字、首末时刻、帧数），然后是覆盖与限制，最后逐帧（按时刻）列出每个文字框的
    文字、坐标（归一化 x1,y1,x2,y2）和置信度。视频的行不再带帧 E 编号，用 source + time_s 定位。"""

    data = result.to_dict()
    summary = result.text_summary()
    rows = []
    for row in sorted(data.get("results") or [], key=lambda r: (r.get("time_s") is None, r.get("time_s") or 0.0)):
        out: dict[str, Any] = {}
        if row.get("time_s") is None:
            out["image_evidence_id"] = row.get("image_evidence_id")
        else:
            out["time_s"] = round(float(row["time_s"]), 3)
        out["items"] = [{"text": item.get("text"),
                         "box": [round(float(v), 4) for v in item.get("box") or ()],
                         "confidence": None if item.get("confidence") is None else round(float(item["confidence"]), 3)}
                        for item in row.get("items") or ()]
        rows.append(out)
    record: dict[str, Any] = {"kind": "ocr", "operation": "ocr", "source_evidence_ids": list(sources), "summary": summary}
    if data.get("coverage"):
        record["coverage"] = data["coverage"]
    for key in ("backend", "limitations"):
        if key in context:
            record[key] = context[key]
    record["results"] = rows
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def ocr_label(result: Any, sources: tuple[str, ...]) -> str:
    """OCR 结果 E 的卡片标签：来源、计数和出现最多的几条文字（卡片常驻视图，学生不读记录也知道里面有什么字）。"""

    summary = result.text_summary(limit=4)
    texts = ", ".join(json.dumps(str(e["text"])[:40], ensure_ascii=False) for e in summary["texts"])
    boxes, distinct = int(summary["text_boxes"]), int(summary["distinct_texts"])
    return (f"OCR of {', '.join(sources) or '?'}: {boxes} text box{'' if boxes == 1 else 'es'}, "
            f"{distinct} distinct text{'' if distinct == 1 else 's'}"
            + (f" ({texts}{', …' if summary['distinct_texts'] > len(summary['texts']) else ''})" if texts else "; no text recognised"))


class InspectService:
    """Workspace 可调用的唯一 INSPECT 门面。"""

    def __init__(self, *, store: GlobalStore, resolver: EvidenceResolver, dispatcher: InspectDispatcher, policy: InspectPolicy, acquirer: Any | None = None, owned_resources: tuple[Any, ...] = (), enabled_operations: frozenset[str] | None = None) -> None:
        """注入 Store、E→文件解析器、操作分发器和可选媒体获取器。"""

        self.store, self.resolver, self.dispatcher, self.policy = store, resolver, dispatcher, policy
        self.acquirer = acquirer
        # 10-13 规格 compact_tool_params（hi@0.10.0 起）由 run_cases 打开：compare_media 不列 mode，画面对不上时只比声音
        self.compact_params = False
        # 未指定时只开放默认操作；diff_media 这类需显式开启的操作必须由调用方（规格 harness_params）列入。
        self.enabled_operations = frozenset(enabled_operations) if enabled_operations is not None else DEFAULT_OPERATIONS
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
        if call.request.operation not in self.enabled_operations:
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
        if isinstance(params, (LocateFramesParams, LocateParams)) and isinstance(params.query, (ImageFrameQuery, TextImageFrameQuery)):
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
        public_operation = operation
        try:
            operation, params = self._translate(operation, params, main_prepared)
        except InspectValidationError as exc:
            return InspectResponse(public_operation, "failed", None, None, tuple(self._dedupe(cards + profile_cards)),
                                   (InspectFeedback("error", "validate", "UNSUPPORTED_MEDIA" if "requires" in str(exc) else "INVALID_INPUT",
                                                    str(exc), main_prepared[0].evidence_id),))
        if operation != public_operation or params is not call.request.params:
            details["effective_operation"] = operation
        call = InspectCall(call.targets, InspectRequest(operation, params)) if operation != public_operation or params is not call.request.params else call
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
            return InspectResponse(public_operation, "failed", None, None, tuple(self._dedupe(cards + profile_cards)), (validation,))
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
            return InspectResponse(public_operation, "failed", None, None, tuple(self._dedupe(cards + profile_cards)), (*selection_feedback, self._feedback(exc, call.targets[0])))
        feedback = selection_feedback + [self._feedback(issue, call.targets[0]) for issue in outcome.issues]
        handle_map: dict[str, str] = {}; artifact_cards: list[EvidenceCard] = []
        created_ids: list[str] = []; reused_ids: list[str] = []
        for artifact in outcome.artifacts:
            if operation == "ocr":
                # 10-15（A7）：OCR 的采样帧只是识别的输入，不再存成 E（旧缓存条目里仍带着这些帧，这里一并跳过）
                continue
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
                        derived_from=tuple(dict.fromkeys((artifact.parent_evidence_id, *artifact.extra_parent_evidence_ids))),
                        modalities=("image",), label=artifact.label,
                        content=_json(dict(artifact.locator)), locator=_json(dict(artifact.locator)), acquired_by=artifact.acquired_by,
                        discovered_by=workspace_id, production_method=artifact.production_method,
                        artifact_bytes=artifact.data, media_type=artifact.media_type, suffix=artifact.suffix, role=artifact.role,
                    )
                    created_ids.append(evidence.evidence_id)
                handle_map[artifact.handle] = evidence.evidence_id; artifact_cards.append(evidence.card())
            except Exception as exc:
                feedback.append(InspectFeedback("error", "save", "SAVE_FAILED", f"Artifact save failed: {type(exc).__name__}"))
        result = self._public_result(operation, outcome, handle_map, tuple(x.evidence_id for x in main_prepared))
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
            label = tr(f"INSPECT {operation} 结果", f"INSPECT {operation} result")
            if operation in CONCLUSION_RECORD_OPERATIONS:
                content = conclusion_record(operation, result, analysis_context)
            elif isinstance(result, OcrResult):
                sources = tuple(x.evidence_id for x in main_prepared)
                content, label = ocr_record(result, analysis_context, sources), ocr_label(result, sources)
            else:
                content = _json({**analysis_context, **result.to_dict()})
            try:
                result_evidence = self.store.commit_derived_evidence(
                    derived_from=derived, modalities=("text",), label=label,
                    content=content, locator=_json(self._result_locator(result)),
                    acquired_by=f"INSPECT/{operation}", discovered_by=workspace_id,
                    production_method="structured_analysis",
                )
            except Exception as exc:
                feedback.append(InspectFeedback("error", "save", "SAVE_FAILED", f"result save failed: {type(exc).__name__}")); result = None; usable = False
        if not usable:
            # 后端没产出可用对象时，即使只有 warning 也要补一个明确 error。
            if not feedback: feedback.append(InspectFeedback("error", "process", "PROCESS_FAILED", "operation produced no usable result"))
            return InspectResponse(public_operation, "failed", None, None, tuple(self._dedupe(cards + profile_cards + artifact_cards)), tuple(feedback))
        if result_evidence:
            artifact_cards.append(result_evidence.card())
            created_ids.append(result_evidence.evidence_id)
        status = cast(Literal["ready", "partial", "failed"], "partial" if any(x.severity == "error" for x in feedback) else "ready")
        details.update(dict(outcome.details)); details["created_evidence_ids"] = created_ids
        if reused_ids:
            details["reused_evidence_ids"] = list(dict.fromkeys(reused_ids))
        response = InspectResponse(public_operation, status, result, result_evidence.evidence_id if result_evidence else None, tuple(self._dedupe(cards + profile_cards + artifact_cards)), tuple(feedback))
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
        video_capable = operation in {"get_frames", "ocr", "asr", "audio_qa", "locate_frames", "align_media", "diff_media",
                                      "compare_media", "video_structure", "locate"}
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
        # 只有 SEARCH 确认的视频 Candidate 来源页 E 才能触发下载（10-13：含 SEARCH 已跟到 YouTube 视频页的网页候选）。
        if not is_video_candidate(candidate) or not is_page_trigger:
            if direct is not None:
                return direct, None
            raise EvidenceResolutionError("MATERIAL_UNAVAILABLE", "Evidence is not an eligible video source page")
        needs_sound = operation in SOUND_OPERATIONS
        for kind, status, cached in reversed(linked):
            # 优先复用最近一次成功获取的视频 Evidence，保证幂等。
            if kind != "video" or status != "ready":
                continue
            if needs_sound and _video_only(cached):
                continue                     # 10-12：低清纯画面兜底下的文件没有声音，转写等要重新整片下载
            try:
                cached_media = self.resolver.resolve(cached.evidence_id)
            except EvidenceResolutionError:
                continue
            if cached_media.media_kind == "video":
                details.setdefault("acquisition", []).append({"trigger_evidence_id": evidence.evidence_id, "candidate_id": candidate_id, "result_evidence_id": cached.evidence_id, "reused": True})
                return cached_media, cached
        if self.acquirer is None:
            raise BackendFailure("MATERIAL_UNAVAILABLE", "no controlled media acquirer is configured", stage="acquire")
        try:
            acquisition_url = self._candidate_media_url(candidate, direct)
        except BackendFailure as exc:
            # 元数据里没有可直下的音画合一格式（只有分离流等）：YouTube 页面仍可交给 yt-dlp（10-05）
            page = self._candidate_download_hints(candidate, direct, "").get("page_url")
            if exc.code != "MATERIAL_UNAVAILABLE" or not page:
                raise
            acquisition_url = page
        # 10-05：平台元数据给的下载请求头与视频页地址（直链被拒时由 yt-dlp 按页面重下）；旧 acquirer 不认这两个参数时不传
        extra = {key: value for key, value in self._candidate_download_hints(candidate, direct, acquisition_url).items() if value}
        destination = self.store.new_owned_temp_path(".media")
        try:
            acquired: AcquiredMedia = await self.acquirer.acquire(acquisition_url, destination=destination, policy=self.policy,
                                                                  **extra)
        except BackendFailure as first:
            destination.unlink(missing_ok=True)
            # 10-12（原 DETECT _light_download，下沉到 INSPECT）：只要画面的操作在整片下载失败后改下 ≤360p 单路视频，
            # 再失败等一会儿重试一次。evidC27c 里 9 个原片有 4 个整片下载失败（YouTube 403、15-30 min 原片超 384 MB），
            # 同一视频几分钟前还下得下来。
            if needs_sound or first.code not in LOW_RES_FALLBACK_CODES:
                raise
            acquired = await self._low_res_acquire(acquisition_url, destination, extra, first)
            details.setdefault("low_res_fallback", []).append({"candidate_id": candidate_id, "first_error": f"{first.code}: {first}"[:300]})
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

    LOW_RES_HEIGHT = 360
    LOW_RES_RETRY_PAUSE_S = 20.0

    async def fetch_picture(self, candidate_id: str, page_evidence_id: str, destination: Path) -> Path:
        """10-12 SEARCH 内容比对用：视频候选的低清纯画面文件（不登记 Evidence，只给比对用）。

        本候选已有下载好的视频就直接用；YouTube 页面下 ≤360p 单路视频，其他平台按元数据整文件下载。
        走同一个受控 acquirer，所以录制回放照常生效（低清下载单独成键）。"""

        candidate = self.store.candidate(candidate_id)
        if not is_video_candidate(candidate):
            raise BackendFailure("MATERIAL_UNAVAILABLE", "not a video candidate", stage="acquire")
        for kind, status, cached in reversed(self.store.candidate_evidence(candidate_id)):
            if kind == "video" and status == "ready":
                try:
                    resolved = self.resolver.resolve(cached.evidence_id)
                except EvidenceResolutionError:
                    continue
                if resolved.media_kind == "video":
                    return resolved.path
        if self.acquirer is None:
            raise BackendFailure("MATERIAL_UNAVAILABLE", "no controlled media acquirer is configured", stage="acquire")
        try:
            trigger: ResolvedEvidence | None = self.resolver.resolve(page_evidence_id)
        except EvidenceResolutionError:
            trigger = None
        hints = self._candidate_download_hints(candidate, trigger, "")
        page = hints.get("page_url") or candidate.get("source_url")
        try:
            url = self._candidate_media_url(candidate, trigger)
        except BackendFailure:
            if not page:
                raise
            url = page
        extra = {key: value for key, value in self._candidate_download_hints(candidate, trigger, url).items() if value}
        if page and "page_url" not in extra:
            extra["page_url"] = page
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        acquired = await self.acquirer.acquire(url, destination=destination, policy=self.policy,
                                               video_only_height=self.LOW_RES_HEIGHT, **extra)
        return acquired.path

    async def _low_res_acquire(self, url: str, destination: Path, extra: dict[str, Any], first: BackendFailure) -> AcquiredMedia:
        errors = []
        for attempt in range(2):
            if attempt:
                await asyncio.sleep(self.LOW_RES_RETRY_PAUSE_S)
            try:
                return await self.acquirer.acquire(url, destination=destination, policy=self.policy,
                                                   video_only_height=self.LOW_RES_HEIGHT, **extra)
            except TypeError:
                raise first from None            # 旧 acquirer 不认低清参数
            except BackendFailure as exc:
                destination.unlink(missing_ok=True)
                errors.append(f"{exc.code}: {exc}"[:200])
        raise BackendFailure(first.code, f"{first}; low-resolution picture-only retries also failed ({'; '.join(errors)})",
                             stage="acquire", retryable=first.retryable) from first

    @staticmethod
    def _candidate_download_hints(candidate: dict[str, Any], trigger: ResolvedEvidence | None,
                                  acquisition_url: str) -> dict[str, Any]:
        """选中格式的 http_headers 与 YouTube 视频页地址（10-05，evidb7：直链 5/5 被 HTTP 403）。拿不到就返回空。"""

        raw = InspectService._platform_metadata(trigger)
        if raw is None:
            return {}
        extractor = str(raw.get("extractor_key") or raw.get("extractor") or "").casefold()
        headers = None
        for item in raw.get("formats") or ():
            if isinstance(item, dict) and acquisition_url in (item.get("url"), item.get("manifest_url")):
                if isinstance(item.get("http_headers"), dict):
                    headers = {str(k): str(v) for k, v in item["http_headers"].items() if isinstance(v, str)}
                break
        page = raw.get("webpage_url") if extractor in {"youtube"} else None
        return {"headers": headers, "page_url": page if isinstance(page, str) and page.startswith("https://") else None}

    @staticmethod
    def _platform_metadata(trigger: ResolvedEvidence | None) -> dict[str, Any] | None:
        if trigger is None or trigger.artifact.media_type not in {"application/json", "text/json"}:
            return None
        try:
            if trigger.artifact.byte_size > 32 * 1024 * 1024:
                return None
            raw = json.loads(trigger.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            return None
        return raw if isinstance(raw, dict) else None

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

        if english():
            parts = [f"media profile: {profile.media_kind}"]
            if profile.duration_s is not None: parts.append(f"{profile.duration_s:.3f} s")
            if profile.width and profile.height: parts.append(f"{profile.width}×{profile.height}")
            if profile.fps: parts.append(f"{profile.fps:.3f} fps")
            if profile.page_count: parts.append(f"{profile.page_count} pages")
            parts.append(f"{profile.audio_tracks} audio track{'s' if profile.audio_tracks != 1 else ''}")
            return ", ".join(parts) + MEDIA_PROFILE_BOILERPLATE_EN
        parts = [f"媒体属性：{profile.media_kind}"]
        if profile.duration_s is not None: parts.append(f"{profile.duration_s:.3f}秒")
        if profile.width and profile.height: parts.append(f"{profile.width}×{profile.height}")
        if profile.fps: parts.append(f"{profile.fps:.3f}fps")
        if profile.page_count: parts.append(f"{profile.page_count}页")
        parts.append(f"{profile.audio_tracks}条音轨")
        return "，".join(parts) + "；本记录未核验日期、地点或作者等来源元数据，不能据此判断这些信息不存在。"

    def _translate(self, operation: InspectOperation, params: Any, media: tuple[PreparedMedia, ...]) -> tuple[InspectOperation, Any]:
        """合并目录（hi@0.8.0 起）的操作按目标媒体类型改派到具体实现：

        get_frames：PDF → get_pages，图片 + box → crop（视频 + box 仍是 get_frames，后端裁剪每帧）；
        locate：视频 → locate_frames，图片 → locate_regions（只接受文字查询）；
        asr：合并目录下附带声音时间轴；
        compare_media：compact_params 打开时不带 mode 的请求改成 auto_fallback。"""

        kind = media[0].media_kind if media else None
        if operation == "get_frames" and isinstance(params, GetFramesParams):
            if kind == "document":
                if params.times_s is not None or params.range_s is not None:
                    raise InspectValidationError("get_frames on a PDF takes pages or page_range/count, not times_s/range_s", operation)
                if params.box is not None:
                    raise InspectValidationError("box cannot be combined with PDF pages: render the page first, then call get_frames with box on the page image E", operation)
                return "get_pages", GetPagesParams(params.pages, params.page_range, params.count)
            if kind == "image":
                if params.box is None:
                    raise InspectValidationError("get_frames on an image requires box (a crop); the image itself is already Evidence", operation)
                return "crop", CropParams(params.box)
            if params.pages is not None or params.page_range is not None:
                raise InspectValidationError(f"pages/page_range require a PDF; {media[0].evidence_id} is {kind}", operation)
            return operation, params
        if operation == "locate" and isinstance(params, LocateParams):
            if kind == "image":
                if not isinstance(params.query, TextFrameQuery):
                    raise InspectValidationError("locate on an image takes a text query", operation)
                return "locate_regions", LocateRegionsParams(params.query.text, params.top_k)
            if kind == "video":
                return "locate_frames", LocateFramesParams(params.query, params.range_s, params.top_k)
            raise InspectValidationError(f"locate requires a video or an image; {media[0].evidence_id} is {kind}", operation)
        if operation == "asr" and is_merged_catalog(self.enabled_operations) and getattr(params, "sounds", None) is None:
            return operation, replace(params, sounds=True)
        if operation == "compare_media" and getattr(self, "compact_params", False) and getattr(params, "mode", "x") is None:
            # 10-13 compact_tool_params（hi@0.10.0 起）：默认比对在画面对不上时退到只比声音（原 mode=audio 并入）。
            # 参数变了，工具缓存键也随之不同，旧缓存不会被误用。
            kinds = {x.media_kind for x in media}
            if "audio" in kinds and kinds <= {"audio", "video"}:
                return operation, replace(params, mode="audio")      # 有一边只是音频：只能比声音
            return operation, replace(params, mode="auto_fallback")
        return operation, params

    def _validate_real(self, operation: InspectOperation, params: Any, media: tuple[PreparedMedia, ...], query_media: PreparedMedia | None) -> InspectFeedback | None:
        """使用 probe 后的真实属性校验媒体类型、范围、数量和模型预算。"""

        kinds = tuple(x.media_kind for x in media)
        # 首先校验每种操作允许的媒体组合。
        expected = {"get_frames":{"video"},"get_pages":{"document"},"crop":{"image"},"locate_regions":{"image"},"ocr":{"image","video"},"asr":{"audio","video"},"audio_qa":{"audio","video"},"visual_qa":{"image"},"locate_frames":{"video"},"video_structure":{"video"}}.get(operation)
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
        if operation == "diff_media":
            # 两侧须同为视频或同为图片；图片对只比较一次，没有时间轴可映射。
            if kinds not in {("video", "video"), ("image", "image")}:
                return InspectFeedback("error", "resolve", "UNSUPPORTED_MEDIA", f"diff_media requires two videos or two images; got {kinds}", media[0].evidence_id)
            if kinds == ("image", "image") and (params.mapping is not None or params.range_a_s is not None):
                return InspectFeedback("error", "validate", "INVALID_INPUT", "diff_media image pairs do not accept mapping or range_a_s", media[0].evidence_id)
        if operation == "compare_media":
            mode = params.mode or "auto"
            if mode == "audio":
                if any(x not in {"audio", "video"} for x in kinds):
                    return InspectFeedback("error", "resolve", "UNSUPPORTED_MEDIA", f"compare_media audio requires audio or video on both sides; got {kinds}", media[0].evidence_id)
                if any(x.profile.audio_tracks == 0 for x in media):
                    return InspectFeedback("error", "probe", "NO_AUDIO", "one input has no usable audio track", media[0].evidence_id)
            elif kinds not in {("video", "video"), ("image", "image")}:
                return InspectFeedback("error", "resolve", "UNSUPPORTED_MEDIA", f"compare_media requires two videos or two images{'' if getattr(self, 'compact_params', False) else ' (or mode=audio)'}; got {kinds}", media[0].evidence_id)
            elif kinds == ("image", "image") and params.range_a_s is not None:
                return InspectFeedback("error", "validate", "INVALID_INPUT", "compare_media image pairs do not accept range_a_s", media[0].evidence_id)
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
    def _public_result(operation: InspectOperation, outcome: BackendOutcome, handles: dict[str,str],
                       media_ids: tuple[str, ...] = ()) -> InspectResult | None:
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
            # 10-15（A7）：视频帧不再存成 E，行指向输入视频本身、靠 time_s 定位
            rows=[]
            for x in raw["results"]:
                evidence_id=x.get("image_evidence_id") or handles.get(x.get("image_handle")) or (media_ids[0] if media_ids and x.get("time_s") is not None else None)
                if evidence_id: rows.append(OcrImageResult(evidence_id,x["time_s"],tuple(OcrItem(i["text"],tuple(i["box"]),i["confidence"]) for i in x["items"])))
            return OcrResult(ocr_coverage,tuple(rows))
        if operation == "asr":
            cov=raw["coverage"]
            sounds = tuple(SoundSegment(float(x["start_s"]), float(x["end_s"]), x["kind"], x.get("label")) for x in raw["sounds"]) if raw.get("sounds") is not None else None
            return AsrResult(TimeCoverage(tuple(cov["range_s"]),tuple(tuple(x) for x in cov["processed_ranges"]),tuple(tuple(x) for x in cov["failed_ranges"])),raw["language"],int(raw["audio_track"]),tuple(AsrSegment(float(x["start_s"]),float(x["end_s"]),x["text"]) for x in raw["segments"]), sounds, raw.get("sound_method"))
        if operation == "audio_qa": return AudioQaResult(tuple(raw["range_s"]),int(raw["audio_track"]),raw["answer"],tuple(raw["limitations"]))
        if operation == "visual_qa": return VisualQaResult(tuple(raw["image_evidence_ids"]),raw["answer"],tuple(raw["limitations"]))
        if operation == "locate_frames":
            cov=raw["coverage"]; locate_coverage=LocateFramesCoverage(tuple(cov["range_s"]),int(cov["coarse_frames"]),tuple(tuple(x) for x in cov["refined_ranges"]),tuple(tuple(x) for x in cov["failed_ranges"])); return LocateFramesResult(locate_coverage,tuple(LocatedFrame(handles[x["handle"]],float(x["time_s"]),float(x["score"])) for x in raw["candidates"] if x["handle"] in handles))
        if operation == "diff_media":
            # 对照图 handle 映射为正式 E；保存失败的图片不进入结果。
            images = tuple(DiffImage(handles[x["handle"]], x["kind"], x["segment_index"], x["frame_a"], x["time_a_s"],
                                     x["frame_b"], x["time_b_s"], float(x["changed_area"])) for x in raw["images"] if x["handle"] in handles)
            segments = tuple(DiffSegment(**{**{k: v for k, v in x.items() if k != "image_handles"},
                                            "union_bbox": tuple(x["union_bbox"]),
                                            "image_evidence_ids": tuple(handles[h] for h in x["image_handles"] if h in handles)})
                             for x in raw["segments"])
            return DiffMediaResult(raw["media_kind"], bool(raw["differs"]), raw["summary"], segments, raw["image_comparison"],
                                   raw["timeline"], dict(raw["coverage"]), raw["frame_rate"], raw["mapping"], dict(raw["threshold"]),
                                   float(raw["min_area"]), tuple(raw["region"]) if raw["region"] else None, dict(raw["registration"]),
                                   raw["segment_definition"], images, tuple(raw["limitations"]))
        if operation in {"compare_media", "video_structure"}:
            images = tuple(ReportImage(handles[x["handle"]], str(x.get("kind")), x.get("index"), str(x.get("caption") or ""))
                           for x in raw.get("images", ()) if x.get("handle") in handles)
            report = {k: v for k, v in raw.items() if k not in {"summary_lines", "images", "media_kind"}}
            diff = report.get("frame_diff")
            if isinstance(diff, dict):
                diff = dict(diff)
                diff["images"] = [{**{k: v for k, v in x.items() if k != "handle"}, "evidence_id": handles[x["handle"]]}
                                  for x in diff.get("images", ()) if x.get("handle") in handles]
                for segment in diff.get("segments", ()):
                    if isinstance(segment, dict) and "image_handles" in segment:
                        segment["image_evidence_ids"] = [handles[h] for h in segment.pop("image_handles") if h in handles]
                report["frame_diff"] = diff
            if operation == "video_structure":
                return VideoStructureResult(tuple(raw["summary_lines"]), report, images)
            return CompareMediaResult(raw.get("media_kind", "video"), tuple(raw["summary_lines"]), report, images)
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
        if isinstance(result, DiffMediaResult):
            return int(result.coverage.get("compared_frames", 0)) > 0
        if isinstance(result, (CompareMediaResult, VideoStructureResult)):
            return bool(result.summary_lines)
        return True

    @staticmethod
    def _result_locator(result: InspectResult) -> dict[str,Any]:
        """从结构化结果提取足以定位其分析范围的小型元数据。"""

        value = result.to_dict()
        return {key:value[key] for key in ("coverage","range_s","image_evidence_id","image_evidence_ids") if key in value}
