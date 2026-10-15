"""INSPECT 工具结果缓存：让 OCR / ASR / 对齐 / 抽帧这类昂贵且可能不确定的调用在不同臂之间共用同一份结果。

为什么落在 Processor 层（dispatcher 调 ``processor.process`` 的地方）：
- 这一层是"纯函数"：输入是已经解析到本地文件并 probe 过的媒体、强类型参数和 InspectPolicy，输出是
  ``BackendOutcome``（结果字典 + 内存里的待提交图片 + 问题 + 诊断）。它**不碰 Store、不分配 E 编号**。
- E 编号分配、Artifact 提交、帧 / 页复用（``_reusable_media_image``）、结构化结果 E 的写入全在 InspectService
  里、在 processor 返回之后。缓存命中后照常走这段路径，所以证据记录的形态（derived_from、locator、
  production_method、content）与未命中时完全一致，只是 BackendOutcome 来自缓存。
- 再往下（backend 里具体的 RapidOCR / Whisper 调用）粒度太细，一个 OCR 请求要调几十次引擎；再往上
  （InspectService.execute）已经写了 Store，缓存它等于缓存 E 编号，不同 run 编号不同，会坏证据闭包。

key = (操作名, 各输入媒体文件的 sha256 与媒体类型, 规范化参数, 影响输出的 InspectPolicy 字段, 后端 / 模型标识及版本)。
visual_qa / audio_qa 的提示词里会出现 E 编号和来源上下文，它们也进 key；其余操作输出里的 E 编号只出现在结构位置
（PendingArtifact.parent_evidence_id、标签、image_evidence_id），存缓存时换成 ``⟦media0⟧`` 这样的占位符，
命中时换回本次调用的 E 编号——同一媒体在不同 run 里编号不同也能命中。

value = 结果 JSON + 产出文件（按 sha256 去重进 blobs/）。只缓存成功且没有"可重试问题"的结果：失败和瞬时问题
（某帧解码超时之类）不应被固化下来。

目录（默认 experiments/sft_data/tool_cache/）::

    entries/<key[:2]>/<key>.json   一次调用的结果
    blobs/<sha[:2]>/<sha>          帧 / 页 / 裁剪图等产出文件

模式：``off``（不包装，默认）/ ``on``（命中读、未命中算完写）/ ``refresh``（一律重算并覆盖，用于换了模型之后）。
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Literal, Mapping

from .backends.base import BackendFailure, BackendOutcome, PendingArtifact, PreparedMedia

ToolCacheMode = Literal["off", "on", "refresh"]
TOOL_CACHE_MODES: tuple[str, ...] = ("off", "on", "refresh")
DEFAULT_TOOL_CACHE_ROOT = Path("experiments") / "sft_data" / "tool_cache"
CACHE_FORMAT = 1

# 提示词里带 E 编号 / 来源上下文的操作：编号本身是模型输入，必须进 key，不能占位替换。
_CONTEXT_SENSITIVE = frozenset({"visual_qa", "audio_qa"})
# 各操作读哪些环境变量来选模型（与 backends/local.py 一致）；换模型必须换 key。
_OPERATION_ENV: dict[str, tuple[str, ...]] = {
    "asr": ("FOREN_WHISPER_MODEL", "FOREN_WHISPER_DEVICE"),
    "visual_qa": ("FOREN_VLM_MODEL",),
    "audio_qa": ("FOREN_AUDIO_MODEL",),
    "locate_frames": ("FOREN_CLIP_MODEL",),
    "locate_regions": ("FOREN_GROUNDING_MODEL",),
}
# 各操作依赖的包；版本进 key（装了新版 rapidocr / faster-whisper 后旧缓存自动失效）。
_COMMON_PACKAGES = ("av", "numpy", "Pillow", "opencv-python", "opencv-python-headless")
_OPERATION_PACKAGES: dict[str, tuple[str, ...]] = {
    "ocr": ("rapidocr_onnxruntime", "rapidocr-onnxruntime", "onnxruntime"),
    "asr": ("faster-whisper", "ctranslate2"),
    "get_pages": ("PyMuPDF",),
    "visual_qa": ("transformers", "torch"),
    "audio_qa": ("transformers", "torch"),
    "locate_frames": ("transformers", "torch"),
    "locate_regions": ("transformers", "torch"),
    "align_media": ("scipy",),
    "diff_media": (),          # 只用 PyAV / OpenCV / numpy（已在 _COMMON_PACKAGES）；阈值与配准都是确定性计算
}
# 只影响资源保护、不影响结果内容的策略字段：不进 key，改超时不该让缓存全部失效。
_POLICY_IGNORED = frozenset({"backend_workers", "execution_timeout_s", "acquisition_timeout_s", "maximum_file_bytes",
                             "maximum_duration_s", "maximum_image_pixels", "maximum_download_bytes"})
_RESULT_ID_KEYS = frozenset({"image_evidence_id", "image_evidence_ids"})


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_default)


def _default(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, (tuple, set, frozenset)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    # numpy 标量 / 数组：还原成 Python 数字和列表，不能变成字符串（否则命中时数值类型与未命中不同）。
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _jsonable(value: Any) -> Any:
    """结果 / 诊断 → 纯 JSON 值（元组变列表）；与 Service 最终写 Store 时的 JSON 形态一致。"""

    return json.loads(canonical_json(value))


# ---------------------------------------------------------------- 媒体哈希

class _FileHashes:
    """按 (路径, 大小, mtime) 记住文件哈希：同一个大视频在一次 run 里被多次 INSPECT，不必每次重算。"""

    def __init__(self) -> None:
        self._known: dict[tuple[str, int, int], str] = {}
        self._lock = threading.Lock()

    def sha256(self, path: Path) -> str:
        stat = Path(path).stat()
        marker = (str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns)
        with self._lock:
            if marker in self._known:
                return self._known[marker]
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        with self._lock:
            self._known[marker] = value
        return value


# ---------------------------------------------------------------- 后端标识

def _package_versions(names: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def backend_identity(processor: Any, operation: str) -> dict[str, Any]:
    """后端类名 + 模型环境变量 + 相关包版本。后端可自带 cache_identity(operation) 补充更精确的标识（如模型权重哈希）。"""

    backend = getattr(processor, "backend", processor)
    identity: dict[str, Any] = {
        "backend": f"{type(backend).__module__}.{type(backend).__qualname__}",
        "env": {name: os.environ.get(name) for name in _OPERATION_ENV.get(operation, ())},
        "packages": _package_versions(_COMMON_PACKAGES + _OPERATION_PACKAGES.get(operation, ())),
    }
    extra = getattr(backend, "cache_identity", None)
    if callable(extra):
        identity["backend_extra"] = extra(operation)
    return identity


def policy_view(policy: Any) -> dict[str, Any]:
    raw = policy.to_dict() if hasattr(policy, "to_dict") else dataclasses.asdict(policy)
    return {key: value for key, value in raw.items() if key not in _POLICY_IGNORED}


# ---------------------------------------------------------------- E 编号占位

def _placeholder(index: int, query: bool = False) -> str:
    return "⟦query⟧" if query else f"⟦media{index}⟧"


def _id_pattern(evidence_id: str) -> re.Pattern[str]:
    # E3 不能误伤 E31，也不能误伤 AE3 这类片段。
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(evidence_id)}(?![0-9])")


class _IdMapper:
    """本次调用的输入 E 编号 ↔ 占位符。只替换结构位置，不碰 OCR / ASR 识别出的文字。"""

    def __init__(self, media: tuple[PreparedMedia, ...], query_media: PreparedMedia | None) -> None:
        self.to_placeholder: dict[str, str] = {}
        for index, item in enumerate(media):
            self.to_placeholder.setdefault(item.evidence_id, _placeholder(index))
        if query_media is not None:
            self.to_placeholder.setdefault(query_media.evidence_id, _placeholder(0, query=True))
        self.to_id = {v: k for k, v in self.to_placeholder.items()}

    def exact(self, value: Any, forward: bool) -> Any:
        table = self.to_placeholder if forward else self.to_id
        if isinstance(value, str):
            return table.get(value, value)
        if isinstance(value, list):
            return [self.exact(v, forward) for v in value]
        return value

    def text(self, value: str, forward: bool) -> str:
        table = self.to_placeholder if forward else self.to_id
        for old, new in table.items():
            value = _id_pattern(old).sub(new, value) if forward else value.replace(old, new)
        return value

    def result(self, value: Any, forward: bool) -> Any:
        """递归替换结果里 image_evidence_id / image_evidence_ids 两个键的值。"""

        if isinstance(value, dict):
            return {k: (self.exact(v, forward) if k in _RESULT_ID_KEYS else self.result(v, forward)) for k, v in value.items()}
        if isinstance(value, list):
            return [self.result(v, forward) for v in value]
        return value


# ---------------------------------------------------------------- 缓存本体

class ToolCache:
    """缓存目录 + 命中统计。线程安全：dispatcher 在多个工作线程里并行调 processor。"""

    def __init__(self, root: str | Path = DEFAULT_TOOL_CACHE_ROOT, mode: ToolCacheMode = "on") -> None:
        if mode not in TOOL_CACHE_MODES:
            raise ValueError(f"tool cache mode 应为 {' / '.join(TOOL_CACHE_MODES)}，得到 {mode!r}")
        self.root = Path(root)
        self.mode = mode
        self.hashes = _FileHashes()
        self._lock = threading.Lock()
        self.counts: dict[str, int] = {"hit": 0, "miss": 0, "stored": 0, "refreshed": 0, "not_cacheable": 0,
                                       "corrupt": 0}
        self.by_operation: dict[str, dict[str, int]] = {}
        self.saved_seconds = 0.0

    # ---- 计数
    def count(self, operation: str, name: str) -> None:
        with self._lock:
            self.counts[name] = self.counts.get(name, 0) + 1
            bucket = self.by_operation.setdefault(operation, {})
            bucket[name] = bucket.get(name, 0) + 1

    def summary(self) -> dict[str, Any]:
        lookups = self.counts["hit"] + self.counts["miss"]
        return {"mode": self.mode, "root": str(self.root), "counts": dict(self.counts),
                "by_operation": {k: dict(v) for k, v in self.by_operation.items()},
                "hit_rate": round(self.counts["hit"] / lookups, 4) if lookups else None,
                "saved_seconds": round(self.saved_seconds, 1)}

    # ---- key
    def key_material(self, processor: Any, operation: str, media: tuple[PreparedMedia, ...], params: Any,
                     policy: Any, query_media: PreparedMedia | None) -> dict[str, Any]:
        sensitive = operation in _CONTEXT_SENSITIVE

        def describe(item: PreparedMedia) -> dict[str, Any]:
            value: dict[str, Any] = {"sha256": self.hashes.sha256(item.path), "media_kind": item.media_kind}
            if sensitive:
                value["evidence_id"] = item.evidence_id
                value["source_context"] = item.source_context
            return value

        params_value = _jsonable(params)
        if not sensitive and isinstance(params_value, dict) and isinstance(params_value.get("query"), dict) \
                and query_media is not None and params_value["query"].get("evidence_id") == query_media.evidence_id:
            # locate_frames 的图片查询：查询图的内容已由 query_media 的哈希代表，E 编号不进 key。
            params_value = {**params_value, "query": {**params_value["query"], "evidence_id": _placeholder(0, query=True)}}
        return {"format": CACHE_FORMAT, "operation": operation, "media": [describe(m) for m in media],
                "query_media": describe(query_media) if query_media is not None else None,
                "params": params_value, "policy": policy_view(policy), "backend": backend_identity(processor, operation)}

    @staticmethod
    def key(material: dict[str, Any]) -> str:
        return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()

    # ---- 存取
    def entry_path(self, key: str) -> Path:
        return self.root / "entries" / key[:2] / f"{key}.json"

    def blob_path(self, sha: str) -> Path:
        return self.root / "blobs" / sha[:2] / sha

    def put_blob(self, data: bytes) -> str:
        sha = hashlib.sha256(data).hexdigest()
        target = self.blob_path(sha)
        if not target.is_file():
            _atomic_write(target, data)
        return sha

    def get_blob(self, sha: str) -> bytes:
        data = self.blob_path(sha).read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            raise OSError(f"tool cache blob {sha} is corrupted")
        return data

    def load(self, key: str, mapper: _IdMapper, media: tuple[PreparedMedia, ...]) -> tuple[BackendOutcome, float] | None:
        path = self.entry_path(key)
        if not path.is_file():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            artifacts = tuple(PendingArtifact(
                handle=item["handle"], data=self.get_blob(item["sha256"]), media_type=item["media_type"],
                suffix=item["suffix"], role=item["role"], parent_evidence_id=mapper.exact(item["parent_evidence_id"], False),
                label=mapper.text(item["label"], False), locator=item["locator"], acquired_by=item["acquired_by"],
                production_method=item["production_method"],
                extra_parent_evidence_ids=tuple(mapper.exact(list(item.get("extra_parent_evidence_ids", [])), False)))
                for item in entry["artifacts"])
            issues = tuple(BackendFailure(item["code"], item["message"], stage=item["stage"], retryable=item["retryable"],
                                          locator=item.get("locator"), severity=item.get("severity", "error"))
                           for item in entry["issues"])
            outcome = BackendOutcome(mapper.result(entry["result"], False), artifacts, issues, dict(entry["details"]))
            return outcome, float(entry.get("elapsed_s") or 0.0)
        except (OSError, KeyError, TypeError, ValueError):
            return None

    def store(self, key: str, material: dict[str, Any], outcome: BackendOutcome, mapper: _IdMapper, elapsed_s: float) -> None:
        artifacts = [{"handle": a.handle, "sha256": self.put_blob(a.data), "bytes": len(a.data), "media_type": a.media_type,
                      "suffix": a.suffix, "role": a.role, "parent_evidence_id": mapper.exact(a.parent_evidence_id, True),
                      "label": mapper.text(a.label, True), "locator": _jsonable(dict(a.locator)),
                      "acquired_by": a.acquired_by, "production_method": a.production_method,
                      **({"extra_parent_evidence_ids": mapper.exact(list(a.extra_parent_evidence_ids), True)}
                         if a.extra_parent_evidence_ids else {})} for a in outcome.artifacts]
        issues = [{"code": i.code, "message": str(i), "stage": i.stage, "retryable": i.retryable,
                   "locator": _jsonable(i.locator) if i.locator is not None else None, "severity": i.severity}
                  for i in outcome.issues]
        entry = {"format": CACHE_FORMAT, "key": key, "key_material": material,
                 "result": mapper.result(_jsonable(dict(outcome.result)), True), "artifacts": artifacts, "issues": issues,
                 "details": _jsonable(dict(outcome.details)), "elapsed_s": round(elapsed_s, 3),
                 "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _atomic_write(self.entry_path(key), canonical_json(entry).encode("utf-8"))


def _atomic_write(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=target.parent, prefix=".tmp-")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        os.replace(temp, target)
    finally:
        Path(temp).unlink(missing_ok=True)


def cacheable(outcome: BackendOutcome) -> bool:
    """只固化确定的结果：有可重试问题（某帧解码失败、模型一时出错）的不缓存，下次还有机会算全。"""

    return not any(getattr(issue, "retryable", False) for issue in outcome.issues)


# ---------------------------------------------------------------- Processor 包装

class CachingProcessor:
    """包一个 InspectProcessor；接口与被包的一样，dispatcher 不需要知道缓存存在。"""

    def __init__(self, inner: Any, cache: ToolCache) -> None:
        self.inner, self.cache = inner, cache

    @property
    def operation(self) -> str:
        return self.inner.operation

    @property
    def backend(self) -> Any:
        return getattr(self.inner, "backend", self.inner)

    async def process(self, media: tuple[PreparedMedia, ...], params: Any, policy: Any,
                      query_media: PreparedMedia | None = None) -> BackendOutcome:
        cache, operation = self.cache, self.inner.operation
        mapper = _IdMapper(media, query_media)
        material = cache.key_material(self.inner, operation, media, params, policy, query_media)
        key = cache.key(material)
        if cache.mode == "on":
            loaded = cache.load(key, mapper, media)
            if loaded is not None:
                outcome, elapsed = loaded
                cache.count(operation, "hit")
                with cache._lock:
                    cache.saved_seconds += elapsed
                return _with_cache_detail(outcome, "hit", key)
            if cache.entry_path(key).is_file():
                cache.count(operation, "corrupt")
            cache.count(operation, "miss")
        else:
            cache.count(operation, "refreshed")
        started = time.monotonic()
        outcome = await self.inner.process(media, params, policy, query_media)
        elapsed = time.monotonic() - started
        if cacheable(outcome):
            try:
                cache.store(key, material, outcome, mapper, elapsed)
                cache.count(operation, "stored")
            except OSError:
                cache.count(operation, "not_cacheable")
        else:
            cache.count(operation, "not_cacheable")
        return _with_cache_detail(outcome, "miss" if cache.mode == "on" else "refresh", key)


def _with_cache_detail(outcome: BackendOutcome, status: str, key: str) -> BackendOutcome:
    """执行审计的 details 里标注缓存状态（命中与未命中都标，形态一致）；结果 E 的内容不读这个键。"""

    return dataclasses.replace(outcome, details={**dict(outcome.details), "tool_cache": {"status": status, "key": key}})


def apply_tool_cache(inspect_service: Any, cache: ToolCache | None) -> Any:
    """把 InspectService 的 dispatcher 里每个 processor 换成 CachingProcessor。off / None 时原样返回。"""

    if cache is None or cache.mode == "off":
        return inspect_service
    dispatcher = inspect_service.dispatcher
    dispatcher.processors = {name: processor if isinstance(processor, CachingProcessor) else CachingProcessor(processor, cache)
                             for name, processor in dispatcher.processors.items()}
    return inspect_service


def processors_without_cache(processors: Mapping[str, Any]) -> dict[str, Any]:
    """反向：取出被包的原始 processor（方差脚本要关掉缓存重跑）。"""

    return {name: getattr(p, "inner", p) for name, p in processors.items()}


__all__ = ["CachingProcessor", "DEFAULT_TOOL_CACHE_ROOT", "TOOL_CACHE_MODES", "ToolCache", "ToolCacheMode",
           "apply_tool_cache", "backend_identity", "cacheable", "policy_view", "processors_without_cache"]
