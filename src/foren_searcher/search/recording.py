"""检索录制回放：把一个案例看到的"检索世界"固化成快照（环境固化，论文硬要求）。

为什么要有：同一案例在不同臂（有 / 无某项改动）、不同轮次重跑时，开放网络会漂移——搜索引擎排序在变、页面被改、
视频被删、限流时有时无。两臂结果的差异若混进了网络漂移，对照就不干净。这里把 provider 的每次查询、opener 的每次
打开、INSPECT 受控获取的每个视频文件都写进按案例共享的快照目录；回放时只从快照读，不联网。

写法与 ``retrieval_guard`` 一致：包在 provider / opener / acquirer 外面，不改 SEARCH / INSPECT 的接口。
套的顺序是 guard(filter(replay(base)))——回放在最里层，记录的是"原始世界"（含核查站点、晚于截止日的来源），
案卷过滤与护栏在回放结果上照常生效，所以换护栏参数不需要重录。

四种模式：
- ``live``：不包装（默认，保持现状）。
- ``record``：照常联网，把每次结果写进快照（同 key 后写覆盖先写；内容指纹没变就不重复写）。
- ``replay``：只读快照；未命中抛 ``ReplayMiss``（SearchFailure，reason=replay_miss），写日志，绝不联网。
- ``replay_or_record``：命中就回放，未命中就联网并补录。录下的"瞬时失败"（超时、网络错误、限流）在这个模式下
  当作未命中重新联网；在 ``replay`` 模式下照原样回放（同一个世界里那次就是超时了）。

目录结构（快照按 case_id 共享，不按 run 分）::

    <snapshot_root>/<case_key>/
        index.jsonl            每行一条：kind(search|open|acquire)、key、request、outcome、payload、指纹、run、时间
        blobs/<sha[:2]>/<sha>  正文、原始响应、材料字节、视频文件；按 sha256 去重
        usage/<run>.json       每个 run 的命中 / 未命中 / 新录计数（scripts/snapshot_stats.py 读它算命中率）

请求规范化（同一请求在不同 run 里必须得到同一个 key）：
- 查询：input_type、output_type、offset、limit、NFC + 折叠空白后的查询文本；图片查询不用 E 编号（每个 run 的编号
  可能不同），改用上传给 provider 的图片字节的 sha256。
- 打开：规范化 URL（小写主机、去默认端口、去 fragment、去 utm_* / fbclid / YouTube si= 这类随机追踪参数、查询参数
  排序）、候选类型、图片 / 缩略图 URL、平台视频 ID、请求的输出类型与字幕选项。候选的标题、摘要、排名、extra 这类
  随 provider 响应变化的字段不进 key。
- 内容指纹：去掉时间戳 / 请求 ID / 耗时一类易变字段后再算，用来判断"重录是否真的变了"（统计漂移），不影响回放。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .contracts import (
    CandidateBatch, CandidateSource, OpenedMaterial, OpenedSource, OpenSourceRequest, SearchError, SearchFailure,
    SearchQueryRequest,
)
from .identity import normalize_url

SearchMode = Literal["live", "record", "replay", "replay_or_record"]
SEARCH_MODES: tuple[str, ...] = ("live", "record", "replay", "replay_or_record")
DEFAULT_SNAPSHOT_ROOT = Path("experiments") / "sft_data" / "search_snapshots"
SNAPSHOT_FORMAT = 1

_LOGGER = logging.getLogger(__name__)
# 这些错误码代表"那一刻网络不好"，不是来源本身的属性：replay_or_record 时值得重新联网。
TRANSIENT_CODES = frozenset({"NETWORK_ERROR", "TIMEOUT", "PROVIDER_ERROR", "RATE_LIMITED"})
TRANSIENT_BACKEND_CODES = frozenset({"TIMEOUT", "NETWORK_ERROR"})
# 追踪 / 分享参数每次搜索都可能不同（YouTube 的 si= 是随机串），留着会让同一来源得到不同 key。
_TRACKING_PARAMS = re.compile(
    r"^(utm_[a-z0-9_]+|fbclid|gclid|dclid|msclkid|yclid|mc_cid|mc_eid|_hsenc|_hsmi|igshid|ref_src|si|"
    r"spm|scm|share_source|share_medium|vd_source|feature)$", re.I)
# 内容指纹要忽略的易变字段名（provider 原始响应里的时间戳、请求 ID、耗时；我们自己加的录制时间）。
_VOLATILE_KEYS = re.compile(
    r"^(timestamp|time|ts|date_fetched|fetched_at|retrieved_at|crawled_at|cached_at|created_at|processed_at|"
    r"request_id|requestid|search_id|searchid|id_token|nonce|session|session_id|trace_id|elapsed|elapsed_ms|"
    r"time_taken|total_time_taken|took|latency|search_metadata|recorded_at|number_of_results)$", re.I)
_INLINE_LIMIT = 2048            # 超过这么多字符的文本进 blob，index.jsonl 每行保持小，追加更接近原子


# ---------------------------------------------------------------- 规范化与指纹

def canonical_json(value: Any) -> str:
    """字段排序、紧凑、保留中文的 JSON；非 JSON 类型转字符串（与 SearchService 保存原始响应的做法一致）。"""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (tuple, set, frozenset)):
        return sorted(value, key=str) if isinstance(value, (set, frozenset)) else list(value)
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
    return str(value)


def normalize_text(text: str) -> str:
    """NFC 统一写法 + 折叠所有空白：同一查询多一个空格、全角空格不应变成另一次搜索。"""

    return " ".join(unicodedata.normalize("NFC", text).split())


def snapshot_url(url: str | None) -> str | None:
    """在 identity.normalize_url 之上再去掉随机追踪参数、给查询参数排序，只用于算快照 key。"""

    if not url:
        return None
    try:
        base = normalize_url(url.strip())
        parts = urlsplit(base)
    except ValueError:
        return url.strip()
    query = sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _TRACKING_PARAMS.match(k))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def request_key(request: dict[str, Any]) -> str:
    """规范化请求 → 稳定 key（sha256 前 32 位十六进制，碰撞概率可忽略，文件里也好读）。"""

    return hashlib.sha256(canonical_json(request).encode("utf-8")).hexdigest()[:32]


def strip_volatile(value: Any) -> Any:
    """递归去掉易变字段，给内容指纹用。"""

    if isinstance(value, dict):
        return {k: strip_volatile(v) for k, v in value.items() if not _VOLATILE_KEYS.match(str(k))}
    if isinstance(value, (list, tuple)):
        return [strip_volatile(v) for v in value]
    return value


def fingerprint(payload: Any) -> str:
    return hashlib.sha256(canonical_json(strip_volatile(payload)).encode("utf-8")).hexdigest()[:16]


def search_request_view(request: SearchQueryRequest, *, offset: int, limit: int, image_path: Path | None) -> dict[str, Any]:
    """provider.search 调用 → 规范化请求。图片查询用图片字节哈希代替 E 编号。"""

    query = request.query
    text: str | None
    if isinstance(query, dict):
        text = normalize_text(str(query.get("text", "")))
    elif request.input_type == "image":
        text = None
    else:
        text = normalize_text(str(query))
    image_sha = None
    if image_path is not None and Path(image_path).is_file():
        image_sha = _file_sha256(Path(image_path))
    return {"kind": "search", "input_type": request.input_type, "output_type": request.output_type,
            "query": text, "image_sha256": image_sha, "offset": int(offset), "limit": int(limit)}


def open_request_view(candidate: CandidateSource, request: Any) -> dict[str, Any]:
    """opener.open 调用 → 规范化请求。只取决定"打开什么、要哪些材料"的字段。"""

    view: dict[str, Any] = {
        "kind": "open", "source_url": snapshot_url(candidate.source_url), "candidate_type": candidate.candidate_type,
        "image_url": snapshot_url(candidate.image_url), "thumbnail_url": snapshot_url(candidate.thumbnail_url),
        "platform": (candidate.platform or "").casefold() or None, "platform_video_id": candidate.platform_video_id,
        # Service 在图片候选部分已缓存时注入这个内部字段，让打开器只补缺项；它改变打开结果，所以进 key。
        "cached_materials": sorted(candidate.extra.get("_search_cached_materials", ()) or ()),
        "output_type": getattr(request, "output_type", None),
        "request_kind": "open_source" if isinstance(request, OpenSourceRequest) else "query",
    }
    if isinstance(request, OpenSourceRequest):
        view["include"] = list(request.include)
        view["subtitle_language"] = request.subtitle_language
    return view


def acquire_request_view(source_url: str) -> dict[str, Any]:
    """INSPECT 受控获取视频 → 规范化请求。签名直链只要来自回放的元数据，就与录制时逐字相同。"""

    return {"kind": "acquire", "source_url": snapshot_url(source_url)}


def case_key(case_id: str) -> str:
    """case_id（如 true:4124903）→ 目录名 true_4124903。"""

    return re.sub(r"[^0-9A-Za-z_.-]+", "_", case_id).strip("_") or "case"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------- 快照目录

class ReplayMiss(SearchFailure):
    """回放模式下快照里没有这个请求。reason 固定为 replay_miss，便于日志和统计区分真实的"来源不存在"。"""

    reason = "replay_miss"

    def __init__(self, stage: Literal["search", "open"], request: dict[str, Any]) -> None:
        # NOT_FOUND 不在 Service 的重试集合里：未命中重试也不会变成命中，不要白白耗预算。
        super().__init__(SearchError(stage, "NOT_FOUND",
                                     "No recorded result exists for this request in the fixed retrieval snapshot (replay_miss)"))
        self.request = request


class SnapshotStore:
    """一个案例的快照目录：index.jsonl + blobs/。进程内有内存索引，未命中时增量读入别的进程追加的行。"""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.index_path = self.directory / "index.jsonl"
        self.blob_dir = self.directory / "blobs"
        self.entries: dict[str, dict[str, Any]] = {}
        self._offset = 0
        self.refresh()

    # ---- 索引
    def refresh(self) -> None:
        """从上次读到的位置继续读 index.jsonl（同一案例多个 sample 并行跑时，别的进程可能刚补录过）。"""

        if not self.index_path.is_file():
            return
        with self.index_path.open("rb") as stream:
            stream.seek(self._offset)
            data = stream.read()
        # 只消费完整的行；写到一半的最后一行留到下次。
        end = data.rfind(b"\n") + 1
        for raw in data[:end].splitlines():
            if not raw.strip():
                continue
            try:
                entry = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                _LOGGER.warning("跳过快照索引里损坏的一行", extra={"snapshot_index": str(self.index_path)})
                continue
            if isinstance(entry, dict) and isinstance(entry.get("key"), str):
                self.entries[entry["key"]] = entry           # 同 key 后写覆盖先写
        self._offset += end

    def get(self, key: str) -> dict[str, Any] | None:
        return self.entries.get(key)

    def append(self, entry: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(entry, ensure_ascii=False, separators=(",", ":"), default=_json_default) + "\n").encode("utf-8")
        with self.index_path.open("ab") as stream:
            _lock(stream)
            try:
                stream.write(line)
                stream.flush()
            finally:
                _unlock(stream)
        self.entries[entry["key"]] = entry

    # ---- blob
    def blob_path(self, sha: str) -> Path:
        return self.blob_dir / sha[:2] / sha

    def put_bytes(self, data: bytes) -> str:
        sha = hashlib.sha256(data).hexdigest()
        target = self.blob_path(sha)
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(target, data)
        return sha

    def put_file(self, path: Path) -> str:
        """大文件（视频）流式算哈希再复制，不整份读进内存。"""

        sha = _file_sha256(path)
        target = self.blob_path(sha)
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            handle, temp = tempfile.mkstemp(dir=target.parent, prefix=".tmp-")
            os.close(handle)
            try:
                shutil.copyfile(path, temp)
                os.replace(temp, target)
            finally:
                Path(temp).unlink(missing_ok=True)
        return sha

    def get_bytes(self, sha: str) -> bytes:
        data = self.blob_path(sha).read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            raise OSError(f"snapshot blob {sha} is corrupted")
        return data

    def put_text(self, text: str) -> Any:
        """短文本内联，长文本进 blob。"""

        if len(text) <= _INLINE_LIMIT:
            return text
        return {"blob": self.put_bytes(text.encode("utf-8"))}

    def get_text(self, value: Any) -> str:
        if isinstance(value, dict) and "blob" in value:
            return self.get_bytes(value["blob"]).decode("utf-8")
        return str(value)


def _atomic_write(target: Path, data: bytes) -> None:
    handle, temp = tempfile.mkstemp(dir=target.parent, prefix=".tmp-")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        os.replace(temp, target)
    finally:
        Path(temp).unlink(missing_ok=True)


def _lock(stream: Any) -> None:
    """跨进程追加锁：POSIX 用 flock，Windows 用 msvcrt；两者都没有时退化成无锁（单行追加仍基本安全）。"""

    try:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
    except ImportError:
        try:
            import msvcrt

            # 固定锁第 0 个字节（可以超出文件末尾）：加锁和解锁必须是同一区域；追加模式下写入仍落在文件末尾。
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        except (ImportError, OSError):
            pass


def _unlock(stream: Any) -> None:
    try:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    except ImportError:
        try:
            import msvcrt

            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        except (ImportError, OSError):
            pass


# ---------------------------------------------------------------- 序列化

def _dump_batch(store: SnapshotStore, batch: CandidateBatch) -> dict[str, Any]:
    return {"provider": batch.provider, "engine": batch.engine,
            "candidates": [asdict(candidate) for candidate in batch.candidates],
            "raw_response": store.put_text(canonical_json(batch.raw_response)),
            "request_metadata": batch.request_metadata}


def _load_batch(store: SnapshotStore, payload: dict[str, Any]) -> CandidateBatch:
    candidates = tuple(CandidateSource(**item) for item in payload["candidates"])
    return CandidateBatch(payload["provider"], payload["engine"], candidates,
                          json.loads(store.get_text(payload["raw_response"])), dict(payload.get("request_metadata") or {}))


def _dump_opened(store: SnapshotStore, opened: OpenedSource) -> dict[str, Any]:
    materials = []
    for material in opened.materials:
        materials.append({
            "kind": material.kind, "status": material.status, "content": store.put_text(material.content),
            "artifact_sha256": store.put_bytes(material.artifact_bytes) if material.artifact_bytes else None,
            "media_type": material.media_type, "suffix": material.suffix, "modalities": list(material.modalities),
            "label": material.label, "locator": material.locator, "acquired_by": material.acquired_by,
            "production_method": material.production_method,
            "error": material.error.to_dict() if material.error else None,
        })
    return {"final_url": opened.final_url, "materials": materials, "download_check": opened.download_check,
            "video_material": opened.video_material, "discovered_links": list(opened.discovered_links)}


def _load_opened(store: SnapshotStore, payload: dict[str, Any]) -> OpenedSource:
    materials = []
    for item in payload["materials"]:
        materials.append(OpenedMaterial(
            item["kind"], item["status"], store.get_text(item["content"]),
            store.get_bytes(item["artifact_sha256"]) if item.get("artifact_sha256") else b"",
            item["media_type"], item["suffix"], tuple(item["modalities"]), item["label"], item["locator"],
            item["acquired_by"], item["production_method"],
            SearchError(**item["error"]) if item.get("error") else None))
    return OpenedSource(payload["final_url"], tuple(materials), payload.get("download_check"),
                        payload.get("video_material"), tuple(payload.get("discovered_links") or ()))


def _dump_failure(exc: SearchFailure) -> dict[str, Any]:
    return {"error": exc.error.to_dict(), "retry_after_s": exc.retry_after_s}


def _load_failure(payload: dict[str, Any]) -> SearchFailure:
    return SearchFailure(SearchError(**payload["error"]), retry_after_s=payload.get("retry_after_s"))


# ---------------------------------------------------------------- 录制回放会话

@dataclass
class SearchRecording:
    """一个 run 对一个案例快照的使用：模式、计数、未命中日志。"""

    mode: SearchMode
    snapshot_root: Path
    case_id: str
    run_name: str = ""
    log_path: Path | None = None
    counts: dict[str, int] = field(default_factory=dict)
    store: SnapshotStore | None = None

    def __post_init__(self) -> None:
        if self.mode not in SEARCH_MODES:
            raise ValueError(f"search mode 应为 {' / '.join(SEARCH_MODES)}，得到 {self.mode!r}")
        self.snapshot_root = Path(self.snapshot_root)
        self.counts = {"hit": 0, "miss": 0, "recorded": 0, "recorded_unchanged": 0, "live_calls": 0}
        self.by_kind: dict[str, dict[str, int]] = {}
        if self.mode != "live":
            self.store = SnapshotStore(self.snapshot_root / case_key(self.case_id))

    @property
    def directory(self) -> Path:
        return self.snapshot_root / case_key(self.case_id)

    def _count(self, kind: str, name: str) -> None:
        self.counts[name] = self.counts.get(name, 0) + 1
        bucket = self.by_kind.setdefault(kind, {})
        bucket[name] = bucket.get(name, 0) + 1

    def _log(self, kind: str, event: str, request: dict[str, Any], key: str, **extra: Any) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"kind": kind, "event": event, "key": key, "request": request, **extra},
                                    ensure_ascii=False, default=_json_default) + "\n")

    # ---- 查找与写入
    def lookup(self, kind: str, request: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        """返回 (key, 可回放的条目或 None)。replay_or_record 下瞬时失败不算命中。"""

        assert self.store is not None
        key = request_key(request)
        entry = self.store.get(key)
        if entry is None and self.mode in {"replay", "replay_or_record"}:
            self.store.refresh()
            entry = self.store.get(key)
        if entry is not None and self.mode == "replay_or_record" and entry.get("transient"):
            entry = None
        return key, entry

    def record(self, kind: str, key: str, request: dict[str, Any], outcome: str, payload: dict[str, Any], *,
               transient: bool = False, fingerprint_source: Any = None) -> None:
        """写一条。fingerprint_source 是未进 blob 的结构化内容（原始响应是嵌套字典时才能剥掉里面的时间戳）。"""

        assert self.store is not None
        digest = fingerprint({"outcome": outcome, "payload": payload if fingerprint_source is None else fingerprint_source})
        previous = self.store.get(key)
        if previous is not None and previous.get("fingerprint") == digest and previous.get("outcome") == outcome:
            self._count(kind, "recorded_unchanged")
            return
        self.store.append({"format": SNAPSHOT_FORMAT, "kind": kind, "key": key, "request": request, "outcome": outcome,
                           "transient": transient, "payload": payload, "fingerprint": digest, "run": self.run_name,
                           "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self._count(kind, "recorded")
        if previous is not None and previous.get("fingerprint") != digest:
            self._log(kind, "drift", request, key, previous_fingerprint=previous.get("fingerprint"), fingerprint=digest)

    def miss(self, kind: str, key: str, request: dict[str, Any]) -> None:
        self._count(kind, "miss")
        _LOGGER.warning("检索快照未命中（replay_miss）", extra={"snapshot_kind": kind, "snapshot_key": key,
                                                               "snapshot_case": self.case_id})
        self._log(kind, "replay_miss" if self.mode == "replay" else "miss_recorded_live", request, key)

    def hit(self, kind: str) -> None:
        self._count(kind, "hit")

    # ---- 汇总
    def summary(self) -> dict[str, Any]:
        lookups = self.counts["hit"] + self.counts["miss"]
        return {"search_mode": self.mode, "snapshot_root": str(self.snapshot_root),
                "snapshot_dir": str(self.directory) if self.mode != "live" else None,
                "case_key": case_key(self.case_id), "counts": dict(self.counts),
                "by_kind": {k: dict(v) for k, v in self.by_kind.items()},
                "hit_rate": round(self.counts["hit"] / lookups, 4) if lookups else None}

    def write_usage(self) -> Path | None:
        """把本 run 的计数写到快照目录 usage/<run>.json，供 snapshot_stats 算各 run 命中率。"""

        if self.mode == "live" or not self.run_name:
            return None
        path = self.directory / "usage" / f"{self.run_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**self.summary(), "run": self.run_name,
                                    "written_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path


# ---------------------------------------------------------------- 包装器

class RecordingProvider:
    """包在真实 provider 外面：按模式录制 / 回放 CandidateBatch。"""

    def __init__(self, provider: Any, recording: SearchRecording) -> None:
        self.provider, self.recording = provider, recording

    async def search(self, request: Any, *, offset: int, limit: int, image_path: Path | None = None) -> CandidateBatch:
        rec = self.recording
        view = search_request_view(request, offset=offset, limit=limit, image_path=image_path)
        key, entry = (request_key(view), None) if rec.mode == "record" else rec.lookup("search", view)
        if entry is not None:
            rec.hit("search")
            if entry["outcome"] == "failure":
                raise _load_failure(entry["payload"])
            return _load_batch(rec.store, entry["payload"])  # type: ignore[arg-type]
        if rec.mode == "replay":
            rec.miss("search", key, view)
            raise ReplayMiss("search", view)
        if rec.mode == "replay_or_record":
            rec.miss("search", key, view)
        rec.counts["live_calls"] += 1
        try:
            batch = await self.provider.search(request, offset=offset, limit=limit, image_path=image_path)
        except SearchFailure as exc:
            rec.record("search", key, view, "failure", _dump_failure(exc), transient=exc.error.code in TRANSIENT_CODES)
            raise
        rec.record("search", key, view, "ok", _dump_batch(rec.store, batch),  # type: ignore[arg-type]
                   fingerprint_source={"provider": batch.provider, "engine": batch.engine,
                                       "candidates": [asdict(c) for c in batch.candidates],
                                       "raw_response": batch.raw_response, "request_metadata": batch.request_metadata})
        return batch

    async def aclose(self) -> None:
        close = getattr(self.provider, "aclose", None)
        if callable(close):
            await close()


class RecordingOpener:
    """包在 SourceDispatcher 外面：按模式录制 / 回放 OpenedSource（含全部材料字节）。"""

    def __init__(self, opener: Any, recording: SearchRecording) -> None:
        self.opener, self.recording = opener, recording

    async def open(self, candidate: CandidateSource, request: Any) -> OpenedSource:
        rec = self.recording
        view = open_request_view(candidate, request)
        key, entry = (request_key(view), None) if rec.mode == "record" else rec.lookup("open", view)
        if entry is not None:
            rec.hit("open")
            if entry["outcome"] == "failure":
                raise _load_failure(entry["payload"])
            return _load_opened(rec.store, entry["payload"])  # type: ignore[arg-type]
        if rec.mode == "replay":
            rec.miss("open", key, view)
            raise ReplayMiss("open", view)
        if rec.mode == "replay_or_record":
            rec.miss("open", key, view)
        rec.counts["live_calls"] += 1
        try:
            opened = await self.opener.open(candidate, request)
        except SearchFailure as exc:
            rec.record("open", key, view, "failure", _dump_failure(exc), transient=exc.error.code in TRANSIENT_CODES)
            raise
        rec.record("open", key, view, "ok", _dump_opened(rec.store, opened))  # type: ignore[arg-type]
        return opened


class RecordingAcquirer:
    """包在 INSPECT 的 DirectMediaAcquirer 外面：视频文件也是检索世界的一部分，同样录制 / 回放。"""

    def __init__(self, acquirer: Any, recording: SearchRecording) -> None:
        self.acquirer, self.recording = acquirer, recording

    async def aclose(self) -> None:
        close = getattr(self.acquirer, "aclose", None)
        if callable(close):
            await close()

    async def acquire(self, source_url: str, *, destination: Path, policy: Any) -> Any:
        from ..inspect.acquisition import AcquiredMedia
        from ..inspect.backends.base import BackendFailure

        rec = self.recording
        view = acquire_request_view(source_url)
        key, entry = (request_key(view), None) if rec.mode == "record" else rec.lookup("acquire", view)
        if entry is not None:
            rec.hit("acquire")
            payload = entry["payload"]
            if entry["outcome"] == "failure":
                raise BackendFailure(payload["code"], payload["message"], stage=payload.get("stage", "acquire"),
                                     retryable=bool(payload.get("retryable")))
            path = Path(destination).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            assert rec.store is not None
            shutil.copyfile(rec.store.blob_path(payload["file_sha256"]), path)
            return AcquiredMedia(path, payload["media_type"], payload["suffix"], payload["label"], dict(payload["details"]))
        if rec.mode == "replay":
            rec.miss("acquire", key, view)
            raise BackendFailure("MATERIAL_UNAVAILABLE",
                                 "No recorded media file exists for this URL in the fixed retrieval snapshot (replay_miss)",
                                 stage="acquire", retryable=False)
        if rec.mode == "replay_or_record":
            rec.miss("acquire", key, view)
        rec.counts["live_calls"] += 1
        try:
            acquired = await self.acquirer.acquire(source_url, destination=destination, policy=policy)
        except BackendFailure as exc:
            rec.record("acquire", key, view, "failure", {"code": exc.code, "message": str(exc), "stage": exc.stage,
                                                         "retryable": exc.retryable},
                       transient=exc.code in TRANSIENT_BACKEND_CODES or bool(exc.retryable))
            raise
        assert rec.store is not None
        sha = rec.store.put_file(Path(acquired.path))
        rec.record("acquire", key, view, "ok", {"file_sha256": sha, "media_type": acquired.media_type,
                                                "suffix": acquired.suffix, "label": acquired.label,
                                                "details": dict(acquired.details)})
        return acquired


def apply_search_recording(service: Any, recording: SearchRecording | None) -> Any:
    """把录制回放套在 SearchService 的 provider 与 opener 外面。必须在案卷过滤与护栏之前调用（最里层）。"""

    if recording is None or recording.mode == "live":
        return service
    service.provider = RecordingProvider(service.provider, recording)
    service.opener = RecordingOpener(service.opener, recording)
    return service


__all__ = [
    "DEFAULT_SNAPSHOT_ROOT", "RecordingAcquirer", "RecordingOpener", "RecordingProvider", "ReplayMiss", "SEARCH_MODES",
    "SearchMode", "SearchRecording", "SnapshotStore", "acquire_request_view", "apply_search_recording", "case_key",
    "canonical_json", "fingerprint", "normalize_text", "open_request_view", "request_key", "search_request_view",
    "snapshot_url", "strip_volatile",
]
