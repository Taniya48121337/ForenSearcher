"""单次调查使用的 SQLite 元数据仓库与文件 Artifact 仓库。

SQLite 保存编号、关系、状态和审计事件；大图片、视频、网页原文等字节保存在
``artifact_root``。写媒体时先放入 ``.incoming`` 暂存区，数据库事务成功后才发布，
从而避免出现“数据库说文件存在、磁盘却只有半个文件”的状态。

所有公开编号只是带前缀的数据库主键：A=Artifact、E=Evidence、R=Candidate Source、
F=Finding、W=Workspace。``threading.RLock`` 保护同一进程的复合写操作，SQLite 的
事务和唯一/外键约束保护持久化不变量。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
import threading
import uuid
from pathlib import Path
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Iterable, Literal

from ..contracts import (
    Evidence,
    Finding,
    WorkspaceReport,
    WorkspaceReportDraft,
    WorkspaceTask,
)


# 数据表按领域分组：调查材料/轮次、R/E/Artifact、工具执行审计、最终结果。
# SQL 放在一个常量中，GlobalStore 初始化时用 executescript 幂等创建。
_SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS materials (ref TEXT PRIMARY KEY, payload_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS candidates (
 id INTEGER PRIMARY KEY AUTOINCREMENT, identity_key TEXT NOT NULL UNIQUE,
 source_url TEXT NOT NULL, search_title TEXT, search_snippet TEXT, provider TEXT,
 engine TEXT, provider_rank INTEGER, discovered_execution_id INTEGER,
 raw_response_ref TEXT, final_url TEXT, candidate_type TEXT, image_url TEXT,
 thumbnail_url TEXT, platform TEXT, platform_video_id TEXT,
 extra_json TEXT NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS candidate_aliases (
 identity_key TEXT PRIMARY KEY,
 candidate_id INTEGER NOT NULL REFERENCES candidates(id));
CREATE TABLE IF NOT EXISTS artifacts (
 id INTEGER PRIMARY KEY AUTOINCREMENT, relative_path TEXT NOT NULL UNIQUE,
 media_type TEXT NOT NULL, byte_size INTEGER NOT NULL, role TEXT NOT NULL,
 sha256 TEXT NOT NULL,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS evidence (
 id INTEGER PRIMARY KEY AUTOINCREMENT, modalities_json TEXT NOT NULL,
 label TEXT NOT NULL, content TEXT NOT NULL, source_ref TEXT,
 derived_from_json TEXT NOT NULL, artifact_id INTEGER REFERENCES artifacts(id),
 locator TEXT NOT NULL, acquired_by TEXT NOT NULL, discovered_by TEXT NOT NULL,
 production_method TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS candidate_evidence (
 candidate_id INTEGER NOT NULL REFERENCES candidates(id),
 evidence_id INTEGER NOT NULL REFERENCES evidence(id), material_kind TEXT NOT NULL,
 material_status TEXT NOT NULL, UNIQUE(candidate_id,evidence_id));
CREATE TABLE IF NOT EXISTS findings (finding_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reports (task_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS finding_sequence (
 id INTEGER PRIMARY KEY AUTOINCREMENT);
CREATE TABLE IF NOT EXISTS rounds (
 round_index INTEGER PRIMARY KEY, status TEXT NOT NULL,
 gia_input_json TEXT NOT NULL, gia_decision_json TEXT NOT NULL,
 started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, finished_at TEXT);
CREATE TABLE IF NOT EXISTS workspace_tasks (
 task_id TEXT PRIMARY KEY, round_index INTEGER NOT NULL REFERENCES rounds(round_index),
 task TEXT NOT NULL, materials_json TEXT NOT NULL, run_status TEXT NOT NULL,
 error_json TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 finished_at TEXT);
CREATE TABLE IF NOT EXISTS record_reads (
 id INTEGER PRIMARY KEY AUTOINCREMENT, actor_role TEXT NOT NULL,
 actor_id TEXT NOT NULL, reference_id TEXT NOT NULL,
 round_index INTEGER, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS trajectory_events (
 id INTEGER PRIMARY KEY AUTOINCREMENT, actor_role TEXT NOT NULL,
 actor_id TEXT NOT NULL, event_type TEXT NOT NULL,
 payload_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS final_results (
 id INTEGER PRIMARY KEY CHECK (id=1), decision_json TEXT NOT NULL,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS investigation_meta (
 id INTEGER PRIMARY KEY CHECK (id=1), task TEXT NOT NULL,
 original_material_ids_json TEXT NOT NULL,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS search_executions (
 id INTEGER PRIMARY KEY AUTOINCREMENT, request_json TEXT NOT NULL,
 workspace_id TEXT NOT NULL, started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 finished_at TEXT, stop_reason TEXT, error_json TEXT,
 elapsed_ms INTEGER,
 provider_request_count INTEGER NOT NULL DEFAULT 0,
 candidate_count INTEGER NOT NULL DEFAULT 0,
 successful_source_count INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS provider_calls (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 execution_id INTEGER NOT NULL REFERENCES search_executions(id),
 provider TEXT NOT NULL, engine TEXT NOT NULL, attempt INTEGER NOT NULL,
 elapsed_ms INTEGER NOT NULL, outcome TEXT NOT NULL, raw_response_ref TEXT,
 error_json TEXT, request_metadata_json TEXT NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS search_image_uploads (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 execution_id INTEGER NOT NULL REFERENCES search_executions(id),
 evidence_id INTEGER NOT NULL REFERENCES evidence(id),
 artifact_id INTEGER NOT NULL REFERENCES artifacts(id),
 adaptation_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS candidate_discoveries (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 execution_id INTEGER NOT NULL REFERENCES search_executions(id),
 candidate_id INTEGER NOT NULL REFERENCES candidates(id),
 provider TEXT NOT NULL, engine TEXT NOT NULL, provider_rank INTEGER,
 raw_response_ref TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS source_open_attempts (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 execution_id INTEGER NOT NULL REFERENCES search_executions(id),
 candidate_id INTEGER NOT NULL REFERENCES candidates(id),
 attempt INTEGER NOT NULL, elapsed_ms INTEGER NOT NULL,
 outcome TEXT NOT NULL, error_json TEXT,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS inspect_executions (
 id INTEGER PRIMARY KEY AUTOINCREMENT, request_json TEXT NOT NULL,
 workspace_id TEXT NOT NULL, started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 finished_at TEXT, status TEXT, result_evidence_id INTEGER REFERENCES evidence(id),
 feedback_json TEXT, details_json TEXT NOT NULL DEFAULT '{}');
"""

# 索引只服务真实查询路径，不改变任何业务语义。
_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_search_executions_workspace_started
 ON search_executions(workspace_id,started_at);
CREATE INDEX IF NOT EXISTS idx_provider_calls_execution
 ON provider_calls(execution_id);
CREATE INDEX IF NOT EXISTS idx_candidate_discoveries_execution
 ON candidate_discoveries(execution_id,candidate_id);
CREATE INDEX IF NOT EXISTS idx_candidate_evidence_candidate_kind
 ON candidate_evidence(candidate_id,material_kind);
CREATE INDEX IF NOT EXISTS idx_evidence_source_ref ON evidence(source_ref);
CREATE INDEX IF NOT EXISTS idx_source_open_attempts_execution
 ON source_open_attempts(execution_id,candidate_id);
CREATE INDEX IF NOT EXISTS idx_workspace_tasks_round
 ON workspace_tasks(round_index,task_id);
CREATE INDEX IF NOT EXISTS idx_record_reads_actor
 ON record_reads(actor_role,actor_id,id);
CREATE INDEX IF NOT EXISTS idx_trajectory_events_actor
 ON trajectory_events(actor_role,actor_id,id);
"""

# user_version 用来拒绝“数据库比当前代码更新”的危险降级读取。
_SCHEMA_VERSION = 3
# 这些列是 SEARCH 正常运行的最低要求，用于识别不能安全迁移的旧库。
_SEARCH_REQUIRED_COLUMNS = {
    "candidates": {"id", "identity_key", "source_url", "extra_json"},
    "candidate_aliases": {"identity_key", "candidate_id"},
    "artifacts": {"id", "relative_path", "media_type", "byte_size", "role", "sha256"},
    "evidence": {"id", "modalities_json", "content", "source_ref", "artifact_id"},
    "candidate_evidence": {"candidate_id", "evidence_id", "material_kind", "material_status"},
    "search_executions": {"id", "request_json", "workspace_id", "finished_at", "stop_reason", "elapsed_ms"},
    "provider_calls": {"execution_id", "attempt", "elapsed_ms", "outcome"},
    "candidate_discoveries": {"execution_id", "candidate_id", "provider", "engine"},
    "source_open_attempts": {"execution_id", "candidate_id", "attempt", "elapsed_ms", "outcome"},
}


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    """一个已发布 Artifact 的编号、受控路径、类型、大小、用途和摘要。"""

    artifact_ref: str
    path: Path
    media_type: str
    byte_size: int
    role: str
    sha256: str


@dataclass(frozen=True, slots=True)
class StoreHealth:
    """Store 健康检查结果；任一数据库或文件问题都会把 status 置为 degraded。"""

    status: Literal["ok", "degraded"]
    database_checks: tuple[str, ...]
    foreign_key_violations: tuple[dict[str, Any], ...]
    missing_artifacts: tuple[str, ...]
    size_mismatches: tuple[str, ...]
    content_mismatches: tuple[str, ...]
    orphan_files: tuple[str, ...]
    staged_files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """转换成便于日志、监控和 API 返回的 JSON 兼容对象。"""

        return {
            "status": self.status,
            "database_checks": list(self.database_checks),
            "foreign_key_violations": [dict(item) for item in self.foreign_key_violations],
            "missing_artifacts": list(self.missing_artifacts),
            "size_mismatches": list(self.size_mismatches),
            "content_mismatches": list(self.content_mismatches),
            "orphan_files": list(self.orphan_files),
            "staged_files": list(self.staged_files),
        }


class ArtifactIntegrityError(RuntimeError):
    """Artifact 实际大小或 SHA-256 与数据库记录不一致。"""

    pass


class _ClosingConnection(sqlite3.Connection):
    """离开 ``with`` 后既提交/回滚事务，也一定关闭 SQLite 连接。"""

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None) -> Literal[False]:
        """复用 sqlite3 事务语义，并在 finally 中释放文件句柄。"""

        try:
            super().__exit__(exc_type, exc, traceback)
        finally:
            self.close()
        return False


class GlobalStore:
    """提供按材料原子发布 Artifact、Evidence 和 R→E 关系的持久化门面。"""

    def __init__(
        self,
        db_path: str | Path | None = None,
        artifact_root: str | Path | None = None,
        *,
        busy_timeout_s: float = 10.0,
        verify_artifacts_on_read: bool = True,
    ) -> None:
        """创建/打开一个 Store，执行兼容迁移、schema 校验和索引初始化。"""

        # SQLite 锁等待时间必须有限且为正；布尔值不能误当 0/1 秒。
        if (
            isinstance(busy_timeout_s, bool)
            or not isinstance(busy_timeout_s, (int, float))
            or not math.isfinite(busy_timeout_s)
            or busy_timeout_s <= 0
        ):
            raise ValueError("busy_timeout_s must be positive")
        if not isinstance(verify_artifacts_on_read, bool):
            raise ValueError("verify_artifacts_on_read must be a boolean")
        self.busy_timeout_s = float(busy_timeout_s)
        self.verify_artifacts_on_read = verify_artifacts_on_read
        # 两个路径都省略时创建测试友好的临时 Store，并由实例持有其生命周期。
        self._owned_root: tempfile.TemporaryDirectory[str] | None = None
        if db_path is None and artifact_root is None:
            self._owned_root = tempfile.TemporaryDirectory(prefix="foren-search-")
            root = Path(self._owned_root.name)
            db_path, artifact_root = root / "store.sqlite3", root / "artifacts"
        elif db_path is None:
            # 只给 Artifact 根时，数据库放在它的同级目录。
            assert artifact_root is not None
            db_path = Path(artifact_root).resolve().parent / "store.sqlite3"
        elif artifact_root is None:
            # 只给数据库时，Artifact 使用同级 artifacts 子目录。
            artifact_root = Path(db_path).resolve().parent / "artifacts"
        assert db_path is not None and artifact_root is not None
        # 立即规范化绝对路径并创建目录，后续安全检查使用稳定根路径。
        self.db_path = Path(db_path).resolve()
        self.artifact_root = Path(artifact_root).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        # .incoming 是 Store 唯一认可的“自己拥有、可以移动/清理”的暂存目录。
        self.incoming_root = (self.artifact_root / ".incoming").resolve()
        self.incoming_root.mkdir(parents=True, exist_ok=True)
        # RLock 允许一个复合方法安全调用另一个也需要锁的内部方法。
        self._lock = threading.RLock()
        self._materials: dict[str, Any] = {}
        with self._connect() as connection:
            # 先拒绝未来 schema；随后创建缺失表、补兼容列和内容摘要。
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > _SCHEMA_VERSION:
                raise RuntimeError(
                    f"store schema version {version} is newer than supported version {_SCHEMA_VERSION}"
                )
            connection.executescript(_SCHEMA)
            self._ensure_column(connection, "search_executions", "elapsed_ms", "INTEGER")
            self._ensure_column(connection, "artifacts", "sha256", "TEXT")
            self._backfill_artifact_hashes(connection)
            self._validate_search_schema(connection)
            connection.executescript(_INDEXES)
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection, table: str, column: str, declaration: str,
    ) -> None:
        """为支持的旧 schema 补一个缺失列；已存在时保持不变。"""

        columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    @staticmethod
    def _validate_search_schema(connection: sqlite3.Connection) -> None:
        """检查 SEARCH 必需表列，拒绝无法可靠解释的遗留数据库。"""

        for table, expected in _SEARCH_REQUIRED_COLUMNS.items():
            actual = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
            missing = sorted(expected - actual)
            if missing:
                raise RuntimeError(
                    f"unsupported legacy store schema: {table} is missing {', '.join(missing)}"
                )

    @staticmethod
    def _file_sha256(path: Path) -> str:
        """以 1 MiB 分块计算文件 SHA-256，避免把大媒体整体读入内存。"""

        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _backfill_artifact_hashes(self, connection: sqlite3.Connection) -> None:
        """升级旧 Store 时，为仍存在且位于根目录内的 Artifact 补摘要。"""

        rows = connection.execute(
            "SELECT id,relative_path FROM artifacts WHERE sha256 IS NULL OR sha256=''"
        ).fetchall()
        for row in rows:
            path = (self.artifact_root / str(row["relative_path"])).resolve()
            try:
                # relative_to 阻止旧数据库中的异常相对路径逃出 artifact_root。
                path.relative_to(self.artifact_root)
            except ValueError:
                continue
            if path.is_file():
                try:
                    sha256 = self._file_sha256(path)
                except OSError:
                    continue
                connection.execute(
                    "UPDATE artifacts SET sha256=? WHERE id=?",
                    (sha256, int(row["id"])),
                )

    def _connect(self) -> sqlite3.Connection:
        """创建一次短生命周期 SQLite 连接并启用行对象、外键和锁等待。"""

        connection = sqlite3.connect(
            self.db_path,
            timeout=self.busy_timeout_s,
            factory=_ClosingConnection,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout_s * 1000)}")
        return connection

    def new_owned_temp_path(self, suffix: str = ".tmp") -> Path:
        """在 .incoming 中生成一个尚未创建的随机暂存路径。"""

        suffix = suffix if suffix.startswith(".") else f".{suffix}"
        path = (self.incoming_root / f"{uuid.uuid4().hex}{suffix[:12]}").resolve()
        path.relative_to(self.incoming_root)
        return path

    def _validated_owned_temp_path(self, value: str | Path) -> Path:
        """确认调用方交回的暂存文件确实属于本 Store 且是普通文件。"""

        path = Path(value).resolve(strict=True)
        try:
            path.relative_to(self.incoming_root)
        except ValueError as exc:
            raise ValueError("owned temporary path is outside the Store incoming directory") from exc
        if not path.is_file():
            raise ValueError("owned temporary path is not a regular file")
        return path

    @staticmethod
    def _json(value: Any) -> str:
        """统一生成紧凑标准 JSON；明确拒绝 NaN 和 Infinity。"""

        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )

    @staticmethod
    def _lastrowid(cursor: sqlite3.Cursor) -> int:
        """读取 SQLite 自增主键；缺失时抛出明确内部错误。"""

        if cursor.lastrowid is None:
            raise RuntimeError("database insert did not return a row ID")
        return int(cursor.lastrowid)

    @staticmethod
    def _require_text(value: Any, field_name: str) -> str:
        """校验并清理数据库主键、任务等必要文本字段。"""

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
        return value.strip()

    def add_material(self, material_id: str, value: Any) -> None:
        """新增一条 C 等普通材料，同时写数据库和进程内只读缓存。"""

        material_id = self._require_text(material_id, "material_id")
        if self.has_reference(material_id):
            raise ValueError(f"duplicate global reference ID for material: {material_id}")
        with self._lock, self._connect() as connection:
            connection.execute("INSERT INTO materials(ref,payload_json) VALUES (?,?)", (material_id, self._json(value)))
        self._materials[material_id] = copy.deepcopy(value)

    def create_round(
        self,
        round_index: int,
        *,
        gia_input: dict[str, Any],
        gia_decision: dict[str, Any],
    ) -> None:
        """只创建 running 轮次；保留给兼容路径，通常优先使用 start_round。"""

        if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 1:
            raise ValueError("round_index must be a positive integer")
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO rounds(
                round_index,status,gia_input_json,gia_decision_json
                ) VALUES (?,?,?,?)""",
                (round_index, "running", self._json(gia_input), self._json(gia_decision)),
            )

    def start_round(
        self,
        round_index: int,
        *,
        gia_input: dict[str, Any],
        gia_decision: dict[str, Any],
        tasks: Iterable[WorkspaceTask],
    ) -> None:
        """在同一事务中发布 running 轮次和该轮全部 WorkspaceTask。"""

        if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 1:
            raise ValueError("round_index must be a positive integer")
        # 先物化和验证全部任务，再开始事务，避免写出半轮数据。
        values = tuple(tasks)
        if not values:
            raise ValueError("a running round requires at least one Workspace Task")
        task_ids = [self._require_text(task.task_id, "task_id") for task in values]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Workspace Task IDs must be unique within a round")
        rows = tuple(
            (
                task_id,
                round_index,
                self._require_text(task.task, "task"),
                self._json(list(task.materials)),
                "pending",
            )
            for task_id, task in zip(task_ids, values, strict=True)
        )
        with self._lock, self._connect() as connection:
            # BEGIN IMMEDIATE 提前取得写锁；round 与 tasks 要么全部成功、要么全部回滚。
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT INTO rounds(
                round_index,status,gia_input_json,gia_decision_json
                ) VALUES (?,?,?,?)""",
                (round_index, "running", self._json(gia_input), self._json(gia_decision)),
            )
            connection.executemany(
                """INSERT INTO workspace_tasks(
                task_id,round_index,task,materials_json,run_status
                ) VALUES (?,?,?,?,?)""",
                rows,
            )
            connection.commit()

    def ensure_investigation(
        self, *, task: str, original_material_ids: Iterable[str]
    ) -> None:
        """初始化调查元数据；重复调用时只接受完全相同的任务。"""

        task = self._require_text(task, "task")
        originals = tuple(original_material_ids)
        if any(not isinstance(value, str) or not value for value in originals):
            raise ValueError("original_material_ids must contain non-empty strings")
        payload = self._json(list(originals))
        with self._lock, self._connect() as connection:
            # 固定使用 id=1，因为一次 Store 只承载一次调查。
            row = connection.execute(
                "SELECT task,original_material_ids_json FROM investigation_meta WHERE id=1"
            ).fetchone()
            if row is None:
                # 首次启动负责写入；之后恢复运行只能核对，不能悄悄改题。
                connection.execute(
                    """INSERT INTO investigation_meta(
                    id,task,original_material_ids_json
                    ) VALUES (1,?,?)""",
                    (task, payload),
                )
                return
            # 这项检查能防止把旧数据库误用于另一个问题。
            if row["task"] != task or row["original_material_ids_json"] != payload:
                raise ValueError(
                    "task and original materials do not match the stored investigation"
                )

    def investigation_meta(self) -> dict[str, Any]:
        """读取调查题目与原始材料 ID，并把 JSON 字符串还原为列表。"""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM investigation_meta WHERE id=1"
            ).fetchone()
        if row is None:
            raise KeyError("investigation metadata has not been initialized")
        value = dict(row)
        value["original_material_ids"] = json.loads(
            value.pop("original_material_ids_json")
        )
        return value

    def has_investigation(self) -> bool:
        """判断 Store 是否已经绑定到某次调查。"""

        with self._connect() as connection:
            return connection.execute(
                "SELECT 1 FROM investigation_meta WHERE id=1"
            ).fetchone() is not None

    def has_material(self, material_id: str) -> bool:
        """安全检查一个普通材料 ID 是否存在；非法输入直接返回 False。"""

        if not isinstance(material_id, str) or not material_id:
            return False
        with self._connect() as connection:
            return connection.execute(
                "SELECT 1 FROM materials WHERE ref=?", (material_id,)
            ).fetchone() is not None

    def finish_round(self, round_index: int, *, status: str = "completed") -> None:
        """把运行中的轮次更新为终态，并记录完成时间。"""

        if status not in {"completed", "completed_with_errors", "cancelled"}:
            raise ValueError(f"unknown round status: {status}")
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE rounds SET status=?,finished_at=CURRENT_TIMESTAMP WHERE round_index=?",
                (status, round_index),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown round: {round_index}")

    def round_record(self, round_index: int) -> dict[str, Any]:
        """读取一轮记录，并反序列化当时的 GIA 输入与决策。"""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM rounds WHERE round_index=?", (round_index,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown round: {round_index}")
        value = dict(row)
        value["gia_input"] = json.loads(value.pop("gia_input_json"))
        value["gia_decision"] = json.loads(value.pop("gia_decision_json"))
        return value

    def add_workspace_tasks(
        self, round_index: int, tasks: Iterable[WorkspaceTask]
    ) -> None:
        """向已有轮次批量补充 Workspace Task。"""

        # 生成器只能遍历一次，所以先固化成元组。
        values = tuple(tasks)
        with self._lock, self._connect() as connection:
            # 外键也能阻止错误写入，但这里先给调用者更清楚的异常。
            if connection.execute(
                "SELECT 1 FROM rounds WHERE round_index=?", (round_index,)
            ).fetchone() is None:
                raise KeyError(f"unknown round: {round_index}")
            connection.execute("BEGIN IMMEDIATE")
            for task in values:
                # 每条任务的 materials 以 JSON 保存，读取时再恢复成列表。
                connection.execute(
                    """INSERT INTO workspace_tasks(
                    task_id,round_index,task,materials_json,run_status
                    ) VALUES (?,?,?,?,?)""",
                    (
                        self._require_text(task.task_id, "task_id"),
                        round_index,
                        self._require_text(task.task, "task"),
                        self._json(list(task.materials)),
                        "pending",
                    ),
                )
            connection.commit()

    def set_workspace_run_status(
        self,
        task_id: str,
        status: str,
        *,
        error: dict[str, Any] | None = None,
    ) -> None:
        """更新 Workspace 的运行状态，并在失败时保留结构化错误。"""

        allowed = {"pending", "running", "reported", "runtime_error", "cancelled", "timed_out"}
        if status not in allowed:
            raise ValueError(f"unknown workspace run status: {status}")
        # 只有终态才写 finished_at；从 pending 切到 running 时仍为空。
        terminal = status in {"reported", "runtime_error", "cancelled", "timed_out"}
        sql = """UPDATE workspace_tasks SET run_status=?,error_json=?,
        finished_at=CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE NULL END
        WHERE task_id=?"""
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                sql,
                (status, self._json(error) if error else None, terminal, task_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown Workspace Task: {task_id}")

    def workspace_task_record(self, task_id: str) -> dict[str, Any]:
        """读取一条 Workspace Task，并还原 materials 与 error 字段。"""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM workspace_tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown Workspace Task: {task_id}")
        value = dict(row)
        value["materials"] = json.loads(value.pop("materials_json"))
        error_json = value.pop("error_json")
        value["error"] = json.loads(error_json) if error_json else None
        return value

    def workspace_task_records(self, round_index: int | None = None) -> list[dict[str, Any]]:
        """按插入顺序读取 Workspace Task，可按轮次筛选。"""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT task_id FROM workspace_tasks"
                + (" WHERE round_index=?" if round_index is not None else "") + " ORDER BY rowid",
                (round_index,) if round_index is not None else (),
            ).fetchall()
        return [self.workspace_task_record(row["task_id"]) for row in rows]

    def round_records(self) -> list[dict[str, Any]]:
        """按轮次编号读取完整轮次历史。"""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT round_index FROM rounds ORDER BY round_index"
            ).fetchall()
        return [self.round_record(int(row["round_index"])) for row in rows]

    def record_read(
        self,
        *,
        actor_role: str,
        actor_id: str,
        reference_id: str,
        round_index: int | None = None,
    ) -> None:
        """审计某个角色读取了哪个全局引用。"""

        actor_role = self._require_text(actor_role, "actor_role")
        actor_id = self._require_text(actor_id, "actor_id")
        reference_id = self._require_text(reference_id, "reference_id")
        # 不允许为不存在的引用制造虚假审计记录。
        if not self.has_reference(reference_id):
            raise KeyError(f"unknown Global Store reference: {reference_id}")
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO record_reads(
                actor_role,actor_id,reference_id,round_index
                ) VALUES (?,?,?,?)""",
                (actor_role, actor_id, reference_id, round_index),
            )

    def record_reads(
        self,
        *,
        actor_role: str | None = None,
        actor_id: str | None = None,
        round_index: int | None = None,
    ) -> list[dict[str, Any]]:
        """按可选角色、主体或轮次筛选读取审计记录。"""

        # 只拼接固定列名；用户值始终通过 SQLite 参数传递。
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("actor_role", actor_role),
            ("actor_id", actor_id),
            ("round_index", round_index),
        ):
            if value is not None:
                clauses.append(f"{column}=?")
                params.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM record_reads{where} ORDER BY id", params
            ).fetchall()
        return [dict(row) for row in rows]

    def record_event(
        self,
        *,
        actor_role: str,
        actor_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> int:
        """把一次模型动作或系统事件追加到不可变轨迹中。"""

        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO trajectory_events(
                actor_role,actor_id,event_type,payload_json
                ) VALUES (?,?,?,?)""",
                (
                    self._require_text(actor_role, "actor_role"),
                    self._require_text(actor_id, "actor_id"),
                    self._require_text(event_type, "event_type"),
                    self._json(payload),
                ),
            )
            return self._lastrowid(cursor)

    def trajectory_events(
        self, *, actor_role: str | None = None, actor_id: str | None = None,
        event_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """读取并反序列化轨迹事件，可按角色或具体主体过滤。"""

        clauses: list[str] = []
        params: list[Any] = []
        if actor_role is not None:
            clauses.append("actor_role=?")
            params.append(actor_role)
        if actor_id is not None:
            clauses.append("actor_id=?")
            params.append(actor_id)
        if event_type is not None:
            clauses.append("event_type=?")
            params.append(event_type)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM trajectory_events{where} ORDER BY id", params
            ).fetchall()
        values: list[dict[str, Any]] = []
        for row in rows:
            # 数据库中的 payload_json 对外应表现为真正的 Python 字典。
            value = dict(row)
            value["payload"] = json.loads(value.pop("payload_json"))
            values.append(value)
        return values

    def export_trajectory(self, path: str | Path) -> Path:
        """以 JSON Lines 原子导出完整轨迹，便于审计或离线分析。"""

        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        # 先写同目录临时文件，再原子替换，避免崩溃留下半份导出文件。
        staged = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            with staged.open("x", encoding="utf-8", newline="\n") as handle:
                for event in self.trajectory_events():
                    # JSONL 每一行都是一个独立 JSON 对象，方便流式读取。
                    handle.write(self._json(event))
                    handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(staged, target)
        except Exception:
            staged.unlink(missing_ok=True)
            raise
        return target

    def save_final_result(self, decision: dict[str, Any]) -> None:
        """保存 GIA 的唯一最终决策；主键 id=1 阻止重复终结。"""

        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO final_results(id,decision_json) VALUES (1,?)",
                (self._json(decision),),
            )

    def final_result(self) -> dict[str, Any]:
        """读取最终决策；尚未完成调查时抛出 KeyError。"""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT decision_json FROM final_results WHERE id=1"
            ).fetchone()
        if row is None:
            raise KeyError("investigation has no final result")
        return json.loads(row["decision_json"])

    def has_final_result(self) -> bool:
        """判断调查是否已经产生最终答案。"""

        with self._connect() as connection:
            return connection.execute(
                "SELECT 1 FROM final_results WHERE id=1"
            ).fetchone() is not None

    def healthcheck(self) -> StoreHealth:
        """只读检查数据库完整性以及每个 Artifact 的文件和摘要。"""

        with self._lock, self._connect() as connection:
            # quick_check 检查 SQLite 页结构，foreign_key_check 检查引用关系。
            database_checks = tuple(
                str(row[0]) for row in connection.execute("PRAGMA quick_check")
            )
            foreign_key_violations = tuple(
                dict(row) for row in connection.execute("PRAGMA foreign_key_check")
            )
            artifacts = connection.execute(
                "SELECT id,relative_path,byte_size,sha256 FROM artifacts ORDER BY id"
            ).fetchall()

        missing: list[str] = []
        mismatched: list[str] = []
        content_mismatched: list[str] = []
        registered_paths: set[Path] = set()
        for row in artifacts:
            artifact_ref = f"A{int(row['id'])}"
            path = (self.artifact_root / str(row["relative_path"])).resolve()
            try:
                # 数据库中的相对路径也必须仍然落在 artifact_root 内。
                path.relative_to(self.artifact_root)
            except ValueError:
                missing.append(artifact_ref)
                continue
            registered_paths.add(path)
            if not path.is_file():
                missing.append(artifact_ref)
            else:
                try:
                    # 先比廉价的字节数；只有大小一致才计算 SHA-256。
                    if path.stat().st_size != int(row["byte_size"]):
                        mismatched.append(artifact_ref)
                    elif not row["sha256"] or self._file_sha256(path) != row["sha256"]:
                        content_mismatched.append(artifact_ref)
                except OSError:
                    content_mismatched.append(artifact_ref)

        # SQLite 数据库及 WAL/SHM 文件位于同一目录，但不属于 Artifact。
        database_files = {
            self.db_path,
            Path(f"{self.db_path}-wal"),
            Path(f"{self.db_path}-shm"),
        }
        orphan_files: list[str] = []
        for path in self.artifact_root.rglob("*"):
            # .incoming 是受控暂存区，稍后单独报告而不是算作孤儿文件。
            if not path.is_file() or self.incoming_root in path.parents:
                continue
            resolved = path.resolve()
            if resolved not in registered_paths and resolved not in database_files:
                orphan_files.append(path.relative_to(self.artifact_root).as_posix())
        staged_files = sorted(
            path.relative_to(self.incoming_root).as_posix()
            for path in self.incoming_root.rglob("*")
            if path.is_file()
        )
        # 所有子检查都通过，Store 才能报告为健康。
        healthy = (
            database_checks == ("ok",)
            and not foreign_key_violations
            and not missing
            and not mismatched
            and not content_mismatched
            and not orphan_files
            and not staged_files
        )
        return StoreHealth(
            status="ok" if healthy else "degraded",
            database_checks=database_checks,
            foreign_key_violations=foreign_key_violations,
            missing_artifacts=tuple(missing),
            size_mismatches=tuple(mismatched),
            content_mismatches=tuple(content_mismatched),
            orphan_files=tuple(sorted(orphan_files)),
            staged_files=tuple(staged_files),
        )

    def record_evidence(
        self, *, content: str | None = None, modalities: tuple[str, ...] = ("text",),
        label: str | None = None, source_ref: str | None = None, locator: str,
        acquired_by: str, discovered_by: str,
        derived_from: tuple[str, ...] = (), artifact_ref: str | None = None,
        production_method: str = "tool_output",
    ) -> Evidence:
        """记录纯结构化 Evidence；这个入口不创建 Artifact 文件。"""

        # 必填字符串先统一校验，避免数据库里出现只有空白的“有效”记录。
        content = self._require_text(content, "content")
        acquired_by = self._require_text(acquired_by, "acquired_by")
        locator = self._require_text(locator, "locator")
        discovered_by = self._require_text(discovered_by, "discovered_by")
        self._validate_modalities(modalities)
        # 派生证据必须引用已经存在的上游 Evidence。
        for evidence_id in derived_from:
            self.evidence(evidence_id)
        # Artifact 引用在数据库中存整数外键，对外仍使用 A<n>。
        artifact_id = self._artifact_pk(artifact_ref) if artifact_ref else None
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO evidence(modalities_json,label,content,source_ref,
                derived_from_json,artifact_id,locator,acquired_by,discovered_by,
                production_method) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (self._json(list(modalities)), label or content[:120], content, source_ref,
                 self._json(list(derived_from)), artifact_id, locator, acquired_by,
                 discovered_by, production_method),
            )
            evidence_pk = self._lastrowid(cursor)
            evidence_id = f"E{evidence_pk}"
            # 虽然不同类型分表保存，对外引用仍共享同一个命名空间。
            if (
                connection.execute(
                    "SELECT 1 FROM materials WHERE ref=?", (evidence_id,)
                ).fetchone()
                or connection.execute(
                    "SELECT 1 FROM findings WHERE finding_id=?", (evidence_id,)
                ).fetchone()
                or connection.execute(
                    "SELECT 1 FROM reports WHERE task_id=?", (evidence_id,)
                ).fetchone()
            ):
                raise ValueError(f"duplicate global reference ID for Evidence: {evidence_id}")
            return self._evidence_from_pk(connection, evidence_pk)

    def commit_material_evidence(
        self, *, candidate_id: str, material_kind: str, material_status: str,
        artifact_bytes: bytes, media_type: str, suffix: str,
        modalities: tuple[str, ...], label: str, content: str, locator: str,
        acquired_by: str, discovered_by: str,
        production_method: str = "source_excerpt", role: str = "source_material",
    ) -> Evidence:
        """先可靠落盘，再原子提交 Artifact、Evidence 与 R→E 关系。"""

        # Candidate 必须存在；先解析 ID 再读一次可给出一致的错误信息。
        candidate_pk = self._candidate_pk(candidate_id)
        self.candidate(candidate_id)
        self._validate_modalities(modalities)
        material_kind = self._require_text(material_kind, "material_kind")
        material_status = self._require_text(material_status, "material_status")
        media_type = self._require_text(media_type, "media_type")
        label = self._require_text(label, "label")
        content = self._require_text(content, "content")
        locator = self._require_text(locator, "locator")
        acquired_by = self._require_text(acquired_by, "acquired_by")
        discovered_by = self._require_text(discovered_by, "discovered_by")
        production_method = self._require_text(production_method, "production_method")
        role = self._require_text(role, "role")
        if not artifact_bytes:
            raise ValueError("artifact_bytes must not be empty")
        # 摘要与内容一起生成，后续读取和 healthcheck 都可检测静默损坏。
        sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        # 后缀只用于便于识别文件类型，随机 UUID 才负责避免重名。
        suffix = suffix if suffix.startswith(".") else f".{suffix}"
        relative_path = f"{uuid.uuid4().hex}{suffix[:12]}"
        final_path = (self.artifact_root / relative_path).resolve()
        temp_path = self.artifact_root / f".{uuid.uuid4().hex}.tmp"
        if self.artifact_root not in final_path.parents:
            raise ValueError("artifact path escaped store root")
        try:
            # 文件系统没有和 SQLite 共用的事务，因此先写临时文件并 fsync。
            with temp_path.open("xb") as handle:
                handle.write(artifact_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, final_path)
            # 尽早确认替换后的文件可读，再开始数据库事务。
            with final_path.open("rb") as handle:
                handle.read(1)
            with self._lock, self._connect() as connection:
                # 以下三张表必须一起成功，不能留下“有 E 无关系”的中间态。
                connection.execute("BEGIN IMMEDIATE")
                artifact_pk = self._lastrowid(connection.execute(
                    """INSERT INTO artifacts(
                    relative_path,media_type,byte_size,role,sha256
                    ) VALUES (?,?,?,?,?)""",
                    (relative_path, media_type, len(artifact_bytes), role, sha256),
                ))
                evidence_pk = self._lastrowid(connection.execute(
                    """INSERT INTO evidence(modalities_json,label,content,source_ref,
                    derived_from_json,artifact_id,locator,acquired_by,discovered_by,
                    production_method) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (self._json(list(modalities)), label, content, candidate_id, "[]",
                     artifact_pk, locator, acquired_by, discovered_by, production_method),
                ))
                evidence_id = f"E{evidence_pk}"
                if (
                    connection.execute(
                        "SELECT 1 FROM materials WHERE ref=?", (evidence_id,)
                    ).fetchone()
                    or connection.execute(
                        "SELECT 1 FROM findings WHERE finding_id=?", (evidence_id,)
                    ).fetchone()
                    or connection.execute(
                        "SELECT 1 FROM reports WHERE task_id=?", (evidence_id,)
                    ).fetchone()
                ):
                    raise ValueError(f"duplicate global reference ID for Evidence: {evidence_id}")
                connection.execute(
                    "INSERT INTO candidate_evidence(candidate_id,evidence_id,material_kind,material_status) VALUES (?,?,?,?)",
                    (candidate_pk, evidence_pk, material_kind, material_status),
                )
                connection.commit()
                return self._evidence_from_pk(connection, evidence_pk)
        except Exception:
            # 数据库回滚由连接上下文负责；这里补偿删除尚未被数据库接纳的文件。
            temp_path.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)
            raise

    def artifact_path(self, artifact_ref: str) -> Path:
        """解析并验证 Artifact 路径，必要时同时校验大小和 SHA-256。"""

        artifact_pk = self._artifact_pk(artifact_ref)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT relative_path,byte_size,sha256 FROM artifacts WHERE id=?",
                (artifact_pk,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown Artifact reference: {artifact_ref}")
        path = (self.artifact_root / row["relative_path"]).resolve()
        # 同时检查目录逃逸和文件存在性，不把不安全路径交给调用方。
        if self.artifact_root not in path.parents or not path.is_file():
            raise KeyError(f"Artifact is unavailable: {artifact_ref}")
        if not row["sha256"]:
            raise ArtifactIntegrityError(
                f"Artifact has no committed content digest: {artifact_ref}"
            )
        if self.verify_artifacts_on_read:
            try:
                # 每次读取都验证最安全；超大文件环境可通过配置关闭以换取性能。
                matches = (
                    path.stat().st_size == int(row["byte_size"])
                    and self._file_sha256(path) == row["sha256"]
                )
            except OSError:
                matches = False
            if not matches:
                raise ArtifactIntegrityError(
                    f"Artifact content does not match committed metadata: {artifact_ref}"
                )
        return path

    def artifact(self, artifact_ref: str) -> ArtifactMetadata:
        """返回 Artifact 的可信元数据，其中 path 已经过完整性校验。"""

        artifact_pk = self._artifact_pk(artifact_ref)
        with self._connect() as connection:
            row = connection.execute(
                """SELECT relative_path,media_type,byte_size,role,sha256
                FROM artifacts WHERE id=?""",
                (artifact_pk,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown Artifact reference: {artifact_ref}")
        path = self.artifact_path(artifact_ref)
        return ArtifactMetadata(
            artifact_ref,
            path,
            row["media_type"],
            int(row["byte_size"]),
            row["role"],
            row["sha256"],
        )

    def _commit_evidence_payload(
        self, *, payload: bytes | Path | None, take_ownership: bool, media_type: str | None,
        suffix: str, role: str, modalities: tuple[str, ...], label: str, content: str,
        source_ref: str | None, derived_from: tuple[str, ...], locator: str,
        acquired_by: str, discovered_by: str, production_method: str,
        candidate_id: str | None = None, material_kind: str | None = None,
        material_status: str = "ready",
    ) -> Evidence:
        """统一提交可选文件载荷，是各种 Evidence 提交入口的公共底层。"""

        # 先验证所有元数据，尽量在创建文件之前失败。
        self._validate_modalities(modalities)
        label = self._require_text(label, "label")
        content = self._require_text(content, "content")
        locator = self._require_text(locator, "locator")
        acquired_by = self._require_text(acquired_by, "acquired_by")
        discovered_by = self._require_text(discovered_by, "discovered_by")
        production_method = self._require_text(production_method, "production_method")
        role = self._require_text(role, "role")
        for evidence_id in derived_from:
            self.evidence(evidence_id)
        # 如果这是某个搜索候选的材料，需要同时准备 R→E 关联外键。
        candidate_pk = self._candidate_pk(candidate_id) if candidate_id else None
        if candidate_id:
            self.candidate(candidate_id)
        if payload is None:
            # 没有载荷时只能创建纯文本/结构化 Evidence，不能虚构 media_type。
            if media_type is not None:
                raise ValueError("media_type requires an Artifact payload")
            return self.record_evidence(
                content=content, modalities=modalities, label=label, source_ref=source_ref,
                derived_from=derived_from, locator=locator, acquired_by=acquired_by,
                discovered_by=discovered_by, production_method=production_method,
            )
        suffix = suffix if suffix.startswith(".") else f".{suffix}"
        relative_path = f"{uuid.uuid4().hex}{suffix[:12]}"
        final_path = (self.artifact_root / relative_path).resolve()
        staged_path = self.artifact_root / f".{uuid.uuid4().hex}.tmp"
        source_path = payload if isinstance(payload, Path) else None
        if self.artifact_root not in final_path.parents:
            raise ValueError("artifact path escaped store root")
        try:
            if isinstance(payload, bytes):
                # 内存字节直接写入暂存文件。
                if not payload:
                    raise ValueError("Artifact payload must not be empty")
                with staged_path.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                byte_size = len(payload)
            else:
                # 文件输入按块复制，避免一次把视频等大文件全部读进内存。
                source_path = payload.resolve(strict=True)
                with source_path.open("rb") as source, staged_path.open("xb") as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
                byte_size = staged_path.stat().st_size
                if byte_size == 0:
                    raise ValueError("Artifact payload must not be empty")
            os.replace(staged_path, final_path)
            with final_path.open("rb") as handle:
                handle.read(1)
            sha256 = self._file_sha256(final_path)
            with self._lock, self._connect() as connection:
                # Artifact、Evidence 和可选 Candidate 关系位于同一数据库事务。
                connection.execute("BEGIN IMMEDIATE")
                artifact_pk = self._lastrowid(connection.execute(
                    """INSERT INTO artifacts(
                    relative_path,media_type,byte_size,role,sha256
                    ) VALUES (?,?,?,?,?)""",
                    (
                        relative_path,
                        self._require_text(media_type, "media_type"),
                        byte_size,
                        role,
                        sha256,
                    ),
                ))
                evidence_pk = self._lastrowid(connection.execute(
                    """INSERT INTO evidence(modalities_json,label,content,source_ref,
                    derived_from_json,artifact_id,locator,acquired_by,discovered_by,
                    production_method) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (self._json(list(modalities)), label, content, source_ref,
                     self._json(list(derived_from)), artifact_pk, locator, acquired_by,
                     discovered_by, production_method),
                ))
                if candidate_pk is not None:
                    connection.execute(
                        "INSERT INTO candidate_evidence(candidate_id,evidence_id,material_kind,material_status) VALUES (?,?,?,?)",
                        (candidate_pk, evidence_pk, self._require_text(material_kind, "material_kind"), material_status),
                    )
                connection.commit()
                return self._evidence_from_pk(connection, evidence_pk)
        except Exception:
            # 任一阶段失败都清理未提交文件，保持数据库与文件目录一致。
            staged_path.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)
            raise
        finally:
            # take_ownership 表示调用方把临时文件的生命周期交给 Store。
            if take_ownership and source_path is not None:
                source_path.unlink(missing_ok=True)

    def commit_ingested_evidence(
        self, *, source_ref: str, media_type: str, modalities: tuple[str, ...],
        label: str, original_filename: str, data: bytes | None = None,
        path: str | Path | None = None, discovered_by: str = "HARNESS",
    ) -> Evidence:
        """把用户输入的原始媒体登记成 source_material Evidence。"""

        # data 与 path 必须二选一；两者都有或都没有都容易造成歧义。
        if (data is None) == (path is None):
            raise ValueError("exactly one of data or path is required")
        payload: bytes | Path = data if data is not None else Path(path)  # type: ignore[arg-type]
        byte_size = len(data) if data is not None else Path(path).stat().st_size  # type: ignore[arg-type]
        # Evidence.content 保存一份人和模型都能读取的基础描述。
        content = self._json(
            {
                "original_filename": original_filename,
                "media_type": media_type,
                "byte_size": byte_size,
            }
        )
        return self._commit_evidence_payload(
            payload=payload, take_ownership=False, media_type=media_type,
            suffix=Path(original_filename).suffix or ".bin", role="source_material",
            modalities=modalities, label=label, content=content, source_ref=source_ref,
            derived_from=(), locator=self._json({"scope": "complete_media"}),
            acquired_by="HARNESS/ingest", discovered_by=discovered_by,
            production_method="source_material",
        )

    def commit_candidate_file_evidence(
        self, *, candidate_id: str, owned_temp_path: str | Path, media_type: str,
        suffix: str, label: str, content: str, locator: str,
        discovered_by: str,
    ) -> Evidence:
        """接管 INSPECT 下载的临时视频，并关联到所属 Candidate。"""

        return self._commit_evidence_payload(
            payload=self._validated_owned_temp_path(owned_temp_path), take_ownership=True, media_type=media_type,
            suffix=suffix, role="source_material", modalities=("video",), label=label,
            content=content, source_ref=candidate_id, derived_from=(), locator=locator,
            acquired_by="INSPECT/acquire", discovered_by=discovered_by,
            production_method="source_material", candidate_id=candidate_id,
            material_kind="video", material_status="ready",
        )

    def commit_derived_evidence(
        self, *, derived_from: tuple[str, ...], modalities: tuple[str, ...], label: str,
        content: str, locator: str, acquired_by: str, discovered_by: str,
        production_method: str = "tool_output", artifact_bytes: bytes | None = None,
        owned_temp_path: str | Path | None = None, media_type: str | None = None,
        suffix: str = ".bin", role: str = "derived_material",
    ) -> Evidence:
        """提交由工具生成的派生 Evidence，可附带字节或受控临时文件。"""

        if artifact_bytes is not None and owned_temp_path is not None:
            raise ValueError("derived Evidence accepts only one Artifact payload")
        payload: bytes | Path | None = artifact_bytes
        take_ownership = False
        if owned_temp_path is not None:
            # 只接管由 new_owned_temp_path 创建、仍位于 .incoming 的文件。
            payload, take_ownership = self._validated_owned_temp_path(owned_temp_path), True
        return self._commit_evidence_payload(
            payload=payload, take_ownership=take_ownership, media_type=media_type,
            suffix=suffix, role=role, modalities=modalities, label=label, content=content,
            source_ref=None, derived_from=derived_from, locator=locator,
            acquired_by=acquired_by, discovered_by=discovered_by,
            production_method=production_method,
        )

    def find_ingested_evidence(self, source_ref: str) -> Evidence | None:
        """按 source_ref 查找已经摄入的原始材料，用于幂等恢复。"""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM evidence WHERE source_ref=? AND acquired_by='HARNESS/ingest' AND derived_from_json='[]' ORDER BY id",
                (source_ref,),
            ).fetchall()
            if len(rows) > 1:
                # 同一输入只应有一份摄入记录；多条意味着数据不变量已损坏。
                raise RuntimeError(f"multiple ingested Evidence records for {source_ref}")
            return self._evidence_from_pk(connection, rows[0]["id"]) if rows else None

    def find_derived_evidence(
        self, *, derived_from: tuple[str, ...], acquired_by: str, locator: str | None = None
    ) -> tuple[Evidence, ...]:
        """按来源、工具和定位器查找已有派生结果，避免重复计算。"""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM evidence WHERE derived_from_json=? AND acquired_by=? AND (? IS NULL OR locator=?) ORDER BY id DESC",
                (self._json(list(derived_from)), acquired_by, locator, locator),
            ).fetchall()
            return tuple(self._evidence_from_pk(connection, row["id"]) for row in rows)

    def create_inspect_execution(self, request: dict[str, Any], workspace_id: str) -> int:
        """登记一次刚开始的 INSPECT 调用并返回内部执行编号。"""

        with self._lock, self._connect() as connection:
            return self._lastrowid(connection.execute(
                "INSERT INTO inspect_executions(request_json,workspace_id) VALUES (?,?)",
                (self._json(request), self._require_text(workspace_id, "workspace_id")),
            ))

    def finish_inspect_execution(
        self, execution_id: int, *, status: str, result_evidence_id: str | None,
        feedback: list[dict[str, Any]], details: dict[str, Any],
    ) -> None:
        """为 INSPECT 执行补全终态、结果 Evidence、反馈和细节。"""

        # 数据库外键使用 Evidence 数字主键，API 层使用 E<n>。
        result_pk = int(result_evidence_id[1:]) if result_evidence_id else None
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """UPDATE inspect_executions SET finished_at=CURRENT_TIMESTAMP,status=?,
                result_evidence_id=?,feedback_json=?,details_json=? WHERE id=?""",
                (status, result_pk, self._json(feedback), self._json(details), execution_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown INSPECT execution: {execution_id}")

    def inspect_execution(self, execution_id: int) -> dict[str, Any]:
        """读取 INSPECT 审计记录，并把 JSON 与 Evidence ID 恢复成外部格式。"""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM inspect_executions WHERE id=?", (execution_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown INSPECT execution: {execution_id}")
        result = dict(row)
        for field in ("request_json", "feedback_json", "details_json"):
            # 去掉字段名的 _json 后缀，让存储细节不泄漏给上层。
            result[field[:-5]] = (
                json.loads(result.pop(field)) if result[field] else None
            )
        if result["result_evidence_id"] is not None:
            result["result_evidence_id"] = f"E{result['result_evidence_id']}"
        return result

    def save_artifact(self, data: bytes, *, media_type: str, suffix: str, role: str) -> str:
        """保存不直接绑定 Evidence 的文件，例如搜索服务原始响应。"""

        if not isinstance(data, bytes):
            raise ValueError("artifact data must be bytes")
        media_type = self._require_text(media_type, "media_type")
        role = self._require_text(role, "role")
        suffix = suffix if suffix.startswith(".") else f".{suffix}"
        relative_path = f"{uuid.uuid4().hex}{suffix[:12]}"
        final_path = (self.artifact_root / relative_path).resolve()
        temp_path = self.artifact_root / f".{uuid.uuid4().hex}.tmp"
        if self.artifact_root not in final_path.parents:
            raise ValueError("artifact path escaped store root")
        sha256 = hashlib.sha256(data).hexdigest()
        try:
            # 与 Evidence 文件一样采用“暂存、fsync、原子替换、登记”的顺序。
            with temp_path.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, final_path)
            with self._lock, self._connect() as connection:
                artifact_pk = self._lastrowid(connection.execute(
                    """INSERT INTO artifacts(
                    relative_path,media_type,byte_size,role,sha256
                    ) VALUES (?,?,?,?,?)""",
                    (relative_path, media_type, len(data), role, sha256),
                ))
            return f"A{artifact_pk}"
        except Exception:
            temp_path.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)
            raise

    def create_search_execution(self, request: dict[str, Any], workspace_id: str) -> int:
        """登记一次刚开始的 SEARCH 调用。"""

        with self._lock, self._connect() as connection:
            return self._lastrowid(connection.execute(
                "INSERT INTO search_executions(request_json,workspace_id) VALUES (?,?)",
                (self._json(request), self._require_text(workspace_id, "workspace_id")),
            ))

    def finish_search_execution(
        self, execution_id: int, *, stop_reason: str, error: dict[str, Any] | None,
        provider_request_count: int, candidate_count: int, successful_source_count: int,
        elapsed_ms: int | None = None,
    ) -> None:
        """记录 SEARCH 为什么停止以及请求量、候选量和耗时。"""

        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """UPDATE search_executions SET finished_at=CURRENT_TIMESTAMP,
                stop_reason=?,error_json=?,provider_request_count=?,candidate_count=?,
                successful_source_count=?,elapsed_ms=? WHERE id=?""",
                (stop_reason, self._json(error) if error else None, provider_request_count,
                 candidate_count, successful_source_count, elapsed_ms, execution_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown SEARCH execution: {execution_id}")

    def search_execution(self, execution_id: int) -> dict[str, Any]:
        """读取一条 SEARCH 执行记录并反序列化请求与错误。"""

        with self._connect() as connection:
            row = connection.execute("SELECT * FROM search_executions WHERE id=?", (execution_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown SEARCH execution: {execution_id}")
        result = dict(row)
        result["request"] = json.loads(result.pop("request_json"))
        error_json = result.pop("error_json")
        result["error"] = json.loads(error_json) if error_json else None
        return result

    def record_provider_call(
        self, execution_id: int, *, provider: str, engine: str, attempt: int,
        elapsed_ms: int, outcome: str, raw_response_ref: str | None = None,
        error: dict[str, Any] | None = None, request_metadata: dict[str, Any] | None = None,
    ) -> None:
        """审计一次外部搜索 Provider 调用及重试序号、耗时和结果。"""

        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO provider_calls(execution_id,provider,engine,attempt,
                elapsed_ms,outcome,raw_response_ref,error_json,request_metadata_json) VALUES (?,?,?,?,?,?,?,?,?)""",
                (execution_id, provider, engine, attempt, elapsed_ms, outcome,
                 raw_response_ref, self._json(error) if error else None, self._json(request_metadata or {})),
            )

    def provider_calls(self, execution_id: int) -> list[dict[str, Any]]:
        """按发生顺序读取某次 SEARCH 的全部 Provider 调用。"""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM provider_calls WHERE execution_id=? ORDER BY id", (execution_id,),
            ).fetchall()
        values: list[dict[str, Any]] = []
        for row in rows:
            value = dict(row)
            error_json = value.pop("error_json")
            value["error"] = json.loads(error_json) if error_json else None
            value["request_metadata"] = json.loads(value.pop("request_metadata_json"))
            values.append(value)
        return values

    def record_source_open_attempt(
        self, execution_id: int, candidate_id: str, *, attempt: int,
        elapsed_ms: int, outcome: str, error: dict[str, Any] | None = None,
    ) -> None:
        """记录 SEARCH 获取候选网页正文的一次尝试。"""

        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO source_open_attempts(
                execution_id,candidate_id,attempt,elapsed_ms,outcome,error_json
                ) VALUES (?,?,?,?,?,?)""",
                (execution_id, self._candidate_pk(candidate_id), attempt, elapsed_ms,
                 outcome, self._json(error) if error else None),
            )

    def source_open_attempts(self, execution_id: int) -> list[dict[str, Any]]:
        """读取正文打开尝试，并额外提供 R<n> 形式的候选引用。"""

        with self._connect() as connection:
            rows = connection.execute(
                """SELECT soa.*, 'R' || soa.candidate_id AS candidate_ref
                FROM source_open_attempts soa WHERE execution_id=? ORDER BY soa.id""",
                (execution_id,),
            ).fetchall()
        values: list[dict[str, Any]] = []
        for row in rows:
            value = dict(row)
            error_json = value.pop("error_json")
            value["error"] = json.loads(error_json) if error_json else None
            values.append(value)
        return values

    def record_search_image_upload(
        self,
        execution_id: int,
        evidence_id: str,
        artifact_ref: str,
        adaptation: dict[str, Any],
    ) -> None:
        """记录反向图片搜索上传了哪个 Artifact，以及做过何种适配。"""

        # 先通过公开读取方法验证 E 与 A 都真实存在。
        evidence_pk = int(self.evidence(evidence_id).evidence_id[1:])
        artifact_pk = self._artifact_pk(artifact_ref)
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO search_image_uploads(
                execution_id,evidence_id,artifact_id,adaptation_json
                ) VALUES (?,?,?,?)""",
                (execution_id, evidence_pk, artifact_pk, self._json(adaptation)),
            )

    def record_candidate_discovery(
        self, execution_id: int, candidate_id: str, *, provider: str, engine: str,
        provider_rank: int | None, raw_response_ref: str | None,
    ) -> None:
        """记录某 Provider 在某次执行中发现 Candidate 的事实和排名。"""

        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO candidate_discoveries(execution_id,candidate_id,provider,engine,
                provider_rank,raw_response_ref) VALUES (?,?,?,?,?,?)""",
                (execution_id, self._candidate_pk(candidate_id), provider, engine,
                 provider_rank, raw_response_ref),
            )

    def candidate_discoveries(self, execution_id: int) -> list[dict[str, Any]]:
        """读取候选发现轨迹，并提供外部 R<n> 引用。"""

        with self._connect() as connection:
            rows = connection.execute(
                """SELECT cd.*, 'R' || cd.candidate_id AS candidate_ref
                FROM candidate_discoveries cd WHERE execution_id=? ORDER BY cd.id""",
                (execution_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def provider_call_count(self, execution_id: int) -> int:
        """返回某次 SEARCH 实际消耗的 Provider 请求次数。"""

        with self._connect() as connection:
            return int(connection.execute(
                "SELECT COUNT(*) FROM provider_calls WHERE execution_id=?", (execution_id,),
            ).fetchone()[0])

    def execution_candidate_count(self, execution_id: int) -> int:
        """返回某次 SEARCH 发现的去重 Candidate 数。"""

        with self._connect() as connection:
            return int(connection.execute(
                "SELECT COUNT(DISTINCT candidate_id) FROM candidate_discoveries WHERE execution_id=?",
                (execution_id,),
            ).fetchone()[0])

    def get_or_create_candidate(
        self, *, identity_key: str, **fields: Any
    ) -> tuple[str, bool]:
        """按规范化身份键查找或创建 Candidate，并报告是否为新记录。"""

        with self._lock, self._connect() as connection:
            # 并发调用必须串行完成“先查再插”，否则可能重复创建同一来源。
            connection.execute("BEGIN IMMEDIATE")
            # alias 表把原始 URL、跳转后 URL 等多个身份键指向同一 R。
            row = connection.execute(
                """SELECT candidate_id AS id FROM candidate_aliases
                WHERE identity_key=?""",
                (identity_key,),
            ).fetchone()
            if row is None:
                row = connection.execute(
                    "SELECT id FROM candidates WHERE identity_key=?", (identity_key,)
                ).fetchone()
            if row is not None:
                connection.commit()
                return f"R{row['id']}", False
            # 只有确实不存在时才写 Candidate 主记录。
            cursor = connection.execute(
                """INSERT INTO candidates(identity_key,source_url,search_title,
                search_snippet,provider,engine,provider_rank,discovered_execution_id,
                raw_response_ref,final_url,candidate_type,image_url,thumbnail_url,
                platform,platform_video_id,extra_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (identity_key, fields["source_url"], fields.get("search_title"),
                 fields.get("search_snippet"), fields.get("provider"), fields.get("engine"),
                 fields.get("provider_rank"), fields.get("discovered_execution_id"),
                 fields.get("raw_response_ref"), fields.get("final_url"),
                 fields.get("candidate_type"), fields.get("image_url"),
                 fields.get("thumbnail_url"), fields.get("platform"),
                 fields.get("platform_video_id"), self._json(fields.get("extra", {}))),
            )
            candidate_pk = self._lastrowid(cursor)
            candidate_id = f"R{candidate_pk}"
            if (
                connection.execute(
                    "SELECT 1 FROM materials WHERE ref=?", (candidate_id,)
                ).fetchone()
                or connection.execute(
                    "SELECT 1 FROM findings WHERE finding_id=?", (candidate_id,)
                ).fetchone()
                or connection.execute(
                    "SELECT 1 FROM reports WHERE task_id=?", (candidate_id,)
                ).fetchone()
            ):
                raise ValueError(f"duplicate global reference ID for Candidate Source: {candidate_id}")
            # 创建记录的同时登记第一个身份别名。
            connection.execute(
                "INSERT INTO candidate_aliases(identity_key,candidate_id) VALUES (?,?)",
                (identity_key, candidate_pk),
            )
            connection.commit()
            return candidate_id, True

    def update_candidate_final_url(
        self, candidate_id: str, final_url: str, final_identity_key: str
    ) -> str:
        """登记跳转后的 URL；若已存在别名，则返回既有规范 Candidate。"""

        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            candidate_pk = self._candidate_pk(candidate_id)
            alias = connection.execute(
                """SELECT candidate_id FROM candidate_aliases
                WHERE identity_key=?""",
                (final_identity_key,),
            ).fetchone()
            # 跳转目标可能早已由另一个搜索结果创建；此时合并到那个 R。
            canonical_pk = int(alias["candidate_id"]) if alias is not None else candidate_pk
            if alias is None:
                connection.execute(
                    """INSERT INTO candidate_aliases(identity_key,candidate_id)
                    VALUES (?,?)""",
                    (final_identity_key, candidate_pk),
                )
            connection.execute(
                "UPDATE candidates SET final_url=? WHERE id=?",
                (final_url, canonical_pk),
            )
            connection.commit()
            return f"R{canonical_pk}"

    def candidate(self, candidate_id: str) -> dict[str, Any]:
        """读取一个 Candidate，并恢复 R<n> ID 与 extra 字典。"""

        pk = self._candidate_pk(candidate_id)
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM candidates WHERE id=?", (pk,)).fetchone()
        if row is None:
            raise KeyError(f"unknown Candidate Source ID: {candidate_id}")
        result = dict(row)
        result["candidate_id"] = candidate_id
        result["extra"] = json.loads(result.pop("extra_json"))
        result.pop("id")
        return result

    def candidate_evidence(
        self, candidate_id: str, material_kind: str | None = None
    ) -> list[tuple[str, str, Evidence]]:
        """读取 Candidate 关联的 Evidence，可按材料种类筛选。"""

        params: list[Any] = [self._candidate_pk(candidate_id)]
        sql = """SELECT ce.material_kind,ce.material_status,e.id FROM candidate_evidence ce
        JOIN evidence e ON e.id=ce.evidence_id WHERE ce.candidate_id=?"""
        if material_kind is not None:
            sql += " AND ce.material_kind=?"
            params.append(material_kind)
        sql += " ORDER BY e.id"
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
            return [
                (
                    row["material_kind"],
                    row["material_status"],
                    self._evidence_from_pk(connection, row["id"]),
                )
                for row in rows
            ]

    def add_finding(self, finding: Finding) -> None:
        """保存一条已分配 F ID 的 Finding。"""

        self._validate_finding(finding)
        if self.has_reference(finding.finding_id):
            raise ValueError(f"duplicate global reference ID for Finding: {finding.finding_id}")
        with self._lock, self._connect() as connection:
            connection.execute("INSERT INTO findings(finding_id,payload_json) VALUES (?,?)", (finding.finding_id, self._json(finding.to_dict())))

    def add_workspace_report(self, report: WorkspaceReport) -> None:
        """验证并原子发布 Workspace Report、其中的 Findings 和任务终态。"""

        # COMPLETED 代表确实完成，不应再附带“不完整原因”。
        if report.status not in {"COMPLETED", "INCOMPLETE"}:
            raise ValueError(f"unknown workspace status: {report.status}")
        if report.status == "COMPLETED" and report.status_reason is not None:
            raise ValueError("status_reason is only valid for an INCOMPLETE Workspace Report")
        if report.status_reason is not None:
            self._require_text(report.status_reason, "status_reason")
        for evidence_id in report.selected_evidence_ids:
            # selected_evidence_ids 是报告允许引用的 Evidence 白名单。
            try:
                self.evidence(evidence_id)
            except KeyError as exc:
                raise KeyError(f"Workspace Report cites unknown selected Evidence ID: {evidence_id}") from exc
        seen: set[str] = set()
        for finding in report.findings:
            # 一个报告内部不能重复发布同一个 F ID。
            if finding.finding_id in seen:
                raise ValueError(f"duplicate Finding ID inside Workspace Report: {finding.finding_id}")
            seen.add(finding.finding_id)
            if finding.workspace_id != report.task_id:
                raise ValueError("Finding workspace_id must match Workspace Report task_id")
            self._validate_finding(finding)
            if self.has_reference(finding.finding_id):
                raise ValueError(f"duplicate global reference ID for Finding: {finding.finding_id}")
        selected = set(report.selected_evidence_ids)
        # 每条 Finding 的引证都必须同时出现在报告选择清单中。
        missing_selected = sorted(
            {
                evidence_id
                for finding in report.findings
                for evidence_id in finding.evidence_ids
                if evidence_id not in selected
            }
        )
        if missing_selected:
            raise ValueError(
                "Finding Evidence IDs must be included in selected_evidence_ids: "
                + ", ".join(missing_selected)
            )
        if self.has_reference(report.task_id) and not isinstance(self.resolve(report.task_id), WorkspaceTask):
            raise ValueError(f"duplicate global reference ID for Workspace Report: {report.task_id}")
        with self._lock, self._connect() as connection:
            # Findings、Report 和 Workspace 终态必须一次提交。
            connection.execute("BEGIN IMMEDIATE")
            for finding in report.findings:
                connection.execute("INSERT INTO findings(finding_id,payload_json) VALUES (?,?)", (finding.finding_id, self._json(finding.to_dict())))
            connection.execute("INSERT INTO reports(task_id,payload_json) VALUES (?,?)", (report.task_id, self._json(report.to_dict())))
            connection.execute(
                """UPDATE workspace_tasks SET run_status='reported',error_json=NULL,
                finished_at=CURRENT_TIMESTAMP WHERE task_id=?""",
                (report.task_id,),
            )
            connection.commit()

    def commit_workspace_report(self, draft: WorkspaceReportDraft) -> WorkspaceReport:
        """为 HI 草稿分配 F ID，然后发布经过验证的终态报告。"""

        findings: list[Finding] = []
        for item in draft.findings:
            # 独立序列表确保即使后续报告校验失败，F ID 也不会被复用。
            with self._lock, self._connect() as connection:
                sequence = self._lastrowid(
                    connection.execute("INSERT INTO finding_sequence DEFAULT VALUES")
                )
            findings.append(
                Finding(
                    finding_id=f"F{sequence}",
                    workspace_id=draft.task_id,
                    statement=item.statement,
                    evidence_ids=item.evidence_ids,
                    limitations=item.limitations,
                )
            )
        report = WorkspaceReport(
            task_id=draft.task_id,
            status=draft.status,
            findings=tuple(findings),
            selected_evidence_ids=draft.selected_evidence_ids,
            status_reason=draft.status_reason,
        )
        self.add_workspace_report(report)
        return report

    def _validate_finding(self, finding: Finding) -> None:
        """校验 Finding 的必填字段以及所有 Evidence 引用。"""

        self._require_text(finding.finding_id, "finding_id")
        self._require_text(finding.workspace_id, "workspace_id")
        self._require_text(finding.statement, "statement")
        for limitation in finding.limitations:
            self._require_text(limitation, "Finding limitation")
        if not finding.evidence_ids:
            raise ValueError("Finding must cite at least one Evidence ID")
        for evidence_id in finding.evidence_ids:
            try:
                self.evidence(evidence_id)
            except KeyError as exc:
                raise KeyError(f"Finding cites unknown Evidence ID: {evidence_id}") from exc

    def has_reference(self, reference_id: str) -> bool:
        """检查 E、R、普通材料、F 或 W 引用是否存在。"""

        if not isinstance(reference_id, str) or not reference_id:
            return False
        if reference_id.startswith("E") and reference_id[1:].isdigit():
            # 数字型引用可以直接命中相应主键。
            table, key, value = "evidence", "id", int(reference_id[1:])
        elif reference_id.startswith("R") and reference_id[1:].isdigit():
            table, key, value = "candidates", "id", int(reference_id[1:])
        else:
            # C/F/W 等字符串引用分别存于各自表中。
            for table, key in (("materials", "ref"), ("findings", "finding_id"),
                               ("reports", "task_id"), ("workspace_tasks", "task_id")):
                with self._connect() as connection:
                    if connection.execute(f"SELECT 1 FROM {table} WHERE {key}=?", (reference_id,)).fetchone():
                        return True
            return False
        with self._connect() as connection:
            return connection.execute(f"SELECT 1 FROM {table} WHERE {key}=?", (value,)).fetchone() is not None

    def resolve(self, reference_id: str) -> Any:
        """把任意 Global Store 引用解析为对应领域对象或材料值。"""

        if not isinstance(reference_id, str) or not reference_id:
            raise KeyError(f"unknown Global Store reference: {reference_id!r}")
        if reference_id.startswith("E") and reference_id[1:].isdigit():
            return self.evidence(reference_id)
        if reference_id.startswith("R") and reference_id[1:].isdigit():
            return self.candidate(reference_id)
        if reference_id in self._materials:
            # 返回深拷贝，防止调用方修改进程内缓存中的原值。
            return copy.deepcopy(self._materials[reference_id])
        with self._connect() as connection:
            row = connection.execute("SELECT payload_json FROM materials WHERE ref=?", (reference_id,)).fetchone()
            if row:
                return json.loads(row["payload_json"])
            row = connection.execute("SELECT payload_json FROM findings WHERE finding_id=?", (reference_id,)).fetchone()
            if row:
                # 兼容旧版 Finding：缺少 workspace_id 时标记为 LEGACY。
                p = json.loads(row["payload_json"])
                return Finding(
                    finding_id=p["finding_id"],
                    workspace_id=p.get("workspace_id", "LEGACY"),
                    statement=p["statement"],
                    evidence_ids=tuple(p["evidence_ids"]),
                    limitations=tuple(p.get("limitations", ())),
                )
            row = connection.execute("SELECT payload_json FROM reports WHERE task_id=?", (reference_id,)).fetchone()
            if row:
                return self._report_from_payload(json.loads(row["payload_json"]))
            row = connection.execute(
                "SELECT task,materials_json FROM workspace_tasks WHERE task_id=?", (reference_id,)
            ).fetchone()
            if row:
                return WorkspaceTask(reference_id, row["task"], tuple(json.loads(row["materials_json"])))
        raise KeyError(f"unknown Global Store reference: {reference_id}")

    def resolve_many(self, reference_ids: tuple[str, ...]) -> dict[str, Any]:
        """批量解析引用，并保留输入 ID 作为结果键。"""

        return {reference_id: self.resolve(reference_id) for reference_id in reference_ids}

    def evidence(self, evidence_id: str) -> Evidence:
        """按 E<n> 读取 Evidence；格式错误和记录不存在都视为未知 ID。"""

        if not evidence_id.startswith("E") or not evidence_id[1:].isdigit():
            raise KeyError(f"unknown Evidence ID: {evidence_id}")
        with self._connect() as connection:
            return self._evidence_from_pk(connection, int(evidence_id[1:]))

    def finding(self, finding_id: str) -> Finding:
        """读取 Finding，并额外确认解析结果类型正确。"""

        value = self.resolve(finding_id)
        if not isinstance(value, Finding):
            raise KeyError(f"unknown Finding ID: {finding_id}")
        return value

    def workspace_report(self, task_id: str) -> WorkspaceReport:
        """按 W ID 读取已经发布的 Workspace Report。"""

        value = self.resolve(task_id)
        if not isinstance(value, WorkspaceReport):
            raise KeyError(f"unknown Workspace Report ID: {task_id}")
        return value

    def evidence_records(self) -> tuple[Evidence, ...]:
        """按 E ID 顺序返回全部完整 Evidence。"""

        with self._connect() as connection:
            rows = connection.execute("SELECT id FROM evidence ORDER BY id").fetchall()
            return tuple(self._evidence_from_pk(connection, int(row["id"])) for row in rows)

    def evidence_index_records(
        self, *, content_chars: int | None = 2000
    ) -> tuple[dict[str, Any], ...]:
        """返回有长度上限的 Evidence 索引，供排序和模型视图使用。"""

        # bool 是 int 的子类，必须单独排除 True/False。
        if content_chars is not None and (
            isinstance(content_chars, bool)
            or not isinstance(content_chars, int)
            or content_chars < 1
        ):
            raise ValueError("content_chars must be a positive integer")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT id,modalities_json,label,source_ref,derived_from_json,
                acquired_by,discovered_by,substr(content,1,coalesce(?,length(content))) AS content_excerpt,
                length(content) AS content_chars
                FROM evidence ORDER BY id""",
                (content_chars,),
            ).fetchall()
        values: list[dict[str, Any]] = []
        for row in rows:
            # 这里只带 content_excerpt，不把可能很大的全文塞进模型上下文。
            values.append(
                {
                    "evidence_id": f"E{int(row['id'])}",
                    "modalities": json.loads(row["modalities_json"]),
                    "label": row["label"],
                    "source_ref": row["source_ref"],
                    "derived_from": json.loads(row["derived_from_json"]),
                    "acquired_by": row["acquired_by"],
                    "discovered_by": row["discovered_by"],
                    "content_excerpt": row["content_excerpt"],
                    "content_chars": int(row["content_chars"]),
                }
            )
        return tuple(values)

    def candidate_records(self) -> tuple[dict[str, Any], ...]:
        """按 R ID 顺序返回全部 Candidate 完整记录。"""

        with self._connect() as connection:
            rows = connection.execute("SELECT id FROM candidates ORDER BY id").fetchall()
        return tuple(self.candidate(f"R{int(row['id'])}") for row in rows)

    def candidate_index_records(self) -> tuple[dict[str, Any], ...]:
        """只返回 QUERY_MEMORY 检索 Candidate 所需的小型字段集合。"""

        with self._connect() as connection:
            rows = connection.execute(
                """SELECT id,source_url,final_url,search_title,search_snippet,candidate_type
                FROM candidates ORDER BY id"""
            ).fetchall()
        return tuple(
            {
                "candidate_id": f"R{int(row['id'])}",
                "source_url": row["source_url"],
                "final_url": row["final_url"],
                "search_title": row["search_title"],
                "search_snippet": row["search_snippet"],
                "candidate_type": row["candidate_type"],
            }
            for row in rows
        )

    def finding_records(self) -> tuple[Finding, ...]:
        """按写入顺序返回全部 Finding。"""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT finding_id FROM findings ORDER BY rowid"
            ).fetchall()
        return tuple(self.finding(row["finding_id"]) for row in rows)

    def workspace_report_records(self) -> tuple[WorkspaceReport, ...]:
        """按写入顺序返回全部 Workspace Report。"""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT task_id FROM reports ORDER BY rowid"
            ).fetchall()
        return tuple(self.workspace_report(row["task_id"]) for row in rows)

    def _evidence_from_pk(self, connection: sqlite3.Connection, pk: int) -> Evidence:
        """使用已有连接把内部整数主键转换成公开 Evidence 对象。"""

        row = connection.execute(
            """SELECT e.*,a.id AS artifact_pk FROM evidence e
            LEFT JOIN artifacts a ON a.id=e.artifact_id WHERE e.id=?""",
            (pk,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown Evidence ID: E{pk}")
        # 数据库中的 JSON 和整数外键在这里统一恢复为领域层格式。
        return Evidence(
            evidence_id=f"E{pk}",
            modalities=tuple(json.loads(row["modalities_json"])),
            label=row["label"],
            content=row["content"],
            source_ref=row["source_ref"],
            derived_from=tuple(json.loads(row["derived_from_json"])),
            artifact_ref=f"A{row['artifact_pk']}" if row["artifact_pk"] else None,
            locator=row["locator"],
            acquired_by=row["acquired_by"],
            discovered_by=row["discovered_by"],
            production_method=row["production_method"],
        )

    @staticmethod
    def _candidate_pk(candidate_id: str) -> int:
        """严格解析 R<n>，返回 SQLite 使用的整数主键。"""

        if (
            not isinstance(candidate_id, str)
            or not candidate_id.startswith("R")
            or not candidate_id[1:].isdigit()
        ):
            raise KeyError(f"unknown Candidate Source ID: {candidate_id}")
        return int(candidate_id[1:])

    @staticmethod
    def _artifact_pk(artifact_ref: str) -> int:
        """严格解析 A<n>，返回 SQLite 使用的整数主键。"""

        if (
            not isinstance(artifact_ref, str)
            or not artifact_ref.startswith("A")
            or not artifact_ref[1:].isdigit()
        ):
            raise KeyError(f"unknown Artifact reference: {artifact_ref}")
        return int(artifact_ref[1:])

    @staticmethod
    def _validate_modalities(modalities: Iterable[str]) -> None:
        """要求模态非空、不重复，而且只来自系统支持的四种类型。"""

        values = tuple(modalities)
        if (
            not values
            or len(values) != len(set(values))
            or any(value not in {"text", "image", "audio", "video"} for value in values)
        ):
            raise ValueError(
                "modalities must contain unique text, image, audio, or video values"
            )

    @staticmethod
    def _report_from_payload(payload: dict[str, Any]) -> WorkspaceReport:
        """把数据库 JSON 兼容地恢复成 WorkspaceReport 领域对象。"""

        findings = tuple(
            Finding(
                finding_id=item["finding_id"],
                workspace_id=item.get("workspace_id", payload["task_id"]),
                statement=item["statement"],
                evidence_ids=tuple(item["evidence_ids"]),
                limitations=tuple(item.get("limitations", ())),
            )
            for item in payload["findings"]
        )
        # 兼容早期数据库使用的小写状态值。
        status = {"completed": "COMPLETED", "incomplete": "INCOMPLETE"}.get(
            payload["status"], payload["status"]
        )
        status_reason = payload.get("status_reason")
        if status == "COMPLETED":
            status_reason = None
        return WorkspaceReport(
            task_id=payload["task_id"],
            status=status,
            findings=findings,
            selected_evidence_ids=tuple(payload["selected_evidence_ids"]),
            status_reason=status_reason,
        )
