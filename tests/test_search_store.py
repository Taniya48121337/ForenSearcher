"""SEARCH 数据表、Candidate 并发去重和 Artifact 提交测试。

这些用例直接检查数据库最终状态，确保审计数据与文件系统不会出现半提交。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
import sqlite3

import pytest

from foren_searcher.store import GlobalStore


def test_candidate_can_exist_without_evidence_and_later_link_multiple_materials() -> None:
    """验证 candidate can exist without evidence and later link multiple materials 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    candidate_id, _ = store.get_or_create_candidate(identity_key="url:https://example.org/", source_url="https://example.org/", candidate_type="page")
    assert candidate_id == "R1"
    assert store.candidate_evidence(candidate_id) == []
    page = store.commit_material_evidence(candidate_id=candidate_id, material_kind="page", material_status="ready",
        artifact_bytes=b"<html>body</html>", media_type="text/html", suffix=".html", modalities=("text",),
        label="page", content="[1] body", locator="paragraph 1", acquired_by="SEARCH/open/http", discovered_by="W1-1")
    image = store.commit_material_evidence(candidate_id=candidate_id, material_kind="image", material_status="ready",
        artifact_bytes=b"image bytes", media_type="image/jpeg", suffix=".jpg", modalities=("image",),
        label="image", content='{"width":1,"height":1}', locator="complete image", acquired_by="SEARCH/open/image", discovered_by="W1-2")
    assert {item[2].evidence_id for item in store.candidate_evidence(candidate_id)} == {page.evidence_id, image.evidence_id}
    assert store.artifact_path(page.artifact_ref).read_bytes() == b"<html>body</html>"


def test_artifact_failure_does_not_publish_evidence() -> None:
    """验证 artifact failure does not publish evidence 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    candidate_id, _ = store.get_or_create_candidate(identity_key="url:https://example.org/", source_url="https://example.org/")
    with patch("foren_searcher.store.os.replace", side_effect=OSError("disk unavailable")):
        with pytest.raises(OSError):
            store.commit_material_evidence(candidate_id=candidate_id, material_kind="page", material_status="ready",
                artifact_bytes=b"x", media_type="text/plain", suffix=".txt", modalities=("text",), label="x",
                content="x", locator="complete", acquired_by="SEARCH/open", discovered_by="W1-1")
    assert store.candidate_evidence(candidate_id) == []


def test_concurrent_workspaces_receive_unique_evidence_ids() -> None:
    """验证 concurrent workspaces receive unique evidence ids 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore()
    def write(index: int) -> str:
        """测试辅助函数 write；为用例提供可重复的输入或可控行为。"""

        return store.record_evidence(content=f"item {index}", source_ref="V0", locator=f"frame {index}",
            acquired_by="INSPECT", discovered_by=f"W1-{index + 1}").evidence_id
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(write, range(40)))
    assert len(ids) == len(set(ids)) == 40
    assert {int(value[1:]) for value in ids} == set(range(1, 41))


def test_sqlite_and_artifacts_survive_store_reopen(tmp_path: Path) -> None:
    """验证 sqlite and artifacts survive store reopen 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    db = tmp_path / "investigation.sqlite3"; artifacts = tmp_path / "artifacts"
    first = GlobalStore(db, artifacts)
    candidate_id, _ = first.get_or_create_candidate(identity_key="url:https://example.org/", source_url="https://example.org/")
    evidence = first.commit_material_evidence(candidate_id=candidate_id, material_kind="page", material_status="ready",
        artifact_bytes=b"persistent", media_type="text/plain", suffix=".txt", modalities=("text",),
        label="persistent", content="persistent", locator="complete", acquired_by="SEARCH/open", discovered_by="W1-1")
    second = GlobalStore(db, artifacts)
    assert second.evidence(evidence.evidence_id).source_ref == candidate_id
    assert second.artifact_path(evidence.artifact_ref).read_bytes() == b"persistent"


def test_candidate_discovery_provenance_and_execution_counts_are_persistent(tmp_path: Path) -> None:
    """验证 candidate discovery provenance and execution counts are persistent 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    db = tmp_path / "investigation.sqlite3"; artifacts = tmp_path / "artifacts"
    first = GlobalStore(db, artifacts)
    execution = first.create_search_execution({"input_type": "text", "query": "q", "output_type": "text"}, "W1-1")
    candidate_id, _ = first.get_or_create_candidate(identity_key="url:https://example.org/", source_url="https://example.org/")
    first.record_candidate_discovery(execution, candidate_id, provider="one", engine="web", provider_rank=1, raw_response_ref=None)
    first.record_candidate_discovery(execution, candidate_id, provider="two", engine="news", provider_rank=4, raw_response_ref=None)
    second = GlobalStore(db, artifacts)
    assert second.execution_candidate_count(execution) == 1
    assert [(item["provider"], item["provider_rank"]) for item in second.candidate_discoveries(execution)] == [("one", 1), ("two", 4)]


def test_unversioned_search_execution_schema_is_upgraded_additively(tmp_path: Path) -> None:
    """验证 unversioned search execution schema is upgraded additively 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    db = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(db) as connection:
        connection.execute("""CREATE TABLE search_executions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, request_json TEXT NOT NULL,
            workspace_id TEXT NOT NULL, started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            finished_at TEXT, stop_reason TEXT, error_json TEXT,
            provider_request_count INTEGER NOT NULL DEFAULT 0,
            candidate_count INTEGER NOT NULL DEFAULT 0,
            successful_source_count INTEGER NOT NULL DEFAULT 0)""")
    store = GlobalStore(db, tmp_path / "artifacts")
    with sqlite3.connect(db) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(search_executions)")}
        version = connection.execute("PRAGMA user_version").fetchone()[0]
    assert "elapsed_ms" in columns and version == 3
    execution = store.create_search_execution({"input_type": "text", "query": "q", "output_type": "text"}, "W1-1")
    store.finish_search_execution(execution, stop_reason="candidates_exhausted", error=None,
        provider_request_count=0, candidate_count=0, successful_source_count=0, elapsed_ms=12)
    assert store.search_execution(execution)["elapsed_ms"] == 12


def test_newer_store_schema_is_rejected_before_writes(tmp_path: Path) -> None:
    """验证 newer store schema is rejected before writes 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    db = tmp_path / "future.sqlite3"
    with sqlite3.connect(db) as connection:
        connection.execute("PRAGMA user_version = 999")
    with pytest.raises(RuntimeError, match="newer than supported"):
        GlobalStore(db, tmp_path / "artifacts")


def test_candidate_identity_is_atomic_across_store_instances(tmp_path: Path) -> None:
    """验证 candidate identity is atomic across store instances 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    db = tmp_path / "shared.sqlite3"; artifacts = tmp_path / "artifacts"
    stores = (GlobalStore(db, artifacts), GlobalStore(db, artifacts))

    def create(index: int) -> tuple[str, bool]:
        """测试辅助函数 create；为用例提供可重复的输入或可控行为。"""

        return stores[index % 2].get_or_create_candidate(
            identity_key="url:https://example.org/same",
            source_url="https://example.org/same",
            candidate_type="page",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(create, range(40)))
    assert {candidate_id for candidate_id, _created in results} == {"R1"}
    assert sum(created for _candidate_id, created in results) == 1
