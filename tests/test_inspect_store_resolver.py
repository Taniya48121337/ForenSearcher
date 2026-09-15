"""GlobalStore 与 EvidenceResolver 的 INSPECT 接口测试。

验证 Artifact 路径安全、媒体嗅探和 E→文件解析错误能稳定工作。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from foren_searcher.evidence_resolver import EvidenceResolutionError, EvidenceResolver
from foren_searcher.store import GlobalStore

from inspect_helpers import image_bytes, ingest


def test_ingest_copies_user_file_and_reopens_persistently(tmp_path: Path):
    """验证 ingest copies user file and reopens persistently 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    original = tmp_path / "user.png"; original.write_bytes(image_bytes())
    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    evidence = store.commit_ingested_evidence(source_ref="I0", media_type="image/png", modalities=("image",), label="input", original_filename=original.name, path=original)
    assert original.exists()
    assert store.find_ingested_evidence("I0") == evidence
    reopened = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    resolved = EvidenceResolver(reopened).resolve(evidence.evidence_id)
    assert resolved.media_kind == "image" and resolved.path.read_bytes().startswith(b"\x89PNG")


def test_derived_media_and_text_use_one_global_e_namespace(tmp_path: Path):
    """验证 derived media and text use one global e namespace 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    source = ingest(store, image_bytes(), source="I0", filename="input.png", media_type="image/png", modality="image")
    crop = store.commit_derived_evidence(derived_from=(source.evidence_id,), modalities=("image",), label="crop", content="{}", locator="{}", acquired_by="INSPECT/crop", discovered_by="W1-1", artifact_bytes=image_bytes(size=(20,20)), media_type="image/png", suffix=".png")
    result = store.commit_derived_evidence(derived_from=(crop.evidence_id,), modalities=("text",), label="result", content="{}", locator="{}", acquired_by="INSPECT/ocr", discovered_by="W1-1")
    assert (source.evidence_id, crop.evidence_id, result.evidence_id) == ("E1", "E2", "E3")
    assert store.find_derived_evidence(derived_from=(source.evidence_id,), acquired_by="INSPECT/crop", locator="{}")[0] == crop
    with pytest.raises(EvidenceResolutionError, match="no media Artifact"):
        EvidenceResolver(store).resolve(result.evidence_id)


def test_missing_artifact_is_not_resolved(tmp_path: Path):
    """验证 missing artifact is not resolved 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

    store = GlobalStore(tmp_path / "case.sqlite3", tmp_path / "artifacts")
    evidence = ingest(store, image_bytes(), source="I0", filename="input.png", media_type="image/png", modality="image")
    store.artifact_path(evidence.artifact_ref).unlink()
    with pytest.raises(EvidenceResolutionError, match="unavailable"):
        EvidenceResolver(store).resolve(evidence.evidence_id)
