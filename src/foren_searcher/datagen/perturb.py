"""证据扰动器（"抽积木"）：在证据库副本上做可控改动，生成自带标准答案的对照状态。

思路来自 REAL（arXiv 2609.08943，逐轮删证据看结论是否跟着变）和 Divide-and-Cooperate
的硬正例增广（往证据里塞弱相关材料）。大白话：老师傅写结论时桌上摆着 E1…E9，我们把其中
一块积木抽掉 / 换掉 / 多放几块无关的，再让他从同一步重做。因为我们**事先就知道**正确的
结论应该怎么变，所以每个扰动都自带期望，不需要请模型来出错题。

三种扰动：
- mask   ：把 E 的内容换成"不可用"提示（模拟来源失效），派生自它的 E 一并遮蔽。
- replace：把 E 的内容换成给定文本（结论相反 / 口径不同 / 编号错位）。
- inject ：往证据库里加若干条弱相关的干扰 E。

所有操作只在 **副本** 上进行（snapshot() 先复制 sqlite + artifacts），原始运行记录不动。
每个扰动返回 PerturbationRecord，其中 expected 用可程序判定的断言表达"正确结论应如何变化"。
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..store.global_store import GlobalStore
from .errors import ErrorType

MASK_NOTICE = "[本条证据在对照实验中已被遮蔽：来源不可访问，内容不可用]"

PerturbationKind = Literal["mask", "replace", "inject"]


class PerturbationError(ValueError):
    """扰动无法安全执行（比如目标 E 已被 Finding 引用，或是初始材料）。"""


@dataclass(frozen=True)
class Expectation:
    """一条可程序判定的期望。kind 决定 params 的含义。"""

    kind: Literal[
        "finding_must_not_cite",        # params: {evidence_ids}  重做后的报告不得引用这些 E 作为支持
        "finding_must_cite_any",        # params: {evidence_ids}  重做后的报告应引用其中之一
        "status_should_be",             # params: {status}        报告状态应为 COMPLETED/INCOMPLETE
        "answer_unchanged",             # params: {}              最终结论不应变化（加噪场景）
        "answer_should_change",         # params: {}              最终结论应变化（替换为相反证据）
        "citations_unchanged",          # params: {evidence_ids}  引用集合不应变化（加噪场景）
    ]
    params: dict[str, Any] = field(default_factory=dict)
    if_violated: ErrorType | None = None   # 违反这条期望对应哪种错误类型

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "params": self.params,
                "if_violated": self.if_violated.value if self.if_violated else None}


@dataclass(frozen=True)
class PerturbationRecord:
    kind: PerturbationKind
    target_ids: tuple[str, ...]            # 被改动/新增的 E
    cascaded_ids: tuple[str, ...]          # 因派生关系被连带遮蔽的 E
    expected: tuple[Expectation, ...]
    detail: str
    store_dir: str                         # 扰动后的副本目录

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "target_ids": list(self.target_ids), "cascaded_ids": list(self.cascaded_ids),
                "expected": [e.to_dict() for e in self.expected], "detail": self.detail, "store_dir": self.store_dir}


# --------------------------------------------------------------------------- 副本

def snapshot(db_path: str | Path, artifact_root: str | Path, destination: str | Path) -> tuple[Path, Path]:
    """把一个 store（sqlite + artifacts 目录）完整复制到 destination，返回副本的 (db, artifacts)。"""

    src_db, src_art, dst = Path(db_path), Path(artifact_root), Path(destination)
    if not src_db.is_file():
        raise PerturbationError(f"store 数据库不存在: {src_db}")
    if dst.exists() and any(dst.iterdir()):
        raise PerturbationError(f"目标目录非空: {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    # 用 sqlite 的 backup 接口复制，避免 WAL 未合并导致副本不完整。
    with sqlite3.connect(src_db) as source, sqlite3.connect(dst / "store.sqlite3") as target:
        source.backup(target)
    if src_art.is_dir():
        shutil.copytree(src_art, dst / "artifacts")
    else:
        (dst / "artifacts").mkdir()
    return dst / "store.sqlite3", dst / "artifacts"


# --------------------------------------------------------------------------- 内部工具

def _pk(evidence_id: str) -> int:
    if not (evidence_id.startswith("E") and evidence_id[1:].isdigit() and int(evidence_id[1:]) > 0):
        raise PerturbationError(f"不是合法的 E 编号: {evidence_id!r}")
    return int(evidence_id[1:])


def _connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _assert_exists(connection: sqlite3.Connection, evidence_id: str) -> sqlite3.Row:
    row = connection.execute("SELECT * FROM evidence WHERE id=?", (_pk(evidence_id),)).fetchone()
    if row is None:
        raise PerturbationError(f"{evidence_id} 不存在")
    return row


def _referenced_by_findings(connection: sqlite3.Connection, evidence_ids: set[str]) -> list[str]:
    hits = []
    for row in connection.execute("SELECT finding_id, payload_json FROM findings"):
        cited = set(json.loads(row["payload_json"]).get("evidence_ids", ()))
        if cited & evidence_ids:
            hits.append(str(row["finding_id"]))
    return hits


def _derived_closure(connection: sqlite3.Connection, roots: set[str]) -> set[str]:
    """返回从 roots 派生出来的全部 E（不含 roots 本身）。"""

    result: set[str] = set()
    frontier = set(roots)
    rows = connection.execute("SELECT id, derived_from_json FROM evidence").fetchall()
    graph: dict[str, list[str]] = {}
    for row in rows:
        for parent in json.loads(row["derived_from_json"]):
            graph.setdefault(parent, []).append(f"E{int(row['id'])}")
    while frontier:
        nxt: set[str] = set()
        for parent in frontier:
            for child in graph.get(parent, ()):
                if child not in result and child not in roots:
                    result.add(child)
                    nxt.add(child)
        frontier = nxt
    return result


def _is_initial_material(connection: sqlite3.Connection, evidence_id: str) -> bool:
    """用户给的原始材料（V0/I0/…）登记成的 E 不允许遮蔽——那是题目本身。"""

    row = connection.execute("SELECT source_ref, acquired_by FROM evidence WHERE id=?", (_pk(evidence_id),)).fetchone()
    source = str(row["source_ref"] or "")
    # 运行时用 HARNESS/ingest 把用户材料（source_ref 形如 V0/I0/A0/D0）登记为 E。
    return bool(source) and source[0] in "VIAD" and source[1:].isdigit() and str(row["acquired_by"]) == "HARNESS/ingest"


def _healthy(db_path: Path, artifact_root: Path) -> None:
    """扰动后用 GlobalStore 自己的健康检查确认副本仍然一致。"""

    store = GlobalStore(db_path, artifact_root)
    health = store.healthcheck()
    if health.status != "ok":
        raise PerturbationError(f"扰动后的副本未通过健康检查: {health.to_dict()}")


# --------------------------------------------------------------------------- 三种扰动

def mask(db_path: str | Path, artifact_root: str | Path, evidence_ids: list[str] | tuple[str, ...], *,
         allow_cited: bool = False) -> PerturbationRecord:
    """遮蔽若干 E：内容换成 MASK_NOTICE，artifact 解绑，派生 E 一并遮蔽。

    期望：重做后的报告不得再把这些 E 当作支持；若原报告完全依赖它们，状态应变 INCOMPLETE。
    """

    db, art = Path(db_path), Path(artifact_root)
    targets = {eid for eid in evidence_ids}
    if not targets:
        raise PerturbationError("mask 至少需要一个 E")
    with _connect(db) as connection:
        for eid in targets:
            _assert_exists(connection, eid)
            if _is_initial_material(connection, eid):
                raise PerturbationError(f"{eid} 是用户原始材料，不能遮蔽")
        cascaded = _derived_closure(connection, targets)
        all_ids = targets | cascaded
        cited = _referenced_by_findings(connection, all_ids)
        if cited and not allow_cited:
            raise PerturbationError(f"这些 E 已被 Finding 引用（{', '.join(cited)}）；扰动应作用在报告提交前的状态。"
                                    " 如确需继续，传 allow_cited=True")
        for eid in sorted(all_ids, key=_pk):
            connection.execute(
                "UPDATE evidence SET content=?, label=?, artifact_id=NULL, modalities_json=?, production_method=? WHERE id=?",
                (MASK_NOTICE, "[已遮蔽的证据]", json.dumps(["text"]), "masked_for_contrast", _pk(eid)),
            )
            # 解绑依赖 artifact 的关联表，避免核验器把遮蔽的 E 当图片去读。
            connection.execute("DELETE FROM search_image_uploads WHERE evidence_id=?", (_pk(eid),))
        connection.commit()
    _healthy(db, art)
    return PerturbationRecord(
        kind="mask", target_ids=tuple(sorted(targets, key=_pk)), cascaded_ids=tuple(sorted(cascaded, key=_pk)),
        expected=(
            Expectation("finding_must_not_cite", {"evidence_ids": sorted(all_ids, key=_pk)},
                        if_violated=ErrorType.UNSUPPORTED_ASSERTION),
        ),
        detail=f"遮蔽 {sorted(targets, key=_pk)}，连带 {sorted(cascaded, key=_pk)}",
        store_dir=str(db.parent),
    )


def replace(db_path: str | Path, artifact_root: str | Path, evidence_id: str, new_content: str, *,
            new_label: str | None = None, contradicts: bool = True, allow_cited: bool = False) -> PerturbationRecord:
    """把一条 E 的内容换成 new_content（通常是结论相反或口径不同的版本）。

    contradicts=True 时期望结论应随之改变；False 表示只是口径变化，期望引用保持但数值应更新。
    """

    db, art = Path(db_path), Path(artifact_root)
    if not new_content.strip():
        raise PerturbationError("new_content 不能为空")
    with _connect(db) as connection:
        _assert_exists(connection, evidence_id)
        if _is_initial_material(connection, evidence_id):
            raise PerturbationError(f"{evidence_id} 是用户原始材料，不能替换")
        cited = _referenced_by_findings(connection, {evidence_id})
        if cited and not allow_cited:
            raise PerturbationError(f"{evidence_id} 已被 Finding 引用（{', '.join(cited)}）；扰动应作用在报告提交前的状态")
        connection.execute(
            "UPDATE evidence SET content=?, label=?, artifact_id=NULL, production_method=? WHERE id=?",
            (new_content, new_label or new_content[:120], "replaced_for_contrast", _pk(evidence_id)),
        )
        connection.execute("DELETE FROM search_image_uploads WHERE evidence_id=?", (_pk(evidence_id),))
        connection.commit()
    _healthy(db, art)
    expected = (Expectation("answer_should_change", {}, if_violated=ErrorType.WRONG_CITATION),) if contradicts else (
        Expectation("finding_must_cite_any", {"evidence_ids": [evidence_id]}, if_violated=ErrorType.SLOPPY_NUMERIC),)
    return PerturbationRecord(
        kind="replace", target_ids=(evidence_id,), cascaded_ids=(), expected=expected,
        detail=f"替换 {evidence_id} 的内容（{'结论相反' if contradicts else '口径变化'}）", store_dir=str(db.parent),
    )


def inject(db_path: str | Path, artifact_root: str | Path, noise: list[dict[str, Any]], *,
           protected_ids: list[str] | tuple[str, ...] = ()) -> PerturbationRecord:
    """往证据库塞干扰 E。noise 每项至少含 content，可选 label / source_ref / locator。

    期望：结论与原有引用集合都不变；若新 E 被引用，即 WRONG_CITATION。
    """

    db, art = Path(db_path), Path(artifact_root)
    if not noise:
        raise PerturbationError("inject 至少需要一条干扰证据")
    new_ids: list[str] = []
    with _connect(db) as connection:
        for item in noise:
            content = str(item.get("content", "")).strip()
            if not content:
                raise PerturbationError("干扰证据的 content 不能为空")
            cursor = connection.execute(
                """INSERT INTO evidence(modalities_json,label,content,source_ref,derived_from_json,artifact_id,
                   locator,acquired_by,discovered_by,production_method) VALUES (?,?,?,?,?,NULL,?,?,?,?)""",
                (json.dumps(list(item.get("modalities", ["text"]))), item.get("label") or content[:120], content,
                 item.get("source_ref"), "[]", item.get("locator", "paragraph:1"),
                 item.get("acquired_by", "SEARCH/injected_noise"), item.get("discovered_by", "W0"),
                 "injected_noise"),
            )
            new_ids.append(f"E{cursor.lastrowid}")
        connection.commit()
    _healthy(db, art)
    return PerturbationRecord(
        kind="inject", target_ids=tuple(new_ids), cascaded_ids=(),
        expected=(
            Expectation("finding_must_not_cite", {"evidence_ids": new_ids}, if_violated=ErrorType.WRONG_CITATION),
            Expectation("answer_unchanged", {}, if_violated=ErrorType.PREMATURE_FINISH),
            *((Expectation("citations_unchanged", {"evidence_ids": list(protected_ids)},
                           if_violated=ErrorType.EVIDENCE_DROPPED),) if protected_ids else ()),
        ),
        detail=f"注入 {len(new_ids)} 条干扰证据 {new_ids}", store_dir=str(db.parent),
    )


# --------------------------------------------------------------------------- 期望判定

def evaluate(record: PerturbationRecord, *, report: dict[str, Any] | None = None,
             answer_changed: bool | None = None) -> list[dict[str, Any]]:
    """把重做后的结果对照期望，返回每条期望的判定 {kind, ok, error_type, detail}。

    report：重做后的 Workspace Report（dict，含 status 与 findings[].evidence_ids）。
    answer_changed：最终结论是否变化（由调用方比较前后 FINISH.answer 得出）。
    没有给到相应输入的期望返回 ok=None（未判定）。
    """

    cited: set[str] = set()
    if report:
        for finding in report.get("findings", ()):
            cited.update(finding.get("evidence_ids", ()))
    results = []
    for exp in record.expected:
        ok: bool | None
        if exp.kind == "finding_must_not_cite":
            ok = None if report is None else not (cited & set(exp.params["evidence_ids"]))
        elif exp.kind == "finding_must_cite_any":
            ok = None if report is None else bool(cited & set(exp.params["evidence_ids"]))
        elif exp.kind == "status_should_be":
            ok = None if report is None else report.get("status") == exp.params["status"]
        elif exp.kind == "citations_unchanged":
            ok = None if report is None else set(exp.params["evidence_ids"]) <= cited
        elif exp.kind == "answer_unchanged":
            ok = None if answer_changed is None else not answer_changed
        elif exp.kind == "answer_should_change":
            ok = None if answer_changed is None else answer_changed
        else:  # pragma: no cover
            raise PerturbationError(f"未知期望 {exp.kind}")
        results.append({"kind": exp.kind, "ok": ok,
                        "error_type": (exp.if_violated.value if (ok is False and exp.if_violated) else None),
                        "detail": exp.params})
    return results


__all__ = ["Expectation", "MASK_NOTICE", "PerturbationError", "PerturbationRecord", "evaluate", "inject", "mask",
           "replace", "snapshot"]
