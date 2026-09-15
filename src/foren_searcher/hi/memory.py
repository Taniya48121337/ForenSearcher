"""HI 的局部工作记忆：证据视图（EvidenceViewBuilder）。

从原 memory.py 拆出（2026-09-15 拆包）。RecordAccess、MemoryPolicy 等两个角色共用的
记录访问逻辑在 store/records.py，这里只放 HI 专属的部分，并为方便重新导出常用名字。
"""

from __future__ import annotations

from typing import Any, Iterable

from ..store.global_store import GlobalStore
from ..store.records import (  # noqa: F401  (re-export for hi.workspace)
    MemoryPolicy,
    RecordAccess,
    RecordPage,
    _evidence_index,
    _evidence_index_card,
    _relevance,
    compact_reference,
    trim_visual_observation_summaries,
    workspace_visual_observations,
)

class EvidenceViewBuilder:
    """为一个 HI Workspace 选择本轮最值得放进上下文的 Evidence 卡片。"""

    def __init__(self, store: GlobalStore, policy: MemoryPolicy) -> None:
        """保存 Store 与容量策略。"""

        self.store = store
        self.policy = policy

    def build(
        self,
        *,
        task: str,
        workspace_id: str,
        explicit_material_ids: Iterable[str],
        focus_evidence_ids: Iterable[str],
        records: Iterable[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """按显式材料、focus、任务相关性、本 Workspace 发现和新旧程度排序。"""

        # set 使后续成员检查为常数时间，并自然忽略重复输入。
        explicit = set(explicit_material_ids)
        focus = set(focus_evidence_ids)
        ranked: list[tuple[int, int, float, int, int, dict[str, Any]]] = []
        # 测试或调用方可以传入同一时点快照；否则直接读取 Store 当前索引。
        evidence_records = (
            tuple(records) if records is not None else self.store.evidence_index_records()
        )
        for evidence in evidence_records:
            ranked.append(
                (
                    int(evidence["evidence_id"] in explicit),
                    int(evidence["evidence_id"] in focus),
                    _relevance(
                        task, evidence["label"], evidence["content_excerpt"]
                    ),
                    int(evidence["discovered_by"] == workspace_id),
                    _evidence_index(evidence["evidence_id"]),
                    evidence,
                )
            )
        # 前五项就是上述五级排序键；最后一项保留原记录用于构造卡片。
        ranked.sort(key=lambda item: item[:5], reverse=True)
        return tuple(
            _evidence_index_card(item[5])
            for item in ranked[: self.policy.max_evidence_cards]
        )

