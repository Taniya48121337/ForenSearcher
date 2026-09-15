"""GlobalStore 基础引用与 Workspace Report 测试。

验证 C/E/F/W 全局命名空间、引用解析、防御性拷贝以及报告原子发布规则。
"""

from __future__ import annotations

import unittest

from foren_searcher.contracts import Finding, WorkspaceReport
from foren_searcher.store import GlobalStore


class GlobalStoreTest(unittest.TestCase):
    """测试辅助类 GlobalStoreTest；用可控状态替代真实外部依赖。"""

    def setUp(self) -> None:
        """测试辅助函数 setUp；为用例提供可重复的输入或可控行为。"""

        self.store = GlobalStore()
        self.store.add_material("V0", {"path": "input.mp4"})

    def test_material_resolution_returns_an_isolated_copy(self) -> None:
        """验证 material resolution returns an isolated copy 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

        resolved = self.store.resolve("V0")
        resolved["path"] = "changed.mp4"
        self.assertEqual(self.store.resolve("V0"), {"path": "input.mp4"})

    def test_tool_evidence_can_reference_prior_evidence(self) -> None:
        """验证 tool evidence can reference prior evidence 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

        first = self.store.record_evidence(
            content="00:08出现XX NEWS文字。",
            source_ref="V0",
            locator="V0@00:08",
            acquired_by="INSPECT/ocr",
            discovered_by="W1-1",
        )
        second = self.store.record_evidence(
            content="候选画面与E1中的画面相同。",
            source_ref=None,
            locator="V0@00:08 ↔ R1@00:03",
            acquired_by="INSPECT/align_media",
            discovered_by="W1-1",
            derived_from=(first.evidence_id,),
        )
        self.assertEqual(second.derived_from, ("E1",))

    def test_finding_must_reference_existing_evidence(self) -> None:
        """验证 finding must reference existing evidence 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

        with self.assertRaisesRegex(KeyError, "unknown Evidence ID"):
            self.store.add_finding(
                Finding(
                    finding_id="F1-1",
                    workspace_id="W1-1",
                    statement="视频早于Claim中的事件出现。",
                    evidence_ids=("E99",),
                )
            )

    def test_workspace_report_is_resolvable_by_task_id(self) -> None:
        """验证 workspace report is resolvable by task id 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

        evidence = self.store.record_evidence(
            content="页面日期为2022-04-18。",
            source_ref="https://example.test/video",
            locator="发布时间字段",
            acquired_by="SEARCH/open",
            discovered_by="W1-1",
        )
        finding = Finding(
            finding_id="F1-1",
            workspace_id="W1-1",
            statement="视频至少在2022年已经出现。",
            evidence_ids=(evidence.evidence_id,),
        )
        report = WorkspaceReport(
            task_id="W1-1",
            status="INCOMPLETE",
            findings=(finding,),
            selected_evidence_ids=(evidence.evidence_id,),
            status_reason="不能确认最初上传者。",
        )

        self.store.add_workspace_report(report)

        self.assertEqual(self.store.resolve("F1-1"), finding)
        self.assertEqual(self.store.resolve("W1-1"), report)

    def test_duplicate_material_id_is_rejected(self) -> None:
        """验证 duplicate material id is rejected 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

        with self.assertRaisesRegex(ValueError, "duplicate global reference ID"):
            self.store.add_material("V0", {"path": "other.mp4"})

    def test_global_reference_ids_cannot_shadow_other_record_types(self) -> None:
        """验证 global reference ids cannot shadow other record types 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

        self.store.add_material("E1", {"kind": "badly_named_material"})
        with self.assertRaisesRegex(ValueError, "duplicate global reference ID"):
            self.store.record_evidence(
                content="可定位观察。",
                source_ref="V0",
                locator="V0@00:01",
                acquired_by="INSPECT/ocr",
                discovered_by="W1-1",
            )

    def test_invalid_report_does_not_partially_add_findings(self) -> None:
        """验证 invalid report does not partially add findings 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

        evidence = self.store.record_evidence(
            content="页面日期为2022-04-18。",
            source_ref="https://example.test/video",
            locator="发布时间字段",
            acquired_by="SEARCH/open",
            discovered_by="W1-1",
        )
        valid_finding = Finding(
            finding_id="F1-1",
            workspace_id="W1-1",
            statement="视频至少在2022年已经出现。",
            evidence_ids=(evidence.evidence_id,),
        )
        invalid_finding = Finding(
            finding_id="F1-2",
            workspace_id="W1-1",
            statement="这条判断没有真实依据。",
            evidence_ids=("E99",),
        )
        report = WorkspaceReport(
            task_id="W1-1",
            status="INCOMPLETE",
            findings=(valid_finding, invalid_finding),
            selected_evidence_ids=(evidence.evidence_id,),
        )

        with self.assertRaises(KeyError):
            self.store.add_workspace_report(report)
        self.assertFalse(self.store.has_reference("F1-1"))
        self.assertFalse(self.store.has_reference("W1-1"))

    def test_workspace_report_status_is_only_completed_or_incomplete(self) -> None:
        """验证 workspace report status is only completed or incomplete 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

        report = WorkspaceReport(
            task_id="W1-1",
            status="failed",  # type: ignore[arg-type]
            findings=(),
            selected_evidence_ids=(),
            status_reason="工具执行失败。",
        )

        with self.assertRaisesRegex(ValueError, "unknown workspace status"):
            self.store.add_workspace_report(report)

    def test_status_reason_cannot_be_blank(self) -> None:
        """验证 status reason cannot be blank 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

        report = WorkspaceReport(
            task_id="W1-1",
            status="INCOMPLETE",
            findings=(),
            selected_evidence_ids=(),
            status_reason="   ",
        )

        with self.assertRaisesRegex(ValueError, "status_reason"):
            self.store.add_workspace_report(report)


if __name__ == "__main__":
    unittest.main()
