"""核心领域契约测试。

覆盖 E/F/W 对象序列化、GIA/HI 决策严格字段、引用格式和非法组合拒绝行为。
"""

from __future__ import annotations

import unittest

from foren_searcher.contracts import (
    Evidence,
    Finding,
    GiaFinishDecision,
    GiaInvestigateDecision,
    HiActionDecision,
    WorkspaceReport,
    WorkspaceReportDraft,
    parse_hi_decision,
    parse_workspace_report,
    parse_gia_decision,
)
from foren_searcher.ids import (
    assign_workspace_tasks,
    format_workspace_id,
)


class GiaContractTest(unittest.TestCase):
    """测试辅助类 GiaContractTest；用可控状态替代真实外部依赖。"""

    def test_investigate_contains_only_task_and_materials(self) -> None:
        """验证 investigate contains only task and materials 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

        decision = parse_gia_decision(
            {
                "decision": "INVESTIGATE",
                "tasks": [
                    {
                        "task": "找到待核查视频能够确认的最早公开发布时间。",
                        "materials": ["V0", "C0"],
                    },
                    {
                        "task": "核查Claim所说的现实事件是否发生。",
                        "materials": ["C0"],
                    },
                ],
            }
        )

        self.assertIsInstance(decision, GiaInvestigateDecision)
        self.assertEqual(decision.to_dict()["tasks"][0]["materials"], ["V0", "C0"])

    def test_gia_cannot_assign_task_id(self) -> None:
        """验证 gia cannot assign task id 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

        with self.assertRaisesRegex(ValueError, "task 1 fields"):
            parse_gia_decision(
                {
                    "decision": "INVESTIGATE",
                    "tasks": [
                        {
                            "task_id": "W1-1",
                            "task": "调查视频来源。",
                            "materials": ["V0"],
                        }
                    ],
                }
            )

    def test_finish_contains_only_answer(self) -> None:
        """验证 finish contains only answer 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

        decision = parse_gia_decision(
            {
                "decision": "FINISH",
                "answer": "该视频至少在2022年已经公开出现[E17,E31]。",
            }
        )

        self.assertIsInstance(decision, GiaFinishDecision)
        self.assertEqual(
            decision.to_dict(),
            {
                "decision": "FINISH",
                "answer": "该视频至少在2022年已经公开出现[E17,E31]。",
            },
        )

    def test_unconfirmed_gia_fields_are_not_accepted(self) -> None:
        """验证 unconfirmed gia fields are not accepted 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

        with self.assertRaisesRegex(ValueError, "INVESTIGATE fields"):
            parse_gia_decision(
                {
                    "decision": "INVESTIGATE",
                    "tasks": [{"task": "调查来源。", "materials": ["V0"]}],
                    "decision_reason": "需要更多证据",
                }
            )


class HarnessIdTest(unittest.TestCase):
    """测试辅助类 HarnessIdTest；用可控状态替代真实外部依赖。"""

    def test_workspace_ids_are_assigned_by_round_and_order(self) -> None:
        """验证 workspace ids are assigned by round and order 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

        decision = parse_gia_decision(
            {
                "decision": "INVESTIGATE",
                "tasks": [
                    {"task": "任务一。", "materials": ["V0"]},
                    {"task": "任务二。", "materials": ["C0"]},
                ],
            }
        )
        self.assertIsInstance(decision, GiaInvestigateDecision)

        assigned = assign_workspace_tasks(2, decision.tasks)

        self.assertEqual([task.task_id for task in assigned], ["W2-1", "W2-2"])
        self.assertEqual(assigned[0].materials, ("V0",))

    def test_workspace_indices_must_be_positive(self) -> None:
        """验证 workspace indices must be positive 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

        with self.assertRaises(ValueError):
            format_workspace_id(0, 1)
        with self.assertRaises(ValueError):
            format_workspace_id(1, 0)

class EvidenceAndReportTest(unittest.TestCase):
    """测试辅助类 EvidenceAndReportTest；用可控状态替代真实外部依赖。"""

    def test_evidence_and_finding_have_separate_shapes(self) -> None:
        """验证 evidence and finding have separate shapes 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

        evidence = Evidence(
            evidence_id="E17",
            content="页面显示发布日期为2022-04-18。",
            source_ref="https://example.test/video",
            locator="页面发布时间字段",
            acquired_by="SEARCH/open",
            discovered_by="W1-1",
        )
        finding = Finding(
            finding_id="F1-1",
            workspace_id="W1-1",
            statement="待核查视频至少在2022年已经公开出现。",
            evidence_ids=("E17", "E31"),
        )

        self.assertIn("source_ref", evidence.to_dict())
        self.assertNotIn("source", finding.to_dict())
        self.assertEqual(finding.to_dict()["evidence_ids"], ["E17", "E31"])

    def test_workspace_report_keeps_findings_and_evidence_ids_distinct(self) -> None:
        """验证 workspace report keeps findings and evidence ids distinct 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

        report = WorkspaceReport(
            task_id="W1-1",
            status="INCOMPLETE",
            findings=(
                Finding(
                    finding_id="F1-1",
                    workspace_id="W1-1",
                    statement="视频至少在2022年已经出现。",
                    evidence_ids=("E17",),
                    limitations=("只能确认最晚公开时间。",),
                ),
            ),
            selected_evidence_ids=("E17",),
            status_reason="尚不能确认最初上传者。",
        )

        payload = report.to_dict()
        self.assertEqual(payload["findings"][0]["finding_id"], "F1-1")
        self.assertEqual(payload["selected_evidence_ids"], ["E17"])
        self.assertNotIn("unresolved", payload)
        self.assertEqual(payload["status_reason"], "尚不能确认最初上传者。")

    def test_completed_report_omits_optional_status_reason(self) -> None:
        """验证 completed report omits optional status reason 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

        report = WorkspaceReport(
            task_id="W1-1",
            status="COMPLETED",
            findings=(),
            selected_evidence_ids=(),
        )

        self.assertNotIn("status_reason", report.to_dict())

    def test_hi_action_has_action_specific_arguments_and_focus_snapshot(self) -> None:
        """验证 hi action has action specific arguments and focus snapshot 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

        decision = parse_hi_decision(
            {
                "action": "inspect",
                "arguments": {
                    "targets": ["E1"],
                    "request": {"operation": "ocr", "params": {}},
                },
                "focus_evidence_ids": ["E1"],
            }
        )
        self.assertIsInstance(decision, HiActionDecision)
        self.assertEqual(decision.action, "INSPECT")
        self.assertEqual(decision.focus_evidence_ids, ("E1",))

    def test_workspace_report_is_a_direct_hi_terminal_output(self) -> None:
        """验证 workspace report is a direct hi terminal output 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

        report = parse_workspace_report(
            {
                "task_id": "W1-1",
                "status": "COMPLETED",
                "findings": [
                    {
                        "statement": "页面明确给出发布时间。",
                        "evidence_ids": ["E1"],
                        "limitations": [],
                    }
                ],
                "selected_evidence_ids": ["E1"],
            }
        )
        self.assertIsInstance(report, WorkspaceReportDraft)
        self.assertNotIn("action", report.to_dict())

    def test_completed_report_rejects_status_reason(self) -> None:
        """验证 completed report rejects status reason 场景；准备代码构造条件，末尾断言给出必须成立的结果。"""

        with self.assertRaisesRegex(ValueError, "Workspace Report fields"):
            parse_workspace_report(
                {
                    "task_id": "W1-1",
                    "status": "COMPLETED",
                    "findings": [],
                    "selected_evidence_ids": [],
                    "status_reason": "不应出现",
                }
            )


if __name__ == "__main__":
    unittest.main()
