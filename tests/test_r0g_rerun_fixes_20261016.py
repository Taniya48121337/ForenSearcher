"""10-16 r0g 收尾：上一对话（5''）待重跑的 18 条 run 对应的三处 walker 修复——L3b 改读 E1、每工作区一次（门 B8）；兜底搜索
标 substituted、由 S0 写理由（门 B7）；纯视觉任务不把音轨列为缺口（videosham_4126 退步）。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "experiments/sft_data"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

from foren_searcher.datagen import walker as w  # noqa: E402


# test_l3b_reads_given_photo_once_per_workspace → tests/test_r0plus_20261016.py::test_l3b_is_the_step_right_before_report
# （10-16 r0+：L3b 改成交报告前的最后一步，_view_refs_before_report 换成 _l3b_before_report）


def test_search_fallback_is_substituted():
    report = w.Decision("report", "C9_report", {"task_id": "W1-1", "status": "INCOMPLETE"})
    slot = w.Decision("next_slot", "C4_next_step", {}, slot={"name": "S3_next", "choices": ["search", "report"],
                      "unread": [], "undated": [], "offsets": {}, "previous": [],
                      "claim": "Video shows flooding in the city centre last week", "report_decision": report.to_dict()})
    got = w.resolve_slot_choice(slot, {"next": "read", "evidence_id": "E9", "rationale": "Read E9."})
    assert got.rule == "C4_search_fallback" and got.slot.get("substituted")
    assert got.output["arguments"]["query"].startswith("Video shows flooding")


def test_visual_only_task_lists_no_audio_gap():
    pa, pb = {"audio": 1}, {"audio": 0}
    visual = SimpleNamespace(task="Determine every time interval during which the person in black is visible in B but not in A.",
                             inspect_ops={"asr"})
    assert not [m for m in w._missing_parts(visual, {}, pa, pb, []) if "audio" in m]
    sound = SimpleNamespace(task="Compare A with B, including whether the speech or sound differs.", inspect_ops={"asr"})
    gaps = [m for m in w._missing_parts(sound, {}, pa, pb, []) if "audio" in m]
    assert any("no audio track" in m for m in gaps) and any("not transcribed" in m for m in gaps)
