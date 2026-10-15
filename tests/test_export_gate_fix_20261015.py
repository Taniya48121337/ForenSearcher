"""10-15 对话 6（export 与门小修）：B7 理由出处（S3 / S1 槽 → walker_slot）与门的分母、C6 频道只比训练案卷元数据（整词、词典词）、
B2 抽查不足 20 条时写明、A10 冻结文件的变体 overrides（freeze_check / recheck_freeze / pack_r0 run_freeze）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from foren_searcher.datagen.export import walker_slots
from foren_searcher.datagen.run_cases import allowed_variant, freeze_check, recheck_freeze
from foren_searcher.datagen.runlog import Decision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_gate  # noqa: E402
import pack_r0  # noqa: E402

from test_export_gate_20261015 import _gate, _row, _write  # noqa: E402

V9 = ["max_evidence_cards=6", "max_history_entries=4"]


# --------------------------------------------------------------------------- B7：export 标注

def _step(output: dict) -> Decision:
    return Decision(request_id="request_0008", role="HI", actor_id="W1-1", turn=3, round_index=None, kind="step", output=output,
                    status="executed", system_prompt="sys", payload={}, image_evidence_ids=[], image_notes=[], image_paths=[])


def _request(tmp: Path, slots: dict[str, dict], kind: str = "action") -> SimpleNamespace:
    rdir = tmp / "requests" / "request_0008"
    rdir.mkdir(parents=True)
    (rdir / "transport.json").write_text(json.dumps({"transport": "oracle_walker", "oracle": {"kind": kind, "rule": "C5_read_page"}}),
                                         encoding="utf-8")
    for name, body in slots.items():
        (rdir / name).write_text(json.dumps(body), encoding="utf-8")
    return SimpleNamespace(run_dir=tmp)


def test_s3_and_s1_slot_rationale_is_walker_slot(tmp_path: Path) -> None:
    said = "Reading E4 to confirm the 2018 context."
    out = {"action": "READ_RECORD", "arguments": {"record_id": "E4"}, "rationale": said}
    s3 = walker_slots(_request(tmp_path / "a", {"slot_S3_next_choice.json": {"next": "read", "rationale": said}}), _step(out))
    assert (s3["rationale_source"], s3["rationale_slot"]) == ("walker_slot", "s3_teacher")
    s1 = walker_slots(_request(tmp_path / "b", {"slot_S1_query_choice.json": {"query": "q", "rationale": said}}), _step(out))
    assert (s1["rationale_source"], s1["rationale_slot"]) == ("walker_slot", "s1_teacher")


def test_template_when_slot_rationale_not_used_or_missing(tmp_path: Path) -> None:
    # S3 写了理由但动作走了程序兜底（C4_search_fallback）：导出的是模板句，仍记 walker_template
    out = {"action": "SEARCH", "arguments": {}, "rationale": "Search for reports of the event the claim describes."}
    fallback = walker_slots(_request(tmp_path / "a", {"slot_S3_next_choice.json": {"rationale": "Something else."}}), _step(out))
    assert fallback["rationale_source"] == "walker_template" and "rationale_slot" not in fallback
    none = walker_slots(_request(tmp_path / "b", {}), _step(out))
    assert none["rationale_source"] == "walker_template"
    # S0 槽文件优先
    s0 = walker_slots(_request(tmp_path / "c", {"slot_S0_rationale.json": {"used": True},
                                                "slot_S3_next_choice.json": {"rationale": out["rationale"]}}), _step(out))
    assert s0["rationale_source"] == "s0_teacher" and "rationale_slot" not in s0


# --------------------------------------------------------------------------- B7：门

def _b7_rows() -> list[dict]:
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo", "lima"]

    def step(i: int, **meta) -> dict:     # 数字会被归一成 #，各句用不同的词，免得撞"模板句 ≤ 5 次"
        return _row(output={"action": "INSPECT", "arguments": {"targets": ["E1"]},
                            "rationale": f"Check the {words[i % 12]} {words[(i // 10) % 12]} detail in E1 now."}, turn=i, **meta)
    rows = [step(i, rationale_source="s0_teacher") for i in range(5)]
    rows += [step(10 + i, rationale_source="walker_slot", rationale_slot="s3_teacher") for i in range(3)]
    rows += [step(20, rationale_source="walker_slot", rationale_slot="s1_teacher"), step(30, rationale_source="walker_template")]
    rows.append(_row(unit="report", output={"status": "COMPLETED", "findings": [{"statement": "s", "evidence_ids": ["E1"]}]},
                     rationale_source="walker_slot"))      # 报告：walker_slot 但不是动作行，不进分母
    return rows


def test_b7_counts_s3_s1_slot_rationales_as_teacher(tmp_path: Path) -> None:
    b7 = _gate(_write(tmp_path, _b7_rows()))["B7"]
    assert b7["value"]["walker_step_rows"] == 10 and b7["value"]["teacher_rationale_rows"] == 9
    assert b7["value"]["s0_adoption"] == 0.9 and b7["value"]["s0_slot_only"] == 0.5
    assert b7["value"]["by_source"] == {"s0_teacher": 5, "s3_teacher": 3, "s1_teacher": 1, "walker_template": 1}
    assert b7["status"] == "通过"


def test_b7_fails_below_threshold(tmp_path: Path) -> None:
    rows = _b7_rows()
    rows[5]["metadata"].update(rationale_source="walker_template", rationale_slot=None)     # 一条 S3 改回模板：8/10
    b7 = _gate(_write(tmp_path, rows))["B7"]
    assert b7["value"]["s0_adoption"] == 0.8 and b7["status"] == "不通过"


# --------------------------------------------------------------------------- B2

def test_b2_short_sample_is_stated(tmp_path: Path) -> None:
    rows = [_row(turn=i, hidden_context=["Kisumu"], hidden_context_released={"Kisumu": "E1"}) for i in range(3)]
    b2 = _gate(_write(tmp_path, rows))["B2"]
    assert b2["value"]["review_sample"] == 3 and b2["value"]["review_sample_short"] is True
    assert "不足 20" in b2["note"]
    assert "本次只有 3 条，不足 20 条" in (tmp_path / "gate_b2_review.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- C6 频道

def _c6_setup(tmp: Path, *, notes: dict[str, str], uploader: dict[str, str] | None = None) -> tuple[list[dict], argparse.Namespace]:
    (tmp / "cases").mkdir()
    for name, text in notes.items():
        case = {"case_id": f"fmnv:{name}", "dataset": "fmnv", "source_id": name, "split": "dev_train", "family_key": f"fmnv:{name}",
                "media": [], "notes": text}
        if uploader and name in uploader:
            case["source_meta"] = {"uploader": uploader[name]}
        (tmp / "cases" / f"fmnv_{name}.json").write_text(json.dumps(case), encoding="utf-8")
    table = tmp / "channels.json"
    table.write_text(json.dumps({"video_131": "Young", "video_132": "Mana", "video_133": "Storyful News & Weather",
                                 "video_134": "Underworld"}), encoding="utf-8")
    # 训练行全文里有 "young soldier""managed"——旧门的子串匹配会报，新门不扫行文
    rows = [{"messages": [{"role": "user", "content": "a young soldier managed the musulmana crowd; Underworld"}], "images": [],
             "metadata": {"bucket": "accepted", "case_id": f"fmnv:{n}", "case_family": f"fmnv:{n}", "split": "dev_train"}} for n in notes]
    args = argparse.Namespace(cases=str(tmp / "cases"), evid_bench=None, evid_cases=None, evid_range=[131, 222],
                              evid_channels=str(table), phash=False, evid_videos=None, phash_max_distance=6)
    return rows, args


def test_c6_channel_ignores_row_text_and_common_words(tmp_path: Path) -> None:
    rows, args = _c6_setup(tmp_path, notes={"a": "D2 案卷，YouTube 原片，一个 young 士兵", "b": ""})
    c6 = export_gate.check_c6(rows, args)
    assert c6["value"]["channel_hits"] == [] and c6["value"]["channel"].startswith("已检查")
    assert c6["value"]["channel_compare"]["train_cases"] == 2


def test_c6_channel_hits_metadata_and_whole_word_notes(tmp_path: Path) -> None:
    rows, args = _c6_setup(tmp_path, notes={"a": "原片来自 storyful news  & weather 频道", "b": "underworldly", "c": "频道 Underworld 上传"},
                           uploader={"b": "Storyful News & Weather"})
    words = tmp_path / "words.txt"
    words.write_text("young\nmana\n", encoding="utf-8")
    args.dictionary = str(words)
    c6 = export_gate.check_c6(rows, args)
    hits = {(h["case"], h["where"], h["channel"]) for h in c6["value"]["channel_hits"]}
    assert hits == {("fmnv:a", "案卷 notes", "storyful news & weather"), ("fmnv:b", "上传者 / 频道字段", "Storyful News & Weather"),
                    ("fmnv:c", "案卷 notes", "underworld")}
    assert c6["status"] == "不通过"
    assert sorted(c6["value"]["channel_compare"]["notes_skipped_names"]) == ["mana", "young"]


def test_c6_without_dictionary_skips_single_word_names_in_notes(tmp_path: Path) -> None:
    rows, args = _c6_setup(tmp_path, notes={"c": "频道 Underworld 上传"}, uploader={"c": "underworld"})
    args.dictionary = str(tmp_path / "missing_words.txt")
    dictionary, _ = export_gate._load_dictionary(args.dictionary)
    c6 = export_gate.check_c6(rows, args)
    where = [h["where"] for h in c6["value"]["channel_hits"]]
    if dictionary:                          # 本机有系统词典时 underworld 可能是词典词；结构化字段照样命中
        assert "上传者 / 频道字段" in where
    else:
        assert where == ["上传者 / 频道字段"]


# --------------------------------------------------------------------------- A10 冻结变体

FREEZE = {"freeze_id": "r0", "spec_hash": {"hi": "h", "gia": "g"}, "tool_layer": {"hash": "t"},
          "checks": {"harness_overrides": [], "state_renderer": "compact_16k"}, "allowed_variant_overrides": {"v9": V9}}


def _actual(overrides: list[str]) -> dict:
    return {"spec_hash.hi": "h", "spec_hash.gia": "g", "tool_layer_hash": "t", "harness_overrides": overrides, "state_renderer": "compact_16k"}


def test_freeze_check_accepts_declared_variant_only() -> None:
    assert freeze_check(FREEZE, _actual([]))["ok"] and "variant" not in freeze_check(FREEZE, _actual([]))
    v9 = freeze_check(FREEZE, _actual(list(reversed(V9))))
    assert v9["ok"] and v9["variant"] == "v9" and v9["items"]["harness_overrides"]["variant"] == "v9"
    for other in (V9 + ["image_input_limit=16"], V9[:1], ["max_evidence_cards=8", "max_history_entries=4"]):
        res = freeze_check(FREEZE, _actual(other))
        assert not res["ok"] and res["failed"] == ["harness_overrides"] and "variant" not in res
    assert allowed_variant({"checks": {}}, V9) is None


def test_recheck_freeze_upgrades_old_v9_records_only() -> None:
    old = freeze_check({**FREEZE, "allowed_variant_overrides": {}}, _actual(V9))          # 跑时冻结文件还没声明变体
    assert not old["ok"]
    new = recheck_freeze(old, FREEZE)
    assert new["ok"] and new["variant"] == "v9" and new["failed"] == [] and new["rechecked"]
    both = freeze_check({**FREEZE, "allowed_variant_overrides": {}}, {**_actual(V9), "tool_layer_hash": "other"})
    assert recheck_freeze(both, FREEZE)["failed"] == ["tool_layer_hash"] and not recheck_freeze(both, FREEZE)["ok"]
    stray = freeze_check({**FREEZE, "allowed_variant_overrides": {}}, _actual(["image_input_limit=16"]))
    assert recheck_freeze(stray, FREEZE) is stray
    assert recheck_freeze(old, {**FREEZE, "freeze_id": "r1"}) is old
    assert recheck_freeze({"applies": False}, FREEZE) == {"applies": False}


def test_pack_r0_run_freeze_accepts_v9(tmp_path: Path) -> None:
    old = freeze_check({**FREEZE, "allowed_variant_overrides": {}}, _actual(V9))
    for run, record in (("vcsl_pair_5_v9_r0a_1", old),
                        ("other_r0a_1", freeze_check({**FREEZE, "allowed_variant_overrides": {}}, _actual(["image_input_limit=16"])))):
        (tmp_path / run).mkdir()
        (tmp_path / run / "run_config.json").write_text(json.dumps({"freeze": record}), encoding="utf-8")
    prov = {p["run"]: p for p in pack_r0.run_provenance({"vcsl_pair_5_v9_r0a_1", "other_r0a_1"}, tmp_path, FREEZE)}
    assert prov["vcsl_pair_5_v9_r0a_1"]["freeze_ok"] is True and prov["vcsl_pair_5_v9_r0a_1"]["freeze_variant"] == "v9"
    assert prov["other_r0a_1"]["freeze_ok"] is False and prov["other_r0a_1"]["freeze_variant"] is None


R0A_V9 = [ROOT / "experiments" / "sft_data" / "runs" / name / "run_config.json" for name in ("vcsl_pair_5_v9_r0a_1", "videosham_1028_v9_r0a_1")]


@pytest.mark.skipif(not all(p.is_file() for p in R0A_V9), reason="需要本机 r0a 的两条 V9 run")
def test_real_r0a_v9_runs_pass_with_current_freeze_file() -> None:
    frozen = json.loads((ROOT / "specs" / "r0_freeze.json").read_text(encoding="utf-8"))
    assert frozen["allowed_variant_overrides"] == {"v9": V9}
    for path in R0A_V9:
        record = json.loads(path.read_text(encoding="utf-8"))["freeze"]
        assert record["ok"] is False and record["failed"] == ["harness_overrides"]        # 跑时记录不改
        rechecked = recheck_freeze(record, frozen)
        assert rechecked["ok"] is True and rechecked["variant"] == "v9"
