"""scripts/behavior_metrics.py 与 scripts/eval_r0.py 的离线测试（2026-10-15，r0 评测口径）。

用合成的 run 目录（最小 store.sqlite3 + trajectory.jsonl）核对四个指标的定义：
SEARCH 自动打开的候选不算"打开"；compare_media 指向候选 page E 同时算打开与比对；打开与比对只认模型输出
（store 里 HARNESS 的 INSPECT、--auto-tools on 的自动比对都不算）；
重复 = 同一工作区逐字相同的 HI 动作；格式拒率 = HI、GIA 各自拒率的平均。
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bm = _load("behavior_metrics")
er = _load("eval_r0")


def _store(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE candidates (id INTEGER PRIMARY KEY, identity_key TEXT, source_url TEXT, candidate_type TEXT,
                                 platform_video_id TEXT, final_url TEXT);
        CREATE TABLE candidate_evidence (candidate_id INTEGER, evidence_id INTEGER, material_kind TEXT, material_status TEXT);
        CREATE TABLE inspect_executions (id INTEGER PRIMARY KEY, request_json TEXT, workspace_id TEXT);
        CREATE TABLE search_executions (id INTEGER PRIMARY KEY, request_json TEXT, workspace_id TEXT);
    """)
    # R1 = 原片（GT_SRC），R2、R3 = 其他视频候选；SEARCH 自动打开三个，各有 page E（E11 / E12 / E13）
    con.executemany("INSERT INTO candidates VALUES (?,?,?,?,?,?)", [
        (1, "a", "https://www.youtube.com/watch?v=GT_SRC", "video", "GT_SRC", None),
        (2, "b", "https://www.youtube.com/watch?v=OTHER2", "video", "OTHER2", None),
        (3, "c", "https://www.youtube.com/watch?v=OTHER3", "video", "OTHER3", None),
        (4, "d", "https://example.com/page", "page", None, None)])
    con.executemany("INSERT INTO candidate_evidence VALUES (?,?,?,?)", [
        (1, 11, "page", "ready"), (2, 12, "page", "ready"), (3, 13, "page", "ready"), (1, 21, "video", "ready")])
    con.executemany("INSERT INTO inspect_executions (request_json, workspace_id) VALUES (?,?)", [
        (json.dumps({"targets": ["E1"], "request": {"operation": "get_frames"}}), "HARNESS"),
        (json.dumps({"targets": ["E1", "E13"], "request": {"operation": "compare_media"}}), "HARNESS"),   # 运行时的不算
        (json.dumps({"targets": ["E11"], "request": {"operation": "get_frames"}}), "W1-1"),            # 打开原片（下载出 E21）
        (json.dumps({"targets": ["E1", "E11"], "request": {"operation": "compare_media"}}), "W1-1"),   # 比对原片
        (json.dumps({"targets": ["E12"], "request": {"operation": "ocr"}}), "W1-1")])                  # 打开 R2，不比对
    con.executemany("INSERT INTO search_executions (request_json, workspace_id) VALUES (?,?)", [
        (json.dumps({"input_type": "text", "query": "x", "output_type": "video"}), "W1-1")])
    con.commit()
    con.close()


def _trajectory(path: Path) -> None:
    events = []

    def out(role, actor, action, arguments):
        events.append({"event_type": "raw_model_output", "actor_role": role, "actor_id": actor, "payload": {}})
        events.append({"event_type": "model_output", "actor_role": role, "actor_id": actor,
                       "payload": {"output": {"action": action, "arguments": arguments}}})

    out("HI", "W1-1", "SEARCH", {"input_type": "text", "query": "x", "output_type": "video"})
    out("HI", "W1-1", "SEARCH", {"input_type": "text", "query": "x", "output_type": "video"})   # 逐字重复
    # SEARCH 观察里带运行时自动比对（--auto-tools on）：不算模型比对
    events.append({"event_type": "observation", "actor_role": "HI", "actor_id": "W1-1", "payload": {"observation": {
        "action": "SEARCH", "details": {"auto_compare": {"candidate_id": "R3", "request": {
            "targets": ["E1", "E13"], "request": {"operation": "compare_media", "params": {}}}, "status": "ready"}}}}})
    out("HI", "W1-1", "INSPECT", {"targets": ["E11"], "request": {"operation": "get_frames"}})          # 打开原片 R1
    out("HI", "W1-1", "INSPECT", {"targets": ["E12"], "request": {"operation": "ocr"}})                 # 打开 R2，不比对
    out("HI", "W1-1", "INSPECT", {"targets": ["E1", "E11"], "request": {"operation": "compare_media"}})  # 比对原片
    out("HI", "W1-2", "SEARCH", {"input_type": "text", "query": "x", "output_type": "video"})   # 另一个工作区，不算重复
    # HI 原始输出 6 + 1 次被拒；GIA 原始输出 2，被拒 1
    events.append({"event_type": "raw_model_output", "actor_role": "HI", "actor_id": "W1-1", "payload": {}})
    events.append({"event_type": "raw_model_output", "actor_role": "GIA", "actor_id": "GIA", "payload": {}})
    out("GIA", "GIA", "READ_RECORD", {"record_id": "R3"})          # GIA 读 R3 = 打开 R3
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "evid_video_002_r0base-s1_1"
    run.mkdir()
    _store(run / "store.sqlite3")
    _trajectory(run / "trajectory.jsonl")
    (run / "run_config.json").write_text(json.dumps({
        "case_id": "evid:video_002", "round": "r0base-s1", "sample": 1, "instruction_mode": "student",
        "configured_model": {"model": "student", "temperature": 0.6},
        "search_snapshot": {"search_mode": "replay_or_record", "snapshot_root": "x", "counts": {"hit": 3, "miss": 1}},
        "freeze": {"applies": True, "ok": False, "failed": ["instruction_mode"]}}), encoding="utf-8")
    return run


def test_run_metrics_definitions(tmp_path):
    row = bm.run_metrics(_run(tmp_path), ["GT_SRC"])
    assert row["case_id"] == "evid:video_002" and row["round"] == "r0base-s1" and row["sample"] == 1
    assert row["pool"] == 3                       # 视频候选 3 个（page 类型的 R4 不进池）
    assert row["opened"] == 3                     # R1（get_frames E11）、R2（ocr E12）、R3（GIA READ_RECORD R3）
    assert row["compared"] == 1 and row["compared_of_opened"] == 1   # 只有模型自己的 compare_media 指向 E11
    assert row["auto_compare_calls"] == 1 and row["auto_compared"] == 1   # 运行时代跑的 [E1, E13] 单列，不算比对
    assert row["src_in_pool"] and row["src_opened"] and row["src_compared"] and row["src_acquired"]
    assert row["hi_actions"] == 6 and row["hi_repeats"] == 1
    assert row["raw_outputs"] == {"HI": 7, "GIA": 2} and row["rejected_outputs"] == {"HI": 1, "GIA": 1}


def test_search_auto_open_is_not_open(tmp_path):
    run = _run(tmp_path)
    con = sqlite3.connect(run / "store.sqlite3")
    con.execute("DELETE FROM inspect_executions WHERE workspace_id='W1-1'")
    con.commit()
    con.close()
    (run / "trajectory.jsonl").write_text("", encoding="utf-8")
    row = bm.run_metrics(run, ["GT_SRC"])
    assert row["pool"] == 3 and row["opened"] == 0 and row["src_in_pool"] and not row["src_opened"]


def test_summary_and_format_reject_is_role_mean(tmp_path):
    row = bm.run_metrics(_run(tmp_path), ["GT_SRC"])
    summary = bm.summarize([row])
    assert summary["open_rate"]["value"] == 1.0 and summary["open_rate"]["judgeable"] is False   # 分母 1 < 5
    assert summary["compare_rate"]["value"] == round(1 / 3, 4)
    assert summary["repeat_rate"]["value"] == round(1 / 6, 4)
    assert summary["format_reject_rate"]["value"] == round((1 / 7 + 1 / 2) / 2, 4)
    assert summary["companions"]["format_reject_rate_pooled"]["value"] == round(2 / 9, 4)
    assert summary["companions"]["auto_compare_runs"]["value"] == 1.0
    assert summary["snapshot"]["new_record_rate"]["value"] == 0.25
    assert summary["freeze"]["failed_items"] == {"instruction_mode": 1}


def test_criteria_fallback_regression_and_overall():
    criteria = json.loads((ROOT / "experiments" / "eval" / "r0" / "success_criteria.json").read_text(encoding="utf-8"))

    def group(open_k, open_n, runs, cmp_k, cmp_n, rep, rej, pts):
        r = bm.ratio
        return {"open_rate": r(open_k, open_n), "compare_rate": r(cmp_k, cmp_n), "repeat_rate": r(int(rep * 1000), 1000),
                "format_reject_rate": r(int(rej * 1000), 1000),
                "companions": {"open_rate_runs": r(open_k, runs)}, "score": {"point_accuracy": r(pts, 162)}}

    base = {"all": group(1, 2, 40, 10, 50, 0.55, 0.12, 50), "evid_ctrl20": group(1, 2, 40, 10, 50, 0.55, 0.12, 50)}
    after = {"all": group(3, 4, 40, 30, 60, 0.15, 0.04, 60), "evid_ctrl20": group(3, 4, 40, 30, 60, 0.15, 0.04, 60)}
    # 缺 run_config（没有 settings）→ 不可比
    assert bm.evaluate_criteria(criteria, after, base)["overall"]["verdict"] == "not_comparable"
    settings = {"tool_layer_hashes": {"sha256:a": 40}, "specs": {"x": 40}, "temperatures": {"0.6": 40},
                "instruction_modes": {"student": 40}}
    for g in (base, after):
        g["all"]["settings"] = settings
        g["all"]["freeze"] = {"failed_items": {"instruction_mode": 40}}
    result = bm.evaluate_criteria(criteria, after, base)
    b1 = result["items"][0]
    assert b1["status"] == "fallback" and b1["used_metric"] == "open_rate_runs" and b1["pass"] is False   # 3/40 < 10%
    assert [i["pass"] for i in result["items"][1:]] == [True, True, True]
    assert result["score"]["checks"] == {"abs_floor": True, "gain": False}     # 60/162 = 37.0%，比 50/162 = 30.9% 只多 6.2 个点
    assert result["overall"]["verdict"] == "diagnostic_success"
    # 倒退：重复率比基线高 6 个点（容差 5）→ 该项不通过，总判定 fail
    worse = {k: dict(v, repeat_rate=bm.ratio(610, 1000)) for k, v in after.items()}
    result = bm.evaluate_criteria(criteria, worse, base)
    assert result["items"][2]["regressed"] is True and result["overall"]["verdict"] == "fail"
    # 同口径核对：工具层 hash 与 E2 不同 → 不可比
    after["all"]["settings"] = dict(settings, tool_layer_hashes={"sha256:b": 40})
    result = bm.evaluate_criteria(criteria, after, base)
    assert result["overall"]["verdict"] == "not_comparable"
    after["all"]["settings"] = settings
    assert bm.evaluate_criteria(criteria, after, base)["overall"]["verdict"] == "diagnostic_success"
    after["all"]["freeze"] = {"failed_items": {"instruction_mode": 40, "tool_layer_hash": 3}}
    assert bm.evaluate_criteria(criteria, after, base)["overall"]["verdict"] == "not_comparable"
    after["all"]["freeze"] = {"failed_items": {"instruction_mode": 40}}
    base["all"]["freeze"] = {"failed_items": {"spec_hash.hi": 2}}          # 基线一边口径不对也不可比
    assert bm.evaluate_criteria(criteria, after, base)["overall"]["verdict"] == "not_comparable"


def test_eval_r0_commands_and_case_key():
    cfg = json.loads((ROOT / "experiments" / "eval" / "r0" / "eval_config.json").read_text(encoding="utf-8"))
    commands = er.build_commands(cfg, "baseline", "http://127.0.0.1:8002/v1", "student", 4, [1, 2])
    assert len(commands) == 4 and {c["round"] for c in commands} == {"r0base-s1", "r0base-s2"}
    argv = commands[0]["argv"]
    assert argv[argv.index("--hi-spec") + 1] == "hi@0.10.0" and argv[argv.index("--gia-spec") + 1] == "gia@0.9.0"
    assert argv[argv.index("--search-mode") + 1] == "replay_or_record" and "--state-renderer" not in argv
    assert argv[argv.index("--instruction-mode") + 1] == "student" and "--no-think" in argv
    ids = argv[argv.index("--ids") + 1: argv.index("--round")]
    assert len(ids) == 20 and ids[0] == "evid:video_002" and "evid:video_027" not in ids
    assert er.case_key("vcsl:pair_4") == "vcsl_pair_4" and er.case_key("evid:video_002") == "evid_video_002"
    assert bm._RUN_SUFFIX.match("evid_video_002_r0base-s1_1").group("round") == "r0base-s1"
