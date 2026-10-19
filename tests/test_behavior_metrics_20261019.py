"""behavior_metrics@1.1.0 的两个报告项与 sft0 评测口径（2026-10-19）。

⑤ 每工作区步数分布：步数 = HI 工作区内 model_output / observation 的最大 turn_index（没有 turn_index 时退回 model_output 条数）；
⑥ 提前交报告：模型自己写的 REPORT 所在回合 < 回合上限的工作区 / 全部工作区；撞上限被截断（运行时代交的 turn limit 空报告、
WorkspaceTurnLimit）另列；回合上限按 轨迹 runtime_constraints.max_turns > run_config.runtime_limits > harness_params_applied 取，
都没有时记 self_report_limit_unknown（在分母、不进分子）。四个已登记指标的数值不因新增项改变。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "eval" / "r0" / "eval_config.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bm = _load("behavior_metrics", ROOT / "scripts" / "behavior_metrics.py")
er = _load("eval_r0", ROOT / "scripts" / "eval_r0.py")
old_tests = _load("test_behavior_metrics_20261015", ROOT / "tests" / "test_behavior_metrics_20261015.py")


def _ev(kind, actor, payload=None, role="HI"):
    return {"event_type": kind, "actor_role": role, "actor_id": actor, "payload": payload or {}}


def _turn(events, actor, turn, *, limit=5, report=False, next_output=None):
    constraints = {"turn_index": turn, "max_turns": limit, "remaining_turns": limit - turn}
    if next_output:
        constraints["next_output"] = next_output
    events.append(_ev("model_input", actor, {"turn_index": turn, "model_view": {"runtime_constraints": constraints}}))
    events.append(_ev("raw_model_output", actor))
    if report:
        events.append(_ev("model_output", actor, {"turn_index": turn, "output": {"task_id": actor, "status": "COMPLETE",
                                                                                 "findings": [], "status_reason": "done"}}))
        events.append(_ev("workspace_report", actor, {"task_id": actor, "status": "COMPLETE", "status_reason": "done"}))
    else:
        events.append(_ev("model_output", actor, {"turn_index": turn, "output": {"action": "SEARCH",
                                                                                 "arguments": {"query": f"{actor}-{turn}"}}}))
        events.append(_ev("observation", actor, {"turn_index": turn, "observation": {"action": "SEARCH"}}))


def _run(tmp_path: Path, name: str, events: list, run_config: dict | None) -> Path:
    run = tmp_path / name
    run.mkdir()
    (run / "trajectory.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    if run_config is not None:
        (run / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
    return run


def _five_workspaces() -> list:
    events: list = []
    for t in (1, 2):                                   # W1-1：第 3 回合自己交报告（上限 5）→ 提前交报告
        _turn(events, "W1-1", t)
    _turn(events, "W1-1", 3, report=True)
    for t in range(1, 6):                              # W1-2：5 个回合都是工具动作，运行时代交 → 到上限被截断
        _turn(events, "W1-2", t)
    events.append(_ev("auto_report", "W1-2", {"reason": "turn limit reached"}))
    events.append(_ev("workspace_report", "W1-2", {"status": "INCOMPLETE", "status_reason":
                                                   "auto_report: turn limit reached; the workspace produced no report of its own"}))
    for t in range(1, 5):                              # W2-1：最后一回合（第 5 回合）只能交报告时交的 → 用满回合后自交
        _turn(events, "W2-1", t)
    _turn(events, "W2-1", 5, report=True, next_output="WORKSPACE REPORT only: this is the last turn")
    _turn(events, "W2-2", 1)                           # W2-2：重复用完后"只可交报告"回合（不耗回合，turn_index 记成 1）
    _turn(events, "W2-2", 1, report=True, next_output="WORKSPACE REPORT only: identical actions were repeated")
    _turn(events, "W3-1", 1)                           # W3-1：一个回合后崩了，没有报告
    events.append(_ev("runtime_error", "W3-1", {"turn_index": 2, "error": {"type": "TimeoutError", "message": "x"}}))
    events.append(_ev("model_output", "GIA", {"output": {"decision": "FINISH"}}, role="GIA"))   # GIA 不算工作区
    return events


def test_workspace_steps_and_endings(tmp_path):
    row = bm.run_metrics(_run(tmp_path, "evid_video_002_r0base-s1_1", _five_workspaces(), {"case_id": "evid:video_002"}),
                         ["GT_SRC"])
    by_id = {w["workspace_id"]: w for w in row["workspaces"]}
    assert sorted(by_id) == ["W1-1", "W1-2", "W2-1", "W2-2", "W3-1"]
    assert {k: v["steps"] for k, v in by_id.items()} == {"W1-1": 3, "W1-2": 5, "W2-1": 5, "W2-2": 1, "W3-1": 1}
    assert {k: v["ending"] for k, v in by_id.items()} == {
        "W1-1": "early_report", "W1-2": "truncated_at_limit", "W2-1": "report_at_limit", "W2-2": "early_report",
        "W3-1": "no_report"}
    assert all(w["max_turns"] == 5 and w["max_turns_source"] == "trajectory.runtime_constraints" for w in by_id.values())
    assert by_id["W2-2"]["report_only_turn"] and not by_id["W1-1"]["report_only_turn"]
    assert by_id["W1-1"]["tool_actions"] == 2 and by_id["W1-1"]["model_outputs"] == 3

    summary = bm.summarize([row])
    steps = summary["workspace_steps"]
    assert steps["n"] == 5 and steps["min"] == 1 and steps["max"] == 5 and steps["mean"] == 3.0
    assert (steps["p25"], steps["p50"], steps["p75"], steps["p90"]) == (1.0, 3.0, 5.0, 5.0)
    assert steps["histogram"] == {"1": 2, "3": 1, "5": 2}
    early = summary["early_report"]
    assert early["early_report_rate"]["num"] == 2 and early["early_report_rate"]["den"] == 5
    assert early["early_report_rate"]["value"] == 0.4
    assert early["truncated_at_limit_rate"]["value"] == 0.2 and early["report_at_limit_rate"]["value"] == 0.2
    assert early["endings"]["no_report"] == 1 and early["limit_unknown"] == 0
    assert early["early_report_on_report_only_turn"] == 1
    assert early["max_turns"] == {"5": 5}


def test_limit_fallbacks_and_unknown(tmp_path):
    events: list = []
    events.append(_ev("model_output", "W1-1", {"output": {"action": "SEARCH", "arguments": {}}}))      # 旧格式：没有 turn_index
    events.append(_ev("model_output", "W1-1", {"output": {"task_id": "W1-1", "status": "COMPLETE"}}))
    events.append(_ev("workspace_report", "W1-1", {"status": "COMPLETE", "status_reason": "ok"}))
    from_limits = bm.run_metrics(_run(tmp_path, "a_r0base-s1_1", events, {"runtime_limits": {"max_workspace_turns": 12}}), None)
    w = from_limits["workspaces"][0]
    assert w["steps"] == 2 and w["max_turns"] == 12 and w["max_turns_source"] == "run_config.runtime_limits"
    assert w["ending"] == "early_report"
    applied = {"harness_params_applied": {"max_workspace_turns": {"value": 2, "source": "hi@0.11.0"}}}
    w = bm.run_metrics(_run(tmp_path, "b_r0base-s1_1", events, applied), None)["workspaces"][0]
    assert w["max_turns"] == 2 and w["max_turns_source"] == "run_config.harness_params_applied" and w["ending"] == "report_at_limit"
    w = bm.run_metrics(_run(tmp_path, "c_r0base-s1_1", events, None), None)["workspaces"][0]
    assert w["max_turns"] is None and w["ending"] == "self_report_limit_unknown"
    early = bm.summarize([bm.run_metrics(tmp_path / "c_r0base-s1_1", None)])["early_report"]
    assert early["limit_unknown"] == 1 and early["early_report_rate"]["num"] == 0 and early["early_report_rate"]["den"] == 1
    # 没开运行时代交时撞上限：WorkspaceTurnLimit 也算被截断
    capped = [_ev("model_output", "W9", {"turn_index": 3, "output": {"action": "OCR", "arguments": {}}}),
              _ev("runtime_error", "W9", {"error": {"type": "WorkspaceTurnLimit", "message": "x"}})]
    w = bm.run_metrics(_run(tmp_path, "d_r0base-s1_1", capped, {"runtime_limits": {"max_workspace_turns": 3}}), None)["workspaces"][0]
    assert w["ending"] == "truncated_at_limit" and w["steps"] == 3


def test_registered_four_metrics_unchanged(tmp_path):
    """1.0.0 的合成 run（test_behavior_metrics_20261015）上，四项与伴随指标的数值和 1.0.0 测试里写死的一样。"""

    row = bm.run_metrics(old_tests._run(tmp_path), ["GT_SRC"])
    summary = bm.summarize([row])
    assert summary["open_rate"]["value"] == 1.0 and summary["compare_rate"]["value"] == round(1 / 3, 4)
    assert summary["repeat_rate"]["value"] == round(1 / 6, 4)
    assert summary["format_reject_rate"]["value"] == round((1 / 7 + 1 / 2) / 2, 4)
    assert summary["companions"]["format_reject_rate_pooled"]["value"] == round(2 / 9, 4)
    assert summary["workspace_steps"]["n"] == 2          # W1-1、W1-2（GIA 不算）；旧合成数据没有 turn_index，按条数
    assert summary["early_report"]["endings"]["no_report"] == 2


def test_markdown_and_baseline_side_by_side(tmp_path):
    run = _run(tmp_path, "evid_video_002_r0sft-s1_1", _five_workspaces(), {"case_id": "evid:video_002"})
    groups = {"all": bm.summarize([bm.run_metrics(run, None)])}
    old_baseline = {"all": {k: v for k, v in groups["all"].items() if k not in ("workspace_steps", "early_report")}}
    md = bm.markdown("t", groups, None, {"metrics_version": bm.METRICS_VERSION, "script_sha256": "x", "inputs": []}, old_baseline)
    assert "⑤ 每工作区步数分布" in md and "| all | 5 | 1 | 1.0 | 3.0 | 5.0 | 5.0 | 5 | 3.0 |" in md
    assert "| all（E2 基线） | —（无此项）" in md and "40.0%（2/5" in md
    comparison = bm.workspace_comparison(groups, groups)
    assert comparison["all"]["baseline"]["early_report"]["early_report_rate"]["value"] == 0.4
    assert bm.workspace_comparison(groups, None)["all"]["baseline"] is None
    out = tmp_path / "rep"
    assert bm.main([str(run), "--out", str(out), "--no-legacy", "--baseline", str(tmp_path / "missing.json")]) == 0
    report = json.loads(Path(f"{out}.json").read_text(encoding="utf-8"))
    assert report["meta"]["metrics_version"] == "behavior_metrics@1.1.0"
    assert report["groups"]["all"]["early_report"]["early_report_rate"]["value"] == 0.4
    assert "workspace_comparison" in report


def test_eval_config_registration():
    """sft0 口径（10-19）：登记的 hash 与文件一致、指标版本与脚本一致、成功标准没动、评委走 Claude CLI 订阅（与 r1_freeze 同模型）。"""

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for key, rel in cfg["registered_files"].items():
        actual = "sha256:" + hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        assert cfg["registered_hashes"][key] == actual, key
    assert cfg["metrics"]["version"] == bm.METRICS_VERSION
    criteria = json.loads((ROOT / cfg["success_criteria"]).read_text(encoding="utf-8"))
    assert criteria["metrics_version"] == "behavior_metrics@1.0.0" and criteria["criteria_id"] == "r0_success_v1"
    freeze = json.loads((ROOT / cfg["harness"]["freeze_file"]).read_text(encoding="utf-8"))
    assert freeze["specs"] == {"hi": cfg["harness"]["hi_spec"], "gia": cfg["harness"]["gia_spec"]}
    assert freeze["tool_layer"]["hash"] == cfg["harness"]["freeze_tool_layer_hash_at_registration"]
    assert freeze["verifier"]["code_hash"] == cfg["judge"]["dev_eval"]["verifier_code_hash"]
    assert cfg["harness"]["freeze_expected_failures"] == ["instruction_mode"]
    assert cfg["judge"]["evid"]["judges"] == ["claude-cli:sonnet"]
    assert cfg["judge"]["dev_eval"]["judge"] == freeze["verifier"]["judge"] == "claude-cli"
    assert cfg["judge"]["dev_eval"]["model"] == freeze["verifier"]["model_alias"] == "sonnet" and freeze["verifier"]["model"] == "claude-sonnet-5-5"


def test_eval_r0_after_sft_describer_and_score_commands(capsys, monkeypatch):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert er.describer_env(cfg, "http://127.0.0.1:8002/v1/") == {
        "FOREN_DESCRIBE_ENDPOINT": "http://127.0.0.1:8002/v1", "FOREN_DESCRIBE_MODEL": "student"}
    commands = er.build_commands(cfg, "after_sft", "http://127.0.0.1:8002/v1", "HI=hi_sft0_s1,GIA=gia_sft0_s1", 6, [2])
    assert {c["round"] for c in commands} == {"r0sft-s2"}
    argv = commands[0]["argv"]
    assert argv[argv.index("--jobs") + 1] == "6" and argv[argv.index("--model") + 1] == "HI=hi_sft0_s1,GIA=gia_sft0_s1"
    assert argv[argv.index("--hi-spec") + 1] == "hi@0.11.0" and "--no-think" in argv and "--state-renderer" not in argv
    # 10-19：EVID 帧不进任何以图搜——两阶段、两个集合都是 --image-search off
    for phase, model in (("baseline", "student"), ("after_sft", "HI=hi_sft0_s1,GIA=gia_sft0_s1")):
        for item in er.build_commands(cfg, phase, "http://127.0.0.1:8002/v1", model, 6, [1, 2]):
            assert item["argv"][item["argv"].index("--image-search") + 1] == "off", (phase, item["set"])
    import argparse
    er.cmd_score(cfg, argparse.Namespace(phase="baseline", seeds=[1], dry_run=True))
    printed = capsys.readouterr().out
    assert "--judges claude-cli:sonnet" in printed
    assert "--judge claude-cli --model sonnet" in printed


def test_wilson_ratio_above_one_has_no_ci():
    """10-19 基线 metrics 崩在 compare_rate_of_acquired（compared > acquired，p > 1 开方为负）：比值照报、CI 为 None。"""

    assert bm.wilson(5, 3) is None and bm.ratio(5, 3) == {"value": 1.6667, "num": 5, "den": 3, "ci95": None}
    assert bm.wilson(2, 4) is not None and bm.wilson(0, 0) is None
