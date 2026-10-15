"""分叉式轨迹生成（10-11 试点）：关键状态、前缀强制回放、事实表打分、V1 屏蔽原件。"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from foren_searcher.datagen import forks
from foren_searcher.datagen.cases import Case

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "experiments" / "sft_data" / "cases"
RUN = ROOT / "experiments" / "sft_data" / "runs" / "true_20361105_r0e_1"


def _request(run: Path, seq: int, payload: dict[str, Any], output: dict[str, Any], forced: str | None = None) -> None:
    d = run / "requests" / f"request_{seq:04d}"
    (d / "actor").mkdir(parents=True)
    (d / "request.json").write_text(json.dumps({"system_prompt": "S", "payload": payload}), encoding="utf-8")
    (d / "response.json").write_text(json.dumps(output), encoding="utf-8")
    if forced:
        (d / "transport.json").write_text(json.dumps({"forced": forced}), encoding="utf-8")


def _hi(obs: dict[str, Any] | None, evidence: str = "E1", history: int = 0) -> dict[str, Any]:
    return {"workspace_id": "W1-1", "model_view": {"latest_observation": obs, "evidence_view": [{"evidence_id": evidence}],
                                                   "history": [{}] * history}}


def _fake_run(tmp_path: Path) -> Path:
    run = tmp_path / "x_tp0_1"
    _request(run, 1, {"global_view": {"current_materials": [{"evidence_id": "E1"}]}, "memory_history": []},
             {"decision": "INVESTIGATE", "tasks": [{"task": "t", "materials": ["E1"]}]})
    _request(run, 2, _hi(None), {"action": "SEARCH", "arguments": {"input_type": "text", "query": "budweiser", "output_type": "text"}})
    search = {"action": "SEARCH", "details": {"status": "ready", "results": [{"candidate_id": "R1"}, {"candidate_id": "R2"}]}}
    _request(run, 3, _hi(search, "E2", 1), {"action": "READ_RECORD", "arguments": {"record_id": "E2"}})
    _request(run, 4, {**_hi(search, "E2", 1), "validation_error": "bad"}, {"action": "READ_RECORD", "arguments": {"record_id": "E2"}})
    failed = {"action": "INSPECT", "details": {"status": "failed"}}
    _request(run, 5, {"workspace_id": "W1-2", "model_view": {"latest_observation": failed, "evidence_view": []}},
             {"action": "SEARCH", "arguments": {"input_type": "text", "query": "q", "output_type": "video"}})
    _request(run, 6, _hi(None, "E3", 3), {"task_id": "W1-1", "status": "COMPLETED", "findings": [{"statement": "s", "evidence_ids": ["E2"]}]})
    _request(run, 7, {"global_view": {"current_materials": []}, "memory_history": []},
             {"decision": "FINISH", "answer": "The claim is false. It came from a satire site [E2]."})
    return run


def test_classify_and_key_states(tmp_path: Path) -> None:
    run = _fake_run(tmp_path)
    kinds = {d.sequence: forks.classify(d) for d in forks.load_decisions(run)}
    assert kinds == {1: None, 2: None, 3: "candidates", 4: None, 5: "tool_failure", 6: "report", 7: "finish"}  # 4 是修复回合
    states = forks.key_states(run, limit=2)
    assert [(k.sequence, k.kind) for k in states] == [(3, "candidates"), (5, "tool_failure")]   # 不同工作区，先挑分叉性强的
    assert [k.kind for k in forks.key_states(run, limit=3)] == ["candidates", "tool_failure", "finish"]


def test_classify_material_failure_dead_end_and_source_visible(tmp_path: Path) -> None:
    case = Case.load(CASES / "averitec_392.json")
    blocked = {"action": "SEARCH", "details": {"results": [{"source_url": "https://www.example.com/x",
               "materials": [{"kind": "page", "status": "failed"}, {"kind": "document", "status": "not_requested"}]}]}}
    empty = {"action": "SEARCH", "details": {"results": []}}
    visible = {"action": "SEARCH", "details": {"results": [{"source_url": "https://www.timesnownews.com/the-buzz/article/budweiser-employee-peeing-x/1",
               "materials": [{"kind": "page", "status": "failed"}]}, {"source_url": "https://a.org", "materials": [{"kind": "page", "status": "ready"}]}]}}
    run = tmp_path / "y_tp0_1"
    for seq, obs in ((1, blocked), (2, empty), (3, visible)):
        _request(run, seq, _hi(obs), {"action": "READ_RECORD", "arguments": {"record_id": "E1"}})
    kinds = [forks.classify(d, case) for d in forks.load_decisions(run)]
    assert kinds == ["tool_failure", "dead_end", "source_visible"]
    assert forks.classify(forks.load_decisions(run)[2]) == "candidates"          # 不给案卷就只是候选列表


def test_forced_plan_replays_prefix_with_fingerprints(tmp_path: Path) -> None:
    run = _fake_run(tmp_path)
    alt = {"action": "SEARCH", "arguments": {"input_type": "text", "query": "foolish humor budweiser", "output_type": "text"}}
    plan = forks.forced_plan(run, 3, alt)
    assert [o["sequence"] for o in plan["outputs"]] == [1, 2, 3]
    assert [o["kind"] for o in plan["outputs"]] == ["prefix", "prefix", "branch"]
    assert [o["actor"] for o in plan["outputs"]] == ["GIA", "W1-1", "W1-1"]
    assert json.loads(plan["outputs"][2]["raw"]) == alt
    assert plan["outputs"][2]["expect"] == {"workspace_id": "W1-1", "evidence": ["E2"], "history": 1}
    same = forks.forced_plan(run, 3, None)
    assert json.loads(same["outputs"][2]["raw"])["action"] == "READ_RECORD"


def test_normalized_action_ignores_wording() -> None:
    a = {"action": "SEARCH", "arguments": {"query": "x"}, "rationale": "one", "focus_evidence_ids": ["E1"]}
    b = {"action": "SEARCH", "arguments": {"query": "x"}, "rationale": "two"}
    c = {"action": "SEARCH", "arguments": {"query": "y"}}
    assert forks.normalized_action(a) == forks.normalized_action(b) != forks.normalized_action(c)
    r1 = {"status": "COMPLETED", "findings": [{"statement": "a", "evidence_ids": ["E2"]}]}
    r2 = {"status": "COMPLETED", "findings": [{"statement": "b", "evidence_ids": ["E2"]}]}
    assert forks.normalized_action(r1) == forks.normalized_action(r2)
    read = {"action": "READ_RECORD", "arguments": {"record_id": "E26"}}
    assert forks.normalized_action(read) == forks.normalized_action({"action": "READ_RECORD", "arguments": {"record_id": "E26", "offset": 0}})


def _store(run: Path, rows: list[tuple[str, int, str]], tools: list[tuple[int, str]] = ()) -> None:
    db = sqlite3.connect(run / "store.sqlite3")
    db.executescript("CREATE TABLE candidates (id INTEGER PRIMARY KEY, source_url TEXT, final_url TEXT);"
                     "CREATE TABLE evidence (id INTEGER PRIMARY KEY, acquired_by TEXT, label TEXT);"
                     "CREATE TABLE candidate_evidence (candidate_id INTEGER, evidence_id INTEGER, material_kind TEXT, material_status TEXT);")
    for i, (url, eid, status) in enumerate(rows, 1):
        db.execute("INSERT INTO candidates VALUES (?,?,NULL)", (i, url))
        if eid:
            db.execute("INSERT OR IGNORE INTO evidence VALUES (?,?,?)", (eid, "SEARCH/open/http", "page"))
            db.execute("INSERT INTO candidate_evidence VALUES (?,?,?,?)", (i, eid, "page", status))
    for eid, acquired_by in tools:
        db.execute("INSERT INTO evidence VALUES (?,?,?)", (eid, acquired_by, "frame"))
    db.commit()
    db.close()


def test_fact_table_groups_follow_the_case_rule(tmp_path: Path) -> None:
    case = Case.load(CASES / "averitec_392.json")             # any_one_of:五个来源 → 一组，任取其一
    atoms = forks.fact_table(case)
    assert [a.kind for a in atoms] == ["EVIDENCE", "VERDICT"] and atoms[0].detail["mode"] == "any_one_of"
    run = _fake_run(tmp_path)
    _store(run, [("https://www.foolishhumor.com/2020/07/budweiser-employee", 2, "ready"), ("https://example.org/x", 0, "")])
    value = forks.evaluate(run, atoms)
    group = value["atoms"][0]
    assert group["value"] == 1.0 and next(m for m in group["members"] if m["key"] == "origin_site")["cited"]
    verdict = value["atoms"][1]
    assert verdict["status"] == "matches" and verdict["method"] == "substring" and value["value"] == 1.0
    judged = forks.evaluate(run, atoms, verdict_judgment={"status": "absent"})
    assert judged["atoms"][1]["value"] == 0.25 and judged["value"] == 0.625
    progress = forks.evaluate(run, atoms, scope="progress")
    assert progress["atoms"][0]["value"] == 0.2 and progress["atoms"][1]["value"] is None     # 只跑一步：组内取平均（5 份拿到 1 份）


def test_wayback_copy_counts_as_the_original(tmp_path: Path) -> None:
    case = Case.load(CASES / "averitec_392.json")
    run = _fake_run(tmp_path)
    _store(run, [("https://web.archive.org/web/20200702000000/https://www.foolishumor.com/budweiser-employee-9999/", 2, "ready")])
    group = forks.evaluate(run, forks.fact_table(case))["atoms"][0]
    assert next(m for m in group["members"] if m["key"] == "origin_site")["cited"]   # 原站真实拼写 foolishumor（案卷 10-12 已修）


def test_snapshot_as_of_replays_what_the_source_run_saw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from foren_searcher.search.recording import SearchRecording

    rec = SearchRecording("replay_or_record", tmp_path, "c:1", run_name="branch")
    old = {"key": "k", "outcome": "failure", "transient": True, "payload": {}, "run": "src", "recorded_at": "2026-10-11T14:00:00"}
    new = {"key": "k", "outcome": "ok", "transient": False, "payload": {}, "run": "other", "recorded_at": "2026-10-11T21:00:00"}
    for entry in (old, new):
        rec.store.append(entry)
    assert rec.store.get("k") is new
    monkeypatch.setattr("foren_searcher.search.recording.request_key", lambda request: "k")
    monkeypatch.setenv("FOREN_SNAPSHOT_AS_OF", "2026-10-11T15:00:00")
    assert rec.lookup("open", {})[1] is old                     # 源轨迹当时是超时，就回放超时
    mine = {**new, "run": "branch", "recorded_at": "2026-10-12T01:00:00"}
    rec.store.append(mine)
    assert rec.lookup("open", {})[1] is mine                    # 本支自己补录过的优先
    monkeypatch.delenv("FOREN_SNAPSHOT_AS_OF")
    assert rec.lookup("open", {})[1] is mine                    # 平时：取最新一条（旧行为）


def test_tool_evidence_matches_by_acquired_by(tmp_path: Path) -> None:
    case = Case.load(CASES / "videosham_1028.json")            # any_one_of:frames,diff_images；另两项 must_cite=False 只记录
    atoms = forks.fact_table(case)
    assert [(a.key, a.required) for a in atoms if a.kind == "EVIDENCE"] == [
        ("frames|diff_images", True), ("media_profile", False), ("alignment", False)]
    run = _fake_run(tmp_path)
    _store(run, [], tools=[(2, "INSPECT/get_frames")])
    value = forks.evaluate(run, atoms)
    assert value["atoms"][0]["value"] == 1.0                    # E2 由 get_frames 产出且被引用
    assert value["atoms"][1]["value"] == 0.0 and value["n_evidence"] == 1


def test_hidden_variant_keeps_glob_and_guard_blocks_it(tmp_path: Path) -> None:
    from foren_searcher.datagen.run_cases import KeyExclusion

    target, keys = forks.hidden_variant(CASES / "averitec_392.json", tmp_path / "cf")
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["source_policy"]["mode"] == "exclude_original" and data["family_key"] == "averitec:392"
    guard = KeyExclusion(tuple(data["source_policy"]["excluded_keys"]), tmp_path / "x.jsonl")
    assert guard.matches("https://www.timesnownews.com/the-buzz/article/budweiser-employee-peeing-has-a-budweiser/615317")   # tp0 漏掉的
    assert guard.matches("https://web.archive.org/web/2020/https://foolishhumor.com/a")
    assert not guard.matches("https://factcheck.newsmobile.in/did-budweiser-employees-really-pee/")
    assert KeyExclusion(("example.com/a",), tmp_path / "y.jsonl").matches("https://example.com/a/b")      # 无通配的旧键照旧按前缀
    assert not KeyExclusion(("example.com/a",), tmp_path / "y.jsonl").matches("https://example.com/ab")
    atoms = forks.fact_table(Case.load(CASES / "averitec_392.json"), hidden_keys=data["source_policy"]["excluded_keys"])
    assert not atoms[0].knowable and atoms[1].knowable          # 原件屏蔽 ≠ 结论不可知（还有别家核查）
    run = _fake_run(tmp_path)
    _store(run, [("https://www.timesnownews.com/the-buzz/article/budweiser-employee-peeing-x", 2, "ready")])
    assert forks.evaluate(run, atoms)["leaks"] == ["timesnow"]


def test_no_retrieval_variant_demands_restraint(tmp_path: Path) -> None:
    target = forks.no_retrieval_variant(CASES / "averitec_392.json", tmp_path / "v0")
    assert json.loads(target.read_text(encoding="utf-8"))["source_policy"]["mode"] == "provided_media"
    atoms = forks.fact_table(Case.load(CASES / "averitec_392.json"), no_retrieval=True)
    run = _fake_run(tmp_path)                                  # 答案写了 "false"：没检索却下结论
    _store(run, [])
    assert forks.evaluate(run, atoms)["atoms"][1]["value"] == 0.0
    (run / "requests" / "request_0007" / "response.json").write_text(json.dumps({"decision": "FINISH", "answer":
        "The claim could not be verified; no source was available [E2]."}), encoding="utf-8")
    assert forks.evaluate(run, atoms)["atoms"][1]["value"] == 1.0


def test_leaked_terms_only_flag_privileged_words() -> None:
    case = Case.load(CASES / "averitec_392.json")
    payload = {"workspace_id": "W1-1", "model_view": {"task": "Budweiser employee story", "runtime_constraints": {"state_renderer": "legacy"}}}
    ok = {"action": "SEARCH", "arguments": {"query": "Budweiser employee urinating beer tank police investigation"}}
    leak = {"action": "SEARCH", "arguments": {"query": "foolishhumor Budweiser"}}
    assert forks.leaked_terms(payload, ok, case) == []
    assert forks.leaked_terms(payload, leak, case) == ["foolishhumor"]
    seen = {**payload, "model_view": {**payload["model_view"], "evidence_view": [{"excerpt": "first published by FoolishHumor"}]}}
    assert forks.leaked_terms(seen, leak, case) == []           # 学生看得到就不算


def test_cli_generator_uses_forced_plan_and_detects_divergence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sys.path.insert(0, str(ROOT / "experiments" / "sft_data"))
    import cli_generator as C  # type: ignore

    payload = _hi(None, "E2", 1)
    plan = {"outputs": [{"sequence": 1, "raw": json.dumps({"action": "READ_RECORD", "arguments": {"record_id": "E2"}}),
                         "kind": "prefix", "expect": forks.fingerprint(payload)},
                        {"sequence": 2, "raw": "{}", "kind": "branch", "expect": {"workspace_id": "W9-9", "evidence": [], "history": 0}}]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setenv("FOREN_FORCED_PLAN", str(plan_path))
    monkeypatch.setattr(C.shutil, "which", lambda _: "codex")
    calls: list[Any] = []
    monkeypatch.setattr(C.subprocess, "run", lambda *a, **k: calls.append(a) or subprocess.CompletedProcess(a, 1, "", ""))
    gen = C.CliGenerator(store=None, output_dir=tmp_path / "requests")
    monkeypatch.setattr(gen, "_image_message", lambda view: (None, []))
    seen: list[dict] = []
    gen.generation_observer = seen.append
    out = gen._generate("SYS", payload, 1)
    assert out == {"action": "READ_RECORD", "arguments": {"record_id": "E2"}} and calls == []
    transport = json.loads((tmp_path / "requests" / "request_0001" / "transport.json").read_text(encoding="utf-8"))
    assert transport["forced"] == "prefix" and seen[0]["provider"] == "forced_plan:prefix"
    assert (tmp_path / "requests" / "request_0001" / "request.json").is_file()      # 前缀行照常落盘（导出按 forced 标记处理）
    with pytest.raises(RuntimeError, match="FORK_DIVERGED"):
        gen._generate("SYS", payload, 2)


@pytest.mark.skipif(not (RUN / "requests").is_dir(), reason="r0e run 未随仓库入库")
def test_key_states_on_a_real_run() -> None:
    states = forks.key_states(RUN, limit=2)
    assert states and all(k.kind in {"candidates", "tool_failure", "detect_result", "report", "finish"} for k in states)
    assert len({k.workspace_id or "GIA" for k in states}) == len(states)


def test_cli_generator_matches_plan_by_actor_and_stops_at_horizon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sys.path.insert(0, str(ROOT / "experiments" / "sft_data"))
    import cli_generator as C  # type: ignore

    gia = {"global_view": {"current_materials": [{"evidence_id": "E1"}]}, "memory_history": []}
    hi = _hi(None, "E2", 1)
    read = json.dumps({"action": "READ_RECORD", "arguments": {"record_id": "E2"}})
    plan = {"outputs": [{"sequence": 1, "actor": "GIA", "raw": json.dumps({"decision": "INVESTIGATE", "tasks": []}), "kind": "prefix",
                         "expect": forks.fingerprint(gia)},
                        {"sequence": 2, "actor": "W1-1", "raw": read, "kind": "prefix", "expect": forks.fingerprint(hi)},
                        {"sequence": 3, "actor": "W1-1", "raw": read, "kind": "branch", "expect": forks.fingerprint(hi)}]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setenv("FOREN_FORCED_PLAN", str(plan_path))
    monkeypatch.setenv("FOREN_FORK_HORIZON", "step")
    monkeypatch.setattr(C.shutil, "which", lambda _: "codex")
    monkeypatch.setattr(C.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no live call expected")))
    out_dir = tmp_path / "run" / "requests"
    gen = C.CliGenerator(store=None, output_dir=out_dir)
    monkeypatch.setattr(gen, "_image_message", lambda view: (None, []))
    assert gen._generate("SYS", hi, 1)["action"] == "READ_RECORD"            # HI 先到（复核时机变了）也能对上号
    assert gen._generate("SYS", gia, 2)["decision"] == "INVESTIGATE"
    assert gen._generate("SYS", hi, 3)["action"] == "READ_RECORD" and gen.branch_actor == "W1-1"
    with pytest.raises(C.ForkHorizonReached):
        gen._generate("SYS", _hi({"action": "READ_RECORD"}, "E2", 2), 4)
    stop = json.loads((tmp_path / "run" / "fork_stop.json").read_text(encoding="utf-8"))
    assert stop["horizon"] == "step" and stop["stopped_at_request"] == 4
    assert (out_dir / "request_0004" / "request.json").is_file()            # 停下那次的状态（含分叉动作的观察）照常落盘
    with pytest.raises(C.ForkHorizonReached):
        gen._generate("SYS", gia, 5)
    assert not (out_dir / "request_0005").exists()


def test_cli_generator_hi_mismatch_diverges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sys.path.insert(0, str(ROOT / "experiments" / "sft_data"))
    import cli_generator as C  # type: ignore

    plan = {"outputs": [{"sequence": 1, "actor": "W1-1", "raw": "{}", "kind": "prefix", "expect": {"workspace_id": "W1-1", "evidence": ["E9"], "history": 0}}]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setenv("FOREN_FORCED_PLAN", str(plan_path))
    monkeypatch.delenv("FOREN_FORK_HORIZON", raising=False)
    monkeypatch.setattr(C.shutil, "which", lambda _: "codex")
    gen = C.CliGenerator(store=None, output_dir=tmp_path / "requests")
    monkeypatch.setattr(gen, "_image_message", lambda view: (None, []))
    with pytest.raises(RuntimeError, match="FORK_DIVERGED"):
        gen._generate("SYS", _hi(None, "E2", 1), 1)
    assert json.loads((tmp_path / "fork_diverged.json").read_text(encoding="utf-8"))["request"] == 1
    with pytest.raises(C.ForkHorizonReached):                   # 之后的请求（例如 GIA 想现场续跑）立刻失败，不再花教师调用
        gen._generate("SYS", {"global_view": {}, "memory_history": []}, 2)
