"""EVID-Bench baseline（10-04）：案卷转换、答案切点、官方多数票、评分端到端（假评委）、run_cases --instruction-mode。"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


build_cases = _load("_evid_build_cases")
score = _load("_evid_score")

ENTRY = {"id": "video_007", "source": ["abcDEF12345", "zzzYYY00000"], "topic": "sports",
         "task": {"category": "multi_source_editing", "type": "contextual_fabrication"},
         "groundtruth": ["GT one about splicing.", "GT two about false context.", "GT three overall narrative."]}


def test_case_has_no_ground_truth_and_loads(tmp_path: Path) -> None:
    from foren_searcher.datagen.cases import Case

    dataset = tmp_path / "plan-harness-1" / "datasets" / "EVID-Bench"
    (dataset / "videos").mkdir(parents=True)
    (dataset / "videos" / "video_007.mp4").write_bytes(b"x")
    (dataset / "evid_bench.json").write_text(json.dumps([ENTRY]), encoding="utf-8")
    out = tmp_path / "experiments" / "evid" / "cases"
    assert build_cases.main(["--dataset-root", str(dataset), "--out", str(out)]) == 0
    path = out / "evid_video_007.json"
    text = path.read_text(encoding="utf-8")
    assert "GT one" not in text and "abcDEF12345" not in text.split('"family_key"')[0]   # 真值不进任务与材料
    case = Case.load(path)
    assert case.case_id == "evid:video_007" and case.effective_split == "test"
    assert case.family_key == "evid_src:abcDEF12345+zzzYYY00000"
    assert (path.parent / case.media[0].path).resolve() == (dataset / "videos" / "video_007.mp4").resolve()
    assert "numbered list of 3 to 5 forgery points" in case.task_text
    assert build_cases.main(["--dataset-root", str(dataset), "--out", str(out), "--check"]) == 0


@pytest.mark.parametrize("answer, mode, n", [
    ("Summary first.\n1. Point one [E3].\n2. Point two (E4, E5).\n3. Point three.", "numbered", 3),
    ("1) alpha 2) beta 3) gamma", "numbered", 3),
    ("First paragraph.\n\nSecond paragraph.", "paragraphs", 2),
    ("A single blob without structure.", "whole", 1),
    ("", "empty", 0),
])
def test_split_points(answer: str, mode: str, n: int) -> None:
    points, got = score.split_points(answer)
    assert got == mode and len(points) == n
    assert all("E3" not in p and "E4" not in p for p in points)


def test_numbers_inside_text_do_not_split() -> None:
    points, mode = score.split_points("1. At 2. 5 seconds the clip jumps.\n2. Second point mentions 3. things.")
    assert mode == "numbered" and len(points) == 2


def test_official_majority_vote() -> None:
    ok, no = [{"verdict": 1}], [{"verdict": 0}]
    assert score.vote({"a": ok, "b": ok, "c": no}, 1)[0]["passed"] is True       # 3 评委 2 票
    assert score.vote({"a": ok, "b": no, "c": no}, 1)[0]["passed"] is False      # 3 评委 1 票
    assert score.vote({"a": ok, "b": no}, 1)[0]["passed"] is True                # 官方 >= n/2：2 评委 1 票即过


def test_parse_judge_keeps_first_and_ignores_out_of_range() -> None:
    rows = score.parse_judge({"matches": [{"gt_idx": 0, "pred_idx": 1, "verdict": 1}, {"gt_idx": 0, "verdict": 0},
                                          {"gt_idx": 7, "verdict": 1}, {"gt_idx": "x"}]}, 2)
    assert rows[0]["verdict"] == 1 and rows[0]["pred_idx"] == 1
    assert rows[1]["verdict"] == 0 and rows[1].get("missing")


def _fake_run(runs: Path, sample_id: str, answer: str | None, *, trajectory: str = "") -> None:
    run = runs / f"evid_{sample_id}_t0_1"
    run.mkdir(parents=True)
    if answer is not None:
        (run / "result.json").write_text(json.dumps({"decision": {"decision": "FINISH", "answer": answer},
                                                     "cited_evidence": []}), encoding="utf-8")
        (run / "execution_links.json").write_text("{}", encoding="utf-8")
    (run / "trajectory.jsonl").write_text(trajectory, encoding="utf-8")


def test_score_end_to_end_with_fake_judges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from foren_searcher.datagen.verify.judges import CallableJudge

    dataset = tmp_path / "ds"
    dataset.mkdir()
    entries = [dict(ENTRY), {**ENTRY, "id": "video_008", "task": {"category": "ai_generation", "type": "identity_swap"}},
               {**ENTRY, "id": "video_009"}]
    (dataset / "evid_bench.json").write_text(json.dumps(entries), encoding="utf-8")
    runs = tmp_path / "runs"
    _fake_run(runs, "video_007", "1. a\n2. b\n3. c", trajectory='{"url": "https://youtube.com/watch?v=abcDEF12345"}')
    _fake_run(runs, "video_008", "blob only")
    _fake_run(runs, "video_009", None)          # 没跑完

    def make(verdicts: list[int]):
        async def generate(system_prompt, payload, images):
            assert "GT-1" in payload["GT (3-5 points)"] and "one-to-one" in system_prompt
            return {"hits": sum(verdicts), "matches": [{"gt_idx": i, "pred_idx": 0, "verdict": v} for i, v in enumerate(verdicts)]}
        return CallableJudge(generate)

    fakes = {"local:j1": make([1, 1, 0]), "local:j2": make([1, 0, 0]), "local:j3": make([0, 1, 0])}
    monkeypatch.setattr(score, "build_judges", lambda specs, endpoint, cache: {s: fakes[s] for s in specs})
    out = tmp_path / "scores"
    assert score.main(["--round", "t0", "--dataset-root", str(dataset), "--runs-dir", str(runs), "--out-dir", str(out),
                       "--judges", *fakes]) == 0
    summary = json.loads((out / "t0.j1+j2+j3.summary.json").read_text(encoding="utf-8"))
    # video_007 / video_008 都是 GT0、GT1 各两票通过、GT2 零票 → 每个 2/3；video_009 没跑完按 0
    assert summary["all_planned"]["gt_points"] == 9 and summary["all_planned"]["passed_points"] == 4
    assert summary["all_planned"]["video_acc"] == 0.0
    assert summary["finished_only"]["videos"] == 2 and summary["finished_only"]["point_acc"] == round(4 / 6, 4)
    assert summary["run_status"] == {"done": 2, "incomplete": 1}
    assert summary["parse_mode"] == {"numbered": 1, "whole": 1, "no_finish": 1}
    assert summary["source_in_trajectory"] == 1
    assert summary["per_judge"]["local:j1"]["point_acc_alone"] == round(4 / 9, 4)
    assert (out / "t0.j1+j2+j3.md").read_text(encoding="utf-8").startswith("# EVID-Bench 评分：t0")


def test_parse_only_needs_no_judges(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    dataset.mkdir()
    (dataset / "evid_bench.json").write_text(json.dumps([ENTRY]), encoding="utf-8")
    assert score.main(["--round", "t0", "--dataset-root", str(dataset), "--runs-dir", str(tmp_path / "runs"),
                       "--out-dir", str(tmp_path / "out"), "--parse-only"]) == 0
    summary = json.loads((tmp_path / "out" / "t0.parse-only.summary.json").read_text(encoding="utf-8"))
    assert summary["run_status"] == {"missing": 1} and summary["all_planned"]["point_acc"] == 0.0


def test_run_cases_instruction_mode_passes_to_worker() -> None:
    from foren_searcher.datagen import run_cases

    item = run_cases.Planned("c.json", "evid:video_007", "test", 1, "evid_video_007_t0_1", "todo")
    base = dict(round="t0", runs_dir="runs", teacher="endpoint", specs="specs", hi_spec="hi@0.4.1", gia_spec="gia@0.5.0",
                no_think=False, blocklist="exclude", date_cutoff=None, search_mode="live", snapshot_root="s",
                tool_cache="off", tool_cache_root="t", harness_params="apply", inspect_ops="auto", harness_override=None)
    default = run_cases._worker_command(argparse.Namespace(**base, instruction_mode="teacher"), item, "http://x/v1", "m")
    student = run_cases._worker_command(argparse.Namespace(**base, instruction_mode="student"), item, "http://x/v1", "m")
    assert "--instruction-mode" not in default                      # 默认命令行与以前逐字相同
    assert student[-2:] == ["--instruction-mode", "student"]
    assert run_cases.INSTRUCTION_MODES == ("teacher", "student")


def test_launcher_smoke_ids_resolve_to_cases(tmp_path: Path) -> None:
    """10-05 冒烟报"找不到这些案卷"：启动脚本传的是 video_001，run_cases 要 evid:video_001。"""

    from foren_searcher.datagen.run_cases import plan

    launch = _load("_launch_evid")
    dataset = tmp_path / "ds"
    (dataset / "videos").mkdir(parents=True)
    entries = [{**ENTRY, "id": i} for i in launch.EVID_SMOKE]
    for i in launch.EVID_SMOKE:
        (dataset / "videos" / f"{i}.mp4").write_bytes(b"x")
    (dataset / "evid_bench.json").write_text(json.dumps(entries), encoding="utf-8")
    out = tmp_path / "cases"
    assert build_cases.main(["--dataset-root", str(dataset), "--out", str(out)]) == 0
    ids = [f"evid:{i}" for i in launch.EVID_SMOKE]
    planned = plan(out, runs_dir=tmp_path / "runs", round_tag="evidb0", split="test", ids=ids)
    assert [p.run_name for p in planned] == [score.run_name(i, "evidb0") for i in launch.EVID_SMOKE]


def test_launcher_evid_uses_default_specs_and_evid_config() -> None:
    """10-05：EVID baseline 跟 run_cases 默认规格（hi@0.4.2）走；配置用 evid_local.toml（每轮 3 个工作区）。"""

    from foren_searcher.config import load_config
    from foren_searcher.datagen import run_cases

    launch = _load("_launch_evid")
    # 修正优化计划（10-05）：p0 阶段与 run_cases 默认规格一致（r0e 等不受影响）；p1（默认）用新规格 hi@0.6.0 / gia@0.7.0
    assert launch.EVID_STAGE_SPECS["p0"] == run_cases.DEFAULT_SPECS
    assert launch.EVID_STAGE == "p1" and launch.EVID_SPECS == ("hi@0.7.0", "gia@0.8.0")   # 10-05 晚：p1 = P1 + 英文
    assert launch.EVID_STAGE_SPECS["p1zh"] == ("hi@0.6.0", "gia@0.7.0")
    assert all((ROOT / "specs" / name / "spec.json").is_file() for name in launch.EVID_SPECS)
    assert launch.EVID_RENDER == "compact_16k" and launch.EVID_STAGE_ARGS == ()
    assert load_config(ROOT / launch.EVID_CONFIG).runtime.max_workspaces_per_round == 3
    assert load_config(ROOT / "configs" / "evid_local.toml").runtime.max_rounds == 3          # P1.7
    assert len(launch.EVID_AB_IDS) == 20 and "video_027" not in launch.EVID_AB_IDS
    assert all(2 <= int(i.split("_")[1]) <= 130 for i in launch.EVID_AB_IDS)                  # 只用 evidb4 已跑过的 002–130
    assert load_config(ROOT / "configs" / "evid_local.toml").runtime.max_workspaces_per_round == 3     # 8B 档
    assert launch.EVID_ROUND not in ("evidb0", "evidb1")


def test_evid_launcher_is_standalone_and_has_all_jobs() -> None:
    """10-05：EVID 任务从 _launch.py 拆到 _launch_evid.py（_launch.py 被并行会话反复用旧底稿覆盖）；.cmd 都指向新文件。"""

    launch = _load("_launch_evid")
    assert set(launch.JOBS) == {"evid_smoke", "evid_full", "evid_score", "evid_score_sonnet", "evid_ab_p0", "evid_ab_bf16",
                                "evid27_full", "evid27_score_sonnet"}        # 10-05：27B 两个任务；修正计划 P0 的 A/B
    for job in launch.JOBS:
        cmd = (ROOT / "scripts" / f"_{job}.cmd").read_text(encoding="ascii")
        assert f"scripts\\_launch_evid.py {job}" in cmd


def test_attempted_row_excludes_samples_not_yet_run(tmp_path: Path) -> None:
    """10-05：全量没跑完时，"全部计划样本"的分母含 219 个还没跑的，0.2% 看不出东西；另报已开跑的口径。"""

    dataset = tmp_path / "ds"
    dataset.mkdir()
    (dataset / "evid_bench.json").write_text(json.dumps([ENTRY, {**ENTRY, "id": "video_008"}]), encoding="utf-8")
    runs = tmp_path / "runs"
    _fake_run(runs, "video_007", None)          # 开跑过但崩溃
    assert score.main(["--round", "t0", "--dataset-root", str(dataset), "--runs-dir", str(runs),
                       "--out-dir", str(tmp_path / "out"), "--parse-only"]) == 0
    summary = json.loads((tmp_path / "out" / "t0.parse-only.summary.json").read_text(encoding="utf-8"))
    assert summary["attempted"]["videos"] == 1 and summary["all_planned"]["videos"] == 2


def test_degraded_samples_leave_test_and_are_not_scored_by_default(tmp_path: Path) -> None:
    """10-05：冒烟三个样本的轨迹促成了 hi@0.5.0 / gia@0.6.0，按 cases.py 规则降级；重建案卷不会把标记冲掉。"""

    from foren_searcher.datagen.cases import Case

    dataset = tmp_path / "ds"
    (dataset / "videos").mkdir(parents=True)
    entries = [{**ENTRY, "id": i} for i in ("video_027", "video_099")]
    for e in entries:
        (dataset / "videos" / f"{e['id']}.mp4").write_bytes(b"x")
    (dataset / "evid_bench.json").write_text(json.dumps(entries), encoding="utf-8")
    cases = tmp_path / "cases"
    for _ in range(2):                                   # 第二次重建：标记保留
        assert build_cases.main(["--dataset-root", str(dataset), "--out", str(cases)]) == 0
    assert Case.load(cases / "evid_video_027.json").effective_split == "dev_train"
    assert Case.load(cases / "evid_video_099.json").effective_split == "test"
    out = tmp_path / "out"
    assert score.main(["--round", "t0", "--dataset-root", str(dataset), "--runs-dir", str(tmp_path / "runs"),
                       "--cases", str(cases), "--out-dir", str(out), "--parse-only"]) == 0
    summary = json.loads((out / "t0.parse-only.summary.json").read_text(encoding="utf-8"))
    assert summary["all_planned"]["videos"] == 1 and summary["meta"]["excluded_used_for_revision"] == ["video_027"]
    assert score.main(["--round", "t0", "--dataset-root", str(dataset), "--runs-dir", str(tmp_path / "runs"),
                       "--cases", str(cases), "--out-dir", str(out), "--parse-only", "--ids", "video_027"]) == 0
    assert (out / "t0.parse-only.subset.summary.json").is_file()      # 指定样本另存，不覆盖正式汇总


def test_new_specs_are_default_and_runtime_prompts_carry_source_rules() -> None:
    from foren_searcher.datagen import run_cases
    from foren_searcher.datagen.prompt_variants import runtime_gia_variants
    from foren_searcher.local_models import _hi_turn_prompt

    assert run_cases.DEFAULT_SPECS == ("hi@0.5.0", "gia@0.6.0")
    hi = _hi_turn_prompt({"direct_visual_input": True, "runtime_constraints": {"remaining_seconds": 1}}, ())
    assert "搜索引擎看不到本次调查的 E/R/F/W/C 编号" in hi and "output_type=video" in hi
    variants = runtime_gia_variants()
    assert all("寻找原始素材" in text for name, text in variants.items() if name != "wait")


def test_rejected_finish_counts_officially_and_strict_row_excludes_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """10-05 全量：GIA 写了 FINISH 答案但没带 E 编号、修复次数用完 → run 失败。正式口径评这份答案，严格口径按 0。"""

    from foren_searcher.datagen.verify.judges import CallableJudge

    dataset = tmp_path / "ds"
    dataset.mkdir()
    (dataset / "evid_bench.json").write_text(json.dumps([ENTRY, {**ENTRY, "id": "video_008"}]), encoding="utf-8")
    runs = tmp_path / "runs"
    _fake_run(runs, "video_007", "1. a\n2. b\n3. c")
    crashed = runs / score.run_name("video_008", "t0")
    _fake_run(runs, "video_008", None)
    for index, raw in enumerate(['{"action": "SEARCH"}',                                     # HI：没有 decision
                                 '{"decision": "FINISH", "answer": "1. old\\n2. old"}',
                                 '{"decision": "INVESTIGATE", "tasks": []}',
                                 '{"decision": "FINISH", "answer": ["1. x", "2. y", "3. z"]}',  # 最后一份，列表形式
                                 '{"decision": "FINISH", "answer": ""}'], 1):                 # 空答案不算
        directory = crashed / "requests" / f"request_{index:04d}"
        directory.mkdir(parents=True)
        (directory / "response.json").write_text(raw, encoding="utf-8")
    assert score.read_run(crashed, [])["rejected_finish_answer"] == "1. x\n2. y\n3. z"
    assert score.read_run(runs / score.run_name("video_007", "t0"), [])["rejected_finish_answer"] == ""

    async def generate(system_prompt, payload, images):
        return {"hits": 2, "matches": [{"gt_idx": 0, "pred_idx": 0, "verdict": 1}, {"gt_idx": 1, "pred_idx": 1, "verdict": 1},
                                       {"gt_idx": 2, "pred_idx": 2, "verdict": 0}]}

    monkeypatch.setattr(score, "build_judges", lambda specs, endpoint, cache: {s: CallableJudge(generate) for s in specs})
    out = tmp_path / "out"
    assert score.main(["--round", "t0", "--dataset-root", str(dataset), "--runs-dir", str(runs), "--out-dir", str(out),
                       "--judges", "local:j1"]) == 0
    summary = json.loads((out / "t0.j1.summary.json").read_text(encoding="utf-8"))
    assert summary["attempted"]["passed_points"] == 4                      # 正式：被拒收的 FINISH 也评
    assert summary["attempted_strict"]["passed_points"] == 2               # 严格：video_008 按 0
    assert summary["rejected_finish_used"] == 1 and summary["finished_only"]["videos"] == 1
    rows = [json.loads(line) for line in (out / "t0.j1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["answer_source"] for r in rows] == ["final", "rejected_finish"]
    assert "拒收" in (out / "t0.j1.md").read_text(encoding="utf-8")


def test_inline_numbering_after_chinese_punctuation() -> None:
    points, mode = score.split_points("1. 在7.5秒处字幕被改；2. 在15.0秒处插入画面；3. 整体制造假叙事。")
    assert mode == "numbered" and len(points) == 3 and points[1].startswith("在15.0秒处")
    assert score.split_points("在7.5秒处出现字幕，在15.0秒处切换。")[1] == "whole"


def test_task_text_names_no_concrete_manipulation_examples() -> None:
    """10-05（evidb5 起）：任务文字不举具体伪造手法（旧版的换脸 / 加物体被 8B 照抄到剪辑视频上），每点要写误导效果。"""

    text = build_cases.TASK_TEXT
    for example in ("swapped face", "added or replaced object", "synthetic segment", "re-captioning", "reordering"):
        assert example not in text
    assert "false impression or misleading conclusion" in text and "in English" in text
    assert "do not answer that nothing was changed" in text


def test_launcher_round_moved_past_old_task_text() -> None:
    launch = _load("_launch_evid")
    # 10-05 晚起默认 27B：轮次 evidq27a、27B 配置、关思考、同时 2 个
    assert launch.EVID_MODEL_PROFILE == "27b" and launch.EVID_ROUND == "evidq27b" == launch.EVID27_ROUND
    assert launch.EVID8_JOBS == "2"                     # 10-05 晚：8B 也并行 2 个（FOREN_EVID8_JOBS 可改）
    assert launch.EVID_MODEL == "qwen38-27b-32k:latest" and launch.EVID_CONFIG == launch.EVID27_CONFIG
    assert launch.EVID_EXTRA_ARGS == ("--no-think", "--jobs", "2") and launch.EVID_GUARDS == "eval"
    for name, needle in (("_evid_smoke_plus.cmd", "FOREN_EVID_GUARDS=eval_plus"), ("_evid_score_sonnet_plus.cmd", "evidq27p")):
        text = (ROOT / "scripts" / name).read_bytes().decode("ascii")
        assert needle in text and "FOREN_EVID_ROUND=evidq27p" in text
