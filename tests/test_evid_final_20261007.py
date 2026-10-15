"""EVID 正式 baseline 入口（scripts/_evid_final.py，10-07）：只跑 131–222，8B 再 27B，共用快照，断点续跑。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_final_ids_are_131_to_222_without_demoted_samples():
    import _evid_final as F

    ids = F.final_ids()
    numbers = [int(i.split("_")[-1]) for i in ids]
    assert min(numbers) >= 131 and max(numbers) == 222
    assert "video_151" not in ids                       # 已降级出 test
    assert not any(n <= 130 for n in numbers)            # 002–130 看过答案，不进正式分


def test_legs_use_current_tools_and_a_shared_snapshot():
    import _evid_final as F

    commands = [F.run_command(leg, ("video_131",)) for leg in F.LEGS]
    for command in commands:
        assert "--search-arm" not in command                                       # 找原片实验结论：工具层不进 baseline
        assert command[command.index("--instruction-mode") + 1] == "student"
        assert command[command.index("--snapshot-root") + 1].endswith("search_snapshots_final")
        assert command[command.index("--hi-spec") + 1] == "hi@0.7.0"
    assert commands[0][commands[0].index("--model") + 1] == F.L.EVID8_BF16_MODEL
    assert commands[1][commands[1].index("--model") + 1] == F.L.EVID27_MODEL and "--no-think" in commands[1]


def _stub(monkeypatch, tmp_path: Path, *, fail_27b: bool = False):
    import _evid_final as F

    monkeypatch.setattr(F.N, "ROOT", tmp_path)
    monkeypatch.setattr(F, "ROOT", tmp_path)
    monkeypatch.setattr(F, "env_setup", lambda: None)
    monkeypatch.setattr(F, "final_ids", lambda: ("video_131", "video_132", "video_133"))
    monkeypatch.setattr(F.L, "ollama_models", lambda: [F.L.EVID8_BF16_MODEL, F.L.EVID27_MODEL])
    monkeypatch.setattr(F.L, "bf16_preflight", lambda log: True)
    monkeypatch.setattr(F.L, "evid27_probe", lambda log: True)
    monkeypatch.setattr(F.L, "_ollama_ps", lambda log, label: None)
    monkeypatch.delenv("FOREN_FINAL_MODELS", raising=False)
    for leg in F.LEGS:
        leg.status, leg.notes = "pending", []
    scores = tmp_path / F.SCORES
    scores.mkdir(parents=True)
    calls: list[list[str]] = []

    def fake_run(self, label, argv, *, show=False):
        calls.append(list(argv))
        joined = " ".join(map(str, argv))
        if "foren_searcher.datagen.run_cases" in joined:
            round_tag = argv[argv.index("--round") + 1]
            if fail_27b and round_tag == "evidF27":
                return 1
            for case in argv[argv.index("--ids") + 1:]:
                run = tmp_path / "experiments" / "evid" / "runs" / f"evid_{case.split(':', 1)[1]}_{round_tag}_1"
                run.mkdir(parents=True, exist_ok=True)
                (run / "result.json").write_text("{}")
                (run / "execution_links.json").write_text("{}")
        if "_evid_score.py" in joined:
            round_tag = argv[argv.index("--round") + 1]
            (scores / f"{round_tag}.sonnet.subset.summary.json").write_text(json.dumps({
                "all_planned": {"point_acc": 0.3 if round_tag == "evidF8" else 0.5, "video_acc": 0.1, "gt_points": 12,
                                "passed_points": 4},
                "by_category": {"single_source_editing": {"point_acc": 0.3, "video_acc": 0.1, "videos": 2}},
                "funnel": {"source_in_pool": 1}, "behavior": {"repeat_rate": 0.5, "crashes": {}}}))
        return 0

    monkeypatch.setattr(F.N.NightLog, "run", fake_run)
    return F, calls, scores


def test_final_runs_8b_then_27b_with_canary_and_writes_report(monkeypatch, tmp_path: Path):
    F, calls, scores = _stub(monkeypatch, tmp_path)
    assert F.main() == 0
    runs = [c for c in calls if "foren_searcher.datagen.run_cases" in " ".join(c)]
    assert [c[c.index("--round") + 1] for c in runs] == ["evidF8", "evidF8", "evidF27", "evidF27"]   # 试跑 2 + 其余
    assert {leg.key: leg.status for leg in F.LEGS} == {"8b": "done", "27b": "done"}
    report = (tmp_path / F.REPORT).read_text(encoding="utf-8")
    assert "131–222" in report and "| Qwen3-VL-8B bf16 | evidF8 | 3/3 | 30.0% |" in report and "single_source_editing" in report
    calls.clear()
    for leg in F.LEGS:
        leg.status, leg.notes = "pending", []
    assert F.main() == 0                                  # 再双击：都评过分且跑完，不再跑
    assert not [c for c in calls if "foren_searcher.datagen.run_cases" in " ".join(c)]


def test_final_stops_a_leg_whose_canary_fails(monkeypatch, tmp_path: Path):
    F, calls, _ = _stub(monkeypatch, tmp_path, fail_27b=True)
    F.main()
    assert {leg.key: leg.status for leg in F.LEGS} == {"8b": "done", "27b": "failed_canary"}
    monkeypatch.setenv("FOREN_FINAL_MODELS", "27b")
    for leg in F.LEGS:
        leg.status, leg.notes = "pending", []
    calls.clear()
    F.main()
    assert all(c[c.index("--round") + 1] == "evidF27" for c in calls if "--round" in c and "run_cases" in " ".join(c))
