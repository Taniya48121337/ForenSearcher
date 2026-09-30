"""对抗探针集（verify/adversarial.py）：从真实 run 生成四类探针、落盘可重跑、假核验模型跑通、judge=none 下程序层能识破的部分。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from foren_searcher.datagen.cases import load_cases
from foren_searcher.datagen.verify.plan_judged import PLAN_SYSTEM_PROMPT
from foren_searcher.datagen.verify.scope import overlap
from foren_searcher.datagen.verify.verdict import Verdict

from test_datagen_judged_units import CASES, RUN, VERDICTS, _replay_from_base, needs_run
from test_datagen_sufficiency_scope import UNITS, ScopeJudge, _matches, needs_units

CASE_ID = "true:4124903"


def _read(path: Path) -> list[Verdict]:
    return [Verdict.from_dict(json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


class ProbeJudge(ScopeJudge):
    """报告：ScopeJudge（充分性按"引了匹配必需证据的 E 就算覆盖"——正是含糊 Finding 要钻的口径）；plan：targets 按文字重合。"""

    async def judge(self, *, system_prompt: str, payload: dict[str, Any], images: list[Path], schema: dict[str, Any] | None = None) -> dict[str, Any]:
        if system_prompt == PLAN_SYSTEM_PROMPT:
            covered = set(payload["covered_before_this_round"])
            tasks = []
            for t in payload["tasks"]:
                targets = sorted(r["id"] for r in payload["requirements"] if overlap(r["text"], t["task"]) >= 0.25)
                tasks.append({"index": t["index"], "targets": targets, "gap_driven": bool(set(targets) - covered), "premises": []})
            return {"tasks": tasks, "notes": "fake plan"}
        return await super().judge(system_prompt=system_prompt, payload=payload, images=images, schema=schema)


@needs_run
@needs_units
def test_adversarial_probes_generate_persist_and_rerun(tmp_path: Path):
    from foren_searcher.datagen.verify.adversarial import PATHS, generate_probes, read_probes, run_probes, write_probes

    base, units = _read(VERDICTS), _read(UNITS)
    cases = load_cases(CASES)
    case = cases[CASE_ID]
    probes = generate_probes(RUN, case, base, units, paths=list(PATHS), per_path=2, stack=5, seed=7)
    by_path = {p: [x for x in probes if x["path"] == p] for p in PATHS}
    assert all(by_path[p] for p in PATHS), {p: len(v) for p, v in by_path.items()}
    for p in probes:
        for key in ("probe_id", "path", "unit", "run_dir", "original", "mutated", "change", "expected", "condition"):
            assert key in p, key
    assert len({p["probe_id"] for p in probes}) == len(probes)
    # (b) 堆满 5 条、都是本工作区可见的已有 E；(d) 合成下一轮、留了 must_cite 缺口
    assert all(len(p["stacked"]) == 5 for p in by_path["b"])
    assert all(p["condition"]["round_index"] == 2 and p["gap"] not in p["condition"]["covered_before"] for p in by_path["d"])
    assert all(p["mutated"]["status"] == "INCOMPLETE" and p["original"]["status"] == "COMPLETED" for p in by_path["c"])
    # 落盘 → 读回 → 重跑，与直接跑一致
    path = write_probes(tmp_path / "probes.jsonl", probes)
    assert read_probes(path) == json.loads(json.dumps(probes, ensure_ascii=False))
    judge = ProbeJudge(_replay_from_base(base), case, _matches(case))
    summary = asyncio.run(run_probes(read_probes(path), cases, base, judge, judge_name="fake", units=units, say=lambda _: None))
    rows = summary["rows"]
    assert summary["n_probes"] == len(probes) and set(summary["by_path"]) == set(PATHS)
    assert summary["by_path"]["b"]["caught_rate"] == 1.0 and summary["by_path"]["c"]["caught_rate"] == 1.0
    assert summary["by_path"]["d"]["caught_rate"] == 1.0
    # (c) 识破了也不拦：incomplete_though_answerable 是 minor，处置仍是 positive（S4：INCOMPLETE 是免死牌）
    assert summary["by_path"]["c"]["rejected_rate"] == 0.0
    assert all(r["hits"] == ["task_not_gap_driven"] for r in rows if r["path"] == "d")
    # (a)：假充分性按引用算覆盖 → 只把帧 / youtube 的 Finding 改空话骗过了它；W1-2 改空话丢了"耶拿桥"，被第一层实体检查识破
    a_rows = [r for r in rows if r["path"] == "a"]
    assert any(r["detected"] and r["hits"] == ["omitted_required_fact"] for r in a_rows)
    assert any(not r["detected"] for r in a_rows)
    assert summary["controls"]


@needs_run
@needs_units
def test_adversarial_exec_only_and_cli(tmp_path: Path):
    from foren_searcher.datagen.verify.adversarial import main

    out = tmp_path / "adv.json"
    assert main(["--runs", str(RUN), "--cases", str(CASES), "--verdicts", str(VERDICTS), "--units", str(UNITS), "--judge", "none",
                 "--per-path", "1", "--out", str(out)]) == 0
    probes_file = tmp_path / "adv.probes.jsonl"
    assert probes_file.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    bp = data["by_path"]
    # 程序层只能靠"空话顺带删掉了必需实体"识破 (a)；(b)(c)(d) 全靠模型层 → 全部 inconclusive
    for p in ("b", "c", "d"):
        assert bp[p]["n"] >= 1 and bp[p]["detected"] == 0 and bp[p]["inconclusive"] == bp[p]["n"], p
    # 重跑已落盘的探针（换核验器的用法），只跑 (c)
    out2 = tmp_path / "adv2.json"
    assert main(["--probes-in", str(probes_file), "--paths", "c", "--cases", str(CASES), "--verdicts", str(VERDICTS), "--units", str(UNITS),
                 "--judge", "none", "--out", str(out2)]) == 0
    data2 = json.loads(out2.read_text(encoding="utf-8"))
    assert data2["n_probes"] == bp["c"]["n"] and data2["by_path"]["a"]["n"] == 0
