"""g3（分支补量）批处理：照 scripts/g2/g2_run.py 跑（同一调度器 g1_run、同一停派条件、同一冻结核对、同一教师覆盖机制），只换：

- 轮次 r1g3（walker 的 r1 行为按 round.startswith("r1") 判；写成 g3 会静默走 r0_freeze / hi@0.10.0）；批次 / 分支 / --origin-tag 叫 g3；
- 教师槽表 specs/g1_freeze.json（10-19 合入 g2 后；槽表与 g1_teacher.json 逐键相同）；闸门状态与 g2 共用 g1 那一份（.wt_g1/.g1/state）：
  槽表的 codex_bin 指向 .wt_g1/.g1/bin/codex_gate.exe，
  g1 / g2 / g3 共用一套"同时在飞"槽位、退避与整批暂停，不另开一份（同一个 Codex 订阅，别抢）；同时在飞默认 ≤ 3；
- 循环内检查的 Claude CLI 也走 .wt_g1/.g1/bin/claude_gate.exe（限额一次整批停 20 min）；
- Codex 周用量 ≥ 60% 停派（--usage-stop 默认 60）；
- 清单 walker_g3_cases.json；run_config.teacher_override 记 g1_freeze.json 与 sha256、batch g3（导出 / 打包同 g2：--freeze-file / --freeze specs/g1_freeze.json）；
- 环境变体：案例名以 _v9 结尾（walker_run.VARIANTS）时，案卷取底案、run 名 <底案>_v9_r1g3_1（与 d1 --variants 同名），
  子进程带 --override max_evidence_cards=6 max_history_entries=4（run_config 记 variant_overrides，r1_freeze 的
  allowed_variant_overrides 认 v9）。

    python scripts\\g3\\g3_run.py --groups P1_A_fmnv --jobs 2
    python scripts\\g3\\g3_run.py --cases fmnv_a_xxx videosham_4002_v9 --jobs 1
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WT_G1 = ROOT.parent / ".wt_g1"
sys.path.insert(0, str(ROOT / "scripts" / "g1"))
os.environ.setdefault("G1_CODEX_MAX_INFLIGHT", "3")

import g1_run as g  # noqa: E402

g.ROUND = "r1g3"
g.STATE = WT_G1 / ".g1" / "state"
os.environ["G1_STATE"] = str(g.STATE)
os.environ["FOREN_CLAUDE_CLI"] = str(WT_G1 / ".g1" / "bin" / "claude_gate.exe")
# 10-19 合入 g2 后（load_freeze 按轮次选：r1g* → specs/g1_freeze.json）：教师槽表直接用 g1_freeze.json（槽表与 g1_teacher.json 逐键相同，
# g2 已核），run_config.freeze 就按 g1_freeze 记，不用 g2 那样事后 g2_freeze_recheck；导出 / 打包同 g2：
# export --freeze-file specs/g1_freeze.json --require-freeze --origin-tag g3、pack_r0 --round g3 --freeze specs/g1_freeze.json
g.TEACHER = ROOT / "specs" / "g1_freeze.json"
os.environ["FOREN_TEACHER_FREEZE"] = str(g.TEACHER)


class Runner(g.Runner):
    def launch(self, index: int, total: int, case: str, attempt: int) -> tuple[int, float]:
        base, overrides = g.wr.split_variant(case)
        if not overrides:
            return super().launch(index, total, case, attempt)
        run_name = f"{case}_{g.ROUND}_1"
        batch_log = g.wr.SFT / "batch_logs" / f"{run_name}{'' if attempt == 1 else '.retry'}.log"
        batch_log.parent.mkdir(parents=True, exist_ok=True)
        batch_log.write_text("", encoding="utf-8")
        started = time.monotonic()
        code = self.log.run(f"[{index}/{total}] {run_name}{'（重试）' if attempt > 1 else ''}（变体 {' '.join(overrides)}；日志 {batch_log}）",
                            [g.wr.PY, "scripts/walker_run.py", "one", str(g.wr.CASES_DIR / f"{base}.json"), run_name,
                             "--round", g.ROUND, "--override", *overrides],
                            env=self.env, log_file=batch_log)
        return code, round(time.monotonic() - started, 1)

    def record(self, case, run_dir, code, wall, attempt, failed):  # noqa: ANN001
        """= g1_run.Runner.record，只把写死的 g1_teacher.json 换成 g3 实际用的 g1_freeze.json（sha256 由 g1_run 按 g.TEACHER 算）。"""

        wr = g.wr
        cfg_path = run_dir / "run_config.json"
        walker_cfg: dict = {}
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                cfg["teacher_override"] = {"file": "specs/g1_freeze.json", "sha256": self.teacher_sha,
                                           "base_freeze": "specs/g1_freeze.json", "batch": "g3",
                                           "note": "槽表 = g1_freeze.json（load_freeze 按轮次 r1g* 选中），不是覆盖"}
                cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
                walker_cfg = cfg.get("walker") or {}
            except (OSError, ValueError):
                pass
        row = {"run": run_dir.name, "case_id": case, "round": g.ROUND, "teacher": "oracle_walker", "returncode": code,
               "status": wr.run_status(run_dir) if run_dir.exists() else "missing", "wall_seconds": wall,
               "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "attempt": attempt, "failed": failed,
               "teacher_model": walker_cfg.get("teacher_model") or wr.teacher_model(),
               "slot_models": walker_cfg.get("slot_models"), "teacher_role": "primary", "teacher_backup": None,
               "teacher_role_slots": walker_cfg.get("teacher_role"), "teacher_override": "specs/g1_freeze.json",
               **(wr.run_stats(run_dir) if run_dir.exists() else {}), **(wr.walker_stats(run_dir) if run_dir.exists() else {})}
        with self.lock:
            with self.manifest.open("a", encoding="utf-8") as h:
                h.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.log.say(f"      {run_dir.name}：{row['status']}；{wall or 0:.0f}s；教师调用 {row.get('llm_calls')} 次")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--groups", nargs="*", default=[])
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--deadline", default=None)
    parser.add_argument("--usage-stop", type=float, default=60.0)
    parser.add_argument("--disk-floor-gb", type=float, default=60.0)
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()
    cases = list(args.cases)
    if args.groups:
        groups = json.loads((g.wr.ROOT / g.wr.SFT / "walker_g3_cases.json").read_text(encoding="utf-8"))["groups"]
        for name in args.groups:
            cases += [c for c in groups.get(name, []) if c not in cases]
    if not cases:
        print("没有案例")
        return 2
    os.chdir(ROOT)
    log = g.wr.Log(f"_walker_{g.ROUND}日志.txt", f"g3 批 {g.ROUND}：{len(cases)} 个案例（S0–S5 Codex，g1_teacher.json；不核验）")
    log.say(f"开始 {g.now()}；E: 剩余 {g.disk_free_gb():.1f} GB；jobs {args.jobs}；同时在飞 ≤ {os.environ['G1_CODEX_MAX_INFLIGHT']}；"
            f"用量停派 {args.usage_stop}%；闸门状态 {g.STATE}")
    if not args.skip_preflight and not g.preflight(log):
        return 1
    Runner(args, log).run(cases)
    log.say(f"结束 {g.now()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
