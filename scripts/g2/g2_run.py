"""g2（覆盖补量）批处理：照 scripts/g1/g1_run.py 跑（同一调度器、同一停派条件、同一冻结核对），只换：

- 轮次 r1g2（walker 的 r1 行为按 round.startswith("r1") 判）；
- 闸门状态用 g1 的那一份（.wt_g1/.g1/state）：g1_teacher.json 的 codex_bin 指向 .wt_g1/.g1/bin/codex_gate.exe，两个工作树共用一套
  "同时在飞"槽位、退避与整批暂停；本批默认同时在飞 ≤ 3（G1_CODEX_MAX_INFLIGHT=3）；
- 循环内检查的 Claude CLI 也走 .wt_g1/.g1/bin/claude_gate.exe（限额一次整批停 20 min）；
- Codex 周用量 ≥ 60% 停派（--usage-stop 默认 60）；
- 清单 walker_g2_cases.json；run_config.teacher_override.batch 记 g2（文件名 / sha256 / base_freeze 与 g1 相同，pack 的覆盖核对认它）。

    python scripts\\g2\\g2_run.py --groups B_videosham --jobs 3
    python scripts\\g2\\g2_run.py --cases fmnv_c_xxx ... --jobs 3
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WT_G1 = ROOT.parent / ".wt_g1"
sys.path.insert(0, str(ROOT / "scripts" / "g1"))
os.environ.setdefault("G1_CODEX_MAX_INFLIGHT", "3")

import g1_run as g  # noqa: E402

g.ROUND = "r1g2"
g.STATE = WT_G1 / ".g1" / "state"
os.environ["G1_STATE"] = str(g.STATE)
os.environ["FOREN_CLAUDE_CLI"] = str(WT_G1 / ".g1" / "bin" / "claude_gate.exe")


class Runner(g.Runner):
    def record(self, case, run_dir, code, wall, attempt, failed):  # noqa: ANN001
        super().record(case, run_dir, code, wall, attempt, failed)
        cfg_path = run_dir / "run_config.json"
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if (cfg.get("teacher_override") or {}).get("batch") != "g2":
                cfg["teacher_override"]["batch"] = "g2"
                cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--groups", nargs="*", default=[])
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--deadline", default=None)
    parser.add_argument("--usage-stop", type=float, default=60.0)
    parser.add_argument("--disk-floor-gb", type=float, default=60.0)
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()
    cases = list(args.cases)
    if args.groups:
        groups = json.loads((g.wr.ROOT / g.wr.SFT / "walker_g2_cases.json").read_text(encoding="utf-8"))["groups"]
        for name in args.groups:
            cases += [c for c in groups.get(name, []) if c not in cases]
    if not cases:
        print("没有案例")
        return 2
    os.chdir(ROOT)
    log = g.wr.Log(f"_walker_{g.ROUND}日志.txt", f"g2 批 {g.ROUND}：{len(cases)} 个案例（S0–S5 Codex，g1_teacher.json；不核验）")
    log.say(f"开始 {g.now()}；E: 剩余 {g.disk_free_gb():.1f} GB；jobs {args.jobs}；同时在飞 ≤ {os.environ['G1_CODEX_MAX_INFLIGHT']}；"
            f"用量停派 {args.usage_stop}%；闸门状态 {g.STATE}")
    if not args.skip_preflight and not g.preflight(log):
        return 1
    Runner(args, log).run(cases)
    log.say(f"结束 {g.now()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
