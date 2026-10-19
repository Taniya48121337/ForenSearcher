"""g3a 金标（g3 目标分支专项）：照 _gold_sample.py 抽，不改它——只把 --new-action-quota 的四类（r1 的 E/L 图片比对、C 帧 / 转写）
换成 g3 的三类，答案只从 A 族 run 里抽。

- A_video：A 族（fmnv_a_ / fmnv_fv_ / groundlie360_a_）的 Finding 引了 compare_media 记录（找原片 → 与原片比对写出的结论）；
- locate：QVHighlights 的 Finding（INSPECT locate / Q 族定位链写出的结论）；
- query_memory：V9 run 里真发过 Q1_query_memory_before_report 的 Finding（查回掉出视图的证据再写报告）。
三类各 5 条（类内核验器标 major/critical 的先取，每 run 至多 1 条），共 15 条 Finding；答案 5 条只取 A 族 run，positive / negative 平衡。

    python scripts\\g3\\g3_gold.py
输出 experiments/sft_data/gold_set/g3a/（items.json、strata.json、media/、标注.html），格式与 r1a / g1a / g2a 相同。
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import _gold_sample as gs  # noqa: E402

SFT = ROOT / "experiments" / "sft_data"
VERDICTS = SFT / "verdicts_r1g3_judged.claude-cli-claude-sonnet-5-5.80runs-be437e21.jsonl"
UNITS = SFT / "verdicts_units_r1g3_judged.claude-cli-claude-sonnet-5-5.80runs-be437e21.jsonl"
A_PREFIX = ("fmnv_a_", "fmnv_fv_", "groundlie360_a_")
KINDS = ("A_video", "locate", "query_memory")
_Q1: dict[str, bool] = {}


def has_q1(run: str) -> bool:
    if run not in _Q1:
        _Q1[run] = any("Q1_query_memory_before_report" in p.read_text(encoding="utf-8", errors="replace")
                       for p in (SFT / "runs" / run / "requests").glob("*/transport.json"))
    return _Q1[run]


def kind_of(log: Any, run: str, fid: str) -> str | None:
    try:
        f = log.store.finding(fid)
    except KeyError:
        return None
    acq = set()
    for e in f.evidence_ids:
        try:
            t = log.lookup.get(e)
        except Exception:  # noqa: BLE001
            continue
        acq.add(str(t.acquired_by or "").lower() + "|" + str(t.label or "").lower())
    if run.startswith(A_PREFIX) and any("compare_media" in a for a in acq):
        return "A_video"
    if run.startswith("qvhighlights_"):
        return "locate"
    if "_v9_" in run and has_q1(run):
        return "query_memory"
    return None


def picks(cands: list[dict[str, Any]], quota: int, seed: str) -> list[dict[str, Any]]:
    """= _gold_sample.new_action_picks，去掉它只认 averimatec / fmnv / groundlie360 run 的前缀过滤，类换成 KINDS。"""

    logs: dict[str, Any] = {}
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    try:
        for c in cands:
            if c["run"] not in logs:
                logs[c["run"]] = gs.load_run(gs.run_dir(c["run"]))
            kind = kind_of(logs[c["run"]], c["run"], c["finding"])
            if kind:
                pools[kind].append({**c, "new_action": kind})
    finally:
        for log in logs.values():
            log.close()
    print({k: len(v) for k, v in pools.items()})
    out: list[dict[str, Any]] = []
    per = [quota // len(KINDS) + (1 if i < quota % len(KINDS) else 0) for i in range(len(KINDS))]
    for kind, n in zip(KINDS, per):
        pool = pools.get(kind) or []
        got = gs.stratified([c for c in pool if c["verifier_flagged"]], n, seed, per_run=1)
        got += gs.stratified([c for c in pool if not c["verifier_flagged"]], n - len(got), seed, per_run=1)
        out += got
    short = quota - len(out)
    if short > 0:
        taken = {(c["run"], c["finding"]) for c in out}
        rest = [c for k in KINDS for c in pools.get(k) or [] if (c["run"], c["finding"]) not in taken]
        out += gs.stratified(rest, short, seed, per_run=1)
    return out


def main() -> int:
    gs.NEW_ACTION_KINDS = KINDS
    gs.new_action_picks = picks
    # 答案只取 A 族 run：把 unit 判定先筛成只含 A 族答案单元的临时文件（plan 不抽）
    rows = [json.loads(l) for l in UNITS.read_text(encoding="utf-8").splitlines() if l.strip()]
    keep = [r for r in rows if r.get("unit") == "answer" and str((r.get("subject") or {}).get("run") or "").startswith(A_PREFIX)]
    tmp = Path(tempfile.mkdtemp()) / "units_a_answers.jsonl"
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
    print(f"A 族答案单元 {len(keep)}（positive {sum(1 for r in keep if r.get('disposition') == 'positive')}）")
    sys.argv = ["_gold_sample.py", "--findings", "15", "--new-action-quota", "15", "--oversample-flagged",
                "--answers", "5", "--answers-balanced", "--plans", "0", "--seed", "20261019g3",
                "--round-prefix", "r1g3", "--only-cases", "walker_r1g3_done_cases.json",
                "--report-verdicts", str(VERDICTS), "--unit-verdicts", str(tmp), "--out", str(SFT / "gold_set" / "g3a")]
    return gs.main()


if __name__ == "__main__":
    sys.exit(main())
