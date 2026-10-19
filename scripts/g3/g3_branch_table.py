"""g3 目标分支前后对照（任务书第 13 步）：g3 导出 accepted 行里每个目标分支 / walker 规则的行数、来自几个案例、是否达到"可学标准"
（accepted ≥ 15 行且 ≥ 5 个不同 case_id，《r0r1训练数据可训性评估》二节），与 r1 包的数并排；held / rejected 里同分支的行数一并列出（看没达标是
没触发还是被 held）。V9 run 的 case_id 与底案相同（不计新案例）。

    python scripts\\g3\\g3_branch_table.py
输出 claude/_g3_work/branch_table.json 与 .md。
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT.parent
G3 = ROOT / "experiments/sft_data/export/r1g3_80runs-be437e21_hi@0.11.0_gia@0.9.0_compact_16k.claude-cli-claude-sonnet-5-5.g1freeze"
R1 = MAIN / "experiments/sft_data/export/r1_134runs-6a8cb9f4_hi@0.11.0_gia@0.9.0_compact_16k.claude-cli-claude-sonnet-5-5"
OUT = ROOT / "claude" / "_g3_work" / "branch_table"

# 目标（任务书 P1–P4）：按 metadata.branch 或 walker_rule 前缀计
TARGETS = [
    ("P1", "SEARCH text→video", ("branch", "SEARCH text→video")),
    ("P1", "SEARCH text_image→video", ("branch", "SEARCH text_image→video")),
    ("P1", "A 族后续：compare_media（M1）", ("rule_a", "M1_compare_media")),
    ("P1", "A 族后续：取帧（M3_frames_*）", ("rule_a", "M3_frames")),
    ("P1", "A 族后续：查原片日期（A6）", ("rule", "A6_dates_of_original")),
    ("P1", "A 族报告（M7_report）", ("rule_a", "M7_report")),
    ("P2", "QUERY_MEMORY (HI)", ("branch", "QUERY_MEMORY (HI)")),
    ("P3", "INSPECT locate（text / image / text_image）", ("branch_prefix", "INSPECT locate")),
    ("P3", "Q 族（Q1–Q4）", ("rule_prefix", ("Q1_locate", "Q1b_", "Q1c_", "Q2_", "Q3_", "Q4_"))),
    ("P3", "F 族（F7 / F8）", ("rule_prefix", ("F7_", "F8_"))),
    ("P4", "READ_RECORD offset>0（C5_read_more）", ("branch", "READ_RECORD offset>0")),
    ("P4", "C8 打开页内视频（C8 / C8b / C8c）", ("rule_prefix", ("C8_", "C8b_", "C8c_"))),
    ("P4", "SEARCH open video", ("branch", "SEARCH open video")),
]
A_PREFIX = ("fmnv_a_", "fmnv_fv_", "groundlie360_a_")


def rows(export: Path, bucket: str) -> list[dict]:
    path = export / f"hi_{bucket}.jsonl"
    return [json.loads(l)["metadata"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.is_file() else []


def match(meta: dict, how: tuple) -> bool:
    kind, val = how
    branch, rule, run = str(meta.get("branch") or ""), str(meta.get("walker_rule") or ""), str(meta.get("run") or "")
    if kind == "branch":
        return branch == val
    if kind == "branch_prefix":
        return branch.startswith(val)
    if kind == "rule":
        return rule == val
    if kind == "rule_a":
        return rule.startswith(val) and run.startswith(A_PREFIX)
    if kind == "rule_prefix":
        return rule.startswith(val)
    return False


def stat(metas: list[dict], how: tuple) -> dict:
    hit = [m for m in metas if match(m, how)]
    return {"rows": len(hit), "cases": len({m.get("case_id") for m in hit}),
            "weight": round(sum(float(m.get("sample_weight") or 0) for m in hit), 2)}


def main() -> int:
    g3 = {b: rows(G3, b) for b in ("accepted", "held", "rejected")}
    r1 = rows(R1, "accepted")
    held_reasons: dict[str, Counter] = defaultdict(Counter)
    table = []
    for p, name, how in TARGETS:
        a, r = stat(g3["accepted"], how), stat(r1, how)
        h, j = stat(g3["held"], how), stat(g3["rejected"], how)
        for m in g3["held"]:
            if match(m, how):
                held_reasons[name][str(m.get("bucket_reason") or "?").split(":")[0][:60]] += 1
        learn = a["rows"] >= 15 and a["cases"] >= 5
        table.append({"P": p, "branch": name, "g3": a, "g3_held": h["rows"], "g3_rejected": j["rows"], "r1": r,
                      "learnable_g3": learn, "r1_plus_g3": {"rows": a["rows"] + r["rows"]}})
    OUT.with_suffix(".json").write_text(json.dumps({"table": table, "held_reasons": {k: dict(v) for k, v in held_reasons.items()}},
                                                   ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["| P | 分支 | g3 accepted 行 / 案例 | g3 可学？ | g3 held / rejected | r1 accepted 行 / 案例 | r1 + g3 行 |", "|---|---|---|---|---|---|---|"]
    for t in table:
        lines.append(f"| {t['P']} | {t['branch']} | {t['g3']['rows']} / {t['g3']['cases']} | {'✅' if t['learnable_g3'] else '—'} | "
                     f"{t['g3_held']} / {t['g3_rejected']} | {t['r1']['rows']} / {t['r1']['cases']} | {t['r1_plus_g3']['rows']} |")
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps({k: dict(v) for k, v in held_reasons.items()}, ensure_ascii=False, indent=1)[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
