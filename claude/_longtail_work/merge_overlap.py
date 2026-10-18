"""三包（r0g / r1 / g1）accepted 行：按分支的行数 / 案例数（各自与去重合并）、重复 case_id、规格 / 工具层 / 核验器版本差异。只读。

python claude/_longtail_work/merge_overlap.py > claude/_longtail_work/merge_overlap.json
去重规则（候选，给合包对话定）：同一 case_id 在 r0g 与 r1 都有时只留 r1（同一案卷、r1 是新规格 / 新工具层 / 新 search）。
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/sft_data/export"
SOURCES = {
    "r0g": EXP / "r0g_107runs-646e51cd_hi@0.10.0_gia@0.9.0_compact_16k.claude-cli-claude-sonnet-5-5",
    "r1": EXP / "r1_134runs-6a8cb9f4_hi@0.11.0_gia@0.9.0_compact_16k.claude-cli-claude-sonnet-5-5",
    "g1": ROOT / ".wt_g1/experiments/sft_data/export/r1g1_147runs-9460dc9a_hi@0.11.0_gia@0.9.0_compact_16k.claude-cli-claude-sonnet-5-5",
}


def family_of(case_id: str) -> str:
    ds, _, src = case_id.partition(":")
    if ds == "fmnv":
        return "A" if src.startswith("a_") else "C"
    if ds == "groundlie360":
        return "C"
    if ds in ("videosham", "vcsl", "qvhighlights"):
        return "B"
    if ds == "averimatec" and src[:1] in ("e", "l"):
        return src[0].upper()
    return "other"


def load(src: Path):
    out = []
    for name in ("hi_accepted.jsonl", "gia_accepted.jsonl"):
        with (src / name).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line)["metadata"])
    return out


def main():
    metas = {k: load(p) for k, p in SOURCES.items()}
    manifests = {k: json.loads((p / "manifest.json").read_text(encoding="utf-8")) for k, p in SOURCES.items()}
    cases = {k: {m["case_id"] for m in v} for k, v in metas.items()}
    runs = {k: sorted({m["run"] for m in v}) for k, v in metas.items()}
    r1_cases = cases["r1"]
    merged = metas["r1"] + metas["g1"] + [m for m in metas["r0g"] if m["case_id"] not in r1_cases]

    def branch_table(rows):
        t = defaultdict(lambda: [0, set()])
        for m in rows:
            b = m.get("branch") or "(none)"
            t[b][0] += 1
            t[b][1].add(m["case_id"])
        return {b: {"rows": n, "cases": len(c)} for b, (n, c) in t.items()}

    tables = {k: branch_table(v) for k, v in metas.items()}
    tables["sft0_dedup"] = branch_table(merged)
    branches = sorted(set().union(*[set(t) for t in tables.values()]))
    overlap = {
        "r0g&r1": sorted(cases["r0g"] & cases["r1"]),
        "r0g&g1": sorted(cases["r0g"] & cases["g1"]),
        "r1&g1": sorted(cases["r1"] & cases["g1"]),
    }
    overlap_family = {k: dict(Counter(family_of(c) for c in v)) for k, v in overlap.items()}
    r0g_only = sorted(cases["r0g"] - cases["r1"])
    versions = {}
    for k, man in manifests.items():
        versions[k] = {
            "spec": man.get("spec"), "spec_hash": man.get("spec_hash"), "verifier": man.get("verifier"),
            "state_renderer": man.get("state_renderer"),
            "tool_layer_hash": list((man.get("freeze") or {}).get("tool_layer_hash", {}).keys()),
            "freeze_id": (man.get("freeze") or {}).get("freeze_id"),
            "slot_models": list((man.get("freeze") or {}).get("slot_models", {}).keys()),
            "export_options": man.get("export_options"),
            "counts": man.get("counts"),
            "judge_models": dict(Counter(m.get("judge_model") for m in metas[k])),
            "origin": dict(Counter(m.get("origin") for m in metas[k])),
            "rows_freeze_ok": dict(Counter(str(m.get("freeze_ok")) for m in metas[k])),
            "teacher_model": dict(Counter(m.get("teacher_model") for m in metas[k])),
            "rows_by_family": dict(Counter(family_of(m["case_id"]) for m in metas[k])),
        }
    # r0g 行里与 r1 重复 case 的行数（去重时要丢的）
    dropped = Counter(family_of(m["case_id"]) for m in metas["r0g"] if m["case_id"] in r1_cases)
    kept_r0g = Counter(family_of(m["case_id"]) for m in metas["r0g"] if m["case_id"] not in r1_cases)
    merged_fam = Counter(family_of(m["case_id"]) for m in merged)
    merged_ds = Counter(m["dataset"] for m in merged if m["role"] == "hi")
    merged_ds_w = Counter()
    for m in merged:
        if m["role"] == "hi":
            merged_ds_w[m["dataset"]] += float(m.get("sample_weight") or 0)
    tw = sum(merged_ds_w.values())
    out = {
        "rows": {k: len(v) for k, v in metas.items()} | {"sft0_dedup": len(merged)},
        "hi_rows": {k: sum(1 for m in v if m["role"] == "hi") for k, v in metas.items()} | {"sft0_dedup": sum(1 for m in merged if m["role"] == "hi")},
        "gia_rows": {k: sum(1 for m in v if m["role"] == "gia") for k, v in metas.items()} | {"sft0_dedup": sum(1 for m in merged if m["role"] == "gia")},
        "cases": {k: len(v) for k, v in cases.items()},
        "runs": {k: len(v) for k, v in runs.items()},
        "overlap_counts": {k: len(v) for k, v in overlap.items()}, "overlap_family": overlap_family, "overlap": overlap,
        "r0g_only_cases": r0g_only,
        "r0g_rows_dropped_by_dedup": dict(dropped), "r0g_rows_kept": dict(kept_r0g),
        "sft0_rows_by_family": dict(merged_fam),
        "sft0_hi_dataset_rows": dict(merged_ds),
        "sft0_hi_dataset_weight_share": {k: round(v / tw, 3) for k, v in merged_ds_w.items()},
        "versions": versions,
        "branches": {b: {k: tables[k].get(b, {"rows": 0, "cases": 0}) for k in tables} for b in branches},
    }
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
