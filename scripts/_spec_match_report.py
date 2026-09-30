"""runtime ≠ spec 诊断：逐行比较运行时 system 与规格 student_prompt，分"去掉教师指令前 / 后"两种口径，
并把仍不一致的行按提示变体归类、给出第一处分歧。结果写到项目根目录 _spec一致性检查.txt。

    python scripts/_spec_match_report.py [--runs 名字 ...] [--hi-spec hi@0.2.0] [--gia-spec gia@0.3.0]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foren_searcher.datagen.export import spec_system_match, spec_system_prompt  # noqa: E402
from foren_searcher.datagen.runlog import load_run  # noqa: E402
from foren_searcher.datagen.specs import Spec  # noqa: E402

DEFAULT_RUNS = ("true_4124903_r1", "true_4124903_r2", "true_17838107_r1", "true_20205316753904_validation",
                "true_20361105_visibility_validation", "true_24171541709824_r1", "true_39132300_validation_r3",
                "videosham_0069_r1", "videosham_1007_validation", "videosham_1028_r2", "videosham_2010_r1",
                "videosham_4126_split_intervals_r7")


def variant(role: str, text: str) -> str:
    if role == "HI":
        kind = "multimodal" if text.startswith("你是独立 Workspace 内的多模态调查代理") else "text"
        return kind + ("+time_note" if "remaining_seconds 是含模型排队" in text else "")
    if "正在评估仍有 HI 运行的一轮调查" in text:
        return "wait"
    if "当前或历史轮次存在可汇总的 Finding" in text:
        return "findings" + ("+budget_out" if "调查轮次预算已经用完" in text else "")
    if "调查轮次预算已经用完" in text:
        return "budget_out"
    tag = "dispatch" if "本次首先决定哪些核验工作需要由 HI 完成" in text else "default"
    return tag + ("+empty_material" if "特别注意：当前视图和记忆动作中没有任何可用的 C/E 引用" in text else "")


def first_diff(a: str, b: str) -> str:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return f"@{n}: 运行时…{a[n:n + 60]!r} | 规格…{b[n:n + 60]!r}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    parser.add_argument("--hi-spec", default="hi@0.2.0")
    parser.add_argument("--gia-spec", default="gia@0.3.0")
    parser.add_argument("--out", default=str(ROOT / "_spec一致性检查.txt"))
    args = parser.parse_args()
    specs = {"HI": Spec.load(ROOT / "specs" / args.hi_spec), "GIA": Spec.load(ROOT / "specs" / args.gia_spec)}
    before, after = Counter(), Counter()
    mismatch_variants: dict[str, Counter] = defaultdict(Counter)
    matched_variants: dict[str, Counter] = defaultdict(Counter)
    examples: dict[tuple[str, str], str] = {}
    per_run: list[str] = []
    for name in args.runs:
        log = load_run(ROOT / "experiments" / "sft_data" / "runs" / name)
        run_counts = Counter()
        for d in log.decisions:
            spec = specs[d.role]
            _, old = spec_system_prompt(spec, dataclasses.replace(d, student_system_prompt=None))
            text, new, variant_name, repair = spec_system_match(spec, d)
            matched_variants[d.role][(variant_name or "未对上") + ("+repair" if repair else "")] += 1
            before[(d.role, old)] += 1
            after[(d.role, new)] += 1
            run_counts[(d.role, new)] += 1
            if not new:
                v = variant(d.role, d.student_system)
                mismatch_variants[d.role][v] += 1
                examples.setdefault((d.role, v), f"{name}/{d.request_id} " + first_diff(d.student_system.strip(), text))
        log.close()
        per_run.append(f"  {name}: HI {run_counts[('HI', True)]}/{run_counts[('HI', True)] + run_counts[('HI', False)]}，"
                       f"GIA {run_counts[('GIA', True)]}/{run_counts[('GIA', True)] + run_counts[('GIA', False)]} 一致")
    lines = [f"规格：{args.hi_spec} / {args.gia_spec}；{len(args.runs)} 个 run", ""]
    for role in ("HI", "GIA"):
        lines.append(f"{role}：原口径（system 含教师指令）一致 {before[(role, True)]} / 不一致 {before[(role, False)]}；"
                     f"去掉教师指令后一致 {after[(role, True)]} / 不一致 {after[(role, False)]}")
    lines += ["", "逐 run（去掉教师指令后）：", *per_run, "", "按规格变体归类（去掉教师指令与修复附注后）："]
    for role, counter in matched_variants.items():
        lines.append(f"  {role}: " + "，".join(f"{k} {v}" for k, v in counter.most_common()))
    lines += ["", "仍不一致的行按运行时提示特征："]
    for role, counter in mismatch_variants.items():
        for v, n in counter.most_common():
            lines.append(f"  {role} {v}: {n} 行；例 {examples[(role, v)]}")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
