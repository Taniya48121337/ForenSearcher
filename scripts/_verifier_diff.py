"""两版核验结果对照（2026-09-30，verifier@0.3.2 → 0.3.3 重核 r0c 用）。只读。

    python scripts/_verifier_diff.py --old 旧verdicts.jsonl [旧units.jsonl ...] --new 新verdicts.jsonl [新units.jsonl ...] [--out 对照.md]

参数可以是 glob。按 (run, 工作区或请求, 单元) 配对，列出：
- 处置转移表（positive / negative / uncertain 之间怎么变）；
- 每个变了处置的单元：去掉了哪些错误、新增了哪些错误（类型、严重度、说明前 80 字）；
- answer 单元的标签判定（label_checks）。
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load(patterns: list[str]) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            if not Path(path).is_file():
                continue
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                v = json.loads(line)
                s = v.get("subject") or {}
                key = (str(s.get("run")), str(s.get("workspace_id") or s.get("request") or ""), str(v.get("unit") or "report"))
                out[key] = v
    return out


def err_key(e: dict[str, Any]) -> tuple[str, str, str]:
    return (e.get("type", ""), e.get("severity", ""), str(e.get("span") or e.get("detail") or "")[:80])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", nargs="+", required=True)
    parser.add_argument("--new", nargs="+", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    old, new = load(args.old), load(args.new)
    keys = sorted(set(old) & set(new))
    lines = [f"# 核验结果对照：{len(keys)} 个单元（旧 {len(old)}，新 {len(new)}）", ""]
    versions = Counter(v.get("verifier") for v in new.values())
    lines.append(f"新结果的核验器版本：{dict(versions)}")
    lines += ["", "## 处置转移", "", "| 单元 | 旧 → 新 | 个数 |", "|---|---|---|"]
    moves = Counter((k[2], old[k].get("disposition"), new[k].get("disposition")) for k in keys)
    for (unit, a, b), n in sorted(moves.items()):
        lines.append(f"| {unit} | {a} → {b}{'' if a == b else ' ★'} | {n} |")
    lines += ["", "## 变了处置的单元", ""]
    for k in keys:
        a, b = old[k], new[k]
        if a.get("disposition") == b.get("disposition"):
            continue
        ea, eb = Counter(map(err_key, a.get("errors", []))), Counter(map(err_key, b.get("errors", [])))
        lines.append(f"### {k[0]} {k[1]}（{k[2]}）：{a.get('disposition')} → {b.get('disposition')}")
        for e, n in (ea - eb).items():
            lines.append(f"- 去掉 {e[0]}（{e[1]}）×{n}：{e[2]}")
        for e, n in (eb - ea).items():
            lines.append(f"- 新增 {e[0]}（{e[1]}）×{n}：{e[2]}")
        lines.append("")
    labels = [(k, c) for k in keys if k[2] == "answer" for c in (new[k].get("extra") or {}).get("label_checks", [])]
    if labels:
        lines += ["## answer 标签判定", "", "| run | 标签 | 子串命中 | 方法 | 判定 | 摘录 |", "|---|---|---|---|---|---|"]
        for k, c in labels:
            lines.append(f"| {k[0]} | {c.get('label')} | {c.get('program_hit')} | {c.get('method')} | {c.get('status', c.get('judge_failed', '—'))} "
                         f"| {str(c.get('quote') or '')[:40]} |")
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
