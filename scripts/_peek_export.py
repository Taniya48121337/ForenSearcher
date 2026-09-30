"""把一份导出目录变成人能读的样例页（2026-09-30）：看"新规格下训练数据长什么样"。只读导出，不改任何东西。

    python scripts/_peek_export.py <导出目录，可带 * 通配，取最新的一个> [--hi 12] [--seed 20260930] [--out <导出目录>/样例.md]
                                   [--compare-verdicts 旧判定.jsonl 新判定.jsonl]

输出：
- 总览：各桶行数、按单元 / 动作 / 数据集的 accepted 分布、带 rationale 的比例、监督 token 中位数；
- HI accepted 按动作分层抽 --hi 条、GIA accepted 全部：每条给出上下文（系统提示只给长度；用户消息保留头尾）和教师输出全文；
- --compare-verdicts：两份报告判定逐份对照（同一 run + 工作区），看换核验器后哪些报告翻了。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HEAD, TAIL = 1500, 2500


def load(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "[图片]") if isinstance(c, dict) else str(c) for c in content)
    return str(content)


def clip(text: str) -> str:
    if len(text) <= HEAD + TAIL + 200:
        return text
    return f"{text[:HEAD]}\n\n…（中间省略 {len(text) - HEAD - TAIL} 字）…\n\n{text[-TAIL:]}"


def render_row(i: int, row: dict[str, Any]) -> str:
    m = row["metadata"]
    msgs = row["messages"]
    system = next((text_of(x["content"]) for x in msgs if x["role"] == "system"), "")
    user = "\n\n".join(text_of(x["content"]) for x in msgs if x["role"] == "user")
    target = text_of(msgs[-1]["content"]) if msgs and msgs[-1]["role"] == "assistant" else ""
    try:
        target = json.dumps(json.loads(target), ensure_ascii=False, indent=1)
    except ValueError:
        pass
    head = (f"### {i}. {m['role'].upper()} · {m['unit']} · {m.get('action') or ''} · {m['run']} "
            f"{m.get('workspace') or ''} {m['request']}\n\n"
            f"权重 {m['sample_weight']} · 判定 {m.get('verdict')} · 错误 {', '.join(m.get('error_types') or []) or '无'} · "
            f"图片 {len(row.get('images') or [])} 张 · 输入约 {m.get('input_tokens_est')} token · 监督约 {m.get('supervised_tokens_est')} token\n")
    return (head + f"\n<details><summary>输入（系统提示 {len(system)} 字，略；用户消息 {len(user)} 字）</summary>\n\n```text\n"
            + clip(user) + "\n```\n</details>\n\n**教师输出（训练目标）**\n\n```json\n" + target + "\n```\n")


def compare(old: Path, new: Path) -> list[str]:
    def key(v: dict[str, Any]) -> tuple[Any, ...]:
        return v["subject"].get("run"), v["subject"].get("workspace_id")

    a = {key(v): v for v in load(old) if v.get("unit", "report") == "report"}
    b = {key(v): v for v in load(new) if v.get("unit", "report") == "report"}
    table = Counter((a[k]["disposition"], b[k]["disposition"]) for k in a.keys() & b.keys())
    lines = [f"## 报告判定对照：{old.name} → {new.name}", "", f"共同报告 {sum(table.values())} 份", "",
             "| 旧 \\ 新 | positive | negative | uncertain |", "|---|---|---|---|"]
    for d in ("positive", "negative", "uncertain"):
        lines.append(f"| {d} | " + " | ".join(str(table.get((d, e), 0)) for e in ("positive", "negative", "uncertain")) + " |")
    lines.append("")
    return lines


def newest(pattern: str) -> Path:
    """路径里带 * 时按通配找，取修改时间最新的一个；不带就原样返回。"""

    if "*" not in pattern:
        return Path(pattern)
    path = Path(pattern)
    anchor = next(i for i, part in enumerate(path.parts) if "*" in part)
    base = Path(*path.parts[:anchor]) if anchor else Path(".")
    found = sorted(base.glob(str(Path(*path.parts[anchor:]))), key=lambda x: x.stat().st_mtime)
    if not found:
        raise SystemExit(f"没有匹配 {pattern} 的文件或目录")
    return found[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("export")
    parser.add_argument("--hi", type=int, default=12)
    parser.add_argument("--seed", default="20260930")
    parser.add_argument("--out", default=None)
    parser.add_argument("--compare-verdicts", nargs=2, default=None)
    args = parser.parse_args()
    root = newest(args.export)
    buckets = {f"{r}_{b}": load(root / f"{r}_{b}.jsonl") for r in ("hi", "gia") for b in ("accepted", "held", "rejected")}
    acc = buckets["hi_accepted"] + buckets["gia_accepted"]
    lines = [f"# 导出样例：{root.name}", ""]
    if acc:
        m0 = acc[0]["metadata"]
        lines.append(f"教师 {m0.get('teacher_model')} · 核验 {m0.get('judge_model')} · {m0.get('verifier')} · 规格 "
                     f"{', '.join(sorted({r['metadata']['spec'] for r in acc}))} · 渲染 {m0.get('state_renderer')}")
        lines.append("")
    lines += ["| 桶 | 行数 |", "|---|---|"] + [f"| {k} | {len(v)} |" for k, v in buckets.items()] + [""]
    if acc:
        sup = [r["metadata"].get("supervised_tokens_est") or 0 for r in acc]
        rat = sum(1 for r in acc if r["metadata"].get("has_rationale"))
        lines.append(f"accepted 共 {len(acc)} 行：带 rationale {rat}（{rat / len(acc):.0%}），监督 token 中位 {statistics.median(sup):.0f}，"
                     f"沾染核查站点的目标 {sum(1 for r in acc if r['metadata'].get('fact_check_in_target'))} 行")
        lines.append("")
        for label, keyf in (("单元", lambda m: f"{m['role']}/{m['unit']}"), ("动作", lambda m: m.get("action") or "—"),
                            ("数据集", lambda m: m.get("dataset") or "—")):
            c = Counter(keyf(r["metadata"]) for r in acc)
            lines.append(f"- 按{label}：" + "，".join(f"{k} {n}" for k, n in c.most_common()))
        lines.append("")
    if args.compare_verdicts:
        lines += compare(newest(args.compare_verdicts[0]), newest(args.compare_verdicts[1]))
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in buckets["hi_accepted"]:
        by_action[f"{r['metadata']['unit']}:{r['metadata'].get('action')}"].append(r)
    picks: list[dict[str, Any]] = []
    groups = [sorted(v, key=lambda r: hashlib.sha1((args.seed + r["metadata"]["run"] + r["metadata"]["request"]).encode()).hexdigest())
              for _, v in sorted(by_action.items())]
    while len(picks) < args.hi and any(groups):
        for g in groups:
            if g and len(picks) < args.hi:
                picks.append(g.pop(0))
    lines += ["## HI accepted 抽样（按动作轮转）", ""] + [render_row(i, r) for i, r in enumerate(picks, 1)]
    lines += ["## GIA accepted 全部", ""] + [render_row(i, r) for i, r in enumerate(buckets["gia_accepted"], 1)]
    out = Path(args.out) if args.out else root / "样例.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"\n写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
