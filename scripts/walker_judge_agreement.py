"""核验器一致性（2026-10-14 晚）：同一轮 run，Sonnet（本机 Claude CLI，基准）与第二个核验模型（OpenRouter 上的 Qwen，
走服务器本地核验器同一条 --judge local 路径）的判定逐单元对照。

    python scripts\\walker_judge_agreement.py --round wk13

读 verdicts[_units]_<轮次>_judged.claude-cli-*.jsonl 与 verdicts[_units]_<轮次>qv_judged.local-*.jsonl（各取最新）。
单元：report（每个工作区报告）、step / plan / answer。比的是 disposition（positive / negative / uncertain），
外加"会不会进 accepted"这一档（positive 且没有 major / critical 错误，与 export 的终端行严格入选一致）。
产物：experiments/sft_data/walker_<轮次>_judges.md / .json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "experiments" / "sft_data"


def _latest(pattern: str) -> Path | None:
    files = [p for p in SFT.glob(pattern) if ".summary" not in p.name and not p.name.endswith(".pre_revise")]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _rows(path: Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _key(row: dict[str, Any]) -> tuple[str, str, str]:
    subject = row.get("subject") or {}
    unit = str(row.get("unit") or "report")
    if unit == "report":
        return unit, str(subject.get("run")), str(subject.get("workspace_id"))
    return unit, str(subject.get("run")), str(subject.get("request"))


def _strict(row: dict[str, Any]) -> bool:
    return row.get("disposition") == "positive" and not any(
        e.get("severity") in ("major", "critical") for e in row.get("errors") or [])


def _kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe >= 1 else round((po - pe) / (1 - pe), 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", required=True)
    args = parser.parse_args()
    tag = args.round
    ref_files = [_latest(f"verdicts_{tag}_judged.claude-cli-*.jsonl"), _latest(f"verdicts_units_{tag}_judged.claude-cli-*.jsonl")]
    alt_files = [_latest(f"verdicts_{tag}qv_judged.*.jsonl"), _latest(f"verdicts_units_{tag}qv_judged.*.jsonl")]
    ref = {_key(r): r for f in ref_files for r in _rows(f)}
    alt = {_key(r): r for f in alt_files for r in _rows(f)}
    model = next((r.get("judge_model") for r in alt.values() if r.get("judge_model")), "?")
    common = sorted(set(ref) & set(alt))
    by_unit: dict[str, dict[str, Any]] = {}
    lines = [f"# 核验器一致性 {tag}：Sonnet（基准） vs {model}", "",
             f"基准文件：{', '.join(p.name for p in ref_files if p)}", f"对照文件：{', '.join(p.name for p in alt_files if p)}", "",
             "| 单元 | 共同 | disposition 一致 | 入选（positive 且无 major）一致 | kappa（入选） | Sonnet 入选 | 对照入选 | 对照放进了 Sonnet 不收的 | 对照漏掉了 Sonnet 收的 |",
             "|---|---|---|---|---|---|---|---|---|"]
    disagreements: list[dict[str, Any]] = []
    for unit in ("report", "answer", "plan", "step"):
        keys = [k for k in common if k[0] == unit]
        if not keys:
            continue
        same = sum(ref[k].get("disposition") == alt[k].get("disposition") for k in keys)
        pairs = [(_strict(ref[k]), _strict(alt[k])) for k in keys]
        loose = sum(1 for a, b in pairs if b and not a)
        tight = sum(1 for a, b in pairs if a and not b)
        confusion = Counter(f"{ref[k].get('disposition')}→{alt[k].get('disposition')}" for k in keys)
        by_unit[unit] = {"n": len(keys), "disposition_agree": same, "accept_agree": sum(a == b for a, b in pairs),
                         "kappa_accept": _kappa(pairs), "ref_accept": sum(a for a, _ in pairs), "alt_accept": sum(b for _, b in pairs),
                         "alt_looser": loose, "alt_stricter": tight, "confusion": dict(confusion)}
        u = by_unit[unit]
        lines.append(f"| {unit} | {u['n']} | {same}/{len(keys)} | {u['accept_agree']}/{len(keys)} | {u['kappa_accept']} | "
                     f"{u['ref_accept']} | {u['alt_accept']} | {loose} | {tight} |")
        for k in keys:
            a, b = _strict(ref[k]), _strict(alt[k])
            if a != b and unit in ("report", "answer"):
                disagreements.append({"unit": unit, "run": k[1], "id": k[2], "sonnet": ref[k].get("disposition"),
                                      "alt": alt[k].get("disposition"),
                                      "sonnet_errors": [f"{e.get('type')}/{e.get('severity')}" for e in ref[k].get("errors") or []
                                                        if e.get("severity") != "minor"],
                                      "alt_errors": [f"{e.get('type')}/{e.get('severity')}" for e in alt[k].get("errors") or []
                                                     if e.get("severity") != "minor"]})
    lines += ["", "混淆（Sonnet→对照）：" + "；".join(f"{u}: {json.dumps(v['confusion'], ensure_ascii=False)}" for u, v in by_unit.items()),
              "", "## 报告 / 答案判定不一致的", ""]
    for d in disagreements:
        lines.append(f"- {d['unit']} {d['run']} {d['id']}：Sonnet {d['sonnet']} {d['sonnet_errors']} ／ 对照 {d['alt']} {d['alt_errors']}")
    out = SFT / f"walker_{tag}_judges"
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"round": tag, "alt_model": model, "by_unit": by_unit,
                                                    "disagreements": disagreements}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
