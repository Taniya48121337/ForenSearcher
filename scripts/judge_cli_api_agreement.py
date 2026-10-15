"""核验器换传输的一致率（2026-10-15，对话 5a，清单 A5）：同一轮 run、同一份核验代码（verifier@0.3.6，code_hash 不变），
Sonnet 经本机 Claude CLI（--judge claude-cli，基准）与经 API（--judge api，OpenRouter 上的 anthropic/claude-sonnet-5.5）
两次判定逐单元对照。

    python scripts\\judge_cli_api_agreement.py --round wk13 --api-round wk13api

读（各取最新、跳过 .pre_revise）：
  基准  verdicts_<轮次>_judged.claude-cli-*.jsonl、verdicts_units_<轮次>_judged.claude-cli-*.jsonl、export\\<轮次>_*claude-cli*\\
  对照  verdicts_<api轮次>_judged.api-*.jsonl、verdicts_units_<api轮次>_judged.api-*.jsonl、export\\<api轮次>_*api-*\\
单元：report（工作区报告，三层核验）、answer（逐句锚定）——这两种是模型判的；plan / step 取导出目录的 verdicts_units.jsonl
（程序层 + 模型层合并后的结果；walker 的 plan 没有模型评判、step 只有程序层，所以它们的一致率主要反映"导出时用的报告判定"
有没有变）。另比导出行的去向（accepted / held / rejected），按 report / step / plan / answer 分。
指标：disposition（positive / negative / uncertain）一致、"严格入选"（positive 且无 major / critical）一致与 κ、混淆表、
不一致清单（附双方的非 minor 错误类型）。产物 experiments\\sft_data\\walker_<轮次>_cli_vs_api.md / .json。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "experiments" / "sft_data"
UNITS = ("report", "answer", "plan", "step")


def _latest(pattern: str, base: Path = SFT) -> Path | None:
    files = [p for p in base.glob(pattern) if ".summary" not in p.name and not p.name.endswith(".pre_revise")
             and ".v0" not in p.name]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _rows(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.is_file():
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


def _errors(row: dict[str, Any]) -> list[str]:
    return [f"{e.get('type')}/{e.get('severity')}" for e in row.get("errors") or [] if e.get("severity") != "minor"]


def _export_dir(round_tag: str, label: str) -> Path | None:
    dirs = [p for p in (SFT / "export").glob(f"{round_tag}_*") if p.is_dir() and p.name.split(".", 1)[-1].startswith(label)
            and p.name.split("_", 1)[0] == round_tag]
    return max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None


def _buckets(export: Path | None) -> dict[tuple[str, str], tuple[str, str]]:
    """导出行 → (unit, bucket)，键 (run, request)。"""

    out: dict[tuple[str, str], tuple[str, str]] = {}
    if not export:
        return out
    for role in ("hi", "gia"):
        for bucket in ("accepted", "held", "rejected"):
            path = export / f"{role}_{bucket}.jsonl"
            if not path.is_file():
                continue
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        meta = json.loads(line).get("metadata") or {}
                    except ValueError:
                        continue
                    out[(str(meta.get("run")), str(meta.get("request")))] = (str(meta.get("unit")), bucket)
    return out


def compare(ref: dict[tuple[str, str, str], dict[str, Any]], alt: dict[tuple[str, str, str], dict[str, Any]],
            units: tuple[str, ...]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_unit: dict[str, Any] = {}
    disagreements: list[dict[str, Any]] = []
    common = sorted(set(ref) & set(alt))
    for unit in units:
        keys = [k for k in common if k[0] == unit]
        if not keys:
            by_unit[unit] = {"n": 0, "ref_only": sum(1 for k in ref if k[0] == unit), "alt_only": sum(1 for k in alt if k[0] == unit)}
            continue
        same = sum(ref[k].get("disposition") == alt[k].get("disposition") for k in keys)
        pairs = [(_strict(ref[k]), _strict(alt[k])) for k in keys]
        by_unit[unit] = {
            "n": len(keys), "disposition_agree": same, "disposition_rate": round(same / len(keys), 3),
            "accept_agree": sum(a == b for a, b in pairs), "accept_rate": round(sum(a == b for a, b in pairs) / len(keys), 3),
            "kappa_accept": _kappa(pairs), "cli_accept": sum(a for a, _ in pairs), "api_accept": sum(b for _, b in pairs),
            "api_looser": sum(1 for a, b in pairs if b and not a), "api_stricter": sum(1 for a, b in pairs if a and not b),
            "confusion": dict(Counter(f"{ref[k].get('disposition')}→{alt[k].get('disposition')}" for k in keys)),
            "ref_only": sum(1 for k in ref if k[0] == unit and k not in alt), "alt_only": sum(1 for k in alt if k[0] == unit and k not in ref),
            "cli_error_types": dict(Counter(e.split("/")[0] for k in keys for e in _errors(ref[k]))),
            "api_error_types": dict(Counter(e.split("/")[0] for k in keys for e in _errors(alt[k]))),
        }
        for k in keys:
            if ref[k].get("disposition") != alt[k].get("disposition") or _strict(ref[k]) != _strict(alt[k]):
                disagreements.append({"unit": unit, "run": k[1], "id": k[2], "cli": ref[k].get("disposition"),
                                      "api": alt[k].get("disposition"), "cli_errors": _errors(ref[k]), "api_errors": _errors(alt[k])})
    return by_unit, disagreements


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", required=True, help="基准轮次（claude-cli 判过的），例如 wk13")
    parser.add_argument("--api-round", default=None, help="API 核验的产物轮次标签，默认 <轮次>api")
    args = parser.parse_args()
    tag, api_tag = args.round, args.api_round or f"{args.round}api"
    ref_files = [_latest(f"verdicts_{tag}_judged.claude-cli-*.jsonl"), _latest(f"verdicts_units_{tag}_judged.claude-cli-*.jsonl")]
    alt_files = [_latest(f"verdicts_{api_tag}_judged.api-*.jsonl"), _latest(f"verdicts_units_{api_tag}_judged.api-*.jsonl")]
    ref = {_key(r): r for f in ref_files for r in _rows(f)}
    alt = {_key(r): r for f in alt_files for r in _rows(f)}
    ref_export, alt_export = _export_dir(tag, "claude-cli"), _export_dir(api_tag, "api-")
    # 导出目录里的单元判定（含 plan / step；report 与 answer 用上面的核验文件为准）
    ref_units = {_key(r): r for r in _rows(ref_export / "verdicts_units.jsonl" if ref_export else None)}
    alt_units = {_key(r): r for r in _rows(alt_export / "verdicts_units.jsonl" if alt_export else None)}
    for k, v in ref_units.items():
        if k[0] in ("plan", "step"):
            ref.setdefault(k, v)
    for k, v in alt_units.items():
        if k[0] in ("plan", "step"):
            alt.setdefault(k, v)
    models = (next((r.get("judge_model") for r in ref.values() if r.get("judge_model")), "?"),
              next((r.get("judge_model") for r in alt.values() if r.get("judge_model")), "?"))
    by_unit, disagreements = compare(ref, alt, UNITS)
    # 导出行去向
    rb, ab = _buckets(ref_export), _buckets(alt_export)
    rows_by_unit: dict[str, Any] = {}
    for unit in UNITS:
        keys = [k for k in set(rb) & set(ab) if rb[k][0] == unit]
        if not keys:
            continue
        pairs = [(rb[k][1] == "accepted", ab[k][1] == "accepted") for k in keys]
        rows_by_unit[unit] = {"n": len(keys), "bucket_agree": sum(rb[k][1] == ab[k][1] for k in keys),
                              "accepted_agree": sum(a == b for a, b in pairs), "kappa_accepted": _kappa(pairs),
                              "cli_accepted": sum(a for a, _ in pairs), "api_accepted": sum(b for _, b in pairs),
                              "confusion": dict(Counter(f"{rb[k][1]}→{ab[k][1]}" for k in keys))}
    lines = [f"# 核验器 CLI vs API 一致率（{tag}）：{models[0]}（基准） vs {models[1]}", "",
             "同一批 run、同一份核验代码（verifier@0.3.6，code_hash 不变）；只换传输（claude -p → OpenAI 兼容 API）。", "",
             f"基准：{', '.join(p.name for p in ref_files if p)}；导出 {ref_export.name if ref_export else '（无）'}",
             f"对照：{', '.join(p.name for p in alt_files if p)}；导出 {alt_export.name if alt_export else '（无）'}", "",
             "## 逐单元判定", "",
             "| 单元 | 共同 | disposition 一致 | 严格入选一致 | κ（入选） | CLI 入选 | API 入选 | API 放进 CLI 不收的 | API 漏掉 CLI 收的 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for unit in UNITS:
        u = by_unit.get(unit) or {}
        if not u.get("n"):
            lines.append(f"| {unit} | 0 | — | — | — | — | — | — | —（基准 {u.get('ref_only', 0)} 条 / 对照 {u.get('alt_only', 0)} 条，无共同） |")
            continue
        lines.append(f"| {unit} | {u['n']} | {u['disposition_agree']}/{u['n']}（{u['disposition_rate']:.0%}） | "
                     f"{u['accept_agree']}/{u['n']}（{u['accept_rate']:.0%}） | {u['kappa_accept']} | {u['cli_accept']} | "
                     f"{u['api_accept']} | {u['api_looser']} | {u['api_stricter']} |")
    lines += ["", "混淆（CLI→API）：" + "；".join(f"{k}: {json.dumps(v.get('confusion'), ensure_ascii=False)}"
                                                for k, v in by_unit.items() if v.get("n")), ""]
    if rows_by_unit:
        lines += ["## 导出行去向（accepted / held / rejected）", "",
                  "| 单元 | 共同行 | 去向一致 | accepted 一致 | κ | CLI accepted | API accepted |", "|---|---|---|---|---|---|---|"]
        for unit, u in rows_by_unit.items():
            lines.append(f"| {unit} | {u['n']} | {u['bucket_agree']}/{u['n']} | {u['accepted_agree']}/{u['n']} | {u['kappa_accepted']} | "
                         f"{u['cli_accepted']} | {u['api_accepted']} |")
        lines.append("")
    lines += ["## 错误类型（非 minor，共同单元）", ""]
    for unit in ("report", "answer"):
        u = by_unit.get(unit) or {}
        if u.get("n"):
            lines.append(f"- {unit}：CLI {json.dumps(u['cli_error_types'], ensure_ascii=False)} ／ API {json.dumps(u['api_error_types'], ensure_ascii=False)}")
    lines += ["", "## 判定不一致的单元", ""]
    for d in disagreements:
        lines.append(f"- {d['unit']} {d['run']} {d['id']}：CLI {d['cli']} {d['cli_errors']} ／ API {d['api']} {d['api_errors']}")
    if not disagreements:
        lines.append("（无）")
    out = SFT / f"walker_{tag}_cli_vs_api"
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"round": tag, "api_round": api_tag, "models": models, "by_unit": by_unit,
                                                    "rows_by_unit": rows_by_unit, "disagreements": disagreements,
                                                    "files": {"ref": [str(p) for p in ref_files if p], "alt": [str(p) for p in alt_files if p],
                                                              "ref_export": str(ref_export) if ref_export else None,
                                                              "alt_export": str(alt_export) if alt_export else None}},
                                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 0 if alt else 2


if __name__ == "__main__":
    raise SystemExit(main())
