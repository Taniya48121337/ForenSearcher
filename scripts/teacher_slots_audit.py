"""分槽教师 + 核验器的核对（2026-10-15，对话 5a；对话 5 改为本机订阅口径）：每个槽走的是不是冻结的后端、核验是不是冻结的核验器。

    python scripts\\teacher_slots_audit.py --round r0s1

逐 run 查：
- 每次槽调用（requests/*/transport.json 的 slots，及核验后重写 slot_S5_revise.json 的 slots）：backend 必须是该槽后端的类型
  （run_config.walker.slot_teachers[Sk].type：10-15 对话 5 起 S0–S3 = endpoint（OpenRouter Qwen），S4 / S5 = codex（Codex CLI
  订阅）；strong 换备用时 S4 / S5 = endpoint），槽号 → 模型必须与 run_config.walker.slot_models 一致；codex 槽的 model_served
  有值时须与请求的模型一致；Codex 痕迹文件（slot_*_events.jsonl、slot_*_stderr.txt）只许出现在 codex 槽；不能有
  backend=claude、没有 backend 的老记录；
- run_config：walker.slot_backend = slots、slot_models 每个槽都有、freeze.ok；
- 核验：verdicts_<轮次>_judged.* 与 verdicts_units_<轮次>_judged.* 的 judge_model 前缀全是冻结的 verifier.judge
  （10-15 对话 5 起 claude-cli:…）；
- 导出：每行都有 slot_teacher / terminal_teacher，且与 run_config 一致。
产物：experiments\\sft_data\\teacher_slots_audit_<轮次>.md / .json；有问题时退出码 1。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "experiments" / "sft_data"
RUNS = SFT / "runs"
SLOT = re.compile(r"^(S[0-9])")
CLI_FILE = re.compile(r"^slot_(S[0-9].*?)(?:_attempt_\d+_events\.jsonl|_stderr\.txt)$")
FREEZE = ROOT / "specs" / "r0_freeze.json"


def frozen_judge() -> str:
    """冻结的核验器前缀（verifier.judge）：10-15 对话 5 起 claude-cli；读不到时按 claude-cli。"""

    return str(((_load(FREEZE) or {}).get("verifier") or {}).get("judge") or "claude-cli")


def _same_model(actual: str | None, expected: str | None) -> bool:
    def norm(m: str | None) -> str:
        return str(m or "").rsplit("/", 1)[-1].lower().replace(".", "-")

    return bool(actual and expected) and (norm(actual) == norm(expected) or norm(actual).startswith(norm(expected) + "-"))


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def audit_run(run_dir: Path) -> dict[str, Any]:
    cfg = _load(run_dir / "run_config.json") or {}
    walker = cfg.get("walker") or {}
    expected = walker.get("slot_models") or {}
    kinds = {k: str((v or {}).get("type") or "endpoint") for k, v in (walker.get("slot_teachers") or {}).items()}
    problems: list[str] = []
    calls: Counter[str] = Counter()
    cost = 0.0
    if walker.get("slot_backend") != "slots":
        problems.append(f"run_config.walker.slot_backend = {walker.get('slot_backend')!r}（应为 slots）")
    missing = [k for k in ("S0", "S1", "S2", "S3", "S4", "S5") if not expected.get(k)]
    if missing:
        problems.append(f"run_config.walker.slot_models 缺 {missing}")
    if (cfg.get("freeze") or {}).get("ok") is not True:
        problems.append(f"run_config.freeze.ok = {(cfg.get('freeze') or {}).get('ok')}（failed {(cfg.get('freeze') or {}).get('failed')}）")
    records: list[tuple[str, dict[str, Any]]] = []
    for transport in sorted((run_dir / "requests").glob("request_*/transport.json")):
        meta = _load(transport) or {}
        records += [(transport.parent.name, s) for s in meta.get("slots") or [] if isinstance(s, dict)]
        if (meta.get("configured_model") or {}).get("model_provider") not in (None, "slot_table"):
            problems.append(f"{transport.parent.name}: configured_model.model_provider = {(meta.get('configured_model') or {}).get('model_provider')}")
    for revise in sorted((run_dir / "requests").glob("request_*/slot_S5_revise.json")):
        records += [(revise.parent.name + "/revise", s) for s in (_load(revise) or {}).get("slots") or [] if isinstance(s, dict)]
    for where, slot in records:
        name = str(slot.get("name") or "?")
        key = (SLOT.match(name) or [None, name])[1]
        model = slot.get("model")
        calls[f"{key}|{slot.get('backend')}|{slot.get('slot_backend')}|{model}"] += 1
        cost += float(slot.get("cost_usd") or 0.0)
        want = kinds.get(key, "endpoint")
        if slot.get("backend") != want:
            problems.append(f"{where} {name}: backend = {slot.get('backend')!r}（该槽后端应为 {want}）")
        if expected.get(key) and model != expected[key]:
            problems.append(f"{where} {name}: model {model} ≠ run_config 的 {key} = {expected[key]}")
        if slot.get("backend") == "codex" and slot.get("model_served") and not _same_model(slot["model_served"], model):
            problems.append(f"{where} {name}: Codex 实际用的模型 {slot['model_served']} ≠ 请求的 {model}")
    cli_files = sorted(p.relative_to(run_dir).as_posix() for pattern in ("slot_*_events.jsonl", "slot_*_stderr.txt")
                       for p in (run_dir / "requests").rglob(pattern))
    stray = []
    for rel in cli_files:
        match = CLI_FILE.match(rel.rsplit("/", 1)[-1])
        if kinds.get((SLOT.match(match.group(1)) or [None, "?"])[1] if match else "?") != "codex":
            stray.append(rel)
    if stray:
        problems.append(f"非 Codex 槽有 CLI 痕迹文件：{stray[:5]}")
    return {"run": run_dir.name, "status": (_load(run_dir / "result.json") or {}).get("status"), "slot_models": expected,
            "teacher_model": cfg.get("teacher_model"), "teacher_role": walker.get("teacher_role"), "judge_planned": walker.get("judge_planned"),
            "slot_calls": dict(calls), "llm_calls": len(records), "slot_cost_usd": round(cost, 4), "problems": problems}


def audit_verdicts(round_tag: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in ("verdicts", "verdicts_units"):
        files = [p for p in SFT.glob(f"{kind}_{round_tag}_judged.*.jsonl") if not p.name.endswith(".pre_revise")]
        judges: Counter[str] = Counter()
        for path in files:
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    judges[str(json.loads(line).get("judge_model"))] += 1
                except ValueError:
                    continue
        out[kind] = {"files": [p.name for p in files], "judge_models": dict(judges)}
    return out


def audit_export(round_tag: str, runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dirs = [p for p in (SFT / "export").glob(f"{round_tag}_*") if p.is_dir()]
    if not dirs:
        return {"dir": None}
    export = max(dirs, key=lambda p: p.stat().st_mtime)
    rows, missing, mismatch = 0, 0, []
    pairs: Counter[str] = Counter()
    for path in export.glob("*_*.jsonl"):
        if path.name.startswith("verdicts"):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                meta = json.loads(line).get("metadata") or {}
            except ValueError:
                continue
            rows += 1
            if not meta.get("slot_teacher") or not meta.get("terminal_teacher"):
                missing += 1
                continue
            pairs[f"{meta['slot_teacher']} | {meta['terminal_teacher']}"] += 1
            models = (runs.get(str(meta.get("run"))) or {}).get("slot_models") or {}
            if models and (meta["slot_teacher"] != models.get("S1") or models.get("S5") not in meta["terminal_teacher"].split("+")):
                mismatch.append(f"{meta.get('run')} {meta.get('request')}")
    return {"dir": export.name, "rows": rows, "rows_without_fields": missing, "slot|terminal": dict(pairs), "mismatch": mismatch[:10]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", required=True)
    args = parser.parse_args()
    runs = sorted(p for p in RUNS.glob(f"*_{args.round}_*") if p.is_dir() and ".failed-" not in p.name)
    results = {r.name: audit_run(r) for r in runs}
    verdicts = audit_verdicts(args.round)
    export = audit_export(args.round, results)
    problems = [f"{name}: {p}" for name, r in results.items() for p in r["problems"]]
    prefix = frozen_judge() + ":"
    for kind, info in verdicts.items():
        bad = [j for j in info["judge_models"] if not j.startswith(prefix) and j != "None"]
        if bad:
            problems.append(f"{kind}: judge_model 不是冻结的核验器 {prefix}…：{bad}")
        if not info["files"]:
            problems.append(f"{kind}: 没有 {kind}_{args.round}_judged.* 文件（核验没跑？）")
    if export.get("dir") is None:
        problems.append("没有导出目录")
    elif export["rows_without_fields"] or export["mismatch"]:
        problems.append(f"导出：{export['rows_without_fields']} 行缺 slot_teacher / terminal_teacher；不一致 {export['mismatch']}")
    lines = [f"# 分槽教师 + 核验器核对：{args.round}", "", f"run {len(results)} 个；问题 {len(problems)} 条", ""]
    for name, r in results.items():
        lines.append(f"## {name}（{r['status']}）")
        lines.append(f"- teacher_model {r['teacher_model']}（{r['teacher_role']}）；核验计划 {r['judge_planned']}；槽调用 {r['llm_calls']} 次，"
                     f"OpenRouter 记账 ${r['slot_cost_usd']}（Codex 槽走订阅，记 0）")
        lines.append("- 槽|后端|分槽后端|模型 → 次数：" + "；".join(f"{k} ×{v}" for k, v in sorted(r["slot_calls"].items())))
        lines.append("")
    lines += ["## 核验", ""] + [f"- {k}：{json.dumps(v, ensure_ascii=False)}" for k, v in verdicts.items()]
    lines += ["", "## 导出", "", f"- {json.dumps(export, ensure_ascii=False)}", "", "## 问题", ""]
    lines += [f"- {p}" for p in problems] or ["（无：每个槽走冻结的后端，核验全是冻结的核验器）"]
    out = SFT / f"teacher_slots_audit_{args.round}"
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"round": args.round, "runs": results, "verdicts": verdicts, "export": export,
                                                    "problems": problems}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
