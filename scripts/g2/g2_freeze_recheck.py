"""g2：事后把 60 个 r1g2 run 归到 specs/g1_freeze.json，并逐个按它重算冻结核对（只追加字段，不改原有运行记录）（2026-10-18）。

    python scripts\\g2\\g2_freeze_recheck.py [--runs-glob "*_r1g2_1"] [--freeze specs/g1_freeze.json] [--dry-run]

背景：g2 的 run 跑时 g1_freeze.json 还不存在（定案 ⑥ 晚于开跑），run_config.freeze 是对 r1_freeze 记的、教师槽读的是
specs/g1_teacher.json（内容与 g1_freeze 的 teacher 一节相同）。这里**不改** run_config 里任何已有字段，只追加：
- freeze_file_assigned_post_hoc = "specs/g1_freeze.json"、freeze_file_assigned_at = 时间；
- freeze_recheck = 按 g1_freeze 逐项重算的结果：
  * run_cases.freeze_check(g1_freeze, actual)——actual 取 run_config.freeze.items 里跑时记下的实际值（spec_hash.hi / .gia、
    tool_layer_hash、state_renderer、image_input_limit、harness_overrides …），即三个 hash 与各项 checks；
  * teacher：run_config.walker.slot_models 逐槽 = g1_freeze teacher.slots → backends 的模型；slot_teachers 的后端名逐槽一致；
  * teacher_override：run_config.teacher_override 的 sha256 = 当前 specs/g1_teacher.json 的 sha256，且 g1_teacher.json 的 teacher 一节
    与 g1_freeze 的 teacher 一节去掉说明字段后逐键相同（跑时读的槽表就是 g1_freeze 登记的槽表）；
  * ok = 三项全过。
已有 freeze_recheck 的 run 不重复追加（幂等；--force 重算）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from foren_searcher.datagen.run_cases import freeze_check, recheck_freeze  # noqa: E402

RUNS = ROOT / "experiments" / "sft_data" / "runs"
NOTE_KEYS = {"decided", "note", "slot_meaning", "slot_name_rule", "base_freeze", "base_freeze_teacher_differs_from", "server",
             "endpoint_alternative"}


def strip_notes(teacher: dict) -> dict:
    out = {}
    for k, v in teacher.items():
        if k in NOTE_KEYS:
            continue
        out[k] = {b: {kk: vv for kk, vv in (cfg or {}).items() if kk not in NOTE_KEYS} for b, cfg in v.items()} if k == "backends" else v
    return out


def recheck(cfg: dict, freeze: dict, teacher_file: Path) -> dict:
    run_freeze = cfg.get("freeze") or {}
    actual = {k: (v or {}).get("actual") for k, v in (run_freeze.get("items") or {}).items()}
    fc = recheck_freeze(freeze_check(freeze, actual), freeze)
    ft = freeze.get("teacher") or {}
    walker = cfg.get("walker") or {}
    models, backends_used = walker.get("slot_models") or {}, {k: (v or {}).get("backend") for k, v in (walker.get("slot_teachers") or {}).items()}
    slot_problems = []
    for slot, backend in (ft.get("slots") or {}).items():
        want = ((ft.get("backends") or {}).get(backend) or {}).get("model")
        if models.get(slot) != want:
            slot_problems.append(f"{slot}: {models.get(slot)} ≠ {want}")
        if backends_used.get(slot) not in (None, backend):
            slot_problems.append(f"{slot}: 后端 {backends_used.get(slot)} ≠ {backend}")
    to = cfg.get("teacher_override") or {}
    sha_now = hashlib.sha256(teacher_file.read_bytes()).hexdigest()
    same_table = strip_notes(json.loads(teacher_file.read_text(encoding="utf-8")).get("teacher") or {}) == strip_notes(ft)
    override_ok = to.get("sha256") == sha_now and Path(str(to.get("file") or "")).name == teacher_file.name and same_table
    return {"against": "specs/g1_freeze.json", "freeze_id": freeze.get("freeze_id"), "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "freeze_check": {"ok": fc.get("ok"), "failed": fc.get("failed"), "variant": fc.get("variant")},
            "teacher_slots": {"ok": not slot_problems, "problems": slot_problems, "slot_models": models},
            "teacher_override": {"ok": override_ok, "run_sha256": to.get("sha256"), "g1_teacher_sha256_now": sha_now,
                                 "teacher_table_equals_g1_freeze": same_table},
            "ok": bool(fc.get("ok")) and not slot_problems and override_ok}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-glob", default="*_r1g2_1")
    ap.add_argument("--freeze", default="specs/g1_freeze.json")
    ap.add_argument("--teacher-file", default="specs/g1_teacher.json")
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    freeze = json.loads((ROOT / args.freeze).read_text(encoding="utf-8"))
    results = {}
    for run in sorted(p for p in Path(args.runs_dir).glob(args.runs_glob) if p.is_dir() and ".failed-" not in p.name):
        path = run / "run_config.json"
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if cfg.get("freeze_recheck") and not args.force:
            results[run.name] = cfg["freeze_recheck"]["ok"]
            continue
        res = recheck(cfg, freeze, ROOT / args.teacher_file)
        results[run.name] = res["ok"]
        if not args.dry_run:
            cfg.setdefault("freeze_file_assigned_post_hoc", args.freeze.replace("\\", "/"))
            cfg.setdefault("freeze_file_assigned_at", res["at"])
            cfg["freeze_recheck"] = res
            path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        if not res["ok"]:
            print(f"{run.name}: 不过 {json.dumps({k: res[k] for k in ('freeze_check', 'teacher_slots', 'teacher_override')}, ensure_ascii=False)[:400]}")
    bad = [r for r, ok in results.items() if not ok]
    print(json.dumps({"runs": len(results), "ok": len(results) - len(bad), "not_ok": bad[:20], "dry_run": args.dry_run}, ensure_ascii=False))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
