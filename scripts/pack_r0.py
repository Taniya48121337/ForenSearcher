"""把第一次 SFT 的数据与配置打成 r0 数据包，带校验清单（清单 E6）。

    # 打包（默认严格：任何一项和冻结口径对不上就不出包）
    python scripts/pack_r0.py --export experiments/sft_data/export/<r0 导出目录>
    # 试打包（不合口径也出包，r0_pack.json 记 status=trial，不能拿去训练正式 r0）
    python scripts/pack_r0.py --export <目录> --out experiments/sft_data/r0_pack/trial --allow-mismatch
    # 校验一个已有的包（服务器解包后、开训前各跑一次）
    python scripts/pack_r0.py --verify experiments/sft_data/r0_pack/r0
    # 同时打一个 tar.gz 带去服务器
    python scripts/pack_r0.py --export <目录> --tar dist

包的结构（默认 experiments/sft_data/r0_pack/r0/，正好是 training/configs/*_r0_9b.json 的 export_dir 的上一级）：
    data/            hi_accepted.jsonl、gia_accepted.jsonl、manifest.json（导出原样）、images/（只复制被行引用的图）
    configs/         hi_r0_9b.json、gia_r0_9b.json、r0_sft.txt（训练配置与任务清单，原样）
    freeze/          r0_freeze.json（冻结口径原样）
    provenance/      runs.jsonl：每个 run 的 teacher_model / teacher_role / spec_hash / tool_layer_hash / freeze.ok（取自 run_config）
    r0_pack.json     身份与核对结果：pack_id、spec_hash、tool_layer_hash、教师分布、核验器版本与代码 hash、训练超参、逐项 checks
    SHA256SUMS       包内除它自己以外每个文件的 sha256（sha256sum -c 可直接用）

核对项（r0_pack.json.checks，每项 ok / detail；严格模式下任何一项 ok=false 就不出包）：
    rows_parse        两份 jsonl 每行可解析、role 与文件一致、sample_weight > 0、引用的图都在
    spec_hash         每行 metadata.spec_hash 按角色只有一个值，且等于冻结值；manifest 一致；当前 specs/ 重算也一致
    state_renderer    每行 state_renderer = 冻结的 compact_16k
    verifier          每行 verifier = 冻结版本；judge_model 是冻结的核验模型且前缀是冻结的 judge（10-15 对话 5 起 claude-cli:）；
                      当前 datagen/verify 代码 hash = 冻结值
    tool_layer_hash   行 / manifest / run_config 里的工具层 hash 只有一个值且等于冻结值（r0e 来源豁免，单独列出）；当前代码重算也一致
    run_freeze        每个 run 的 run_config.freeze.ok == true（r0e 豁免）
    teacher           分槽冻结（teacher.slots，10-15 起）：每个 run 的 run_config.walker.slot_models 逐槽核对——槽 Sk 的模型必须是
                      该槽后端（teacher.slots[Sk]）的主模型（backends）或备用模型（backup_backends），只换一个后端的 run 也认；
                      没有 slot_models 的 run 不过（r0e 豁免）。老格式冻结仍按 run 级 teacher_model ∈ {primary, backup}。
                      用了备用的 run 的行占比 > 20% 报警（不算失败）
    gia_wait_rows     system_variant=wait 的 GIA 行为 0（定时复核关）
    train_configs     两份配置角色对、export_dir 指向本包 data/、种子 ≥ 2、sample_weight 开、lora_r ≤ 32、max_tokens ≥ 32768
    counts            HI ≥ 500、GIA ≥ 60、FINISH ≥ 30（清单 C1；只报警——数量门由 export_gate 管）

不改任何行；行的入选与权重在 export 里已定。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_CONFIGS = ["training/configs/hi_r0_9b.json", "training/configs/gia_r0_9b.json", "training/jobs/r0_sft.txt"]
EXEMPT_ORIGINS = {"r0e"}                 # 冻结前跑的来源，靠"当前规格重导出"进来（《r0冻结口径》§5）
MIN_COUNTS = {"hi": 500, "gia": 60, "gia_finish": 30}
MAX_LORA_RANK = 32                       # server/loop_env.sh FS_LORA_RANK
BACKUP_SHARE_WARN = 0.20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_exempt(md: dict[str, Any]) -> bool:
    origin = str(md.get("origin") or "")
    run = str(md.get("run") or "")
    return origin in EXEMPT_ORIGINS or any(run.startswith(f"{o}_") or f"_{o}_" in run for o in EXEMPT_ORIGINS)


class Checks:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, name: str, ok: bool, detail: Any, *, warn: bool = False) -> None:
        self.items.append({"name": name, "ok": bool(ok), "level": "warn" if warn else "error", "detail": detail})

    @property
    def failed(self) -> list[str]:
        return [c["name"] for c in self.items if not c["ok"] and c["level"] == "error"]

    @property
    def warnings(self) -> list[str]:
        return [c["name"] for c in self.items if not c["ok"] and c["level"] == "warn"]


def current_code_hashes() -> dict[str, Any]:
    """按 run_cases / walker_run 的同一口径重算当前代码的工具层 hash 与核验器代码 hash（只读）。"""

    out: dict[str, Any] = {}
    try:
        from foren_searcher.datagen.run_cases import _hash_files, _layer_files, tool_layer_hash
        out["tool_layer"] = tool_layer_hash()
        package = ROOT / "src" / "foren_searcher"
        out["verifier_code_hash"] = _hash_files(package, _layer_files(package, ("datagen/verify", "datagen/errors.py")))
    except Exception as exc:                     # 依赖不全时只记下来，不中断（核对项会显示 None）
        out["error"] = f"{type(exc).__name__}: {exc}"
    try:
        from foren_searcher.datagen.specs import Spec
        out["spec_hash"] = {}
        for role in ("hi", "gia"):
            frozen = json.loads((ROOT / "specs" / "r0_freeze.json").read_text(encoding="utf-8"))["specs"][role]
            out["spec_hash"][role] = Spec.load(ROOT / "specs" / frozen).hash()
    except Exception as exc:
        out["spec_error"] = f"{type(exc).__name__}: {exc}"
    return out


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            res = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20)
            return res.stdout.strip() or None if res.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
    return {"head": run("rev-parse", "HEAD"), "describe": run("describe", "--tags", "--always", "--dirty"),
            "tags_at_head": (run("tag", "--points-at", "HEAD") or "").split() or None}


def run_provenance(runs: set[str], runs_dir: Path) -> list[dict[str, Any]]:
    out = []
    for run in sorted(runs):
        path = runs_dir / run / "run_config.json"
        if not path.is_file():
            out.append({"run": run, "run_config": None})
            continue
        cfg = json.loads(path.read_text(encoding="utf-8"))
        freeze = cfg.get("freeze") or {}
        walker = cfg.get("walker") or {}
        out.append({"run": run, "run_config": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                    "teacher_model": cfg.get("teacher_model") or (cfg.get("configured_model") or {}).get("model") or walker.get("teacher_model"),
                    "teacher_role": walker.get("teacher_role"), "slot_models": walker.get("slot_models"),
                    "slot_backends": {k: (v or {}).get("backend") for k, v in (walker.get("slot_teachers") or {}).items()} or None,
                    "spec_hash": cfg.get("spec_hash"),
                    "tool_layer_hash": cfg.get("tool_layer_hash"), "render_profile": cfg.get("render_profile"),
                    "freeze_ok": freeze.get("ok"), "freeze_failed": freeze.get("failed"), "freeze_id": freeze.get("freeze_id")})
    return out


def normalize_model(model: str | None) -> str:
    """与 oracle_generator.normalize_model 相同：去 org 前缀、小写、点换横线。"""

    return str(model or "").strip().rsplit("/", 1)[-1].lower().replace(".", "-")


def model_matches(actual: str | None, expected: str | None) -> bool:
    a, e = normalize_model(actual), normalize_model(expected)
    return bool(a and e) and (a == e or a.startswith(e + "-") or a.startswith(e + ":"))


def slot_teacher_check(prov: list[dict[str, Any]], ft: dict[str, Any]) -> dict[str, Any]:
    """10-15（5a 遗留，r0 成本压缩落实）：按 run_config.walker.slot_models 逐槽核对教师。
    返回 {"ok", "runs": {run: role}, "problems": [...], "slot_models": {Sk: {模型: run 数}}, "roles": {role: run 数}}。"""

    slots = ft.get("slots") or {}
    backends = ft.get("backends") or {}
    backups = ft.get("backup_backends") or {}
    problems: list[str] = []
    roles: dict[str, str | None] = {}
    seen: dict[str, Counter] = {k: Counter() for k in slots}
    for item in prov:
        run = item["run"]
        models = item.get("slot_models") or {}
        if not models:
            problems.append(f"{run}: run_config.walker.slot_models 缺失")
            roles[run] = None
            continue
        per = []
        for key, backend in slots.items():
            actual = models.get(key)
            seen[key][str(actual)] += 1
            primary = (backends.get(backend) or {}).get("model")
            backup = (backups.get(backend) or {}).get("model")
            role = "primary" if model_matches(actual, primary) else "backup" if model_matches(actual, backup) else None
            if role is None:
                problems.append(f"{run}: {key} = {actual}，冻结的 {backend} 后端是 {primary}（备用 {backup}）")
            declared = (item.get("slot_backends") or {}).get(key)
            if declared and declared != backend:
                problems.append(f"{run}: {key} 走了 {declared} 后端，冻结表是 {backend}")
                role = None
            per.append(role)
        roles[run] = (None if None in per else "primary" if set(per) == {"primary"} else "backup" if set(per) == {"backup"}
                      else "mixed")
    return {"ok": not problems and bool(prov), "runs": roles, "problems": problems[:40], "n_problems": len(problems),
            "slot_models": {k: dict(v) for k, v in seen.items()}, "roles": dict(Counter(str(r) for r in roles.values()))}


def check_rows(rows: dict[str, list[dict[str, Any]]], export_dir: Path, checks: Checks) -> set[str]:
    bad: list[str] = []
    images: set[str] = set()
    for role, items in rows.items():
        for i, row in enumerate(items):
            md = row.get("metadata") or {}
            if md.get("role") != role:
                bad.append(f"{role}#{i}: metadata.role={md.get('role')}")
            if not isinstance(row.get("messages"), list) or not row["messages"] or row["messages"][-1].get("role") != "assistant":
                bad.append(f"{role}#{i}: 最后一条不是 assistant")
            try:
                if float(md.get("sample_weight")) <= 0:
                    bad.append(f"{role}#{i}: sample_weight={md.get('sample_weight')}")
            except (TypeError, ValueError):
                bad.append(f"{role}#{i}: 缺 sample_weight")
            for rel in row.get("images") or []:
                images.add(rel)
                if not (export_dir / rel).is_file():
                    bad.append(f"{role}#{i}: 图不在 {rel}")
    checks.add("rows_parse", not bad, {"rows": {r: len(v) for r, v in rows.items()}, "images": len(images), "problems": bad[:30],
                                       "n_problems": len(bad)})
    return images


def pack(args: argparse.Namespace) -> int:
    export_dir = (ROOT / args.export).resolve() if not Path(args.export).is_absolute() else Path(args.export)
    out = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    freeze_path = ROOT / args.freeze
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    runs_dir = ROOT / args.runs_dir
    checks = Checks()

    files = {"hi": export_dir / "hi_accepted.jsonl", "gia": export_dir / "gia_accepted.jsonl"}
    missing = [str(p) for p in [*files.values(), export_dir / "manifest.json"] if not p.is_file()]
    if missing:
        raise SystemExit(f"导出目录缺文件：{missing}")
    rows = {role: read_jsonl(path) for role, path in files.items()}
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    all_md = [(role, r.get("metadata") or {}) for role, items in rows.items() for r in items]
    images = check_rows(rows, export_dir, checks)
    code = current_code_hashes()

    # ---- spec_hash
    spec_sets = {role: sorted({md.get("spec_hash") for r, md in all_md if r == role}) for role in ("hi", "gia")}
    frozen_spec = freeze.get("spec_hash") or {}
    ok = all(spec_sets[r] == [frozen_spec.get(r)] for r in ("hi", "gia") if rows[r])
    ok &= all((manifest.get("spec_hash") or {}).get(r) == frozen_spec.get(r) for r in ("hi", "gia"))
    current_spec = code.get("spec_hash") or {}
    ok &= all(current_spec.get(r) == frozen_spec.get(r) for r in ("hi", "gia"))
    checks.add("spec_hash", ok, {"rows": spec_sets, "manifest": manifest.get("spec_hash"), "current_specs_dir": current_spec or code.get("spec_error"),
                                 "frozen": frozen_spec})

    # ---- renderer
    frozen_renderer = (freeze.get("checks") or {}).get("state_renderer")
    renderers = Counter(md.get("state_renderer") for _, md in all_md)
    checks.add("state_renderer", set(renderers) == {frozen_renderer}, {"rows": dict(renderers), "frozen": frozen_renderer})

    # ---- verifier
    fv = freeze.get("verifier") or {}
    versions = Counter(md.get("verifier") for _, md in all_md)
    judges = Counter(md.get("judge_model") for _, md in all_md)
    # None = 只过了程序层（如 GIA plan），没有模型核验；模型按身份比（claude-cli:claude-sonnet-5-5 ≡ claude-sonnet-5-5）。
    # 10-15 对话 5（A5）：核验器改回 Claude CLI 订阅，只认 claude-cli: 前缀；5a / 5b 的 api:… 判定（同提示同模型）不再算 r0
    prefix = f"{fv.get('judge')}:"
    judge_ok = all(j is None or (str(j).startswith(prefix) and model_matches(str(j)[len(prefix):], fv.get("model")))
                   for j in judges)
    code_ok = code.get("verifier_code_hash") == fv.get("code_hash")
    manifest_code = manifest.get("verifier_code_hash")
    checks.add("verifier", set(versions) == {fv.get("version")} and judge_ok and code_ok and manifest_code in (None, fv.get("code_hash")),
               {"rows": dict(versions), "judge_models": dict(judges), "frozen": {k: fv.get(k) for k in ("version", "judge", "model", "code_hash")},
                "current_code_hash": code.get("verifier_code_hash"), "manifest_code_hash": manifest_code})

    # ---- runs / tool layer / freeze.ok
    runs = {str(md.get("run")) for _, md in all_md if md.get("run") and not is_exempt(md)}
    exempt_runs = sorted({str(md.get("run")) for _, md in all_md if md.get("run") and is_exempt(md)})
    prov = run_provenance(runs, runs_dir)
    frozen_layer = (freeze.get("tool_layer") or {}).get("hash")
    layer_rows = Counter(md.get("tool_layer_hash") for _, md in all_md if not is_exempt(md))
    layer_runs = Counter(p.get("tool_layer_hash") for p in prov)
    layer_manifest = manifest.get("tool_layer_hash")
    if isinstance(layer_manifest, str):
        layer_manifest = [layer_manifest]
    seen = ({h for h in layer_rows if h} | {h for h in layer_runs if h} | set(layer_manifest or []))
    current_layer = (code.get("tool_layer") or {}).get("hash")
    missing_layer = sum(v for k, v in layer_runs.items() if not k)
    checks.add("tool_layer_hash", seen == {frozen_layer} and not missing_layer and current_layer == frozen_layer,
               {"frozen": frozen_layer, "status": (freeze.get("tool_layer") or {}).get("status"), "rows": dict(layer_rows),
                "run_configs": dict(layer_runs), "manifest": layer_manifest, "current_code": current_layer,
                "runs_without_hash": missing_layer, "exempt_runs": exempt_runs,
                "current_groups": (code.get("tool_layer") or {}).get("groups"), "frozen_groups": (freeze.get("tool_layer") or {}).get("groups"),
                "error": code.get("error")})
    not_ok = [p["run"] for p in prov if p.get("freeze_ok") is not True]
    checks.add("run_freeze", not not_ok, {"runs": len(prov), "not_ok": not_ok[:40], "n_not_ok": len(not_ok),
                                          "missing_run_config": [p["run"] for p in prov if p.get("run_config") is None][:20]})

    # ---- teacher
    ft = freeze.get("teacher") or {}
    teachers = Counter(md.get("teacher_model") for _, md in all_md)
    if ft.get("slots"):
        slot_check = slot_teacher_check(prov, ft)
        run_roles = slot_check["runs"]
        counted = [(r, md) for r, md in all_md if not is_exempt(md)]
        backup_rows = sum(1 for _, md in counted if run_roles.get(str(md.get("run"))) in ("backup", "mixed"))
        checks.add("teacher", slot_check["ok"], {"mode": "per_slot", "rows": dict(teachers), "primary": ft.get("primary"),
                                                 "backup": ft.get("backup"), **{k: v for k, v in slot_check.items() if k != "runs"},
                                                 "teacher_role_rows": dict(Counter(md.get("teacher_role") for _, md in all_md))})
    else:
        allowed = {ft.get("primary"), ft.get("backup")}
        backup_rows = teachers.get(ft.get("backup"), 0)
        checks.add("teacher", set(teachers) <= allowed, {"rows": dict(teachers), "primary": ft.get("primary"), "backup": ft.get("backup"),
                                                         "provisional": ft.get("provisional"),
                                                         "teacher_role_rows": dict(Counter(md.get("teacher_role") for _, md in all_md))})
    share = backup_rows / max(1, len(all_md))
    checks.add("teacher_backup_share", share <= BACKUP_SHARE_WARN, {"backup_rows": backup_rows, "share": round(share, 3)}, warn=True)

    # ---- GIA wait 变体
    wait_rows = sum(1 for r, md in all_md if r == "gia" and md.get("system_variant") == "wait")
    checks.add("gia_wait_rows", wait_rows == 0, {"rows": wait_rows})

    # ---- 训练配置
    cfg_out: dict[str, Any] = {}
    problems: list[str] = []
    for rel in args.configs:
        path = ROOT / rel
        if not path.is_file():
            problems.append(f"缺 {rel}")
            continue
        if path.suffix != ".json":
            continue
        cfg = json.loads(path.read_text(encoding="utf-8"))
        role = cfg.get("role")
        hyper = {k: cfg.get(k) for k in ("role", "model", "export_dir", "max_tokens", "epochs", "max_steps", "lr", "min_lr_ratio",
                                         "warmup_steps", "grad_accum", "lora_r", "lora_alpha", "lora_dropout", "lora_targets",
                                         "seeds", "use_sample_weight", "save_each_epoch", "logits_on_supervised_only")}
        cfg_out[rel] = hyper
        if not str(cfg.get("export_dir") or "").replace("\\", "/").rstrip("/").endswith(f"{out.name}/data"):
            problems.append(f"{rel}: export_dir={cfg.get('export_dir')} 不是本包的 data/")
        if len(cfg.get("seeds") or []) < 2:
            problems.append(f"{rel}: 种子少于 2 个")
        if not cfg.get("use_sample_weight", True):
            problems.append(f"{rel}: use_sample_weight 关着")
        if int(cfg.get("lora_r") or 0) > MAX_LORA_RANK:
            problems.append(f"{rel}: lora_r={cfg.get('lora_r')} > vLLM max-lora-rank {MAX_LORA_RANK}")
        if int(cfg.get("max_tokens") or 0) < 32768:
            problems.append(f"{rel}: max_tokens={cfg.get('max_tokens')} < 32768")
        if role not in ("hi", "gia"):
            problems.append(f"{rel}: role={role}")
    roles = {h.get("role") for h in cfg_out.values()}
    if roles != {"hi", "gia"}:
        problems.append(f"训练配置角色不全：{sorted(r for r in roles if r)}")
    checks.add("train_configs", not problems, {"configs": cfg_out, "problems": problems})

    # ---- 数量（只报警）
    finish = sum(1 for r, md in all_md if r == "gia" and (md.get("action") == "FINISH" or md.get("unit") == "answer"))
    counts = {"hi": len(rows["hi"]), "gia": len(rows["gia"]), "gia_finish": finish}
    checks.add("counts", all(counts[k] >= v for k, v in MIN_COUNTS.items()), {"counts": counts, "min": MIN_COUNTS}, warn=True)

    strict_fail = checks.failed
    status = "ok" if not strict_fail else "trial"
    if strict_fail and not args.allow_mismatch:
        for c in checks.items:
            mark = "OK " if c["ok"] else ("WARN" if c["level"] == "warn" else "FAIL")
            print(f"[{mark}] {c['name']}")
        print(f"与冻结口径不一致：{strict_fail}。没有出包。要看细节或做试包，加 --allow-mismatch --out <别的目录>。")
        report = out.parent / f"{out.name}.check_failed.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({"export": str(export_dir), "checks": checks.items}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"核对明细：{report}")
        return 1

    # ---- 写包
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} 已存在；确认要覆盖加 --overwrite")
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    for name in ("hi_accepted.jsonl", "gia_accepted.jsonl", "manifest.json"):
        shutil.copy2(export_dir / name, out / "data" / name)
    for rel in sorted(images):
        if not (export_dir / rel).is_file():              # 只在试包时会走到（严格模式下 rows_parse 已经拦住）
            continue
        target = out / "data" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(export_dir / rel, target)
    (out / "configs").mkdir()
    for rel in args.configs:
        if (ROOT / rel).is_file():
            shutil.copy2(ROOT / rel, out / "configs" / Path(rel).name)
    (out / "freeze").mkdir()
    shutil.copy2(freeze_path, out / "freeze" / freeze_path.name)
    (out / "provenance").mkdir()
    (out / "provenance" / "runs.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in prov), encoding="utf-8")

    data_hashes = {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted((out / "data").rglob("*")) if p.is_file()}
    digest = hashlib.sha256("".join(f"{k}\0{v}\n" for k, v in data_hashes.items()).encode()).hexdigest()
    pack_id = f"r0-{time.strftime('%Y%m%d')}-{digest[:12]}"
    identity = {
        "pack_id": pack_id, "status": status, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "freeze_id": freeze.get("freeze_id"),
        "export_id": manifest.get("export_id"), "source_export_dir": str(export_dir),
        "spec": freeze.get("specs"), "spec_hash": frozen_spec, "state_renderer": frozen_renderer,
        "tool_layer_hash": sorted(h for h in seen if h), "tool_layer_frozen": frozen_layer,
        "teacher_model": dict(teachers), "teacher_frozen": {"primary": ft.get("primary"), "backup": ft.get("backup")},
        "verifier": fv.get("version"), "verifier_code_hash": fv.get("code_hash"), "judge_model": dict(judges),
        "counts": {**counts, "images": len(images)}, "sample_weight_sum": {r: round(sum(float((x.get("metadata") or {}).get("sample_weight") or 0) for x in rows[r]), 3) for r in rows},
        "train_configs": cfg_out, "git": git_info(), "data_digest": digest,
        "checks": checks.items, "failed": strict_fail, "warnings": checks.warnings,
    }
    (out / "r0_pack.json").write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
    sums = [f"{sha256_file(p)}  {p.relative_to(out).as_posix()}" for p in sorted(out.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(f"{'打包完成' if status == 'ok' else '试包（不合冻结口径，不能用于正式 r0）'}：{out}  pack_id={pack_id}")
    for c in checks.items:
        mark = "OK " if c["ok"] else ("WARN" if c["level"] == "warn" else "FAIL")
        print(f"  [{mark}] {c['name']}")
    print(f"  文件 {len(sums)} 个（SHA256SUMS），HI {counts['hi']} 行、GIA {counts['gia']} 行（FINISH {finish}）、图 {len(images)} 张")
    if args.tar:
        tar_dir = ROOT / args.tar
        tar_dir.mkdir(parents=True, exist_ok=True)
        tar_path = tar_dir / f"ForenSearcher_{pack_id}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(out, arcname=f"experiments/sft_data/r0_pack/{out.name}")
        print(f"  tar：{tar_path}（{tar_path.stat().st_size / 1e6:.1f} MB，在仓库根目录解开）")
    return 0


def verify(pack_dir: Path) -> int:
    sums = pack_dir / "SHA256SUMS"
    if not sums.is_file():
        print(f"{pack_dir} 没有 SHA256SUMS")
        return 1
    listed: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.split("  ", 1)
            listed[rel] = digest
    bad, missing = [], []
    for rel, digest in listed.items():
        path = pack_dir / rel
        if not path.is_file():
            missing.append(rel)
        elif sha256_file(path) != digest:
            bad.append(rel)
    extra = [p.relative_to(pack_dir).as_posix() for p in pack_dir.rglob("*")
             if p.is_file() and p.name != "SHA256SUMS" and p.relative_to(pack_dir).as_posix() not in listed]
    identity = json.loads((pack_dir / "r0_pack.json").read_text(encoding="utf-8")) if (pack_dir / "r0_pack.json").is_file() else {}
    data_hashes = {rel: d for rel, d in listed.items() if rel.startswith("data/")}
    digest = hashlib.sha256("".join(f"{k}\0{v}\n" for k, v in sorted(data_hashes.items())).encode()).hexdigest()
    digest_ok = digest == identity.get("data_digest")
    ok = not bad and not missing and not extra and digest_ok
    print(json.dumps({"pack": str(pack_dir), "pack_id": identity.get("pack_id"), "status": identity.get("status"),
                      "files": len(listed), "changed": bad[:20], "missing": missing[:20], "extra": extra[:20],
                      "data_digest_ok": digest_ok, "failed_checks_at_pack_time": identity.get("failed"),
                      "ok": ok}, ensure_ascii=False, indent=2))
    if identity.get("status") != "ok":
        print("注意：这是试包（status != ok），不能用于正式 r0 训练。")
    return 0 if ok and identity.get("status") == "ok" else (2 if ok else 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", help="r0 导出目录（export.py 或 pack_sft.py 的产物）")
    parser.add_argument("--out", default="experiments/sft_data/r0_pack/r0")
    parser.add_argument("--freeze", default="specs/r0_freeze.json")
    parser.add_argument("--runs-dir", default="experiments/sft_data/runs")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--allow-mismatch", action="store_true", help="不合冻结口径也出包（status=trial）")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tar", default=None, help="另打一个 tar.gz 到这个目录（例如 dist）")
    parser.add_argument("--verify", default=None, help="只校验一个已有的包")
    args = parser.parse_args(argv)
    if args.verify:
        path = Path(args.verify)
        return verify(path if path.is_absolute() else ROOT / path)
    if not args.export:
        parser.error("需要 --export 或 --verify")
    return pack(args)


if __name__ == "__main__":
    raise SystemExit(main())
