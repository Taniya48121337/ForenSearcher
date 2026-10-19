"""r0 评测（清单 E2 训前基线 / 训后评测）的一条龙：核对 → 建检索快照 → 跑 → 锁快照 → 评分 → 行为指标与成功标准。

口径全部来自 experiments/eval/r0/eval_config.json（评委、温度、种子、规格、快照）与 eval_set.json（案例）；
成功标准在 success_criteria.json，训前登记、训后不改。说明见 claude/r0评测口径与成功标准_20261015.md。

    python scripts/eval_r0.py check                       # 只读核对：案卷、GT、快照（逐个 blob）、冻结文件、脚本 hash
    python scripts/eval_r0.py seed                        # 把现有快照复制进评测专用快照目录（只复制，不覆盖、不删）
    python scripts/eval_r0.py plan --phase baseline --endpoint http://127.0.0.1:8002/v1 --model student
    python scripts/eval_r0.py run  --phase baseline --endpoint ... --model student --jobs 4
    python scripts/eval_r0.py lock                        # 基线跑完：记下快照 as_of 与每个案例 index 的 hash
    python scripts/eval_r0.py score   --phase baseline    # EVID 20：Sonnet 官方提示评分；dev_eval 4：核验器第 1–2 步
    python scripts/eval_r0.py metrics --phase baseline    # behavior_metrics → reports/r0base.{json,md}
    python scripts/eval_r0.py run --phase after_sft --endpoint ... --model HI=hi_r0,GIA=gia_r0 --jobs 4
    python scripts/eval_r0.py score --phase after_sft && python scripts/eval_r0.py metrics --phase after_sft
                                                          # 训后：自动带 FOREN_SNAPSHOT_AS_OF，并按成功标准逐条判定

两个种子 = 两个轮次名 <tag>-s1、<tag>-s2（各 --samples 1）。_evid_score.py 只认 run 名末尾 _1，这样不用改它；
端点不收随机种子（EndpointTeacher 不发 seed），两次是同温度下的独立抽样。
基线与训后必须在同一台机器上跑（快照 recorded_at 是本地时间，as_of 按字符串比较）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "eval" / "r0" / "eval_config.json"
PY = sys.executable


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str | None:
    try:
        return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def rel(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def case_key(case_id: str) -> str:
    """与 search/recording.case_key 相同：case_id → 快照子目录名。"""

    return re.sub(r"[^0-9A-Za-z_.-]+", "_", case_id).strip("_") or "case"


def blob_refs(entry: dict[str, Any]) -> set[str]:
    text = json.dumps(entry.get("payload"), ensure_ascii=False)
    return set(re.findall(r'"(?:blob|artifact_sha256|file_sha256)": "([0-9a-f]{64})"', text))


def check_snapshot_dir(directory: Path) -> dict[str, Any]:
    index = directory / "index.jsonl"
    if not index.is_file():
        return {"dir": str(directory), "exists": directory.is_dir(), "index": False, "entries": 0, "missing_blobs": []}
    raw = index.read_bytes()
    entries, bad = [], 0
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except ValueError:
            bad += 1
    refs: set[str] = set()
    for entry in entries:
        refs |= blob_refs(entry)
    missing = sorted(s for s in refs if not (directory / "blobs" / s[:2] / s).is_file())
    return {"dir": str(directory), "exists": True, "index": True, "entries": len(entries), "bad_lines": bad,
            "index_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(), "blob_refs": len(refs),
            "missing_blobs": missing, "last_recorded_at": max((str(e.get("recorded_at") or "") for e in entries), default=None)}


# ---------------------------------------------------------------- check

def cmd_check(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    eval_set = load_json(rel(cfg["eval_set"]))
    problems: list[str] = []
    report: dict[str, Any] = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "cases": [], "files": {}}
    bench_path = rel(cfg["judge"]["evid"]["dataset"])
    bench = {e["id"]: e for e in load_json(bench_path)} if bench_path.is_file() else {}
    if not bench:
        problems.append(f"没有 EVID GT：{bench_path}")
    for set_name, spec in eval_set["sets"].items():
        for case in spec["cases"]:
            row: dict[str, Any] = {"set": set_name, "case_id": case["case_id"]}
            case_file = rel(case["case_file"])
            if not case_file.is_file():
                problems.append(f"{case['case_id']}：案卷不在 {case_file}")
            else:
                data = load_json(case_file)
                if data.get("case_id") != case["case_id"]:
                    problems.append(f"{case['case_id']}：案卷里的 case_id 是 {data.get('case_id')}")
                row["case_sha256"] = sha256_file(case_file)
            if case["case_id"].startswith("evid:"):
                sample = case["case_id"].split(":", 1)[1]
                gt = bench.get(sample) or {}
                if len(gt.get("groundtruth") or []) != case.get("n_gt") or list(gt.get("source") or []) != case.get("gt_sources"):
                    problems.append(f"{case['case_id']}：GT 点数或原片 ID 与清单不符")
            source_dir = (case.get("snapshot") or {}).get("source_dir")
            if source_dir:
                snap = check_snapshot_dir(rel(source_dir))
                row["source_snapshot"] = snap
                expected = (case.get("snapshot") or {}).get("index_sha256")
                if expected and snap.get("index_sha256") != expected:
                    row["source_snapshot_changed_since_freeze"] = True     # 只提示：历史快照被别的轮次追加过
                if snap["missing_blobs"]:
                    problems.append(f"{case['case_id']}：源快照缺 {len(snap['missing_blobs'])} 个 blob")
            eval_dir = rel(cfg["search"]["snapshot_root"]) / case_key(case["case_id"])
            if eval_dir.is_dir():
                snap = check_snapshot_dir(eval_dir)
                row["eval_snapshot"] = snap
                if snap["missing_blobs"]:
                    problems.append(f"{case['case_id']}：评测快照缺 {len(snap['missing_blobs'])} 个 blob")
            report["cases"].append(row)
    freeze_path = rel(cfg["harness"]["freeze_file"])
    if freeze_path.is_file():
        freeze = load_json(freeze_path)
        if freeze.get("specs") != {"hi": cfg["harness"]["hi_spec"], "gia": cfg["harness"]["gia_spec"]}:
            problems.append(f"冻结文件的规格 {freeze.get('specs')} 与评测配置不一致")
        report["freeze"] = {"specs": freeze.get("specs"), "tool_layer_hash": (freeze.get("tool_layer") or {}).get("hash"),
                            "tool_layer_tag": (freeze.get("tool_layer") or {}).get("tag")}
    else:
        problems.append(f"没有冻结文件 {freeze_path}")
    for key, path in (cfg.get("registered_files") or {}).items():
        report["files"][key] = {"path": path, "sha256": sha256_file(rel(path))}
        if report["files"][key]["sha256"] is None:
            problems.append(f"缺文件 {path}")
    for key, value in (cfg.get("registered_hashes") or {}).items():
        actual = (report["files"].get(key) or {}).get("sha256")
        if value and actual and value != actual:
            problems.append(f"{key}（{report['files'][key]['path']}）与登记时的 hash 不同：评测口径被改过（登记 {value[7:19]}…，现在 {actual[7:19]}…）")
    report["problems"] = problems
    out = rel(cfg["reports_dir"]) / f"check_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    seen = sum(1 for r in report["cases"] if r.get("source_snapshot", {}).get("index"))
    blobs = sum(r.get("source_snapshot", {}).get("blob_refs", 0) for r in report["cases"])
    print(f"案例 {len(report['cases'])} 个；有源快照 {seen} 个，共引用 blob {blobs} 个；问题 {len(problems)} 条。详见 {out}")
    for line in problems:
        print("  ✗ " + line)
    return 1 if problems else 0


# ---------------------------------------------------------------- seed / lock

def cmd_seed(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    eval_set = load_json(rel(cfg["eval_set"]))
    root = rel(cfg["search"]["snapshot_root"])
    root.mkdir(parents=True, exist_ok=True)
    plan = []
    for spec in eval_set["sets"].values():
        for case in spec["cases"]:
            target = root / case_key(case["case_id"])
            source = (case.get("snapshot") or {}).get("source_dir")
            plan.append((case["case_id"], rel(source) if source else None, target))
    size = sum(f.stat().st_size for _, s, t in plan if s and s.is_dir() and not t.exists() for f in s.rglob("*") if f.is_file())
    print(f"将复制 {sum(1 for _, s, t in plan if s and not t.exists())} 个快照目录，约 {size / 1e9:.2f} GB → {root}")
    if not args.yes and input("继续？输入 yes：").strip().lower() != "yes":
        return 1
    manifest = []
    for case_id, source, target in plan:
        if target.exists():
            manifest.append({"case_id": case_id, "action": "exists_kept"})
            continue
        if source and (source / "index.jsonl").is_file():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("usage"))
            manifest.append({"case_id": case_id, "action": "copied", "from": str(source)})
        else:
            target.mkdir(parents=True)
            manifest.append({"case_id": case_id, "action": "created_empty"})
    (root / "seed_manifest.json").write_text(json.dumps({"seeded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                          "cases": manifest}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("完成；接着跑 python scripts/eval_r0.py check")
    return 0


def cmd_lock(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    root = rel(cfg["search"]["snapshot_root"])
    lock_path = rel(cfg["search"]["lock_file"])
    if lock_path.exists() and not args.force:
        print(f"{lock_path} 已存在（基线快照只锁一次）；确实要重锁加 --force，并在文档里记一笔")
        return 1
    cases, last = {}, ""
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        snap = check_snapshot_dir(directory)
        lines = (directory / "index.jsonl").read_bytes().count(b"\n") if snap.get("index") else 0
        cases[directory.name] = {"entries": snap["entries"], "lines": lines, "index_sha256": snap.get("index_sha256"),
                                 "missing_blobs": len(snap["missing_blobs"]), "last_recorded_at": snap.get("last_recorded_at")}
        last = max(last, snap.get("last_recorded_at") or "")
    lock = {"locked_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "as_of": last, "cases": cases,
            "use": "训后评测设 FOREN_SNAPSHOT_AS_OF=as_of（eval_r0.py run --phase after_sft 自动设）：共用查询看到的就是基线当时的结果"}
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写入 {lock_path}；as_of = {last}")
    return 0


# ---------------------------------------------------------------- plan / run

def phase_tag(cfg: dict[str, Any], phase: str) -> str:
    return cfg["round_tags"][phase]


def build_commands(cfg: dict[str, Any], phase: str, endpoint: str, model: str, jobs: int,
                   seeds: list[int]) -> list[dict[str, Any]]:
    eval_set = load_json(rel(cfg["eval_set"]))
    h = cfg["harness"]
    freeze_path = rel(h["freeze_file"])
    # auto_tools 跟冻结文件走（清单 A9 可能把它定为非 off）：训练数据怎么生成，评测就怎么跑，否则训练 / 部署不一致
    auto_tools = ((load_json(freeze_path).get("checks") or {}).get("auto_tools") if freeze_path.is_file() else None) or "off"
    commands = []
    for seed in seeds:
        round_tag = f"{phase_tag(cfg, phase)}-s{seed}"
        for set_name, spec in eval_set["sets"].items():
            argv = [PY, "-m", "foren_searcher.datagen.run_cases", "--cases", spec["cases_dir"], "--runs-dir", cfg["runs_dir"],
                    "--split", spec["run_cases_split"], "--ids", *[c["case_id"] for c in spec["cases"]],
                    "--round", round_tag, "--samples", "1", "--endpoint", endpoint, "--model", model,
                    "--instruction-mode", h["instruction_mode"], "--hi-spec", h["hi_spec"], "--gia-spec", h["gia_spec"],
                    "--search-mode", cfg["search"]["mode"], "--snapshot-root", cfg["search"]["snapshot_root"],
                    "--runtime-guards", h["runtime_guards"], "--blocklist", h["blocklist"], "--jobs", str(jobs),
                    "--retry-failed"]
            if h.get("no_think"):
                argv.append("--no-think")
            if h.get("image_search"):                  # 10-19：EVID 帧不进任何以图搜（负责人定 off，两阶段两集合同一值）
                argv += ["--image-search", h["image_search"]]
            if auto_tools != "off":
                argv += ["--auto-tools", auto_tools]
            commands.append({"seed": seed, "set": set_name, "round": round_tag, "argv": argv})
    return commands


def phase_env(cfg: dict[str, Any], phase: str) -> dict[str, str]:
    env = {"PYTHONPATH": "src", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "FOREN_CONFIG": cfg["harness"]["config"]}
    if phase == "after_sft":
        lock_path = rel(cfg["search"]["lock_file"])
        if not lock_path.is_file():
            raise SystemExit(f"训后评测要先有基线快照锁 {lock_path}（基线跑完后 python scripts/eval_r0.py lock）")
        env["FOREN_SNAPSHOT_AS_OF"] = load_json(lock_path)["as_of"]
    return env


def describer_env(cfg: dict[str, Any], endpoint: str) -> dict[str, str]:
    """INSPECT 比对图的固定描述模型（inspect/describe.py）：训前训后都用被测端点上未挂 LoRA 的底座。不设时它取 run 自己的
    --model，训后是 "HI=…,GIA=…"（服务上没有这个名字），比对图描述整行缺失，两次的观察不一样。"""

    model = (cfg["harness"].get("describer") or {}).get("model")
    return {"FOREN_DESCRIBE_ENDPOINT": endpoint.rstrip("/"), "FOREN_DESCRIBE_MODEL": model} if model else {}


def cmd_plan(cfg: dict[str, Any], args: argparse.Namespace, *, execute: bool = False) -> int:
    env = {**phase_env(cfg, args.phase), **describer_env(cfg, args.endpoint)}
    commands = build_commands(cfg, args.phase, args.endpoint, args.model, args.jobs, args.seeds)
    print("环境变量：" + " ".join(f"{k}={v}" for k, v in env.items()))
    for item in commands:
        print(f"[种子 {item['seed']} / {item['set']}] " + " ".join(item["argv"]))
    if not execute:
        return 0
    if cmd_check(cfg, argparse.Namespace()) != 0 and not args.ignore_check:
        print("check 没过，不开跑（确认问题无关可加 --ignore-check，并在文档里记一笔）")
        return 1
    full_env = {**os.environ, **env}
    log = rel(cfg["reports_dir"]) / f"run_{phase_tag(cfg, args.phase)}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"phase": args.phase, "env": env, "commands": commands}, ensure_ascii=False) + "\n")
        for item in commands:
            print(f"== 种子 {item['seed']} / {item['set']}", flush=True)
            code = subprocess.call(item["argv"], cwd=ROOT, env=full_env)
            stream.write(json.dumps({"seed": item["seed"], "set": item["set"], "exit": code}) + "\n")
            if code != 0:
                print(f"退出码 {code}（--retry-failed：再跑一次会续上）")
    return 0


# ---------------------------------------------------------------- score / metrics

def cmd_score(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    eval_set = load_json(rel(cfg["eval_set"]))
    judge = cfg["judge"]
    env = {**os.environ, "PYTHONPATH": "src", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    evid_ids = [c["case_id"].split(":", 1)[1] for c in eval_set["sets"]["evid_ctrl20"]["cases"]]
    dev = eval_set["sets"]["dev_eval4"]["cases"]
    for seed in args.seeds:
        round_tag = f"{phase_tag(cfg, args.phase)}-s{seed}"
        evid = [PY, judge["evid"]["script"], "--round", round_tag, "--runs-dir", cfg["runs_dir"], "--out-dir", cfg["scores_dir"],
                "--cases", eval_set["sets"]["evid_ctrl20"]["cases_dir"], "--ids", *evid_ids, "--judges", *judge["evid"]["judges"]]
        dev_runs = [str(Path(cfg["runs_dir"]) / f"{case_key(c['case_id'])}_{round_tag}_1") for c in dev]
        verify = [PY, judge["dev_eval"]["script"], "--runs", *dev_runs, "--judge", judge["dev_eval"]["judge"],
                  "--model", judge["dev_eval"]["model"], "--to", str(judge["dev_eval"]["to_step"]),
                  "--round", f"{round_tag}-dev"]
        for argv in (evid, verify):
            print(" ".join(argv))
            if not args.dry_run:
                code = subprocess.call(argv, cwd=ROOT, env=env)
                if code != 0:
                    print(f"退出码 {code}（缓存在，重跑会续上）")
    return 0


def cmd_metrics(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    tag = phase_tag(cfg, args.phase)
    rounds = [f"{tag}-s{seed}" for seed in args.seeds]
    scores = [str(Path(cfg["scores_dir"]) / f"{r}.sonnet.subset.jsonl") for r in rounds]
    present = [s for s in scores if rel(s).is_file()]
    if len(present) < len(scores):
        print(f"还没有评分文件：{[s for s in scores if s not in present]}（先 eval_r0.py score；没有分数时只出行为指标）")
    argv = [PY, cfg["metrics"]["script"], cfg["runs_dir"], *sum((["--round", r] for r in rounds), []),
            "--eval-set", cfg["eval_set"], *(["--scores", *present] if present else []),
            "--out", str(Path(cfg["reports_dir"]) / tag), "--title", f"r0 评测 {args.phase}（{tag}）"]
    if args.phase == "after_sft":
        argv += ["--criteria", cfg["success_criteria"], "--baseline", str(Path(cfg["reports_dir"]) / f"{phase_tag(cfg, 'baseline')}.json")]
    print(" ".join(argv))
    return 0 if args.dry_run else subprocess.call(argv, cwd=ROOT, env={**os.environ, "PYTHONUTF8": "1"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="r0 评测（E2 训前基线 / 训后）")
    parser.add_argument("--config", type=Path, default=CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    seed = sub.add_parser("seed")
    seed.add_argument("--yes", action="store_true")
    lock = sub.add_parser("lock")
    lock.add_argument("--force", action="store_true")
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--phase", choices=["baseline", "after_sft"], required=True)
        p.add_argument("--endpoint", required=True, help="被测模型的 OpenAI 兼容地址（/v1 结尾）")
        p.add_argument("--model", required=True, help="服务上的模型名；两套 LoRA 写 HI=hi_r0,GIA=gia_r0")
        p.add_argument("--jobs", type=int, default=2)
        p.add_argument("--seeds", type=int, nargs="+", default=None)
        p.add_argument("--ignore-check", action="store_true")
    for name in ("score", "metrics"):
        p = sub.add_parser(name)
        p.add_argument("--phase", choices=["baseline", "after_sft"], required=True)
        p.add_argument("--seeds", type=int, nargs="+", default=None)
        p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_json(args.config)
    if getattr(args, "seeds", "absent") is None:
        args.seeds = list(range(1, int(cfg["sampling"]["seeds"]) + 1))
    if args.command == "check":
        return cmd_check(cfg, args)
    if args.command == "seed":
        return cmd_seed(cfg, args)
    if args.command == "lock":
        return cmd_lock(cfg, args)
    if args.command in ("plan", "run"):
        return cmd_plan(cfg, args, execute=args.command == "run")
    if args.command == "score":
        return cmd_score(cfg, args)
    return cmd_metrics(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
