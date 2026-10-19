"""服务器代码副本与某个 git 提交逐文件比对（单卡 Vast，2026-10-19；bootstrap 复用）。git blob hash 用纯 Python 算（与 git hash-object 相同），
服务器上不需要 .git。

    # 本机：列出提交里全部 tracked 文件与 blob hash
    python server/code_parity.py manifest --ref HEAD --out parity_manifest.tsv
    # 服务器：逐文件比对，输出三类
    python server/code_parity.py check --manifest parity_manifest.tsv --root /workspace/repo --json parity_result.json

三类：missing（提交里有、服务器没有）、different（内容不同）、extra（服务器上有、提交里没有，且不在运行时目录白名单里）。
运行时目录白名单（评测 run / 快照 / 报告、模型、数据包、环境链接、日志、缓存）只用于 extra，不影响 missing / different。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

RUNTIME_PREFIXES = (
    ".git/", ".local/", ".models/", "dist/", ".venv-agent", ".venv-vllm", "plan-harness-1", ".pytest-tmp/", "server_logs/",
    "experiments/eval/r0/runs/", "experiments/eval/r0/search_snapshots/", "experiments/eval/r0/reports/", "experiments/eval/r0/scores/",
    "experiments/evid/", "experiments/sft_data/batch_logs/", "experiments/sft_data/run_manifests/", "experiments/sft_data/data/",
    "experiments/sft_data/r1_pack/", "experiments/sft_data/search_snapshots/",
)
RUNTIME_PARTS = ("__pycache__", ".egg-info")


def blob_sha(path: Path) -> str:
    if path.is_symlink():
        data = os.readlink(path).encode("utf-8")
    else:
        data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def manifest(args: argparse.Namespace) -> int:
    out = subprocess.run(["git", "-c", "core.quotepath=false", "ls-tree", "-r", "-z", args.ref], capture_output=True, check=True).stdout
    rows = []
    for item in out.split(b"\0"):
        if not item:
            continue
        meta, path = item.split(b"\t", 1)
        mode, kind, sha = meta.decode().split()
        if kind == "blob":
            rows.append(f"{sha}\t{mode}\t{path.decode('utf-8')}")
    commit = subprocess.run(["git", "rev-parse", args.ref], capture_output=True, text=True, check=True).stdout.strip()
    describe = subprocess.run(["git", "describe", "--tags", "--always", args.ref], capture_output=True, text=True).stdout.strip()
    Path(args.out).write_text(f"# ref {args.ref} commit {commit} describe {describe}\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"{len(rows)} tracked files at {args.ref} ({commit[:10]}, {describe}) -> {args.out}")
    return 0


def check(args: argparse.Namespace) -> int:
    root = Path(args.root)
    lines = Path(args.manifest).read_text(encoding="utf-8").splitlines()
    header = lines[0] if lines and lines[0].startswith("#") else ""
    expected = {}
    for line in lines:
        if line and not line.startswith("#"):
            sha, mode, path = line.split("\t", 2)
            expected[path] = (sha, mode)
    missing, different = [], []
    for path, (sha, _mode) in expected.items():
        p = root / path
        if not (p.is_file() or p.is_symlink()):
            missing.append(path)
        elif blob_sha(p) != sha:
            different.append(path)
    extra = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
        rel_dir = "" if rel_dir == "." else rel_dir + "/"
        dirnames[:] = [d for d in dirnames if not (rel_dir + d + "/").startswith(RUNTIME_PREFIXES)
                       and not any(part in d for part in RUNTIME_PARTS)]
        for name in filenames:
            rel = rel_dir + name
            if rel in expected or rel.startswith(RUNTIME_PREFIXES) or any(part in rel for part in RUNTIME_PARTS):
                continue
            extra.append(rel)
    top = lambda items: dict(Counter(p.split("/", 1)[0] if "/" in p else "(root)" for p in items).most_common())  # noqa: E731
    result = {"manifest": header, "root": str(root), "tracked": len(expected), "same": len(expected) - len(missing) - len(different),
              "missing": len(missing), "different": len(different), "extra": len(extra),
              "missing_by_top": top(missing), "different_by_top": top(different), "extra_by_top": top(extra),
              "missing_files": sorted(missing), "different_files": sorted(different), "extra_files": sorted(extra)}
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{header}\ntracked {result['tracked']}: same {result['same']}, missing {result['missing']}, different {result['different']}; extra {result['extra']}")
    for key in ("missing", "different", "extra"):
        if result[key]:
            print(f"  {key} by top-level dir: {result[key + '_by_top']}")
            for p in result[key + "_files"][: args.show]:
                print(f"    {key}: {p}")
    return 0 if not (missing or different or extra) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--ref", default="HEAD")
    m.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--manifest", required=True)
    c.add_argument("--root", default=".")
    c.add_argument("--json", default=None)
    c.add_argument("--show", type=int, default=40)
    args = ap.parse_args()
    return manifest(args) if args.cmd == "manifest" else check(args)


if __name__ == "__main__":
    raise SystemExit(main())
