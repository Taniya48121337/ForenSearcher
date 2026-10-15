"""Pack everything the server baselines need into one archive (2026-10-12). Double-click scripts\\_pack_server.cmd.

Output: dist\\ForenSearcher_server_<date>.tar.gz (code, specs, configs, tests, server scripts, EVID cases and manifest,
GroundLie360 / FMNV annotations, the vendored facenet-pytorch, FaceNet weights if found). No videos, no model weights
except FaceNet, no runs: the server downloads those itself (server/README_服务器跑baseline.md).
"""

from __future__ import annotations

import hashlib
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREES = ["src", "specs", "configs", "tests", "server", "training", ".models/deepfake/pylib"]   # training: loop LoRA jobs (10-12)
SCRIPT_GLOBS = ["scripts/*.py"]
FILES = ["pyproject.toml", "README.md", "DESIGN.md",
         "plan-harness-1/datasets/EVID-Bench/evid_bench.json",
         "plan-harness-1/datasets/GroundLie360/data.csv.gz", "plan-harness-1/datasets/GroundLie360/README.md",
         "plan-harness-1/datasets/FMNV/data.json", "plan-harness-1/datasets/FMNV/README.md"]
# 10-14 walker on the server: the generator lives in experiments/sft_data (imported by scripts/walker_run.py), the walker
# cases and their local media (B family videos, D2 clips) are needed too; runs / verdicts / exports are not shipped.
CASE_GLOBS = ["experiments/evid/cases/*.json", "experiments/sft_data/cases/*.json", "experiments/sft_data/walker_d2_cases.json",
              "experiments/sft_data/splits_manifest.json", "experiments/sft_data/oracle_generator.py",
              "experiments/sft_data/cli_generator.py"]
MEDIA_TREES: list[str] = []                       # walker media is collected per case by walker_case_files()
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".git"}
SKIP_SUFFIX = {".pyc", ".pyo"}
PREFIX = "ForenSearcher"


def collect() -> list[Path]:
    out: list[Path] = []
    for tree in TREES + MEDIA_TREES:
        base = ROOT / tree
        if not base.exists():
            print(f"!! missing folder {tree}")
            continue
        for p in base.rglob("*"):
            if p.is_file() and not (SKIP_PARTS & set(p.parts)) and p.suffix not in SKIP_SUFFIX and "assets" not in p.relative_to(ROOT).parts[1:2]:
                out.append(p)
    for pattern in SCRIPT_GLOBS + CASE_GLOBS:
        out += [p for p in ROOT.glob(pattern) if p.is_file()]
    for f in FILES:
        p = ROOT / f
        if p.is_file():
            out.append(p)
        else:
            print(f"!! missing file {f}")
    out += walker_case_files()
    return sorted(set(out))


def walker_case_files() -> list[Path]:
    """10-14：walker 案卷里引用的本地文件（材料视频 / 图片、reference.json），路径相对 cases 目录（../data/...、
    ../runs/<老 run>/artifacts/...、../media/...）。逐个案卷解析，存在的都带上；缺的打印出来。"""

    import json

    cases_dir = ROOT / "experiments" / "sft_data" / "cases"
    found: list[Path] = []
    missing: list[str] = []

    def walk(node: object, case: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("path", "reference_path") and isinstance(value, str) and value and "://" not in value:
                    target = (cases_dir / value).resolve()
                    (found.append(target) if target.is_file() else missing.append(f"{case}: {value}"))
                else:
                    walk(value, case)
        elif isinstance(node, list):
            for item in node:
                walk(item, case)

    for case_path in sorted(cases_dir.glob("*.json")):
        try:
            walk(json.loads(case_path.read_text(encoding="utf-8")), case_path.stem)
        except ValueError:
            print(f"!! cannot parse {case_path}")
    for line in missing:
        print(f"!! walker case file missing: {line}")
    print(f"walker case files: {len(found)} ({sum(p.stat().st_size for p in found) / 1e6:.0f} MB)")
    return found


def facenet_weights() -> Path | None:
    candidates = [Path.home() / ".cache/torch/hub/checkpoints/20180402-114759-vggface2.pt"]
    try:
        import torch

        candidates.insert(0, Path(torch.hub.get_dir()) / "checkpoints/20180402-114759-vggface2.pt")
    except Exception:  # noqa: BLE001
        pass
    return next((c for c in candidates if c.is_file()), None)


def main() -> int:
    files = collect()
    weights = facenet_weights()
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    out = dist / f"ForenSearcher_server_{time.strftime('%Y%m%d_%H%M')}.tar.gz"

    def info_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mode = 0o755 if info.name.endswith(".sh") else 0o644
        return info

    total = 0
    with tarfile.open(out, "w:gz") as tar:
        for p in files:
            rel = p.relative_to(ROOT).as_posix()
            data_size = p.stat().st_size
            if p.suffix == ".sh":                       # Windows checkouts may carry CRLF; bash needs LF
                text = p.read_bytes().replace(b"\r\n", b"\n")
                import io

                ti = tarfile.TarInfo(f"{PREFIX}/{rel}")
                ti.size = len(text)
                ti.mtime = int(p.stat().st_mtime)
                tar.addfile(info_filter(ti), io.BytesIO(text))
            else:
                tar.add(p, arcname=f"{PREFIX}/{rel}", filter=info_filter)
            total += data_size
        if weights:
            tar.add(weights, arcname=f"{PREFIX}/server/assets/20180402-114759-vggface2.pt", filter=info_filter)
            total += weights.stat().st_size
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    groups: dict[str, int] = {}
    for p in files:
        key = p.relative_to(ROOT).parts[0]
        groups[key] = groups.get(key, 0) + 1
    print(f"packed {len(files)} files ({total / 1e6:.0f} MB before compression): "
          + ", ".join(f"{k} {v}" for k, v in sorted(groups.items())))
    print(f"FaceNet weights: {'included (' + str(weights) + ')' if weights else 'not found on this PC (the server downloads them from GitHub)'}")
    print(f"archive: {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"sha256:  {digest}")
    (out.with_suffix(".sha256")).write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    print("\nCopy it to the server, for example (in PowerShell):")
    print(f"  scp \"{out}\" <user>@<server>:~/")
    print("then follow server/README_服务器跑baseline.md (step 0 onwards).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
