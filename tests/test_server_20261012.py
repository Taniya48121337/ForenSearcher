"""Server baseline package (10-12): orchestrator commands, preflight sections, packing list."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_run_command_matches_the_evid_full_run_setting():
    R = _load("run_baselines", ROOT / "server" / "run_baselines.py")
    cmd = R.run_command(R.ARMS["evid_merged"], ["video_131"])
    joined = " ".join(cmd)
    # 10-12：DETECT 并入 SEARCH / INSPECT，合并目录 = 规格 hi@0.8.0 / gia@0.9.0（以图搜图默认开）
    for part in ("--no-think", "--instruction-mode student", "--blocklist exclude", "--hi-spec hi@0.8.0", "--gia-spec gia@0.9.0",
                 "--runtime-guards eval", "--state-renderer compact_16k", "--search-mode replay_or_record", "evid:video_131"):
        assert part in joined, part
    assert "--detect" not in joined and "--image-search" not in joined
    assert R.ARMS["evid_detect"] is R.ARMS["evid_merged"]
    base = " ".join(R.run_command(R.ARMS["evid_base"], ["video_131"]))
    assert "--hi-spec hi@0.7.0" in base and "--image-search off" in base and "search_snapshots_server" in base
    assert "g360:123" in " ".join(R.run_command(R.ARMS["g360"], ["123"]))
    assert R.run_dir("evid", "video_131", "srvM27").name == "evid_video_131_srvM27_1"
    assert R.run_dir("fmnv", "abc", "fsrvM27").name == "fmnv_abc_fsrvM27_1"


def test_preflight_has_all_sections():
    P = _load("preflight", ROOT / "server" / "preflight.py")
    assert set(P.SECTIONS) == {"env", "files", "tools", "llm", "net", "detect", "judge", "tests"}
    assert all(callable(fn) for fns in P.SECTIONS.values() for fn in fns)


def test_pack_list_contains_what_the_server_needs():
    K = _load("_pack_server", ROOT / "scripts" / "_pack_server.py")
    files = {p.relative_to(ROOT).as_posix() for p in K.collect()}
    for need in ("server/env.sh", "server/preflight.py", "server/run_baselines.py", "configs/evid_server_27b.toml",
                 "scripts/_bench_common.py", "scripts/_evid_score.py", "src/foren_searcher/inspect/describe.py",
                 "specs/hi@0.8.0/spec.json", "specs/gia@0.9.0/spec.json"):
        assert need in files, need
    assert not any("__pycache__" in f for f in files)
