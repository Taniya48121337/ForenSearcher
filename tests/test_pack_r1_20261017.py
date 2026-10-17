"""10-17 r1：pack_r0.py 按轮次打包的默认值与 provenance 核对。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pack():
    spec = importlib.util.spec_from_file_location("pack_r0", ROOT / "scripts" / "pack_r0.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_round_defaults_and_identity_file(tmp_path: Path) -> None:
    pack = _pack()
    # 校验一个只有 r1_pack.json 的包：身份文件按 r*_pack.json 找
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.jsonl").write_text("{}\n", encoding="utf-8")
    digest = pack.data_digest({"data/x.jsonl": pack.sha256_file(tmp_path / "data" / "x.jsonl")})
    (tmp_path / "r1_pack.json").write_text(json.dumps({"pack_id": "r1-test", "status": "ok", "data_digest": digest}), encoding="utf-8")
    lines = [f"{pack.sha256_file(p)}  {p.relative_to(tmp_path).as_posix()}" for p in sorted(tmp_path.rglob("*")) if p.is_file()]
    (tmp_path / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert pack.main(["--round", "r1", "--verify", str(tmp_path)]) == 0


def test_provenance_flags_untracked_input(tmp_path: Path) -> None:
    pack = _pack()
    probe = ROOT / "experiments" / "sft_data" / "_pytest_untracked_provenance_probe.json"
    probe.write_text("{}", encoding="utf-8")
    try:
        ok, detail = pack.provenance_check([probe.relative_to(ROOT).as_posix(), "no/such/file.json"])
    finally:
        probe.unlink()
    assert not ok
    assert probe.relative_to(ROOT).as_posix() in detail["untracked"]
    assert detail["unmatched_patterns"] == ["no/such/file.json"]
