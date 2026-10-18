"""10-18 g1：pack_r0.py --teacher-override（S0–S3 按覆盖文件核、S4 / S5 仍按冻结文件），身份文件按 *_pack.json 找。"""

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


BASE = {"slots": {"S0": "qwen", "S1": "qwen", "S2": "qwen", "S3": "qwen", "S4": "strong", "S5": "strong"},
        "backends": {"qwen": {"model": "qwen/qwen3.5-397b-a17b"}, "strong": {"model": "gpt-6-astra"}},
        "backup_backends": {"qwen": {"model": "qwen/qwen3.5-122b-a10b"}, "strong": {"model": "anthropic/claude-sonnet-5.5"}}}
OVERRIDE = {"slots": {"S0": "codex_low", "S1": "codex_low", "S2": "codex_low", "S3": "codex_low", "S4": "strong", "S5": "strong"},
            "backends": {"codex_low": {"model": "gpt-6-astra"}, "strong": {"model": "something-else"}}}
CODEX_RUN = {"run": "x_r1g1_1", "slot_models": {k: "gpt-6-astra" for k in BASE["slots"]},
             "slot_backends": {**{k: "codex_low" for k in ("S0", "S1", "S2", "S3")}, "S4": "strong", "S5": "strong"}}


def test_override_checks_s0_s3_only() -> None:
    pack = _pack()
    assert not pack.slot_teacher_check([CODEX_RUN], BASE)["ok"]                 # 不给覆盖：S0–S3 对不上 qwen
    res = pack.slot_teacher_check([CODEX_RUN], BASE, OVERRIDE)
    assert res["ok"] and res["runs"] == {"x_r1g1_1": "primary"}
    # S4 仍按基础冻结文件核：覆盖文件里 strong 写成别的模型也不影响
    bad = {**CODEX_RUN, "slot_models": {**CODEX_RUN["slot_models"], "S4": "something-else"}}
    assert not pack.slot_teacher_check([bad], BASE, OVERRIDE)["ok"]


def test_verify_finds_any_round_identity(tmp_path: Path) -> None:
    pack = _pack()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.jsonl").write_text("{}\n", encoding="utf-8")
    digest = pack.data_digest({"data/x.jsonl": pack.sha256_file(tmp_path / "data" / "x.jsonl")})
    (tmp_path / "g1_pack.json").write_text(json.dumps({"pack_id": "g1-test", "status": "ok", "data_digest": digest}), encoding="utf-8")
    lines = [f"{pack.sha256_file(p)}  {p.relative_to(tmp_path).as_posix()}" for p in sorted(tmp_path.rglob("*")) if p.is_file()]
    (tmp_path / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert pack.main(["--round", "g1", "--verify", str(tmp_path)]) == 0
