"""10-18：冻结文件按轮次显式选（run_cases.load_freeze）——specs/ 里同时有 g1_freeze.json 与 r1_freeze.json（规格相同、g1 文件名排在前）时，
r1 轮次与不给轮次的调用方仍取 r1_freeze；r1g* 取 g1_freeze；FOREN_FREEZE_FILE 覆盖；r0 规格照旧按规格找到 r0_freeze。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foren_searcher.datagen.run_cases import load_freeze  # noqa: E402

R1 = ("hi@0.11.0", "gia@0.9.0")
R0 = ("hi@0.10.0", "gia@0.9.0")


@pytest.fixture()
def specs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("FOREN_FREEZE_FILE", raising=False)
    for name, fid, (hi, gia) in (("g1_freeze.json", "g1", R1), ("r1_freeze.json", "r1", R1), ("r0_freeze.json", "r0", R0)):
        (tmp_path / name).write_text(json.dumps({"freeze_id": fid, "specs": {"hi": hi, "gia": gia}}), encoding="utf-8")
    return tmp_path


def test_r1_round_still_gets_r1_freeze_when_g1_present(specs: Path) -> None:
    assert sorted(p.name for p in specs.glob("*_freeze.json"))[0] == "g1_freeze.json"      # 旧取法会拿到它
    assert load_freeze(specs, *R1, "r1")["freeze_id"] == "r1"
    assert load_freeze(specs, *R1)["freeze_id"] == "r1"                                      # 不给轮次：默认 r1_freeze
    assert load_freeze(specs, *R1, "r0g")["freeze_id"] == "r1"                               # 轮次不在映射里：兜底同上


def test_round_mapping_and_override(specs: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert load_freeze(specs, *R1, "r1g2")["freeze_id"] == "g1"
    assert load_freeze(specs, *R1, "r1g1")["freeze_id"] == "g1"
    assert load_freeze(specs, *R0, "r0g")["freeze_id"] == "r0"                               # r0 规格照旧按规格找
    assert load_freeze(specs, *R0, "r1") is None                                             # 映射到的文件规格对不上：不乱认别的
    monkeypatch.setenv("FOREN_FREEZE_FILE", "g1_freeze.json")
    assert load_freeze(specs, *R1, "r1")["freeze_id"] == "g1"


def test_round_only_file_never_found_by_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOREN_FREEZE_FILE", raising=False)
    (tmp_path / "g1_freeze.json").write_text(json.dumps({"freeze_id": "g1", "specs": {"hi": R1[0], "gia": R1[1]}}), encoding="utf-8")
    assert load_freeze(tmp_path, *R1) is None                                                # 没有 r1_freeze 时也不退到 g1
    assert load_freeze(tmp_path, *R1, "r1g2")["freeze_id"] == "g1"
