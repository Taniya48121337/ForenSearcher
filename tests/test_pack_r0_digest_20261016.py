"""10-16 r0 数据包定稿：pack_r0 的 data_digest 按 posix 路径字符串排序，打包与 --verify 同一口径（以前打包按 Path 排序，
Windows 上不分大小写，fmnv_c__0oux… 与 fmnv_c_DQc… 的先后和 --verify 相反，Windows 打的包校验不过）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pack_r0  # noqa: E402


def test_data_digest_ignores_insertion_order() -> None:
    a = {"data/images/fmnv_c_DQc/x.png": "1", "data/images/fmnv_c__0ou/x.png": "2", "data/hi_accepted.jsonl": "3"}
    b = dict(reversed(list(a.items())))
    assert pack_r0.data_digest(a) == pack_r0.data_digest(b)


def test_verify_accepts_mixed_case_image_dirs(tmp_path: Path) -> None:
    pack = tmp_path / "r0"
    files = {"data/images/fmnv_c_DQcHjdjCdMI_r0g_1/image_01.png": b"a",
             "data/images/fmnv_c__0oux8o9MJo_r0g_1/image_01.png": b"b",
             "data/hi_accepted.jsonl": b"{}\n"}
    for rel, content in files.items():
        (pack / rel).parent.mkdir(parents=True, exist_ok=True)
        (pack / rel).write_bytes(content)
    hashes = {rel: pack_r0.sha256_file(pack / rel) for rel in files}
    (pack / "r0_pack.json").write_text(json.dumps({"pack_id": "t", "status": "ok", "data_digest": pack_r0.data_digest(hashes)}),
                                       encoding="utf-8")
    sums = [f"{pack_r0.sha256_file(p)}  {p.relative_to(pack).as_posix()}" for p in sorted(pack.rglob("*")) if p.is_file()]
    (pack / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    assert pack_r0.verify(pack) == 0
