"""10-19 负责人：门的族口径 averitec 归 C；C2 单一族按权重判、按行同列；C4 不再有"A 族本轮不进"，各族 accepted 行数（含 A）照列。"""

from __future__ import annotations

from pathlib import Path

from test_export_gate_20261015 import _gate, _row, _write, export_gate, walker_coverage


def test_averitec_is_c_family() -> None:
    assert walker_coverage.family_of("averitec:c_2130") == "C"
    assert walker_coverage.family_of("qvhighlights:6") == "B" and walker_coverage.family_of("true:1") == "other"


def test_c2_family_share_judged_by_weight(tmp_path: Path) -> None:
    # C 族 6 行（权重各 0.5）、B / E 族各 2 行（权重各 1）：按行 C 60% > 50%，按权重 C 3 / 7 ≈ 42.9% → 过
    rows = [_row(case=f"groundlie360:v{i}", sample_weight=0.5) for i in range(6)]
    rows += [_row(case=f"videosham:{i}", sample_weight=1.0) for i in range(2)]
    rows += [_row(case=f"averimatec:e{i}", sample_weight=1.0) for i in range(2)]
    c2 = _gate(_write(tmp_path, rows))["C2"]
    assert c2["value"]["max_family_share"] == 0.6 and c2["value"]["max_family_share_weight"] == 0.429
    assert c2["value"]["family_share_rows"]["C"] == 0.6 and c2["status"] == "通过"


def test_c4_lists_family_rows_without_a_allowed_zero(tmp_path: Path) -> None:
    rows = [_row(case="fmnv:a_x", branch="SEARCH text→video"), _row(case="videosham:1")]
    result = walker_coverage.export_branch_table(_write(tmp_path, rows))
    assert result["family_rows"] == {"A": 1, "B": 1}
    assert all(z["branch"] != "A 族链路" for z in result["allowed_zero"])
    assert "A 族本轮不进" not in "\n".join(walker_coverage.branch_table_markdown(result))
    assert "本轮不进" not in export_gate.check_c4(tmp_path)[0]["note"]
