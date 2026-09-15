import pytest
from temporal_metrics import temporal_iou


def test_partial_exact_disjoint_and_uncertainty_envelope():
    assert temporal_iou([45, 51], [38, 52]) == pytest.approx(6 / 14)
    assert temporal_iou([38, 51.5], [38, 52]) == pytest.approx(13.5 / 14)
    assert temporal_iou([38, 52], [38, 52]) == 1
    assert temporal_iou([0, 4], [38, 52]) == 0
    with pytest.raises(ValueError):
        temporal_iou([4, 4], [38, 52])
