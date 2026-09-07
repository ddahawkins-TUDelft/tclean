"""Tests for level-shift data-quality evaluation."""

import pandas as pd

from tclean import TimeGrid
from tclean.data_quality.methods.level_shift import build_details, evaluate


def _grid() -> TimeGrid:
    """Return a sixteen-hour test grid."""
    return TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-01T16:00:00Z", frequency="1h"
    )


def _data(values: list[float | None]) -> pd.DataFrame:
    """Build single-context test data."""
    return pd.DataFrame({"A": values}, index=pd.DatetimeIndex(_grid().target_index))


def _test(*, window_duration: str = "3h", threshold: float = 40) -> dict:
    """Build a validated-style level-shift test."""
    return {
        "name": "level_change",
        "method": "level_shift",
        "window_duration": pd.Timedelta(window_duration),
        "threshold": threshold,
    }


def test_evaluate_localizes_upward_level_shift():
    """Localize one persistent upward level shift."""
    data = _data([100] * 6 + [150] * 10)

    result = evaluate(data, test=_test(), grid=_grid())

    assert result.index[result["A"]].tolist() == [pd.Timestamp("2026-01-01T06:00:00Z")]


def test_evaluate_localizes_downward_level_shift():
    """Preserve the direction of a persistent downward level shift."""
    data = _data([150] * 6 + [100] * 10)

    result = evaluate(data, test=_test(), grid=_grid())

    assert result.index[result["A"]].tolist() == [pd.Timestamp("2026-01-01T06:00:00Z")]

    details = build_details(
        data["A"],
        start=pd.Timestamp("2026-01-01T06:00:00Z"),
        end=pd.Timestamp("2026-01-01T07:00:00Z"),
        test=_test(),
        grid=_grid(),
    )

    assert details["estimated_shift"] == -50.0


def test_evaluate_treats_threshold_as_exclusive():
    """Do not flag a shift exactly equal to the threshold."""
    data = _data([100] * 6 + [150] * 10)

    result = evaluate(data, test=_test(threshold=50), grid=_grid())

    assert not result["A"].any()


def test_evaluate_uses_paired_profile_offset():
    """Detect a coherent offset despite variation within each profile."""
    data = _data(
        [100, 80, 120, 200, 100, 80, 120, 200, 150, 130, 170, 250, 150, 130, 170, 250]
    )

    result = evaluate(
        data, test=_test(window_duration="4h", threshold=40), grid=_grid()
    )

    assert result.index[result["A"]].tolist() == [pd.Timestamp("2026-01-01T08:00:00Z")]


def test_evaluate_does_not_treat_shape_change_as_level_shift():
    """Allow opposing paired changes to cancel rather than mimic a shift."""
    data = _data(
        [100, 80, 120, 200, 100, 80, 120, 200, 80, 120, 200, 100, 80, 120, 200, 100]
    )

    result = evaluate(
        data, test=_test(window_duration="4h", threshold=30), grid=_grid()
    )

    assert not result["A"].any()


def test_evaluate_requires_complete_paired_windows():
    """Do not infer a level shift across incomplete comparison windows."""
    data = _data(
        [
            100,
            100,
            100,
            100,
            None,
            100,
            150,
            150,
            150,
            150,
            150,
            150,
            150,
            150,
            150,
            150,
        ]
    )

    result = evaluate(data, test=_test(), grid=_grid())

    assert not result["A"].any()


def test_evaluate_requires_evidence_on_both_sides():
    """Do not detect shifts too close to a dataset boundary."""
    near_start = _data([100] + [150] * 15)

    near_end = _data([100] * 15 + [150])

    start_result = evaluate(near_start, test=_test(), grid=_grid())

    end_result = evaluate(near_end, test=_test(), grid=_grid())

    assert not start_result["A"].any()
    assert not end_result["A"].any()


def test_evaluate_applies_independently_to_contexts():
    """Evaluate level shifts independently for each context."""
    data = pd.DataFrame(
        {"A": [100] * 6 + [150] * 10, "B": [200] * 9 + [140] * 7},
        index=pd.DatetimeIndex(_grid().target_index),
    )

    result = evaluate(data, test=_test(), grid=_grid())

    assert result.index[result["A"]].tolist() == [pd.Timestamp("2026-01-01T06:00:00Z")]

    assert result.index[result["B"]].tolist() == [pd.Timestamp("2026-01-01T09:00:00Z")]


def test_evaluate_localizes_separate_level_shifts():
    """Return one localized failure for each disconnected shift event."""
    data = _data([100] * 5 + [160] * 6 + [100] * 5)

    result = evaluate(data, test=_test(), grid=_grid())

    assert result.index[result["A"]].tolist() == [
        pd.Timestamp("2026-01-01T05:00:00Z"),
        pd.Timestamp("2026-01-01T11:00:00Z"),
    ]


def test_build_details_uses_full_support_region():
    """Describe evidence from every boundary supporting the localized shift."""
    data = _data([100] * 6 + [150] * 10)

    details = build_details(
        data["A"],
        start=pd.Timestamp("2026-01-01T06:00:00Z"),
        end=pd.Timestamp("2026-01-01T07:00:00Z"),
        test=_test(),
        grid=_grid(),
    )

    assert details["window_duration"] == pd.Timedelta("3h")
    assert details["threshold"] == 40

    assert details["change_point"] == pd.Timestamp("2026-01-01T06:00:00Z")
    assert details["estimated_shift"] == 50.0

    assert details["qualifying_boundary_count"] == 3

    assert details["pre_evidence_start"] == pd.Timestamp("2026-01-01T02:00:00Z")
    assert details["post_evidence_end"] == pd.Timestamp("2026-01-01T10:00:00Z")
