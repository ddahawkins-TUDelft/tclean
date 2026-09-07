"""Tests for low-variability data-quality evaluation."""

import pandas as pd
import pytest

from tclean import TimeGrid
from tclean.data_quality.methods.low_variability import build_details, evaluate


def _grid() -> TimeGrid:
    """Return an eight-hour test grid."""
    return TimeGrid(
        start="2026-01-01T00:00:00Z",
        end="2026-01-01T08:00:00Z",
        frequency="1h",
    )


def _data(values: list[float | None]) -> pd.DataFrame:
    """Build single-context test data."""
    return pd.DataFrame(
        {"A": values},
        index=pd.DatetimeIndex(_grid().target_index),
    )


def _test(
    *,
    window_duration: str = "3h",
    maximum_range: float = 1.0,
) -> dict:
    """Build a validated-style low-variability test."""
    return {
        "name": "low_variability",
        "method": "low_variability",
        "window_duration": pd.Timedelta(window_duration),
        "maximum_range": maximum_range,
    }


def test_evaluate_flags_one_qualifying_window():
    """Flag every observation belonging to a qualifying window."""
    data = _data([10, 1.0, 1.2, 1.4, 10, 20, 30, 40])

    result = evaluate(
        data,
        test=_test(maximum_range=0.5),
        grid=_grid(),
    )

    assert result["A"].tolist() == [
        False,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]


def test_evaluate_does_not_flag_window_above_threshold():
    """Do not flag windows whose range exceeds the threshold."""
    data = _data([10, 1.0, 1.2, 2.0, 10, 20, 30, 40])

    result = evaluate(
        data,
        test=_test(maximum_range=0.5),
        grid=_grid(),
    )

    assert not result["A"].any()


def test_evaluate_treats_threshold_as_inclusive():
    """Allow a window range exactly equal to the threshold."""
    data = _data([10, 1.0, 1.2, 1.5, 10, 20, 30, 40])

    result = evaluate(
        data,
        test=_test(maximum_range=0.5),
        grid=_grid(),
    )

    assert result["A"].tolist() == [
        False,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]


def test_evaluate_combines_overlapping_qualifying_windows():
    """Union observations from overlapping low-variability windows."""
    data = _data([0.0, 0.1, 0.2, 0.3, 0.4, 10, 20, 30])

    result = evaluate(
        data,
        test=_test(maximum_range=0.2),
        grid=_grid(),
    )

    assert result["A"].tolist() == [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_evaluate_handles_separate_low_variability_periods():
    """Keep disconnected qualifying windows as separate failures."""
    data = _data([1.0, 1.1, 1.2, 10, 5.0, 5.1, 5.2, 20])

    result = evaluate(
        data,
        test=_test(maximum_range=0.2),
        grid=_grid(),
    )

    assert result["A"].tolist() == [
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        False,
    ]


def test_evaluate_requires_complete_window():
    """Do not evaluate windows containing missing observations."""
    data = _data([1.0, None, 1.1, 10, 20, 30, 40, 50])

    result = evaluate(
        data,
        test=_test(maximum_range=1.0),
        grid=_grid(),
    )

    assert not result["A"].any()


def test_evaluate_does_not_bridge_missing_values():
    """Allow qualifying windows only after a missing value has left the window."""
    data = _data([1.0, None, 5.0, 5.1, 5.2, 10, 20, 30])

    result = evaluate(
        data,
        test=_test(maximum_range=0.2),
        grid=_grid(),
    )

    assert result["A"].tolist() == [
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_evaluate_applies_independently_to_contexts():
    """Evaluate low variability independently for each context."""
    data = pd.DataFrame(
        {
            "A": [1.0, 1.1, 1.2, 10, 20, 30, 40, 50],
            "B": [10, 20, 2.0, 2.1, 2.2, 30, 40, 50],
        },
        index=pd.DatetimeIndex(_grid().target_index),
    )

    result = evaluate(
        data,
        test=_test(maximum_range=0.2),
        grid=_grid(),
    )

    assert result["A"].tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]

    assert result["B"].tolist() == [
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_build_details_describes_contributing_windows():
    """Distinguish qualifying-window variation from merged-period variation."""
    data = _data([0.0, 0.1, 0.2, 0.3, 0.4, 10, 20, 30])

    details = build_details(
        data["A"],
        start=pd.Timestamp("2026-01-01T00:00:00Z"),
        end=pd.Timestamp("2026-01-01T05:00:00Z"),
        test=_test(maximum_range=0.2),
        grid=_grid(),
    )

    assert details["window_duration"] == pd.Timedelta("3h")
    assert details["maximum_range"] == 0.2
    assert details["qualifying_window_count"] == 3
    assert details["minimum_window_range"] == pytest.approx(0.2)
    assert details["maximum_window_range"] == pytest.approx(0.2)
    assert details["period_observed_minimum"] == 0.0
    assert details["period_observed_maximum"] == 0.4
    assert details["period_observed_range"] == pytest.approx(0.4)


def test_evaluate_handles_floating_point_threshold_boundary():
    """Treat mathematically equal floating-point ranges as at the threshold."""
    data = _data([10, 5.0, 5.1, 5.2, 20, 30, 40, 50])

    result = evaluate(
        data,
        test=_test(maximum_range=0.2),
        grid=_grid(),
    )

    assert result["A"].tolist() == [
        False,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]

