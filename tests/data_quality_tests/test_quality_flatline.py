"""Tests for flatline data-quality evaluation."""

import pandas as pd
import pytest
from _method_helpers import method_details, method_mask

from tclean import TimeGrid
from tclean.data_quality.methods.flatline import build_details, evaluate


def _grid() -> TimeGrid:
    """Return an eight-hour test grid."""
    return TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-01T08:00:00Z", frequency="1h"
    )


def _data(values: list[float | None]) -> pd.DataFrame:
    """Build single-context test data."""
    return pd.DataFrame({"A": values}, index=pd.DatetimeIndex(_grid().target_index))


def _test(*, minimum_duration: str = "3h", tolerance: float = 0.0) -> dict:
    """Build a validated-style flatline test."""
    return {
        "name": "flatline",
        "method": "flatline",
        "minimum_duration": pd.Timedelta(minimum_duration),
        "tolerance": {"value_mode": "fixed", "value": tolerance},
    }


def test_evaluate_flags_exact_flatline_meeting_duration():
    """Flag every observation in an exact qualifying flatline."""
    data = _data([1, 5, 5, 5, 2, 3, 4, 5])

    result = method_mask(evaluate, data, test=_test(), grid=_grid())

    assert result["A"].tolist() == [False, True, True, True, False, False, False, False]


def test_evaluate_does_not_flag_short_flatline():
    """Do not flag a flatline shorter than the configured duration."""
    data = _data([1, 5, 5, 2, 3, 4, 5, 6])

    result = method_mask(evaluate, data, test=_test(), grid=_grid())

    assert not result["A"].any()


def test_evaluate_flags_flatline_longer_than_minimum():
    """Flag the complete run when it exceeds the minimum duration."""
    data = _data([5, 5, 5, 5, 1, 2, 3, 4])

    result = method_mask(evaluate, data, test=_test(), grid=_grid())

    assert result["A"].tolist() == [True, True, True, True, False, False, False, False]


def test_evaluate_handles_multiple_flatlines():
    """Identify multiple qualifying flatline runs independently."""
    data = _data([5, 5, 5, 1, 8, 8, 8, 2])

    result = method_mask(evaluate, data, test=_test(), grid=_grid())

    assert result["A"].tolist() == [True, True, True, False, True, True, True, False]


def test_evaluate_uses_tolerance_between_consecutive_values():
    """Keep small consecutive changes within one flatline run."""
    data = _data([1.0, 5.00, 5.04, 4.98, 2.0, 3.0, 4.0, 5.0])

    result = method_mask(evaluate, data, test=_test(tolerance=0.1), grid=_grid())

    assert result["A"].tolist() == [False, True, True, True, False, False, False, False]


def test_evaluate_breaks_flatline_when_step_exceeds_tolerance():
    """Start a new run when consecutive values differ too much."""
    data = _data([5.00, 5.05, 5.20, 5.25, 5.30, 1.0, 2.0, 3.0])

    result = method_mask(evaluate, data, test=_test(tolerance=0.1), grid=_grid())

    assert result["A"].tolist() == [False, False, True, True, True, False, False, False]


def test_evaluate_breaks_flatline_at_missing_value():
    """Treat a missing observation as a flatline boundary."""
    data = _data([5, 5, None, 5, 5, 5, 1, 2])

    result = method_mask(evaluate, data, test=_test(), grid=_grid())

    assert result["A"].tolist() == [False, False, False, True, True, True, False, False]


def test_evaluate_applies_independently_to_contexts():
    """Evaluate flatlines independently for each context."""
    data = pd.DataFrame(
        {"A": [5, 5, 5, 1, 2, 3, 4, 5], "B": [1, 2, 8, 8, 8, 3, 4, 5]},
        index=pd.DatetimeIndex(_grid().target_index),
    )

    result = method_mask(evaluate, data, test=_test(), grid=_grid())

    assert result["A"].tolist() == [True, True, True, False, False, False, False, False]

    assert result["B"].tolist() == [False, False, True, True, True, False, False, False]


def test_evaluate_does_not_flag_missing_values():
    """Never classify missing observations themselves as flatline failures."""
    data = _data([None, None, None, 1, 2, 3, 4, 5])

    result = method_mask(evaluate, data, test=_test(), grid=_grid())

    assert not result["A"].any()


def test_build_details_reports_flatline_diagnostics():
    """Report duration, variation, and configured flatline threshold."""
    data = _data([1.0, 5.00, 5.04, 4.98, 2.0, 3.0, 4.0, 5.0])

    details = method_details(
        evaluate,
        build_details,
        data["A"],
        start=pd.Timestamp("2026-01-01T01:00:00Z"),
        end=pd.Timestamp("2026-01-01T04:00:00Z"),
        test=_test(tolerance=0.1),
        grid=_grid(),
    )

    assert details["duration"] == pd.Timedelta("3h")
    assert details["minimum_duration"] == pd.Timedelta("3h")
    assert details["tolerance"] == 0.1
    assert details["observed_minimum"] == 4.98
    assert details["observed_maximum"] == 5.04
    assert details["observed_range"] == pytest.approx(0.06)
    assert details["maximum_step_change"] == pytest.approx(0.06)


def test_evaluate_flatline_supports_derived_tolerance():
    """Resolve flatline tolerance independently from focal data."""
    data = _data([1, 1.01, 1.02, 5, 10, 15, 20, 25])
    test = _test()
    test["tolerance"] = {"value_mode": "median", "multiplier": 0.02}

    result = method_mask(evaluate, data, test=test, grid=_grid())

    assert result["A"].tolist()[:3] == [True, True, True]
