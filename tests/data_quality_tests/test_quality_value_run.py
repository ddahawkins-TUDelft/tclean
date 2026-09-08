"""Tests for repeated-value data-quality evaluation."""

import pandas as pd
from _method_helpers import method_mask

from tclean import TimeGrid
from tclean.data_quality.methods.value_run import evaluate


def _grid() -> TimeGrid:
    """Return an eight-hour test grid."""
    return TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-01T08:00:00Z", frequency="1h"
    )


def _data(values: list[float | None]) -> pd.DataFrame:
    """Build single-context test data."""
    return pd.DataFrame({"A": values}, index=pd.DatetimeIndex(_grid().target_index))


def test_evaluate_flags_run_meeting_minimum_duration():
    """Flag every value in a run that meets the duration threshold."""
    data = _data([5, 0, 0, 0, 5, 5, 5, 5])

    result = method_mask(
        evaluate,
        data,
        test={
            "name": "zero_run",
            "method": "value_run",
            "value": {"value_mode": "fixed", "value": 0},
            "minimum_duration": pd.Timedelta("3h"),
            "tolerance": {"value_mode": "fixed", "value": 0.0},
        },
        grid=_grid(),
    )

    expected = pd.DataFrame(
        {"A": [False, True, True, True, False, False, False, False]},
        index=data.index,
        dtype=bool,
    )

    pd.testing.assert_frame_equal(result, expected)


def test_evaluate_does_not_flag_short_run():
    """Do not flag a run shorter than the configured duration."""
    data = _data([5, 0, 0, 5, 5, 5, 5, 5])

    result = method_mask(
        evaluate,
        data,
        test={
            "name": "zero_run",
            "method": "value_run",
            "value": {"value_mode": "fixed", "value": 0},
            "minimum_duration": pd.Timedelta("3h"),
            "tolerance": {"value_mode": "fixed", "value": 0.0},
        },
        grid=_grid(),
    )

    assert not result["A"].any()


def test_evaluate_flags_run_longer_than_minimum_duration():
    """Flag the complete run when it exceeds the minimum duration."""
    data = _data([0, 0, 0, 0, 5, 5, 5, 5])

    result = method_mask(
        evaluate,
        data,
        test={
            "name": "zero_run",
            "method": "value_run",
            "value": {"value_mode": "fixed", "value": 0},
            "minimum_duration": pd.Timedelta("3h"),
            "tolerance": {"value_mode": "fixed", "value": 0.0},
        },
        grid=_grid(),
    )

    assert result["A"].tolist() == [True, True, True, True, False, False, False, False]


def test_evaluate_handles_multiple_qualifying_runs():
    """Flag multiple qualifying runs independently."""
    data = _data([0, 0, 0, 5, 0, 0, 0, 5])

    result = method_mask(
        evaluate,
        data,
        test={
            "name": "zero_run",
            "method": "value_run",
            "value": {"value_mode": "fixed", "value": 0},
            "minimum_duration": pd.Timedelta("3h"),
            "tolerance": {"value_mode": "fixed", "value": 0.0},
        },
        grid=_grid(),
    )

    assert result["A"].tolist() == [True, True, True, False, True, True, True, False]


def test_evaluate_uses_tolerance():
    """Treat values within the configured tolerance as part of one run."""
    data = _data([5, 0.0, 0.05, -0.08, 5, 5, 5, 5])

    result = method_mask(
        evaluate,
        data,
        test={
            "name": "near_zero_run",
            "method": "value_run",
            "value": {"value_mode": "fixed", "value": 0},
            "minimum_duration": pd.Timedelta("3h"),
            "tolerance": {"value_mode": "fixed", "value": 0.1},
        },
        grid=_grid(),
    )

    assert result["A"].tolist() == [False, True, True, True, False, False, False, False]


def test_evaluate_breaks_run_when_value_exceeds_tolerance():
    """Break a run when an observation lies outside tolerance."""
    data = _data([0, 0, 0.2, 0, 0, 5, 5, 5])

    result = method_mask(
        evaluate,
        data,
        test={
            "name": "near_zero_run",
            "method": "value_run",
            "value": {"value_mode": "fixed", "value": 0},
            "minimum_duration": pd.Timedelta("3h"),
            "tolerance": {"value_mode": "fixed", "value": 0.1},
        },
        grid=_grid(),
    )

    assert not result["A"].any()


def test_evaluate_breaks_run_at_missing_value():
    """Treat missing observations as boundaries between value runs."""
    data = _data([0, 0, None, 0, 0, 0, 5, 5])

    result = method_mask(
        evaluate,
        data,
        test={
            "name": "zero_run",
            "method": "value_run",
            "value": {"value_mode": "fixed", "value": 0},
            "minimum_duration": pd.Timedelta("3h"),
            "tolerance": {"value_mode": "fixed", "value": 0.0},
        },
        grid=_grid(),
    )

    assert result["A"].tolist() == [False, False, False, True, True, True, False, False]


def test_evaluate_applies_independently_to_contexts():
    """Evaluate value runs independently for each context."""
    data = pd.DataFrame(
        {"A": [0, 0, 0, 5, 5, 5, 5, 5], "B": [5, 5, 0, 0, 0, 5, 5, 5]},
        index=pd.DatetimeIndex(_grid().target_index),
    )

    result = method_mask(
        evaluate,
        data,
        test={
            "name": "zero_run",
            "method": "value_run",
            "value": {"value_mode": "fixed", "value": 0},
            "minimum_duration": pd.Timedelta("3h"),
            "tolerance": {"value_mode": "fixed", "value": 0.0},
        },
        grid=_grid(),
    )

    assert result["A"].tolist() == [True, True, True, False, False, False, False, False]

    assert result["B"].tolist() == [False, False, True, True, True, False, False, False]


def test_evaluate_value_run_supports_derived_target_value():
    """Resolve a repeated target value from the focal context."""
    data = _data([0, 0, 0, 5, 5, 5, 5, 5])

    result = method_mask(
        evaluate,
        data,
        test={
            "name": "median_run",
            "method": "value_run",
            "value": {"value_mode": "median"},
            "minimum_duration": pd.Timedelta("3h"),
            "tolerance": {"value_mode": "fixed", "value": 0.0},
        },
        grid=_grid(),
    )

    assert result["A"].tolist() == [False, False, False, True, True, True, True, True]
