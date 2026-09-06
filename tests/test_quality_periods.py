"""Tests for data-quality failure-period construction."""

import pandas as pd
import pytest

from tclean import TimeGrid
from tclean.data_quality._periods import failure_mask_to_periods


def _grid() -> TimeGrid:
    """Return a six-hour test grid."""
    return TimeGrid(
        start="2026-01-01T00:00:00Z",
        end="2026-01-01T06:00:00Z",
        frequency="1h",
    )


def _mask(values: list[bool]) -> pd.Series:
    """Build a Boolean mask on the test grid."""
    grid = _grid()

    return pd.Series(
        values,
        index=pd.DatetimeIndex(grid.target_index),
        dtype=bool,
    )


def test_failure_mask_to_periods_returns_empty_for_no_failures():
    """Return no periods when every observation passes."""
    result = failure_mask_to_periods(
        _mask([False, False, False, False, False, False]),
        grid=_grid(),
    )

    assert result == []


def test_failure_mask_to_periods_reports_single_failed_value():
    """Represent one failed value as one grid interval."""
    result = failure_mask_to_periods(
        _mask([False, False, True, False, False, False]),
        grid=_grid(),
    )

    assert result == [
        (
            pd.Timestamp("2026-01-01T02:00:00Z"),
            pd.Timestamp("2026-01-01T03:00:00Z"),
        )
    ]


def test_failure_mask_to_periods_merges_adjacent_failures():
    """Merge consecutive failed values into one half-open period."""
    result = failure_mask_to_periods(
        _mask([False, True, True, True, False, False]),
        grid=_grid(),
    )

    assert result == [
        (
            pd.Timestamp("2026-01-01T01:00:00Z"),
            pd.Timestamp("2026-01-01T04:00:00Z"),
        )
    ]


def test_failure_mask_to_periods_keeps_separate_runs():
    """Return separate periods for non-contiguous failure runs."""
    result = failure_mask_to_periods(
        _mask([True, True, False, True, False, True]),
        grid=_grid(),
    )

    assert result == [
        (
            pd.Timestamp("2026-01-01T00:00:00Z"),
            pd.Timestamp("2026-01-01T02:00:00Z"),
        ),
        (
            pd.Timestamp("2026-01-01T03:00:00Z"),
            pd.Timestamp("2026-01-01T04:00:00Z"),
        ),
        (
            pd.Timestamp("2026-01-01T05:00:00Z"),
            pd.Timestamp("2026-01-01T06:00:00Z"),
        ),
    ]


def test_failure_mask_to_periods_uses_grid_end_for_final_run():
    """Use the exclusive grid end when failures continue to the final value."""
    result = failure_mask_to_periods(
        _mask([False, False, False, True, True, True]),
        grid=_grid(),
    )

    assert result == [
        (
            pd.Timestamp("2026-01-01T03:00:00Z"),
            pd.Timestamp("2026-01-01T06:00:00Z"),
        )
    ]


def test_failure_mask_to_periods_accepts_nullable_boolean_dtype():
    """Accept pandas BooleanDtype when it contains no missing values."""
    grid = _grid()

    mask = pd.Series(
        [False, True, True, False, False, False],
        index=pd.DatetimeIndex(grid.target_index),
        dtype="boolean",
    )

    result = failure_mask_to_periods(mask, grid=grid)

    assert result == [
        (
            pd.Timestamp("2026-01-01T01:00:00Z"),
            pd.Timestamp("2026-01-01T03:00:00Z"),
        )
    ]


def test_failure_mask_to_periods_rejects_non_series():
    """Reject failure masks that are not pandas Series."""
    with pytest.raises(
        TypeError,
        match="must be a pandas Series",
    ):
        failure_mask_to_periods(
            [False, True, False],
            grid=_grid(),
        )


def test_failure_mask_to_periods_rejects_non_boolean_values():
    """Reject masks whose values are not Boolean."""
    grid = _grid()

    mask = pd.Series(
        [0, 1, 1, 0, 0, 0],
        index=pd.DatetimeIndex(grid.target_index),
        dtype=int,
    )

    with pytest.raises(
        TypeError,
        match="must contain boolean values",
    ):
        failure_mask_to_periods(mask, grid=grid)


def test_failure_mask_to_periods_rejects_missing_values():
    """Reject Boolean masks containing unknown failure states."""
    grid = _grid()

    mask = pd.Series(
        [False, True, pd.NA, False, False, False],
        index=pd.DatetimeIndex(grid.target_index),
        dtype="boolean",
    )

    with pytest.raises(
        ValueError,
        match="must not contain missing values",
    ):
        failure_mask_to_periods(mask, grid=grid)


def test_failure_mask_to_periods_rejects_wrong_index():
    """Reject masks that do not use the complete target grid."""
    mask = pd.Series(
        [False, True, True, False, False, False],
        index=pd.date_range(
            "2026-01-01T01:00:00Z",
            periods=6,
            freq="1h",
        ),
        dtype=bool,
    )

    with pytest.raises(
        ValueError,
        match="must exactly match the target time grid",
    ):
        failure_mask_to_periods(mask, grid=_grid())