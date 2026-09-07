"""Tests for range-based data-quality evaluation."""

import pandas as pd
from _method_helpers import method_mask
from pandas.api.types import is_bool_dtype

from tclean import TimeGrid
from tclean.data_quality.methods.range import evaluate


def _grid() -> TimeGrid:
    """Return an eight-hour test grid."""
    return TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-01T08:00:00Z", frequency="1h"
    )


def _data() -> pd.DataFrame:
    """Return example two-context time-series data."""
    index = pd.date_range("2026-01-01T00:00:00Z", periods=5, freq="1h")

    return pd.DataFrame(
        {"A": [-1.0, 0.0, 5.0, 10.0, 11.0], "B": [100.0, 50.0, None, -5.0, 10.0]},
        index=index,
    )


def test_evaluate_range_flags_values_below_minimum():
    """Flag observed values below the configured minimum."""
    result = method_mask(
        evaluate,
        _data(),
        test={"name": "non_negative", "method": "range", "minimum": 0},
        grid=_grid(),
    )

    expected = pd.DataFrame(
        {
            "A": [True, False, False, False, False],
            "B": [False, False, False, True, False],
        },
        index=_data().index,
        dtype=bool,
    )

    pd.testing.assert_frame_equal(result, expected)


def test_evaluate_range_flags_values_above_maximum():
    """Flag observed values above the configured maximum."""
    result = method_mask(
        evaluate,
        _data(),
        test={"name": "maximum_value", "method": "range", "maximum": 10},
        grid=_grid(),
    )

    expected = pd.DataFrame(
        {
            "A": [False, False, False, False, True],
            "B": [True, True, False, False, False],
        },
        index=_data().index,
        dtype=bool,
    )

    pd.testing.assert_frame_equal(result, expected)


def test_evaluate_range_flags_values_outside_both_bounds():
    """Flag observed values outside a bounded interval."""
    result = method_mask(
        evaluate,
        _data(),
        test={"name": "bounded_values", "method": "range", "minimum": 0, "maximum": 10},
        grid=_grid(),
    )

    expected = pd.DataFrame(
        {"A": [True, False, False, False, True], "B": [True, True, False, True, False]},
        index=_data().index,
        dtype=bool,
    )

    pd.testing.assert_frame_equal(result, expected)


def test_evaluate_range_treats_bounds_as_inclusive():
    """Do not fail values exactly equal to configured bounds."""
    data = pd.DataFrame(
        {"A": [0.0, 10.0]},
        index=pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="1h"),
    )

    result = method_mask(
        evaluate,
        data,
        test={
            "name": "inclusive_bounds",
            "method": "range",
            "minimum": 0,
            "maximum": 10,
        },
        grid=_grid(),
    )

    expected = pd.DataFrame({"A": [False, False]}, index=data.index, dtype=bool)

    pd.testing.assert_frame_equal(result, expected)


def test_evaluate_range_does_not_flag_missing_values():
    """Treat missing observations as unevaluated rather than failures."""
    data = pd.DataFrame(
        {"A": [None, 5.0, None]},
        index=pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="1h"),
    )

    result = method_mask(
        evaluate,
        data,
        test={"name": "bounded_values", "method": "range", "minimum": 0, "maximum": 10},
        grid=_grid(),
    )

    expected = pd.DataFrame({"A": [False, False, False]}, index=data.index, dtype=bool)

    pd.testing.assert_frame_equal(result, expected)


def test_evaluate_range_preserves_index_and_columns():
    """Return a Boolean mask aligned exactly to the input frame."""
    data = _data()

    result = method_mask(
        evaluate,
        data,
        test={"name": "non_negative", "method": "range", "minimum": 0},
        grid=_grid(),
    )

    assert result.index.equals(data.index)
    assert result.columns.equals(data.columns)
    assert all(is_bool_dtype(dtype) for dtype in result.dtypes)
