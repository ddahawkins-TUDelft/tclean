"""Tests for relative rate-of-change data-quality evaluation."""

import pandas as pd
from _method_helpers import method_details, method_mask

from tclean import TimeGrid
from tclean.data_quality.methods.relative_rate_of_change import build_details, evaluate


def _grid() -> TimeGrid:
    """Return a six-hour test grid."""
    return TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-01T06:00:00Z", frequency="1h"
    )


def _data(values: list[float | None]) -> pd.DataFrame:
    """Build single-context test data."""
    return pd.DataFrame({"A": values}, index=pd.DatetimeIndex(_grid().target_index))


def _test(
    *, threshold: float = 0.2, reference_magnitude_threshold: float = 0.0
) -> dict:
    """Build a validated-style relative rate-of-change test."""
    return {
        "name": "large_relative_change",
        "method": "relative_rate_of_change",
        "threshold": threshold,
        "reference_magnitude_threshold": (reference_magnitude_threshold),
    }


def test_evaluate_flags_relative_change():
    """Flag a change exceeding the configured proportion."""
    data = _data([100, 130, 135, 140, 145, 150])

    result = method_mask(evaluate, data, test=_test(), grid=_grid())

    assert result["A"].tolist() == [False, True, False, False, False, False]


def test_evaluate_treats_threshold_as_exclusive():
    """Do not flag a relative change exactly equal to the threshold."""
    data = _data([100, 120, 120, 120, 120, 120])

    result = method_mask(evaluate, data, test=_test(), grid=_grid())

    assert not result["A"].any()


def test_evaluate_uses_reference_magnitude():
    """Use magnitude so negative reference values remain meaningful."""
    data = _data([-100, -130, -135, -140, -145, -150])

    result = method_mask(evaluate, data, test=_test(), grid=_grid())

    assert result["A"].tolist() == [False, True, False, False, False, False]


def test_evaluate_excludes_zero_reference_by_default():
    """Do not evaluate relative change from a zero reference value."""
    data = _data([0, 100, 105, 110, 115, 120])

    result = method_mask(evaluate, data, test=_test(), grid=_grid())

    assert not result["A"].any()


def test_evaluate_includes_small_nonzero_reference_by_default():
    """Evaluate any non-zero reference when the default threshold is zero."""
    data = _data([0.01, 1, 1, 1, 1, 1])

    result = method_mask(evaluate, data, test=_test(), grid=_grid())

    assert result["A"].tolist() == [False, True, False, False, False, False]


def test_evaluate_excludes_reference_equal_to_magnitude_threshold():
    """Require reference magnitude to be strictly above its threshold."""
    data = _data([10, 100, 100, 100, 100, 100])

    result = method_mask(
        evaluate, data, test=_test(reference_magnitude_threshold=10), grid=_grid()
    )

    assert not result["A"].any()


def test_evaluate_includes_reference_above_magnitude_threshold():
    """Evaluate references strictly exceeding the magnitude threshold."""
    data = _data([11, 100, 100, 100, 100, 100])

    result = method_mask(
        evaluate, data, test=_test(reference_magnitude_threshold=10), grid=_grid()
    )

    assert result["A"].tolist() == [False, True, False, False, False, False]


def test_evaluate_does_not_apply_threshold_to_current_value():
    """Require only the preceding value to exceed the reference threshold."""
    data = _data([100, 1, 1, 1, 1, 1])

    result = method_mask(
        evaluate, data, test=_test(reference_magnitude_threshold=50), grid=_grid()
    )

    assert result["A"].tolist() == [False, True, False, False, False, False]


def test_evaluate_does_not_bridge_missing_values():
    """Require an observed immediately preceding value."""
    data = _data([100, None, 500, 500, 500, 500])

    result = method_mask(evaluate, data, test=_test(), grid=_grid())

    assert not result["A"].any()


def test_build_details_preserves_adjacent_transitions():
    """Preserve each relative transition in a merged failure period."""
    data = _data([100, 150, 240, 245, 250, 255])

    details = method_details(
        evaluate,
        build_details,
        data["A"],
        start=pd.Timestamp("2026-01-01T01:00:00Z"),
        end=pd.Timestamp("2026-01-01T03:00:00Z"),
        test=_test(),
        grid=_grid(),
    )

    assert details["threshold"] == 0.2
    assert details["reference_magnitude_threshold"] == 0.0
    assert details["transition_count"] == 2

    assert details["transitions"][0]["relative_change"] == 0.5
    assert details["transitions"][1]["relative_change"] == 0.6
