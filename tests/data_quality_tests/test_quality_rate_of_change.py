"""Tests for rate-of-change data-quality evaluation."""

import pandas as pd
from _method_helpers import method_details, method_mask

from tclean import TimeGrid
from tclean.data_quality.methods.rate_of_change import build_details, evaluate


def _grid() -> TimeGrid:
    """Return a six-hour test grid."""
    return TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-01T06:00:00Z", frequency="1h"
    )


def _data(values: list[float | None]) -> pd.DataFrame:
    """Build single-context test data."""
    return pd.DataFrame({"A": values}, index=pd.DatetimeIndex(_grid().target_index))


def _test(
    *,
    difference_mode: str,
    threshold: float | None = None,
    reference_magnitude_threshold: float | None = None,
) -> dict:
    """Build a validated-style rate-of-change test."""
    if threshold is None:
        threshold = 20 if difference_mode == "fixed" else 0.2

    test = {
        "name": "large_change",
        "method": "rate_of_change",
        "difference_mode": difference_mode,
        "threshold": {"value_mode": "fixed", "value": threshold},
    }

    if reference_magnitude_threshold is not None:
        test["reference_magnitude_threshold"] = {
            "value_mode": "fixed",
            "value": reference_magnitude_threshold,
        }
    elif difference_mode == "relative":
        test["reference_magnitude_threshold"] = {"value_mode": "fixed", "value": 0.0}

    return test


def test_evaluate_fixed_flags_upward_change():
    """Flag an upward transition exceeding a fixed threshold."""
    data = _data([100, 130, 135, 140, 145, 150])

    result = method_mask(
        evaluate, data, test=_test(difference_mode="fixed"), grid=_grid()
    )

    assert result["A"].tolist() == [False, True, False, False, False, False]


def test_evaluate_fixed_flags_downward_change():
    """Flag a downward transition based on its magnitude."""
    data = _data([100, 70, 65, 60, 55, 50])

    result = method_mask(
        evaluate, data, test=_test(difference_mode="fixed"), grid=_grid()
    )

    assert result["A"].tolist() == [False, True, False, False, False, False]


def test_evaluate_fixed_treats_threshold_as_exclusive():
    """Do not flag a fixed change exactly equal to the threshold."""
    data = _data([100, 120, 125, 130, 135, 140])

    result = method_mask(
        evaluate, data, test=_test(difference_mode="fixed"), grid=_grid()
    )

    assert not result["A"].any()


def test_evaluate_relative_flags_change():
    """Flag a change exceeding the configured relative threshold."""
    data = _data([100, 130, 135, 140, 145, 150])

    result = method_mask(
        evaluate, data, test=_test(difference_mode="relative"), grid=_grid()
    )

    assert result["A"].tolist() == [False, True, False, False, False, False]


def test_evaluate_relative_treats_threshold_as_exclusive():
    """Do not flag a relative change exactly equal to the threshold."""
    data = _data([100, 120, 120, 120, 120, 120])

    result = method_mask(
        evaluate, data, test=_test(difference_mode="relative"), grid=_grid()
    )

    assert not result["A"].any()


def test_evaluate_relative_uses_reference_magnitude():
    """Use magnitude so negative reference values remain meaningful."""
    data = _data([-100, -130, -135, -140, -145, -150])

    result = method_mask(
        evaluate, data, test=_test(difference_mode="relative"), grid=_grid()
    )

    assert result["A"].tolist() == [False, True, False, False, False, False]


def test_evaluate_relative_excludes_zero_reference_by_default():
    """Do not evaluate relative change from a zero reference value."""
    data = _data([0, 100, 105, 110, 115, 120])

    result = method_mask(
        evaluate, data, test=_test(difference_mode="relative"), grid=_grid()
    )

    assert not result["A"].any()


def test_evaluate_relative_includes_small_nonzero_reference_by_default():
    """Evaluate any non-zero reference when the default floor is zero."""
    data = _data([0.01, 1, 1, 1, 1, 1])

    result = method_mask(
        evaluate, data, test=_test(difference_mode="relative"), grid=_grid()
    )

    assert result["A"].tolist() == [False, True, False, False, False, False]


def test_evaluate_relative_excludes_reference_equal_to_magnitude_threshold():
    """Require reference magnitude to be strictly above its floor."""
    data = _data([10, 100, 100, 100, 100, 100])

    result = method_mask(
        evaluate,
        data,
        test=_test(difference_mode="relative", reference_magnitude_threshold=10),
        grid=_grid(),
    )

    assert not result["A"].any()


def test_evaluate_relative_includes_reference_above_magnitude_threshold():
    """Evaluate references strictly exceeding the magnitude floor."""
    data = _data([11, 100, 100, 100, 100, 100])

    result = method_mask(
        evaluate,
        data,
        test=_test(difference_mode="relative", reference_magnitude_threshold=10),
        grid=_grid(),
    )

    assert result["A"].tolist() == [False, True, False, False, False, False]


def test_evaluate_relative_does_not_apply_reference_floor_to_current_value():
    """Apply the reference floor only to the preceding observation."""
    data = _data([100, 1, 1, 1, 1, 1])

    result = method_mask(
        evaluate,
        data,
        test=_test(difference_mode="relative", reference_magnitude_threshold=50),
        grid=_grid(),
    )

    assert result["A"].tolist() == [False, True, False, False, False, False]


def test_evaluate_does_not_bridge_missing_values_in_either_mode():
    """Require the immediately preceding observation in both modes."""
    data = _data([100, None, 500, 505, 510, 515])

    fixed = method_mask(
        evaluate, data, test=_test(difference_mode="fixed"), grid=_grid()
    )
    relative = method_mask(
        evaluate, data, test=_test(difference_mode="relative"), grid=_grid()
    )

    assert not fixed["A"].any()
    assert not relative["A"].any()


def test_evaluate_applies_independently_to_contexts():
    """Evaluate rate of change independently for each context."""
    data = pd.DataFrame(
        {"A": [100, 130, 135, 140, 145, 150], "B": [100, 105, 110, 80, 85, 90]},
        index=pd.DatetimeIndex(_grid().target_index),
    )

    result = method_mask(
        evaluate, data, test=_test(difference_mode="fixed"), grid=_grid()
    )

    assert result["A"].tolist() == [False, True, False, False, False, False]
    assert result["B"].tolist() == [False, False, False, True, False, False]


def test_build_details_fixed_preserves_adjacent_transitions():
    """Preserve each fixed transition inside a merged failure period."""
    data = _data([100, 130, 170, 175, 180, 185])

    details = method_details(
        evaluate,
        build_details,
        data["A"],
        start=pd.Timestamp("2026-01-01T01:00:00Z"),
        end=pd.Timestamp("2026-01-01T03:00:00Z"),
        test=_test(difference_mode="fixed"),
        grid=_grid(),
    )

    assert details["difference_mode"] == "fixed"
    assert details["threshold"] == 20
    assert details["transition_count"] == 2
    assert "reference_magnitude_threshold" not in details

    assert [transition["timestamp"] for transition in details["transitions"]] == [
        pd.Timestamp("2026-01-01T01:00:00Z"),
        pd.Timestamp("2026-01-01T02:00:00Z"),
    ]
    assert [transition["change"] for transition in details["transitions"]] == [
        30.0,
        40.0,
    ]
    assert all(
        "relative_change" not in transition for transition in details["transitions"]
    )


def test_build_details_relative_preserves_adjacent_transitions():
    """Preserve each relative transition inside a merged failure period."""
    data = _data([100, 150, 240, 245, 250, 255])

    details = method_details(
        evaluate,
        build_details,
        data["A"],
        start=pd.Timestamp("2026-01-01T01:00:00Z"),
        end=pd.Timestamp("2026-01-01T03:00:00Z"),
        test=_test(difference_mode="relative"),
        grid=_grid(),
    )

    assert details["difference_mode"] == "relative"
    assert details["threshold"] == 0.2
    assert details["reference_magnitude_threshold"] == 0.0
    assert details["transition_count"] == 2
    assert [transition["relative_change"] for transition in details["transitions"]] == [
        0.5,
        0.6,
    ]


def test_evaluate_fixed_supports_derived_threshold():
    """Resolve a fixed-mode threshold from typical absolute increments."""
    data = _data([100, 105, 110, 150, 155, 160])
    test = _test(difference_mode="fixed")
    test["threshold"] = {"value_mode": "median_absolute_increment", "multiplier": 2}

    result = method_mask(evaluate, data, test=test, grid=_grid())

    assert result["A"].tolist() == [False, False, False, True, False, False]


def test_evaluate_relative_supports_derived_reference_magnitude_threshold():
    """Resolve the relative-mode reference floor from focal data."""
    data = _data([1, 2, 4, 100, 150, 200])
    test = _test(difference_mode="relative", threshold=0.4)
    test["reference_magnitude_threshold"] = {"value_mode": "median"}

    result = method_mask(evaluate, data, test=test, grid=_grid())

    assert result["A"].tolist() == [False, False, False, False, True, False]
