"""Tests for fixed rate-of-change data-quality evaluation."""

import pandas as pd
from _method_helpers import method_details, method_mask

from tclean import TimeGrid
from tclean.data_quality.methods.fixed_rate_of_change import build_details, evaluate


def _grid() -> TimeGrid:
    """Return a six-hour test grid."""
    return TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-01T06:00:00Z", frequency="1h"
    )


def _data(values: list[float | None]) -> pd.DataFrame:
    """Build single-context test data."""
    return pd.DataFrame({"A": values}, index=pd.DatetimeIndex(_grid().target_index))


def _test(*, threshold: float = 20) -> dict:
    """Build a validated-style fixed rate-of-change test."""
    return {
        "name": "large_change",
        "method": "fixed_rate_of_change",
        "threshold": threshold,
    }


def test_evaluate_flags_upward_change():
    """Flag an upward transition exceeding the threshold."""
    data = _data([100, 130, 135, 140, 145, 150])

    result = method_mask(evaluate, data, test=_test(), grid=_grid())

    assert result["A"].tolist() == [False, True, False, False, False, False]


def test_evaluate_flags_downward_change():
    """Flag a downward transition based on its magnitude."""
    data = _data([100, 70, 65, 60, 55, 50])

    result = method_mask(evaluate, data, test=_test(), grid=_grid())

    assert result["A"].tolist() == [False, True, False, False, False, False]


def test_evaluate_treats_threshold_as_exclusive():
    """Do not flag a change exactly equal to the threshold."""
    data = _data([100, 120, 125, 130, 135, 140])

    result = method_mask(evaluate, data, test=_test(), grid=_grid())

    assert not result["A"].any()


def test_evaluate_does_not_bridge_missing_values():
    """Require the immediately preceding observation to exist."""
    data = _data([100, None, 500, 505, 510, 515])

    result = method_mask(evaluate, data, test=_test(), grid=_grid())

    assert not result["A"].any()


def test_evaluate_applies_independently_to_contexts():
    """Evaluate rate of change independently for each context."""
    data = pd.DataFrame(
        {"A": [100, 130, 135, 140, 145, 150], "B": [100, 105, 110, 80, 85, 90]},
        index=pd.DatetimeIndex(_grid().target_index),
    )

    result = method_mask(evaluate, data, test=_test(), grid=_grid())

    assert result["A"].tolist() == [False, True, False, False, False, False]
    assert result["B"].tolist() == [False, False, False, True, False, False]


def test_build_details_preserves_adjacent_transitions():
    """Preserve each transition inside a merged failure period."""
    data = _data([100, 130, 170, 175, 180, 185])

    details = method_details(
        evaluate,
        build_details,
        data["A"],
        start=pd.Timestamp("2026-01-01T01:00:00Z"),
        end=pd.Timestamp("2026-01-01T03:00:00Z"),
        test=_test(),
        grid=_grid(),
    )

    assert details["threshold"] == 20
    assert details["transition_count"] == 2

    assert [transition["timestamp"] for transition in details["transitions"]] == [
        pd.Timestamp("2026-01-01T01:00:00Z"),
        pd.Timestamp("2026-01-01T02:00:00Z"),
    ]

    assert [transition["change"] for transition in details["transitions"]] == [
        30.0,
        40.0,
    ]
