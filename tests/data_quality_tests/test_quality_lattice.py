"""Tests for contextual reference-lattice construction."""

import pandas as pd
import pytest

from tclean import TimeGrid
from tclean.data_quality.methods._lattice import (
    normalize_reference_orders,
    reference_timestamps,
)


def _grid() -> TimeGrid:
    """Return an hourly grid for reference-order validation."""
    return TimeGrid(
        start="2020-01-01T00:00:00Z", end="2030-01-01T00:00:00Z", frequency="1h"
    )


def test_normalize_reference_orders_accepts_fixed_period():
    """Normalize a fixed-duration reference order."""
    result = normalize_reference_orders([{"period": "7D", "radius": 4}], grid=_grid())

    assert len(result) == 1
    assert result[0].period.kind == "fixed"
    assert result[0].period.value == pd.Timedelta("7D")
    assert result[0].radius == 4


@pytest.mark.parametrize(
    ("period", "kind", "value"),
    [
        ("1y", "years", 1),
        ("2y", "years", 2),
        ("1mo", "months", 1),
        ("6mo", "months", 6),
    ],
)
def test_normalize_reference_orders_accepts_calendar_periods(period, kind, value):
    """Normalize calendar-aware month and year periods."""
    result = normalize_reference_orders([{"period": period, "radius": 2}], grid=_grid())

    assert result[0].period.kind == kind
    assert result[0].period.value == value


def test_normalize_reference_orders_rejects_off_grid_fixed_period():
    """Require fixed reference periods to align with the grid."""
    with pytest.raises(ValueError, match="integer multiple"):
        normalize_reference_orders([{"period": "90min", "radius": 2}], grid=_grid())


@pytest.mark.parametrize("radius", [0, -1])
def test_normalize_reference_orders_requires_positive_radius(radius):
    """Require each reference radius to be at least one."""
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        normalize_reference_orders([{"period": "7D", "radius": radius}], grid=_grid())


@pytest.mark.parametrize("radius", [1.0, "1", True])
def test_normalize_reference_orders_requires_integer_radius(radius):
    """Require reference radius to be an integer."""
    with pytest.raises(ValueError, match="must be an integer"):
        normalize_reference_orders([{"period": "7D", "radius": radius}], grid=_grid())


def test_normalize_reference_orders_rejects_empty_sequence():
    """Require at least one contextual reference order."""
    with pytest.raises(ValueError, match="non-empty ordered sequence"):
        normalize_reference_orders([], grid=_grid())


def test_normalize_reference_orders_rejects_unknown_order_key():
    """Reject unsupported fields within a reference order."""
    with pytest.raises(ValueError, match="unknown keys"):
        normalize_reference_orders(
            [{"period": "7D", "radius": 4, "weight": 2}], grid=_grid()
        )


def test_reference_timestamps_builds_single_order_lattice():
    """Build references on both sides of a single temporal order."""
    target = pd.Timestamp("2026-06-15T12:00:00Z")

    available_index = pd.DatetimeIndex(
        [
            target - pd.Timedelta("14D"),
            target - pd.Timedelta("7D"),
            target,
            target + pd.Timedelta("7D"),
            target + pd.Timedelta("14D"),
        ]
    )

    orders = normalize_reference_orders([{"period": "7D", "radius": 2}], grid=_grid())

    result = reference_timestamps(
        target, orders=orders, available_index=available_index
    )

    assert result.tolist() == [
        target - pd.Timedelta("14D"),
        target - pd.Timedelta("7D"),
        target + pd.Timedelta("7D"),
        target + pd.Timedelta("14D"),
    ]


def test_reference_timestamps_nests_lower_orders_around_higher_centres():
    """Expand every higher-order centre by the complete lower-order lattice."""
    target = pd.Timestamp("2026-06-15T12:00:00Z")

    orders = normalize_reference_orders(
        [{"period": "7D", "radius": 1}, {"period": "1y", "radius": 1}], grid=_grid()
    )

    expected = pd.DatetimeIndex(
        sorted(
            [
                pd.Timestamp("2025-06-08T12:00:00Z"),
                pd.Timestamp("2025-06-15T12:00:00Z"),
                pd.Timestamp("2025-06-22T12:00:00Z"),
                pd.Timestamp("2026-06-08T12:00:00Z"),
                pd.Timestamp("2026-06-22T12:00:00Z"),
                pd.Timestamp("2027-06-08T12:00:00Z"),
                pd.Timestamp("2027-06-15T12:00:00Z"),
                pd.Timestamp("2027-06-22T12:00:00Z"),
            ]
        )
    )

    available_index = expected.insert(4, target)

    result = reference_timestamps(
        target, orders=orders, available_index=available_index
    )

    assert result.equals(expected)


def test_reference_timestamps_constructs_calendar_lattice_top_down():
    """Discard invalid higher-order branches before lower-order expansion."""
    target = pd.Timestamp("2024-02-29T12:00:00Z")

    orders = normalize_reference_orders(
        [{"period": "7D", "radius": 1}, {"period": "1y", "radius": 1}], grid=_grid()
    )

    available_index = pd.DatetimeIndex(
        [
            pd.Timestamp("2023-02-22T12:00:00Z"),
            pd.Timestamp("2023-03-08T12:00:00Z"),
            pd.Timestamp("2024-02-22T12:00:00Z"),
            target,
            pd.Timestamp("2024-03-07T12:00:00Z"),
            pd.Timestamp("2025-02-22T12:00:00Z"),
            pd.Timestamp("2025-03-08T12:00:00Z"),
        ]
    )

    result = reference_timestamps(
        target, orders=orders, available_index=available_index
    )

    assert result.tolist() == [
        pd.Timestamp("2024-02-22T12:00:00Z"),
        pd.Timestamp("2024-03-07T12:00:00Z"),
    ]


def test_reference_timestamps_deduplicates_overlapping_lattice_paths():
    """Count a timestamp once when multiple order paths reach it."""
    target = pd.Timestamp("2026-06-15T12:00:00Z")

    orders = normalize_reference_orders(
        [{"period": "7D", "radius": 1}, {"period": "14D", "radius": 1}], grid=_grid()
    )

    expected = pd.DatetimeIndex(
        [
            target - pd.Timedelta("21D"),
            target - pd.Timedelta("14D"),
            target - pd.Timedelta("7D"),
            target + pd.Timedelta("7D"),
            target + pd.Timedelta("14D"),
            target + pd.Timedelta("21D"),
        ]
    )

    available_index = expected.insert(3, target)

    result = reference_timestamps(
        target, orders=orders, available_index=available_index
    )

    assert result.equals(expected)
