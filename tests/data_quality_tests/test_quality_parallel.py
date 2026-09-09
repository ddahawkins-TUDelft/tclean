"""Tests for parallel contextual data-quality evaluation."""

import numpy as np
import pandas as pd

from tclean import TimeGrid
from tclean.data_quality import evaluate
from tclean.data_quality._parallel import ordered_thread_map
from tclean.data_quality.methods._lattice import (
    build_reference_lattice,
    normalize_reference_orders,
    reference_timestamps,
)


def test_ordered_thread_map_preserves_input_order():
    """Return threaded results in supplied item order."""
    result = ordered_thread_map(lambda value: value * 10, [3, 1, 2], threads=3)

    assert result == [30, 10, 20]


def test_build_reference_lattice_matches_reference_timestamps():
    """Precomputed lattice positions reproduce direct timestamp construction."""
    grid = TimeGrid(
        start="2024-01-01T00:00:00Z", end="2027-01-01T00:00:00Z", frequency="1h"
    )

    available_index = grid.target_index
    targets = pd.DatetimeIndex(
        ["2025-01-15T12:00:00Z", "2026-01-15T12:00:00Z", "2026-06-15T12:00:00Z"]
    )

    orders = normalize_reference_orders(
        [{"period": "7D", "radius": 2}, {"period": "1Y", "radius": 1}], grid=grid
    )

    lattice = build_reference_lattice(
        targets, orders=orders, available_index=available_index
    )

    for target_number, target in enumerate(targets):
        expected_timestamps = reference_timestamps(
            target, orders=orders, available_index=available_index
        )
        expected_positions = available_index.get_indexer(expected_timestamps)

        np.testing.assert_array_equal(
            lattice.positions_for(target_number), expected_positions
        )


def test_contextual_level_threaded_matches_serial():
    """Threaded contextual-level evaluation matches serial output exactly."""
    grid = TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-30T00:00:00Z", frequency="1h"
    )

    source = pd.DataFrame({"A": 100.0, "B": 200.0, "C": 300.0}, index=grid.target_index)

    target = pd.Timestamp("2026-01-15T12:00:00Z")
    source.loc[target, "A"] = 150.0
    source.loc[target, "C"] = 360.0

    tests = [
        {
            "name": "unusual_level",
            "method": "contextual_level",
            "reference_orders": [{"period": "7D", "radius": 2}],
            "robust_deviation_threshold": 6,
        }
    ]

    serial = evaluate({"primary": source}, tests=tests, grid=grid, threads=1)
    threaded = evaluate({"primary": source}, tests=tests, grid=grid, threads=4)

    assert not serial.failures.empty

    pd.testing.assert_frame_equal(serial.failures, threaded.failures)
    pd.testing.assert_frame_equal(serial.issues, threaded.issues)


def test_contextual_profile_threaded_matches_serial():
    """Threaded contextual-profile evaluation matches serial failures and issues."""
    grid = TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-03T16:00:00Z", frequency="1h"
    )

    repeats = len(grid.target_index) // 4
    pattern = np.tile(np.array([0.0, 1.0, 3.0, 1.0]), repeats)

    source = pd.DataFrame(
        {"A": pattern.copy(), "B": pattern.copy() + 10.0, "C": pattern.copy() + 20.0},
        index=grid.target_index,
    )

    target = pd.Timestamp("2026-01-01T20:00:00Z")

    source.loc[target : target + pd.Timedelta("3h"), "A"] = [3, 0, 0, 3]
    source.loc[target : target + pd.Timedelta("3h"), "C"] = [23, 20, 20, 23]
    source.loc[target + pd.Timedelta("1h"), "B"] = np.nan

    tests = [
        {
            "name": "unusual_shape",
            "method": "contextual_profile",
            "profile_duration": "4h",
            "reference_orders": [{"period": "8h", "radius": 2}],
            "robust_deviation_threshold": 6,
        }
    ]

    serial = evaluate({"primary": source}, tests=tests, grid=grid, threads=1)
    threaded = evaluate({"primary": source}, tests=tests, grid=grid, threads=4)

    assert not serial.failures.empty
    assert not serial.issues.empty

    pd.testing.assert_frame_equal(serial.failures, threaded.failures)
    pd.testing.assert_frame_equal(serial.issues, threaded.issues)
