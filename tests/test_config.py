"""Tests for fixed-frequency time-grid behaviour."""

import pandas as pd
import pytest

from tclean.time_grid import TimeGrid


def test_time_grid_builds_expected_index():
    """Build the complete end-exclusive target index."""
    grid = TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-01T03:00:00Z", frequency="1h"
    )

    expected = pd.date_range(
        "2026-01-01T00:00:00Z", periods=3, freq="1h", name="timestamp"
    )

    pd.testing.assert_index_equal(grid.index, expected, exact=False)


def test_time_grid_rejects_incompatible_end():
    """Reject an output window containing a partial time step."""
    with pytest.raises(ValueError, match="integer number of configured time steps"):
        TimeGrid(
            start="2026-01-01T00:00:00Z", end="2026-01-01T03:30:00Z", frequency="1h"
        )


def test_time_grid_uses_start_as_alignment_anchor():
    """Use the configured start timestamp to define grid phase."""
    grid = TimeGrid(
        start="2026-01-01T00:30:00Z", end="2026-01-01T03:30:00Z", frequency="1h"
    )

    assert grid.is_aligned(pd.Timestamp("2025-12-31T23:30:00Z"))
    assert grid.is_aligned(pd.Timestamp("2026-01-01T01:30:00Z"))
    assert not grid.is_aligned(pd.Timestamp("2026-01-01T01:00:00Z"))


def test_time_grid_sparse_index_allows_missing_periods():
    """Allow gaps when validating a sparse grid."""
    grid = TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-02T00:00:00Z", frequency="1h"
    )

    index = pd.DatetimeIndex(
        ["2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z", "2026-01-01T05:00:00Z"],
        name="timestamp",
    )

    grid.validate_aligned_index(index, require_complete=False)


def test_time_grid_complete_index_rejects_missing_period():
    """Reject skipped periods when complete coverage is required."""
    grid = TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-02T00:00:00Z", frequency="1h"
    )

    index = pd.DatetimeIndex(
        ["2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z"], name="timestamp"
    )

    with pytest.raises(ValueError, match="exactly one configured interval apart"):
        grid.validate_aligned_index(index, require_complete=True)
