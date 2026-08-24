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

    pd.testing.assert_index_equal(grid.target_index, expected, exact=False)


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

    grid._validate_index(index, require_complete=False)


def test_time_grid_complete_index_rejects_missing_period():
    """Reject skipped periods when complete coverage is required."""
    grid = TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-02T00:00:00Z", frequency="1h"
    )

    index = pd.DatetimeIndex(
        ["2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z"], name="timestamp"
    )

    with pytest.raises(ValueError, match="exactly one configured interval apart"):
        grid._validate_index(index, require_complete=True)


def test_time_grid_rejects_wrong_phase_even_when_sparse():
    """Reject sparse timestamps that do not lie on the configured grid."""
    grid = TimeGrid(
        start="2026-01-01T00:30:00Z", end="2026-01-01T04:30:00Z", frequency="1h"
    )

    index = pd.DatetimeIndex(
        ["2026-01-01T00:30:00Z", "2026-01-01T02:00:00Z"], name="timestamp"
    )

    with pytest.raises(ValueError, match="align with the configured time grid"):
        grid.validate_sparse_index(index)


def test_time_grid_validates_period_outside_target_window():
    """Allow aligned supporting periods outside the output window."""
    grid = TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-02T00:00:00Z", frequency="1h"
    )

    grid.validate_period(
        start=pd.Timestamp("2025-12-20T00:00:00Z"),
        end=pd.Timestamp("2025-12-21T00:00:00Z"),
    )


def test_time_grid_rejects_misaligned_period_boundary():
    """Reject period boundaries that do not lie on the grid."""
    grid = TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-02T00:00:00Z", frequency="1h"
    )

    with pytest.raises(ValueError, match="Period start"):
        grid.validate_period(
            start=pd.Timestamp("2025-12-20T00:30:00Z"),
            end=pd.Timestamp("2025-12-21T00:00:00Z"),
        )


def test_time_grid_validates_target_coverage_with_extra_context():
    """Accept an index extending beyond the complete target window."""
    grid = TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-01T03:00:00Z", frequency="1h"
    )

    index = pd.date_range(
        "2025-12-31T23:00:00Z", "2026-01-01T03:00:00Z", freq="1h", name="timestamp"
    )

    grid.validate_target_coverage(index)


def test_time_grid_rejects_incomplete_target_coverage():
    """Reject data missing a timestamp from the requested output window."""
    grid = TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-01T03:00:00Z", frequency="1h"
    )

    index = pd.DatetimeIndex(
        ["2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z"], name="timestamp"
    )

    with pytest.raises(ValueError, match="complete requested output window"):
        grid.validate_target_coverage(index)
