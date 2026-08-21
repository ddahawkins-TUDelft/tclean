"""Tests the time.py functionality."""

import pandas as pd
import pytest

from tclean.time import as_utc_timestamp, build_hourly_index


def test_as_utc_timestamp_localises_naive_timestamp():
    """Test that timestamps are localised."""
    result = as_utc_timestamp("2026-01-01 12:00")

    assert result == pd.Timestamp("2026-01-01 12:00", tz="UTC")


def test_as_utc_timestamp_converts_timezone_aware_timestamp():
    """Tests for timezone conversion."""
    result = as_utc_timestamp("2026-01-01 13:00+01:00")

    assert result == pd.Timestamp("2026-01-01 12:00", tz="UTC")


def test_build_hourly_index_is_end_exclusive():
    """Tests endstamp exclusivity."""
    result = build_hourly_index(start="2026-01-01 00:00", end="2026-01-01 03:00")

    expected = pd.DatetimeIndex(
        ["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 02:00"],
        tz="UTC",
        name="time",
    )

    pd.testing.assert_index_equal(result, expected)


def test_build_hourly_index_rejects_equal_start_and_end():
    """Tests end=start rejection."""
    with pytest.raises(
        ValueError, match="The temporal end must be later than its start."
    ):
        build_hourly_index(start="2026-01-01 00:00", end="2026-01-01 00:00")


def test_build_hourly_index_rejects_end_before_start():
    """Tests end<start rejection."""
    with pytest.raises(
        ValueError, match="The temporal end must be later than its start."
    ):
        build_hourly_index(start="2026-01-02 00:00", end="2026-01-01 00:00")
