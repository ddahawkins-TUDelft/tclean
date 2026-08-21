"""Tests for canonical time-index construction."""

import pandas as pd
import pandera.errors
import pytest

from tclean.time import build_hourly_index
from tclean.validation import validate_temporal_range


def test_validate_temporal_range_accepts_valid_range():
    """Validate and normalize a valid temporal range."""
    start, end = validate_temporal_range(
        start="2026-01-01T00:00:00Z", end="2026-01-01T03:00:00Z"
    )

    assert start == pd.Timestamp("2026-01-01T00:00:00Z")
    assert end == pd.Timestamp("2026-01-01T03:00:00Z")


def test_validate_temporal_range_converts_offsets_to_utc():
    """Normalize timezone-offset temporal parameters to UTC."""
    start, end = validate_temporal_range(
        start="2026-01-01T01:00:00+01:00", end="2026-01-01T04:00:00+01:00"
    )

    assert start == pd.Timestamp("2026-01-01T00:00:00Z")
    assert end == pd.Timestamp("2026-01-01T03:00:00Z")


def test_validate_temporal_range_rejects_invalid_timestamp():
    """Reject temporal parameters that cannot be parsed."""
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_temporal_range(start="not-a-date", end="2026-01-01T03:00:00Z")


def test_validate_temporal_range_rejects_equal_bounds():
    """Reject a temporal range whose bounds are equal."""
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_temporal_range(
            start="2026-01-01T00:00:00Z", end="2026-01-01T00:00:00Z"
        )


def test_validate_temporal_range_rejects_reversed_bounds():
    """Reject a temporal range whose end precedes its start."""
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_temporal_range(
            start="2026-01-01T03:00:00Z", end="2026-01-01T00:00:00Z"
        )


def test_build_hourly_index_is_end_exclusive():
    """Build an hourly timestamp index that excludes the end bound."""
    result = build_hourly_index(
        start="2026-01-01T00:00:00Z", end="2026-01-01T03:00:00Z"
    )

    expected = pd.DatetimeIndex(
        ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T02:00:00Z"],
        name="timestamp",
    )

    pd.testing.assert_index_equal(result, expected, exact=False)
