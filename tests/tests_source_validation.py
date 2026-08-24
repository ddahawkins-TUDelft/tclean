"""Tests for advanced source validation."""

import pandas as pd
import pandera.errors
import pytest

from tclean.validation import validate_hourly_timestamp_index, validate_source_periods


def test_validate_hourly_timestamp_index_accepts_valid_index():
    """Accept a canonical hourly UTC timestamp index."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )

    result = validate_hourly_timestamp_index(index)

    pd.testing.assert_index_equal(result, index, exact=False)


def test_validate_hourly_timestamp_index_converts_to_utc():
    """Normalize timezone-aware hourly timestamps to UTC."""
    index = pd.date_range(
        "2026-01-01 01:00", periods=3, freq="h", tz="Europe/Amsterdam", name="timestamp"
    )

    result = validate_hourly_timestamp_index(index)

    assert str(result.tz) == "UTC"
    assert result[0] == pd.Timestamp("2026-01-01T00:00:00Z")


def test_validate_hourly_timestamp_index_rejects_missing_hour():
    """Reject an index containing a break in hourly coverage."""
    index = pd.DatetimeIndex(
        ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T03:00:00Z"],
        name="timestamp",
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_hourly_timestamp_index(index)


def test_validate_hourly_timestamp_index_rejects_unnamed_index():
    """Reject an hourly index without the canonical timestamp name."""
    index = pd.date_range("2026-01-01 00:00", periods=3, freq="h", tz="UTC")

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_hourly_timestamp_index(index)


def test_validate_hourly_timestamp_index_rejects_duplicates():
    """Reject duplicate hourly timestamps."""
    index = pd.DatetimeIndex(
        ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T01:00:00Z"],
        name="timestamp",
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_hourly_timestamp_index(index)


def test_validate_source_periods_accepts_valid_sources():
    """Accept complete valid source-period definitions."""
    source_periods = pd.DataFrame(
        {
            "context": ["GBR", "FRA"],
            "start": ["2025-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "end": ["2026-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "weight": [1, 0.5],
        }
    )

    result = validate_source_periods(source_periods, frequency=pd.Timedelta("1h"))

    assert str(result["start"].dt.tz) == "UTC"
    assert str(result["end"].dt.tz) == "UTC"
    assert result["weight"].dtype == float


def test_validate_source_periods_converts_offsets_to_utc():
    """Normalize source-period timestamps to UTC."""
    source_periods = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T01:00:00+01:00"],
            "end": ["2025-01-02T01:00:00+01:00"],
            "weight": [1],
        }
    )

    result = validate_source_periods(source_periods, frequency=pd.Timedelta("1h"))

    assert result.loc[0, "start"] == pd.Timestamp("2025-01-01T00:00:00Z")
    assert result.loc[0, "end"] == pd.Timestamp("2025-01-02T00:00:00Z")


def test_validate_source_periods_rejects_missing_weight_column():
    """Reject source definitions that omit explicit weights."""
    source_periods = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2026-01-01T00:00:00Z"],
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_source_periods(source_periods, frequency=pd.Timedelta("1h"))


def test_validate_source_periods_rejects_missing_weight_value():
    """Reject source definitions when any source omits its weight."""
    source_periods = pd.DataFrame(
        {
            "context": ["GBR", "FRA"],
            "start": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "end": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "weight": [1.0, None],
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_source_periods(source_periods, frequency=pd.Timedelta("1h"))


def test_validate_source_periods_rejects_zero_weight():
    """Reject source definitions containing a zero weight."""
    source_periods = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2026-01-01T00:00:00Z"],
            "weight": [0],
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_source_periods(source_periods, frequency=pd.Timedelta("1h"))


def test_validate_source_periods_rejects_negative_weight():
    """Reject source definitions containing a negative weight."""
    source_periods = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2026-01-01T00:00:00Z"],
            "weight": [-1],
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_source_periods(source_periods, frequency=pd.Timedelta("1h"))


def test_validate_source_periods_rejects_non_numeric_weight():
    """Reject source definitions containing non-numeric weights."""
    source_periods = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2026-01-01T00:00:00Z"],
            "weight": ["high"],
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_source_periods(source_periods, frequency=pd.Timedelta("1h"))


def test_validate_source_periods_rejects_reversed_period():
    """Reject a source period whose end is not later than its start."""
    source_periods = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2026-01-01T00:00:00Z"],
            "end": ["2025-01-01T00:00:00Z"],
            "weight": [1],
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_source_periods(source_periods, frequency=pd.Timedelta("1h"))


def test_validate_source_periods_rejects_equal_bounds():
    """Reject a source period with identical start and end timestamps."""
    source_periods = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2025-01-01T00:00:00Z"],
            "weight": [1],
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_source_periods(source_periods, frequency=pd.Timedelta("1h"))


def test_validate_source_periods_rejects_extra_column():
    """Reject source definitions containing unsupported fields."""
    source_periods = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2026-01-01T00:00:00Z"],
            "weight": [1],
            "comment": ["example"],
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_source_periods(source_periods, frequency=pd.Timedelta("1h"))


def test_validate_source_periods_rejects_wrong_column_order():
    """Reject source fields supplied in the wrong order."""
    source_periods = pd.DataFrame(
        {
            "context": ["GBR"],
            "weight": [1],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2026-01-01T00:00:00Z"],
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_source_periods(source_periods, frequency=pd.Timedelta("1h"))


def test_validate_source_periods_rejects_missing_context():
    """Reject source definitions containing a missing context."""
    source_periods = pd.DataFrame(
        {
            "context": [None],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2026-01-01T00:00:00Z"],
            "weight": [1],
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_source_periods(source_periods, frequency=pd.Timedelta("1h"))


def test_validate_source_periods_rejects_empty_sources():
    """Reject an empty source-period configuration."""
    source_periods = pd.DataFrame(columns=["context", "start", "end", "weight"])

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_source_periods(source_periods, frequency=pd.Timedelta("1h"))


def test_validate_source_periods_rejects_subhourly_start():
    """Reject source periods whose start is not aligned to a whole hour."""
    source_periods = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T00:30:00Z"],
            "end": ["2026-01-01T00:00:00Z"],
            "weight": [1.0],
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_source_periods(source_periods, frequency=pd.Timedelta("1h"))


def test_validate_source_periods_rejects_subhourly_end():
    """Reject source periods whose end is not aligned to a whole hour."""
    source_periods = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2026-01-01T00:30:00Z"],
            "weight": [1.0],
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_source_periods(source_periods, frequency=pd.Timedelta("1h"))
