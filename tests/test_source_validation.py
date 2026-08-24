"""Tests for timestamp-index and source-period validation."""

import pandas as pd
import pandera.errors
import pytest

from tclean.time_grid import TimeGrid
from tclean.validation import validate_source_periods, validate_timestamp_index


def test_validate_timestamp_index_accepts_valid_index():
    """Accept a canonical fixed-frequency UTC timestamp index."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="1h", tz="UTC", name="timestamp"
    )

    result = validate_timestamp_index(index, grid=TimeGrid(start="2026-01-01T00:00:00Z", end = "2026-12-01T01:00:00Z", frequency="1h"))

    pd.testing.assert_index_equal(result, index, exact=False)


def test_validate_timestamp_index_accepts_half_hour_frequency():
    """Accept timestamps spaced at the configured half-hour frequency."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="30min", tz="UTC", name="timestamp"
    )

    result = validate_timestamp_index(index, grid=TimeGrid(start="2026-01-01T00:00:00Z", end = "2026-12-01T01:00:00Z", frequency="30min"))

    pd.testing.assert_index_equal(result, index, exact=False)


def test_validate_timestamp_index_accepts_shifted_grid():
    """Accept a regular grid that is not aligned to the wall-clock hour."""
    index = pd.date_range(
        "2026-01-01 00:30", periods=3, freq="1h", tz="UTC", name="timestamp"
    )

    result = validate_timestamp_index(index, grid=TimeGrid(start="2026-01-01T00:30:00Z", end = "2026-12-01T01:30:00Z", frequency="1h"))

    pd.testing.assert_index_equal(result, index, exact=False)


def test_validate_timestamp_index_converts_to_utc():
    """Normalize timezone-aware timestamps to UTC."""
    index = pd.date_range(
        "2026-01-01 01:00",
        periods=3,
        freq="1h",
        tz="Europe/Amsterdam",
        name="timestamp",
    )

    result = validate_timestamp_index(index, grid=TimeGrid(start="2026-01-01T00:00:00Z", end = "2026-12-01T01:00:00Z", frequency="1h"))

    assert str(result.tz) == "UTC"
    assert result[0] == pd.Timestamp("2026-01-01T00:00:00Z")


def test_validate_timestamp_index_rejects_missing_interval():
    """Reject an index containing a break in complete coverage."""
    index = pd.DatetimeIndex(
        ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T03:00:00Z"],
        name="timestamp",
    )

    with pytest.raises(ValueError, match="exactly one configured interval apart"):
        validate_timestamp_index(index, grid=TimeGrid(start="2026-01-01T00:00:00Z", end = "2026-12-01T01:00:00Z", frequency="1h"))


def test_validate_timestamp_index_rejects_wrong_frequency():
    """Reject an index whose spacing differs from the configured frequency."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="30min", tz="UTC", name="timestamp"
    )

    with pytest.raises(ValueError, match="exactly one configured interval apart"):
        validate_timestamp_index(index, grid=TimeGrid(start="2026-01-01T00:00:00Z", end = "2026-12-01T01:00:00Z", frequency="1h"))


def test_validate_timestamp_index_rejects_unnamed_index():
    """Reject an index without the canonical timestamp name."""
    index = pd.date_range("2026-01-01 00:00", periods=3, freq="1h", tz="UTC")

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_timestamp_index(index, grid=TimeGrid(start="2026-01-01T00:00:00Z", end = "2026-12-01T01:00:00Z", frequency="1h"))


def test_validate_timestamp_index_rejects_duplicates():
    """Reject duplicate timestamps."""
    index = pd.DatetimeIndex(
        ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T01:00:00Z"],
        name="timestamp",
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_timestamp_index(index, grid=TimeGrid(start="2026-01-01T00:00:00Z", end = "2026-12-01T01:00:00Z", frequency="1h"))


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

    result = validate_source_periods(source_periods, grid=TimeGrid(start="2026-01-01T00:00:00Z", end = "2026-12-01T01:00:00Z", frequency="1h"))

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

    result = validate_source_periods(source_periods, grid=TimeGrid(start="2026-01-01T00:00:00Z", end = "2026-12-01T01:00:00Z", frequency="1h"))

    assert result.loc[0, "start"] == pd.Timestamp("2025-01-01T00:00:00Z")
    assert result.loc[0, "end"] == pd.Timestamp("2025-01-02T00:00:00Z")


def test_validate_source_periods_accepts_half_hour_frequency():
    """Accept periods spanning whole configured half-hour intervals."""
    source_periods = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T00:15:00Z"],
            "end": ["2025-01-01T01:45:00Z"],
            "weight": [1.0],
        }
    )

    result = validate_source_periods(source_periods, grid=TimeGrid(start="2025-01-01T00:15:00Z", end = "2026-12-01T01:45:00Z", frequency="30min"))

    assert len(result) == 1


def test_validate_source_periods_accepts_shifted_hourly_period():
    """Accept hourly periods without requiring wall-clock-hour alignment."""
    source_periods = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T00:30:00Z"],
            "end": ["2025-01-01T03:30:00Z"],
            "weight": [1.0],
        }
    )

    result = validate_source_periods(source_periods, grid=TimeGrid(start="2025-01-01T00:30:00Z", end = "2026-12-01T01:00:00Z", frequency="30min"))

    assert len(result) == 1


def test_validate_source_periods_rejects_incompatible_duration():
    """Reject periods not divisible by the configured frequency."""
    source_periods = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2025-01-01T00:45:00Z"],
            "weight": [1.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="Period end .* does not align",
    ):
        validate_source_periods(source_periods, grid=TimeGrid(start="2026-01-01T00:00:00Z", end = "2026-12-01T01:00:00Z", frequency="30min"))


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
        validate_source_periods(source_periods, grid=TimeGrid(start="2026-01-01T00:00:00Z", end = "2026-12-01T01:00:00Z", frequency="1h"))


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
        validate_source_periods(source_periods, grid=TimeGrid(start="2026-01-01T00:00:00Z", end = "2026-12-01T01:00:00Z", frequency="1h"))


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
        validate_source_periods(source_periods, grid=TimeGrid(start="2026-01-01T00:00:00Z", end = "2026-12-01T01:00:00Z", frequency="1h"))


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
        validate_source_periods(source_periods, grid=TimeGrid(start="2026-01-01T00:00:00Z", end = "2026-12-01T01:00:00Z", frequency="1h"))


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
        validate_source_periods(source_periods, grid=TimeGrid(start="2026-01-01T00:00:00Z", end = "2026-12-01T01:00:00Z", frequency="1h"))


def test_validate_source_periods_rejects_reversed_period():
    """Reject a source period whose end is earlier than its start."""
    source_periods = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2026-01-01T00:00:00Z"],
            "end": ["2025-01-01T00:00:00Z"],
            "weight": [1],
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_source_periods(source_periods, grid=TimeGrid(start="2026-01-01T00:00:00Z", end = "2026-12-01T01:00:00Z", frequency="1h"))


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
        validate_source_periods(source_periods, grid=TimeGrid(start="2026-01-01T00:00:00Z", end = "2026-12-01T01:00:00Z", frequency="1h"))


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
        validate_source_periods(source_periods, grid=TimeGrid(start="2026-01-01T00:00:00Z", end = "2026-12-01T01:00:00Z", frequency="1h"))


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
        validate_source_periods(source_periods, grid=TimeGrid(start="2026-01-01T00:00:00Z", end = "2026-12-01T01:00:00Z", frequency="1h"))


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
        validate_source_periods(source_periods, grid=TimeGrid(start="2026-01-01T00:00:00Z", end = "2026-12-01T01:00:00Z", frequency="1h"))


def test_validate_source_periods_rejects_empty_sources():
    """Reject an empty source-period configuration."""
    source_periods = pd.DataFrame(columns=["context", "start", "end", "weight"])

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_source_periods(source_periods, grid=TimeGrid(start="2026-01-01T00:00:00Z", end = "2026-12-01T01:00:00Z", frequency="1h"))
