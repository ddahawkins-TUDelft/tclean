"""Tests for external profile ingestion."""

from pathlib import Path

import pandas as pd
import pandera.errors
import pytest

from tclean.advanced.methods.external_profile import data_external_profile


def test_data_external_profile_reads_valid_csv(tmp_path: Path):
    """Data a valid external profile as a sorted UTC series."""
    path = tmp_path / "profile.csv"
    path.write_text(
        "timestamp,value\n2026-01-01T01:00:00Z,110\n2026-01-01T00:00:00Z,100\n"
    )

    result = data_external_profile(path)

    expected = pd.Series(
        [100.0, 110.0],
        index=pd.DatetimeIndex(
            ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"], name="timestamp"
        ),
        dtype=float,
        name="value",
    )

    pd.testing.assert_series_equal(
        result, expected, check_dtype=False, check_index_type=False
    )


def test_data_external_profile_converts_offsets_to_utc(tmp_path: Path):
    """Normalize timezone-offset timestamps to UTC."""
    path = tmp_path / "profile.csv"
    path.write_text("timestamp,value\n2026-01-01T01:00:00+01:00,100\n")

    result = data_external_profile(path)

    assert result.index[0] == pd.Timestamp("2026-01-01T00:00:00Z")


def test_data_external_profile_rejects_missing_column(tmp_path: Path):
    """Reject external profiles missing a required column."""
    path = tmp_path / "profile.csv"
    path.write_text("timestamp\n2026-01-01T00:00:00Z\n")

    with pytest.raises(pandera.errors.SchemaErrors):
        data_external_profile(path)


def test_data_external_profile_rejects_extra_column(tmp_path: Path):
    """Reject external profiles containing unexpected columns."""
    path = tmp_path / "profile.csv"
    path.write_text("timestamp,value,comment\n2026-01-01T00:00:00Z,100,test\n")

    with pytest.raises(pandera.errors.SchemaErrors):
        data_external_profile(path)


def test_data_external_profile_rejects_missing_value(tmp_path: Path):
    """Reject external profiles containing missing values."""
    path = tmp_path / "profile.csv"
    path.write_text("timestamp,value\n2026-01-01T00:00:00Z,\n")

    with pytest.raises(pandera.errors.SchemaErrors):
        data_external_profile(path)


def test_data_external_profile_rejects_non_numeric_value(tmp_path: Path):
    """Reject values that cannot be coerced to numbers."""
    path = tmp_path / "profile.csv"
    path.write_text("timestamp,value\n2026-01-01T00:00:00Z,not-a-number\n")

    with pytest.raises(pandera.errors.SchemaErrors):
        data_external_profile(path)


def test_data_external_profile_rejects_duplicate_timestamps(tmp_path: Path):
    """Reject duplicate timestamps after UTC normalization."""
    path = tmp_path / "profile.csv"
    path.write_text(
        "timestamp,value\n2026-01-01T00:00:00Z,100\n2026-01-01T01:00:00+01:00,110\n"
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        data_external_profile(path)


def test_data_external_profile_rejects_subhourly_timestamp(tmp_path: Path):
    """Reject timestamps that are not aligned to whole hours."""
    path = tmp_path / "profile.csv"
    path.write_text("timestamp,value\n2026-01-01T00:30:00Z,100\n")

    with pytest.raises(pandera.errors.SchemaErrors):
        data_external_profile(path)


def test_data_external_profile_rejects_invalid_timestamp(tmp_path: Path):
    """Reject timestamp values that cannot be parsed."""
    path = tmp_path / "profile.csv"
    path.write_text("timestamp,value\ndefinitely-not-a-date,100\n")

    with pytest.raises(pandera.errors.SchemaErrors):
        data_external_profile(path)


def test_data_external_profile_rejects_reversed_columns(tmp_path: Path):
    """Reject external profiles whose columns are in the wrong order."""
    path = tmp_path / "profile.csv"
    path.write_text("value,timestamp\n100,2026-01-01T00:00:00Z\n")

    with pytest.raises(pandera.errors.SchemaErrors):
        data_external_profile(path)


def test_data_external_profile_rejects_whitespace_in_column_name(tmp_path: Path):
    """Reject external profiles with malformed column names."""
    path = tmp_path / "profile.csv"
    path.write_text("timestamp, value\n2026-01-01T00:00:00Z,100\n")

    with pytest.raises(pandera.errors.SchemaErrors):
        data_external_profile(path)
