"""Tests for canonical electricity-demand validation."""

import pandas as pd
import pandera.errors
import pytest

from tclean.validation import infer_regular_timestep, validate_load


def test_validate_load_accepts_hourly_numeric_dataframe():
    """Accept canonical hourly electricity-demand data."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    load = pd.DataFrame(
        {"ALB": [100.0, 101.0, 102.0], "GBR": [200.0, 201.0, 202.0]}, index=index
    )

    result = validate_load(load)

    pd.testing.assert_index_equal(result.index, load.index, exact=False)

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), load.reset_index(drop=True)
    )


def test_validate_load_coerces_numeric_demand():
    """Coerce numeric demand values to floating-point data."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    load = pd.DataFrame({"ALB": ["100", "101", "102"]}, index=index)

    result = validate_load(load)

    assert result["ALB"].dtype == float
    assert result["ALB"].tolist() == [100.0, 101.0, 102.0]


def test_validate_load_allows_missing_demand():
    """Allow missing demand values for subsequent gap cleaning."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    load = pd.DataFrame({"ALB": [100.0, float("nan"), 102.0]}, index=index)

    result = validate_load(load)

    assert pd.isna(result.loc[index[1], "ALB"])


def test_validate_load_rejects_non_numeric_demand():
    """Reject demand values that cannot be coerced to numbers."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    load = pd.DataFrame({"ALB": [100.0, "invalid", 102.0]}, index=index)

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_load(load)


def test_validate_load_rejects_no_demand_columns():
    """Reject demand data containing no regional demand columns."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    load = pd.DataFrame(index=index)

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_load(load)


def test_validate_load_rejects_single_timestamp():
    """Reject demand data with fewer than two timestamps."""
    index = pd.DatetimeIndex(["2026-01-01T00:00:00Z"], name="timestamp")
    load = pd.DataFrame({"ALB": [100.0]}, index=index)

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_load(load)


def test_validate_load_rejects_unsorted_timestamps():
    """Reject demand data whose timestamps are not sorted."""
    index = pd.DatetimeIndex(
        ["2026-01-01T01:00:00Z", "2026-01-01T00:00:00Z"], name="timestamp"
    )
    load = pd.DataFrame({"ALB": [100.0, 101.0]}, index=index)

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_load(load)


def test_validate_load_rejects_duplicate_timestamps():
    """Reject demand data containing duplicate timestamps."""
    index = pd.DatetimeIndex(
        ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], name="timestamp"
    )
    load = pd.DataFrame({"ALB": [100.0, 101.0]}, index=index)

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_load(load)


def test_validate_load_rejects_non_hourly_timestamps():
    """Reject otherwise regular demand data that are not hourly."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="30min", tz="UTC", name="timestamp"
    )
    load = pd.DataFrame({"ALB": [100.0, 101.0, 102.0]}, index=index)

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_load(load)


def test_validate_load_rejects_missing_hour():
    """Reject an hourly demand index containing a missing timestamp."""
    index = pd.DatetimeIndex(
        ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T03:00:00Z"],
        name="timestamp",
    )
    load = pd.DataFrame({"ALB": [100.0, 101.0, 103.0]}, index=index)

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_load(load)


def test_validate_load_converts_timestamp_index_to_utc():
    """Normalize timezone-aware timestamps to UTC."""
    index = pd.DatetimeIndex(
        [
            "2026-01-01T01:00:00+01:00",
            "2026-01-01T02:00:00+01:00",
            "2026-01-01T03:00:00+01:00",
        ],
        name="timestamp",
    )
    load = pd.DataFrame({"ALB": [100.0, 101.0, 102.0]}, index=index)

    result = validate_load(load)

    expected = pd.DatetimeIndex(
        ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T02:00:00Z"],
        name="timestamp",
    )

    pd.testing.assert_index_equal(result.index, expected, exact=False)


def test_infer_regular_timestep_returns_hourly_timestep():
    """Return one hour for a validated hourly index."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )

    result = infer_regular_timestep(index)

    assert result == pd.Timedelta(hours=1)


def test_validate_load_rejects_unnamed_timestamp_index():
    """Reject demand data whose timestamp index is unnamed."""
    index = pd.date_range("2026-01-01 00:00", periods=3, freq="h", tz="UTC")
    load = pd.DataFrame({"ALB": [100.0, 101.0, 102.0]}, index=index)

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_load(load)
