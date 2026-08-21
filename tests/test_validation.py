"""Tests for electricity-demand time-series validation."""

import pandas as pd
import pytest

from tclean.validation import infer_regular_timestep, validate_load


def test_infer_regular_timestep_returns_hourly_timestep():
    """Return one hour for a complete hourly datetime index."""
    index = pd.date_range("2026-01-01 00:00", periods=3, freq="h", tz="UTC")

    result = infer_regular_timestep(index)

    assert result == pd.Timedelta(hours=1)


def test_infer_regular_timestep_rejects_non_datetime_index():
    """Reject indexes that are not pandas DatetimeIndex objects."""
    index = pd.Index([1, 2, 3])

    with pytest.raises(TypeError, match="Load data must use a pandas DatetimeIndex."):
        infer_regular_timestep(index)


def test_infer_regular_timestep_rejects_unsorted_index():
    """Reject datetime indexes that are not sorted."""
    index = pd.DatetimeIndex(["2026-01-01 01:00", "2026-01-01 00:00"], tz="UTC")

    with pytest.raises(ValueError, match="Load timestamps must be sorted"):
        infer_regular_timestep(index)


def test_infer_regular_timestep_rejects_duplicate_timestamps():
    """Reject datetime indexes containing duplicate timestamps."""
    index = pd.DatetimeIndex(["2026-01-01 00:00", "2026-01-01 00:00"], tz="UTC")

    with pytest.raises(ValueError, match="must not contain duplicates"):
        infer_regular_timestep(index)


def test_infer_regular_timestep_rejects_single_timestamp():
    """Reject indexes that contain fewer than two timestamps."""
    index = pd.DatetimeIndex(["2026-01-01 00:00"], tz="UTC")

    with pytest.raises(ValueError, match="At least two timestamps"):
        infer_regular_timestep(index)


def test_infer_regular_timestep_rejects_irregular_index():
    """Reject datetime indexes with inconsistent spacing."""
    index = pd.DatetimeIndex(
        ["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 03:00"], tz="UTC"
    )

    with pytest.raises(ValueError, match="complete, regular time index"):
        infer_regular_timestep(index)


def test_validate_load_accepts_hourly_numeric_dataframe():
    """Accept a non-empty hourly DataFrame containing numeric columns."""
    index = pd.date_range("2026-01-01 00:00", periods=3, freq="h", tz="UTC")
    load = pd.DataFrame(
        {"ALB": [100.0, 101.0, 102.0], "GBR": [200.0, 201.0, 202.0]}, index=index
    )

    validate_load(load)


def test_validate_load_rejects_empty_dataframe():
    """Reject an empty electricity-demand DataFrame."""
    load = pd.DataFrame()

    with pytest.raises(ValueError, match="Load dataframe is empty."):
        validate_load(load)


def test_validate_load_rejects_non_hourly_data():
    """Reject otherwise regular demand data that are not hourly."""
    index = pd.date_range("2026-01-01 00:00", periods=3, freq="30min", tz="UTC")
    load = pd.DataFrame({"ALB": [100.0, 101.0, 102.0]}, index=index)

    with pytest.raises(ValueError, match="currently expects hourly load data"):
        validate_load(load)


def test_validate_load_rejects_non_numeric_columns():
    """Reject demand DataFrames containing non-numeric columns."""
    index = pd.date_range("2026-01-01 00:00", periods=3, freq="h", tz="UTC")
    load = pd.DataFrame({"ALB": ["100", "101", "102"]}, index=index)

    with pytest.raises(TypeError, match="All load columns must be numeric."):
        validate_load(load)
