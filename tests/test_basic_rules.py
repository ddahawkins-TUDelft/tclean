"""Tests for basic gap-filling rules."""

import pandas as pd

from tclean.basic.methods.average_periods import apply_average_periods
from tclean.basic.methods.copy_periods import apply_copy_periods
from tclean.basic.methods.linear_interpolation import apply_linear_interpolation


def test_linear_interpolation_fills_bounded_gap():
    """Fill an eligible bounded gap using linear interpolation."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    data = pd.DataFrame({"ALB": [10.0, float("nan"), 14.0]}, index=index)
    durations = pd.DataFrame(
        {"ALB": [pd.Timedelta(0), pd.Timedelta(hours=1), pd.Timedelta(0)]}, index=index
    )

    filled, newly_filled = apply_linear_interpolation(
        data, max_gap="1h", original_gap_duration=durations
    )

    assert filled.loc[index[1], "ALB"] == 12.0
    assert newly_filled.loc[index[1], "ALB"]


def test_linear_interpolation_respects_max_gap():
    """Leave gaps unresolved when their original duration exceeds the limit."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=4, freq="h", tz="UTC", name="timestamp"
    )
    data = pd.DataFrame({"ALB": [10.0, float("nan"), float("nan"), 16.0]}, index=index)
    durations = pd.DataFrame(
        {
            "ALB": [
                pd.Timedelta(0),
                pd.Timedelta(hours=2),
                pd.Timedelta(hours=2),
                pd.Timedelta(0),
            ]
        },
        index=index,
    )

    filled, newly_filled = apply_linear_interpolation(
        data, max_gap="1h", original_gap_duration=durations
    )

    assert filled["ALB"].isna().sum() == 2
    assert not newly_filled.to_numpy().any()


def test_copy_periods_fills_from_previous_period():
    """Fill a gap using corresponding values from an earlier period."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=5, freq="h", tz="UTC", name="timestamp"
    )
    data = pd.DataFrame({"ALB": [10.0, 11.0, float("nan"), 13.0, 14.0]}, index=index)
    durations = pd.DataFrame(pd.Timedelta(0), index=index, columns=["ALB"])
    durations.loc[index[2], "ALB"] = pd.Timedelta(hours=1)

    filled, newly_filled = apply_copy_periods(
        data, max_gap="1h", source_offset="-2h", original_gap_duration=durations
    )

    assert filled.loc[index[2], "ALB"] == 10.0
    assert newly_filled.loc[index[2], "ALB"]


def test_copy_periods_requires_complete_source_by_default():
    """Leave a whole gap unresolved when its source period is incomplete."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=6, freq="h", tz="UTC", name="timestamp"
    )
    data = pd.DataFrame(
        {"ALB": [10.0, float("nan"), float("nan"), float("nan"), 14.0, 15.0]},
        index=index,
    )
    durations = pd.DataFrame(pd.Timedelta(0), index=index, columns=["ALB"])
    durations.loc[index[2:4], "ALB"] = pd.Timedelta(hours=2)

    filled, newly_filled = apply_copy_periods(
        data, max_gap="2h", source_offset="-2h", original_gap_duration=durations
    )

    assert filled.loc[index[2:4], "ALB"].isna().all()
    assert not newly_filled.loc[index[2:4], "ALB"].any()


def test_average_periods_uses_complete_sources():
    """Fill a gap with the mean of corresponding source-period values."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=5, freq="h", tz="UTC", name="timestamp"
    )
    data = pd.DataFrame({"ALB": [10.0, 12.0, float("nan"), 16.0, 18.0]}, index=index)
    durations = pd.DataFrame(pd.Timedelta(0), index=index, columns=["ALB"])
    durations.loc[index[2], "ALB"] = pd.Timedelta(hours=1)

    filled, newly_filled = apply_average_periods(
        data,
        max_gap="1h",
        source_offsets=["-2h", "2h"],
        original_gap_duration=durations,
    )

    assert filled.loc[index[2], "ALB"] == 14.0
    assert newly_filled.loc[index[2], "ALB"]


def test_average_periods_requires_every_source():
    """Leave a gap unresolved when any configured source value is missing."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=5, freq="h", tz="UTC", name="timestamp"
    )
    data = pd.DataFrame(
        {"ALB": [10.0, 12.0, float("nan"), 16.0, float("nan")]}, index=index
    )
    durations = pd.DataFrame(pd.Timedelta(0), index=index, columns=["ALB"])
    durations.loc[index[2], "ALB"] = pd.Timedelta(hours=1)

    filled, newly_filled = apply_average_periods(
        data,
        max_gap="1h",
        source_offsets=["-2h", "2h"],
        original_gap_duration=durations,
    )

    assert pd.isna(filled.loc[index[2], "ALB"])
    assert not newly_filled.loc[index[2], "ALB"]
