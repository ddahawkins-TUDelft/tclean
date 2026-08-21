"""Tests for orchestration of basic demand cleaning."""

import pandas as pd
import pytest

from tclean.basic.apply import calculate_missing_run_durations, fill_basic_gaps


def test_calculate_missing_run_durations_labels_complete_gap():
    """Assign the full run duration to every value in a missing run."""
    index = pd.date_range("2026-01-01 00:00", periods=5, freq="h", tz="UTC")

    load = pd.DataFrame(
        {"ALB": [10.0, float("nan"), float("nan"), 13.0, 14.0]}, index=index
    )

    result = calculate_missing_run_durations(load)

    expected = pd.DataFrame(
        {
            "ALB": [
                pd.Timedelta(0),
                pd.Timedelta(hours=2),
                pd.Timedelta(hours=2),
                pd.Timedelta(0),
                pd.Timedelta(0),
            ]
        },
        index=index,
    )

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_fill_basic_gaps_applies_rules_in_order():
    """Apply configured cleaning rules sequentially."""
    index = pd.date_range("2026-01-01 00:00", periods=4, freq="h", tz="UTC")

    load = pd.DataFrame({"ALB": [10.0, float("nan"), 14.0, 16.0]}, index=index)

    cleaning_method = pd.DataFrame(
        {"ALB": ["observed_entsoe", pd.NA, "observed_entsoe", "observed_entsoe"]},
        index=index,
        dtype="string",
    )

    rules = [
        {
            "name": "short_interpolation",
            "method": "linear_interpolation",
            "max_gap": "1h",
        }
    ]

    filled, provenance = fill_basic_gaps(
        load, cleaning_method=cleaning_method, rules=rules
    )

    assert filled.loc[index[1], "ALB"] == 12.0
    assert provenance.loc[index[1], "ALB"] == "short_interpolation"


def test_fill_basic_gaps_marks_unresolved_values_missing():
    """Mark unresolved missing values explicitly in provenance."""
    index = pd.date_range("2026-01-01 00:00", periods=4, freq="h", tz="UTC")

    load = pd.DataFrame({"ALB": [10.0, float("nan"), float("nan"), 16.0]}, index=index)

    cleaning_method = pd.DataFrame(
        {"ALB": ["observed_entsoe", pd.NA, pd.NA, "observed_entsoe"]},
        index=index,
        dtype="string",
    )

    rules = [
        {
            "name": "short_interpolation",
            "method": "linear_interpolation",
            "max_gap": "1h",
        }
    ]

    filled, provenance = fill_basic_gaps(
        load, cleaning_method=cleaning_method, rules=rules
    )

    assert filled.loc[index[1:3], "ALB"].isna().all()
    assert (provenance.loc[index[1:3], "ALB"] == "missing").all()


def test_fill_basic_gaps_disabled_preserves_load():
    """Leave demand unchanged when basic cleaning is disabled."""
    index = pd.date_range("2026-01-01 00:00", periods=3, freq="h", tz="UTC")

    load = pd.DataFrame({"ALB": [10.0, float("nan"), 14.0]}, index=index)

    cleaning_method = pd.DataFrame(
        {"ALB": ["observed_entsoe", pd.NA, "observed_entsoe"]},
        index=index,
        dtype="string",
    )

    filled, provenance = fill_basic_gaps(
        load, cleaning_method=cleaning_method, rules=[], enabled=False
    )

    pd.testing.assert_frame_equal(filled, load)
    assert provenance.loc[index[1], "ALB"] == "missing"


def test_fill_basic_gaps_rejects_unknown_method():
    """Reject cleaning rules containing unsupported methods."""
    index = pd.date_range("2026-01-01 00:00", periods=3, freq="h", tz="UTC")

    load = pd.DataFrame({"ALB": [10.0, float("nan"), 14.0]}, index=index)

    cleaning_method = pd.DataFrame(
        {"ALB": ["observed_entsoe", pd.NA, "observed_entsoe"]},
        index=index,
        dtype="string",
    )

    rules = [{"name": "mystery_rule", "method": "not_a_method", "max_gap": "1h"}]

    with pytest.raises(ValueError, match="Unsupported gap-filling method"):
        fill_basic_gaps(load, cleaning_method=cleaning_method, rules=rules)


def test_fill_basic_gaps_rejects_misaligned_provenance():
    """Reject provenance whose index differs from the demand data."""
    index = pd.date_range("2026-01-01 00:00", periods=3, freq="h", tz="UTC")

    load = pd.DataFrame({"ALB": [10.0, 11.0, 12.0]}, index=index)

    cleaning_method = pd.DataFrame(
        {"ALB": ["observed_entsoe"] * 2}, index=index[:2], dtype="string"
    )

    with pytest.raises(ValueError, match="same index"):
        fill_basic_gaps(load, cleaning_method=cleaning_method, rules=[])
