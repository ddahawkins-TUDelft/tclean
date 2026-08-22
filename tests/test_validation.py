"""Tests for canonical validation."""

import pandas as pd
import pandera.errors
import pytest

from tclean.validation import (
    infer_regular_timestep,
    validate_advanced_fill_rules,
    validate_advanced_source,
    validate_cleaning_method,
    validate_time_series,
)


def test_validate_time_series_accepts_hourly_numeric_dataframe():
    """Accept canonical hourly data."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    data = pd.DataFrame(
        {"ALB": [100.0, 101.0, 102.0], "GBR": [200.0, 201.0, 202.0]}, index=index
    )

    result = validate_time_series(data, frequency=pd.Timedelta("1h"))

    pd.testing.assert_index_equal(result.index, data.index, exact=False)

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), data.reset_index(drop=True)
    )


def test_validate_time_series_coerces_numeric_value():
    """Coerce numeric values to floating-point data."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    data = pd.DataFrame({"ALB": ["100", "101", "102"]}, index=index)

    result = validate_time_series(data, frequency=pd.Timedelta("1h"))

    assert result["ALB"].dtype == float
    assert result["ALB"].tolist() == [100.0, 101.0, 102.0]


def test_validate_time_series_allows_missing_value():
    """Allow missing values for subsequent gap cleaning."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    data = pd.DataFrame({"ALB": [100.0, float("nan"), 102.0]}, index=index)

    result = validate_time_series(data, frequency=pd.Timedelta("1h"))

    assert pd.isna(result.loc[index[1], "ALB"])


def test_validate_time_series_rejects_non_numeric_value():
    """Reject values that cannot be coerced to numbers."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    data = pd.DataFrame({"ALB": [100.0, "invalid", 102.0]}, index=index)

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_time_series(data, frequency=pd.Timedelta("1h"))


def test_validate_time_series_rejects_no_value_columns():
    """Reject data containing no context columns."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    data = pd.DataFrame(index=index)

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_time_series(data, frequency=pd.Timedelta("1h"))


def test_validate_time_series_rejects_single_timestamp():
    """Reject data with fewer than two timestamps."""
    index = pd.DatetimeIndex(["2026-01-01T00:00:00Z"], name="timestamp")
    data = pd.DataFrame({"ALB": [100.0]}, index=index)

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_time_series(data, frequency=pd.Timedelta("1h"))


def test_validate_time_series_rejects_unsorted_timestamps():
    """Reject data whose timestamps are not sorted."""
    index = pd.DatetimeIndex(
        ["2026-01-01T01:00:00Z", "2026-01-01T00:00:00Z"], name="timestamp"
    )
    data = pd.DataFrame({"ALB": [100.0, 101.0]}, index=index)

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_time_series(data, frequency=pd.Timedelta("1h"))


def test_validate_time_series_rejects_duplicate_timestamps():
    """Reject data containing duplicate timestamps."""
    index = pd.DatetimeIndex(
        ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], name="timestamp"
    )
    data = pd.DataFrame({"ALB": [100.0, 101.0]}, index=index)

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_time_series(data, frequency=pd.Timedelta("1h"))


def test_validate_time_series_rejects_non_hourly_timestamps():
    """Reject otherwise regular data that are not hourly."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="30min", tz="UTC", name="timestamp"
    )
    data = pd.DataFrame({"ALB": [100.0, 101.0, 102.0]}, index=index)

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_time_series(data, frequency=pd.Timedelta("1h"))


def test_validate_time_series_rejects_missing_hour():
    """Reject an hourly index containing a missing timestamp."""
    index = pd.DatetimeIndex(
        ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T03:00:00Z"],
        name="timestamp",
    )
    data = pd.DataFrame({"ALB": [100.0, 101.0, 103.0]}, index=index)

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_time_series(data, frequency=pd.Timedelta("1h"))


def test_validate_time_series_converts_timestamp_index_to_utc():
    """Normalize timezone-aware timestamps to UTC."""
    index = pd.DatetimeIndex(
        [
            "2026-01-01T01:00:00+01:00",
            "2026-01-01T02:00:00+01:00",
            "2026-01-01T03:00:00+01:00",
        ],
        name="timestamp",
    )
    data = pd.DataFrame({"ALB": [100.0, 101.0, 102.0]}, index=index)

    result = validate_time_series(data, frequency=pd.Timedelta("1h"))

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


def test_validate_time_series_rejects_unnamed_timestamp_index():
    """Reject data whose timestamp index is unnamed."""
    index = pd.date_range("2026-01-01 00:00", periods=3, freq="h", tz="UTC")
    data = pd.DataFrame({"ALB": [100.0, 101.0, 102.0]}, index=index)

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_time_series(data, frequency=pd.Timedelta("1h"))


def test_validate_cleaning_method_accepts_aligned_data():
    """Accept cleaning-method data aligned with canonical values."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )

    data = pd.DataFrame({"GBR": [10.0, 11.0, 12.0]}, index=index)

    cleaning_method = pd.DataFrame(
        {"GBR": ["observed_entsoe", pd.NA, "observed_entsoe"]},
        index=index,
        dtype="string",
    )

    result = validate_cleaning_method(cleaning_method, data=data)

    pd.testing.assert_index_equal(result.index, cleaning_method.index, exact=False)

    pd.testing.assert_index_equal(result.columns, cleaning_method.columns)

    assert result.loc[index[0], "GBR"] == "observed_entsoe"
    assert pd.isna(result.loc[index[1], "GBR"])
    assert result.loc[index[2], "GBR"] == "observed_entsoe"


def test_validate_cleaning_method_rejects_misaligned_index():
    """Reject provenance whose index differs from values."""
    data_index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )

    method_index = pd.date_range(
        "2026-01-02 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )

    data = pd.DataFrame({"GBR": [10.0, 11.0, 12.0]}, index=data_index)

    cleaning_method = pd.DataFrame(
        {"GBR": ["a", "b", "c"]}, index=method_index, dtype="string"
    )

    with pytest.raises(ValueError, match="index must exactly match"):
        validate_cleaning_method(cleaning_method, data=data)


def test_validate_cleaning_method_rejects_misaligned_columns():
    """Reject provenance whose columns differ from values."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )

    data = pd.DataFrame({"GBR": [10.0, 11.0, 12.0]}, index=index)

    cleaning_method = pd.DataFrame(
        {"FRA": ["a", "b", "c"]}, index=index, dtype="string"
    )

    with pytest.raises(ValueError, match="columns must exactly match"):
        validate_cleaning_method(cleaning_method, data=data)


def test_validate_advanced_fill_rules_rejects_unknown_method():
    """Reject unsupported advanced-fill methods."""
    rules = pd.DataFrame(
        {
            "rule_name": ["rule"],
            "method": ["magic"],
            "source": ["test_source"],
            "context": ["GBR"],
            "start": ["2026-01-01T00:00:00Z"],
            "end": ["2026-01-02T00:00:00Z"],
            "scope": ["fill_gaps"],
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_advanced_fill_rules(rules)


def test_validate_advanced_fill_rules_rejects_unknown_scope():
    """Reject unsupported advanced-fill scopes."""
    rules = pd.DataFrame(
        {
            "rule_name": ["rule"],
            "method": ["external_profile"],
            "source": ["test_source"],
            "context": ["GBR"],
            "start": ["2026-01-01T00:00:00Z"],
            "end": ["2026-01-02T00:00:00Z"],
            "scope": ["sometimes"],
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_advanced_fill_rules(rules)


def test_validate_advanced_fill_rules_rejects_duplicate_names():
    """Reject duplicate advanced-fill rule names."""
    rules = pd.DataFrame(
        {
            "rule_name": ["rule", "rule"],
            "method": ["external_profile", "external_profile"],
            "source": ["test_source", "test_source_2"],
            "context": ["GBR", "GBR"],
            "start": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
            "end": ["2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"],
            "scope": ["fill_gaps", "fill_gaps"],
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_advanced_fill_rules(rules)


def test_validate_advanced_fill_rules_requires_source_for_profile_method():
    """Require a source for advanced methods that consume external data."""
    rules = pd.DataFrame(
        {
            "rule_name": ["fill"],
            "method": ["external_profile"],
            "source": [None],
            "context": ["GBR"],
            "start": ["2026-01-01T00:00:00Z"],
            "end": ["2026-01-02T00:00:00Z"],
            "scope": ["fill_gaps"],
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_advanced_fill_rules(rules)


def test_validate_advanced_fill_rules_rejects_source_for_leave_missing():
    """Reject a source on a leave-missing rule."""
    rules = pd.DataFrame(
        {
            "rule_name": ["leave"],
            "method": ["leave_missing"],
            "source": ["unused"],
            "context": ["GBR"],
            "start": ["2026-01-01T00:00:00Z"],
            "end": ["2026-01-02T00:00:00Z"],
            "scope": ["fill_gaps"],
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_advanced_fill_rules(rules)


def test_validate_advanced_fill_rules_allows_source_reuse():
    """Allow multiple advanced rules to reference the same source."""
    rules = pd.DataFrame(
        {
            "rule_name": ["first", "second"],
            "method": ["external_profile", "external_profile"],
            "source": ["winter_profile", "winter_profile"],
            "context": ["GBR", "GBR"],
            "start": ["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"],
            "end": ["2026-01-02T00:00:00Z", "2026-02-02T00:00:00Z"],
            "scope": ["fill_gaps", "overwrite"],
        }
    )

    result = validate_advanced_fill_rules(rules)

    assert result["source"].tolist() == ["winter_profile", "winter_profile"]


def test_validate_advanced_source_accepts_valid_series():
    """Accept a valid advanced time-series source."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )

    source = pd.Series([10.0, 20.0, 30.0], index=index)

    result = validate_advanced_source(source, frequency=pd.Timedelta("1h"))

    pd.testing.assert_series_equal(result, source, check_dtype=False)


def test_validate_advanced_source_coerces_numeric_values():
    """Coerce numeric advanced-source values to floating point."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )

    source = pd.Series(["10", "20", "30"], index=index)

    result = validate_advanced_source(source, frequency=pd.Timedelta("1h"))

    assert result.tolist() == [10.0, 20.0, 30.0]


def test_validate_advanced_source_rejects_missing_values():
    """Reject missing values in an advanced source."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )

    source = pd.Series([10.0, float("nan"), 30.0], index=index)

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_advanced_source(source, frequency=pd.Timedelta("1h"))


def test_validate_advanced_source_rejects_duplicate_timestamps():
    """Reject duplicate advanced-source timestamps."""
    index = pd.DatetimeIndex(
        ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], name="timestamp"
    )

    source = pd.Series([10.0, 20.0], index=index)

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_advanced_source(source, frequency=pd.Timedelta("1h"))


def test_validate_advanced_source_rejects_non_hourly_timestamp():
    """Reject advanced-source timestamps that are not on whole hours."""
    index = pd.DatetimeIndex(["2026-01-01T00:30:00Z"], name="timestamp")

    source = pd.Series([10.0], index=index)

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_advanced_source(source, frequency=pd.Timedelta("1h"))


def test_validate_advanced_source_rejects_non_numeric_values():
    """Reject advanced-source values that cannot be converted to numbers."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=2, freq="h", tz="UTC", name="timestamp"
    )

    source = pd.Series(["10", "not-a-number"], index=index)

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_advanced_source(source, frequency=pd.Timedelta("1h"))


def test_validate_advanced_source_rejects_unsorted_timestamps():
    """Reject advanced-source timestamps that are not sorted."""
    index = pd.DatetimeIndex(
        ["2026-01-01T01:00:00Z", "2026-01-01T00:00:00Z"], name="timestamp"
    )

    source = pd.Series([20.0, 10.0], index=index)

    with pytest.raises(ValueError, match="must be sorted"):
        validate_advanced_source(source, frequency=pd.Timedelta("1h"))
