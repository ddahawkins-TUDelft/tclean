"""Tests for basic cleaning rule validation."""

import pandas as pd
import pytest

from tclean.basic.rule_validation import validate_basic_rule, validate_basic_rules


def test_validate_linear_interpolation_normalizes_max_gap():
    """Normalize the interpolation maximum gap to a timedelta."""
    result = validate_basic_rule(
        {"name": "short_gaps", "method": "linear_interpolation", "max_gap": "3h"},
        frequency=pd.Timedelta("1h"),
    )

    assert result["max_gap"] == pd.Timedelta("3h")


def test_validate_copy_periods_normalizes_timedeltas():
    """Normalize copy-period duration arguments."""
    result = validate_basic_rule(
        {
            "name": "previous_day",
            "method": "copy_periods",
            "max_gap": "6h",
            "source_offset": "-1D",
            "require_complete_source": True,
        },
        frequency=pd.Timedelta("1h"),
    )

    assert result["max_gap"] == pd.Timedelta("6h")
    assert result["source_offset"] == pd.Timedelta("-1D")


def test_validate_average_periods_normalizes_offsets():
    """Normalize all average-period source offsets."""
    result = validate_basic_rule(
        {
            "name": "adjacent_days",
            "method": "average_periods",
            "max_gap": "4h",
            "source_offsets": ["-1D", "1D"],
        },
        frequency=pd.Timedelta("1h"),
    )

    assert result["source_offsets"] == [pd.Timedelta("-1D"), pd.Timedelta("1D")]


def test_validate_basic_rules_preserves_order():
    """Preserve configured rule execution order."""
    rules = [
        {"name": "first", "method": "linear_interpolation", "max_gap": "1h"},
        {"name": "second", "method": "linear_interpolation", "max_gap": "2h"},
    ]

    result = validate_basic_rules(rules, frequency=pd.Timedelta("1h"))

    assert [rule["name"] for rule in result] == ["first", "second"]


def test_validate_basic_rules_rejects_duplicate_names():
    """Reject duplicate basic cleaning rule names."""
    rules = [
        {"name": "duplicate", "method": "linear_interpolation", "max_gap": "1h"},
        {"name": "duplicate", "method": "linear_interpolation", "max_gap": "2h"},
    ]

    with pytest.raises(ValueError, match="must be unique"):
        validate_basic_rules(rules, frequency=pd.Timedelta("1h"))


def test_validate_basic_rule_rejects_unknown_method():
    """Reject unsupported basic cleaning methods."""
    with pytest.raises(ValueError, match="Unsupported basic cleaning method"):
        validate_basic_rule(
            {"name": "unknown", "method": "something_else", "max_gap": "1h"},
            frequency=pd.Timedelta("1h"),
        )


def test_validate_basic_rule_rejects_unknown_argument():
    """Reject arguments unsupported by the selected method."""
    with pytest.raises(ValueError, match="unknown keys"):
        validate_basic_rule(
            {
                "name": "interpolate",
                "method": "linear_interpolation",
                "max_gap": "1h",
                "source_offset": "-1D",
            },
            frequency=pd.Timedelta("1h"),
        )


def test_validate_copy_periods_requires_complete_source_flag():
    """Require explicit source-completeness behavior for copy periods."""
    with pytest.raises(ValueError, match="Missing keys"):
        validate_basic_rule(
            {
                "name": "copy",
                "method": "copy_periods",
                "max_gap": "2h",
                "source_offset": "-1D",
            },
            frequency=pd.Timedelta("1h"),
        )


def test_validate_basic_rule_rejects_non_positive_max_gap():
    """Reject zero or negative maximum-gap durations."""
    with pytest.raises(ValueError, match="greater than zero"):
        validate_basic_rule(
            {"name": "interpolate", "method": "linear_interpolation", "max_gap": "0h"},
            frequency=pd.Timedelta("1h"),
        )


def test_validate_copy_periods_rejects_zero_source_offset():
    """Reject a zero copy-period source offset."""
    with pytest.raises(ValueError, match="must not be zero"):
        validate_basic_rule(
            {
                "name": "copy",
                "method": "copy_periods",
                "max_gap": "2h",
                "source_offset": "0h",
                "require_complete_source": True,
            },
            frequency=pd.Timedelta("1h"),
        )


def test_validate_average_periods_rejects_empty_offsets():
    """Reject averaging rules without source periods."""
    with pytest.raises(ValueError, match="non-empty sequence"):
        validate_basic_rule(
            {
                "name": "average",
                "method": "average_periods",
                "max_gap": "2h",
                "source_offsets": [],
            },
            frequency=pd.Timedelta("1h"),
        )
