"""Tests for basic cleaning rule validation."""

import pandas as pd
import pytest

from tclean.basic.rule_validation import validate_basic_rule, validate_basic_rules
from tclean.time_grid import TimeGrid


def _grid(frequency: str = "1h") -> TimeGrid:
    """Return a test grid with the requested frequency."""
    return TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-03T00:00:00Z", frequency=frequency
    )


def test_validate_linear_interpolation_normalizes_max_gap():
    """Normalize the interpolation maximum gap to a timedelta."""
    result = validate_basic_rule(
        {"name": "short_gaps", "method": "linear_interpolation", "max_gap": "3h"},
        grid=_grid(),
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
        grid=_grid(),
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
        grid=_grid(),
    )

    assert result["source_offsets"] == [pd.Timedelta("-1D"), pd.Timedelta("1D")]


def test_validate_basic_rules_preserves_order():
    """Preserve configured rule execution order."""
    rules = [
        {"name": "first", "method": "linear_interpolation", "max_gap": "1h"},
        {"name": "second", "method": "linear_interpolation", "max_gap": "2h"},
    ]

    result = validate_basic_rules(rules, grid=_grid())

    assert [rule["name"] for rule in result] == ["first", "second"]


def test_validate_basic_rules_rejects_duplicate_names():
    """Reject duplicate basic cleaning rule names."""
    rules = [
        {"name": "duplicate", "method": "linear_interpolation", "max_gap": "1h"},
        {"name": "duplicate", "method": "linear_interpolation", "max_gap": "2h"},
    ]

    with pytest.raises(ValueError, match="must be unique"):
        validate_basic_rules(rules, grid=_grid())


def test_validate_basic_rule_rejects_unknown_method():
    """Reject unsupported basic cleaning methods."""
    with pytest.raises(ValueError, match="Unsupported basic cleaning method"):
        validate_basic_rule(
            {"name": "unknown", "method": "something_else", "max_gap": "1h"},
            grid=_grid(),
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
            grid=_grid(),
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
            grid=_grid(),
        )


def test_validate_basic_rule_rejects_non_positive_max_gap():
    """Reject zero or negative maximum-gap durations."""
    with pytest.raises(ValueError, match="greater than zero"):
        validate_basic_rule(
            {"name": "interpolate", "method": "linear_interpolation", "max_gap": "0h"},
            grid=_grid(),
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
            grid=_grid(),
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
            grid=_grid(),
        )


def test_basic_rule_accepts_frequency_aligned_duration():
    """Accept durations aligned with the configured frequency."""
    result = validate_basic_rule(
        {"name": "interpolate", "method": "linear_interpolation", "max_gap": "90min"},
        grid=_grid("30min"),
    )

    assert result["max_gap"] == pd.Timedelta("90min")


def test_basic_rule_rejects_misaligned_max_gap():
    """Reject maximum gaps not aligned with the configured frequency."""
    with pytest.raises(ValueError, match="integer multiple"):
        validate_basic_rule(
            {
                "name": "interpolate",
                "method": "linear_interpolation",
                "max_gap": "45min",
            },
            grid=_grid("30min"),
        )


def test_copy_periods_rejects_misaligned_source_offset():
    """Reject copy offsets not aligned with the configured frequency."""
    with pytest.raises(ValueError, match="integer multiple"):
        validate_basic_rule(
            {
                "name": "copy",
                "method": "copy_periods",
                "max_gap": "2h",
                "source_offset": "-3h",
                "require_complete_source": True,
            },
            grid=_grid("2h"),
        )


def test_basic_rule_rejects_missing_duration():
    """Reject missing basic-rule durations."""
    with pytest.raises(ValueError, match="must not be missing"):
        validate_basic_rule(
            {
                "name": "interpolate",
                "method": "linear_interpolation",
                "max_gap": pd.NaT,
            },
            grid=_grid(),
        )


def test_basic_rule_rejects_numeric_duration():
    """Reject durations without explicit units."""
    with pytest.raises(TypeError, match="duration string"):
        validate_basic_rule(
            {"name": "interpolate", "method": "linear_interpolation", "max_gap": 3},
            grid=_grid(),
        )


def test_average_periods_rejects_misaligned_source_offset():
    """Reject any average-period offset not aligned with frequency."""
    with pytest.raises(ValueError, match="integer multiple"):
        validate_basic_rule(
            {
                "name": "average",
                "method": "average_periods",
                "max_gap": "2h",
                "source_offsets": ["-4h", "3h"],
            },
            grid=_grid("2h"),
        )


def test_basic_rule_rejects_numeric_duration_string_without_unit():
    """Reject duration strings without an explicit unit."""
    rule = {
        "name": "interpolate",
        "method": "linear_interpolation",
        "max_gap": "3600000000000",
    }

    with pytest.raises(ValueError, match="explicit duration unit"):
        validate_basic_rule(rule, grid=_grid())


def test_copy_periods_accepts_negative_source_offset():
    """Allow source offsets before the target period."""
    rule = {
        "name": "previous_week",
        "method": "copy_periods",
        "max_gap": "6h",
        "source_offset": "-7D",
        "require_complete_source": True,
    }

    result = validate_basic_rule(rule, grid=_grid())

    assert result["source_offset"] == pd.Timedelta("-7D")
