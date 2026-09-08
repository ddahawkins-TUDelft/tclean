"""Tests for data-quality test validation."""

import pandas as pd
import pytest

from tclean.data_quality.rule_validation import (
    validate_quality_test,
    validate_quality_tests,
)
from tclean.time_grid import TimeGrid


def _grid(frequency: str = "1h") -> TimeGrid:
    """Return a test grid with the requested frequency."""
    return TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-03T00:00:00Z", frequency=frequency
    )


def test_validate_range_test_accepts_both_bounds():
    """Accept a range test with minimum and maximum bounds."""
    result = validate_quality_test(
        {
            "name": "plausible_values",
            "method": "range",
            "minimum": {"value_mode": "fixed", "value": 0},
            "maximum": {"value_mode": "fixed", "value": 1000},
        },
        grid=_grid(),
    )

    assert result == {
        "name": "plausible_values",
        "method": "range",
        "minimum": {"value_mode": "fixed", "value": 0},
        "maximum": {"value_mode": "fixed", "value": 1000},
    }


def test_validate_range_test_accepts_minimum_only():
    """Accept a range test with only a lower bound."""
    result = validate_quality_test(
        {
            "name": "non_negative",
            "method": "range",
            "minimum": {"value_mode": "fixed", "value": 0},
        },
        grid=_grid(),
    )

    assert result["minimum"] == {"value_mode": "fixed", "value": 0}
    assert "maximum" not in result


def test_validate_range_test_accepts_maximum_only():
    """Accept a range test with only an upper bound."""
    result = validate_quality_test(
        {
            "name": "upper_bound",
            "method": "range",
            "maximum": {"value_mode": "fixed", "value": 1000.5},
        },
        grid=_grid(),
    )

    assert result["maximum"] == {"value_mode": "fixed", "value": 1000.5}
    assert "minimum" not in result


def test_validate_range_test_accepts_equal_bounds():
    """Allow a range test whose minimum equals its maximum."""
    result = validate_quality_test(
        {
            "name": "exact_value",
            "method": "range",
            "minimum": {"value_mode": "fixed", "value": 5},
            "maximum": {"value_mode": "fixed", "value": 5},
        },
        grid=_grid(),
    )

    assert result["minimum"] == {"value_mode": "fixed", "value": 5}
    assert result["maximum"] == {"value_mode": "fixed", "value": 5}


def test_validate_range_test_normalizes_selectors():
    """Normalize source and context selectors to lists."""
    result = validate_quality_test(
        {
            "name": "selected_range",
            "method": "range",
            "sources": ("entsoe", "opsd"),
            "contexts": ("ALB", "BIH"),
            "minimum": {"value_mode": "fixed", "value": 0},
        },
        grid=_grid(),
    )

    assert result["sources"] == ["entsoe", "opsd"]
    assert result["contexts"] == ["ALB", "BIH"]


def test_validate_quality_tests_preserves_order():
    """Preserve configured quality-test execution order."""
    tests = [
        {
            "name": "first",
            "method": "range",
            "minimum": {"value_mode": "fixed", "value": 0},
        },
        {
            "name": "second",
            "method": "range",
            "maximum": {"value_mode": "fixed", "value": 1000},
        },
    ]

    result = validate_quality_tests(tests, grid=_grid())

    assert [test["name"] for test in result] == ["first", "second"]


def test_validate_quality_tests_accepts_empty_sequence():
    """Allow a quality evaluation plan containing no tests."""
    result = validate_quality_tests([], grid=_grid())

    assert result == []


def test_validate_quality_tests_rejects_duplicate_names():
    """Reject duplicate data-quality test names."""
    tests = [
        {
            "name": "duplicate",
            "method": "range",
            "minimum": {"value_mode": "fixed", "value": 0},
        },
        {
            "name": "duplicate",
            "method": "range",
            "maximum": {"value_mode": "fixed", "value": 1000},
        },
    ]

    with pytest.raises(ValueError, match="must be unique"):
        validate_quality_tests(tests, grid=_grid())


def test_validate_quality_tests_rejects_non_sequence():
    """Reject unordered or non-sequence quality-test collections."""
    tests = {
        "name": "non_negative",
        "method": "range",
        "minimum": {"value_mode": "fixed", "value": 0},
    }

    with pytest.raises(TypeError, match="ordered sequence"):
        validate_quality_tests(tests, grid=_grid())


def test_validate_quality_test_rejects_non_mapping():
    """Reject individual tests that are not mappings."""
    with pytest.raises(TypeError, match="must be a mapping"):
        validate_quality_test(["non_negative", "range", 0], grid=_grid())


def test_validate_quality_test_rejects_blank_name():
    """Reject blank data-quality test names."""
    with pytest.raises(ValueError, match="'name' must be a non-empty string"):
        validate_quality_test(
            {
                "name": "   ",
                "method": "range",
                "minimum": {"value_mode": "fixed", "value": 0},
            },
            grid=_grid(),
        )


def test_validate_quality_test_rejects_blank_method():
    """Reject blank data-quality method names."""
    with pytest.raises(ValueError, match="'method' must be a non-empty string"):
        validate_quality_test(
            {"name": "non_negative", "method": "", "minimum": 0}, grid=_grid()
        )


def test_validate_quality_test_rejects_unknown_method():
    """Reject unsupported data-quality methods."""
    with pytest.raises(ValueError, match="Unsupported data-quality method"):
        validate_quality_test(
            {"name": "unknown", "method": "something_else", "minimum": 0}, grid=_grid()
        )


def test_validate_range_test_rejects_unknown_argument():
    """Reject arguments unsupported by the range method."""
    with pytest.raises(ValueError, match="unknown keys"):
        validate_quality_test(
            {
                "name": "non_negative",
                "method": "range",
                "minimum": {"value_mode": "fixed", "value": 0},
                "tolerance": 0.1,
            },
            grid=_grid(),
        )


def test_validate_range_test_requires_a_bound():
    """Require at least one range boundary."""
    with pytest.raises(ValueError, match="at least one of 'minimum' or 'maximum'"):
        validate_quality_test({"name": "unbounded", "method": "range"}, grid=_grid())


def test_validate_range_test_rejects_reversed_bounds():
    """Reject a minimum greater than the maximum."""
    with pytest.raises(ValueError, match="less than or equal"):
        validate_quality_test(
            {
                "name": "invalid_range",
                "method": "range",
                "minimum": {"value_mode": "fixed", "value": 10},
                "maximum": {"value_mode": "fixed", "value": 5},
            },
            grid=_grid(),
        )


@pytest.mark.parametrize(
    "value", ["zero", None, True, pd.NA, float("nan"), float("inf"), float("-inf")]
)
def test_validate_range_test_rejects_invalid_minimum(value):
    """Reject non-numeric or non-finite lower bounds."""
    with pytest.raises(ValueError, match="finite real number"):
        validate_quality_test(
            {
                "name": "invalid_minimum",
                "method": "range",
                "minimum": {"value_mode": "fixed", "value": value},
            },
            grid=_grid(),
        )


@pytest.mark.parametrize(
    "value",
    ["one_thousand", None, False, pd.NA, float("nan"), float("inf"), float("-inf")],
)
def test_validate_range_test_rejects_invalid_maximum(value):
    """Reject non-numeric or non-finite upper bounds."""
    with pytest.raises(ValueError, match="finite real number"):
        validate_quality_test(
            {
                "name": "invalid_maximum",
                "method": "range",
                "maximum": {"value_mode": "fixed", "value": value},
            },
            grid=_grid(),
        )


@pytest.mark.parametrize("field", ["sources", "contexts"])
def test_validate_quality_test_rejects_string_selector(field):
    """Reject a string where an ordered selector sequence is required."""
    test = {
        "name": "selected_range",
        "method": "range",
        "minimum": {"value_mode": "fixed", "value": 0},
        field: "entsoe",
    }

    with pytest.raises(ValueError, match="non-empty ordered sequence"):
        validate_quality_test(test, grid=_grid())


@pytest.mark.parametrize("field", ["sources", "contexts"])
def test_validate_quality_test_rejects_empty_selector(field):
    """Reject an explicitly empty source or context selector."""
    test = {
        "name": "selected_range",
        "method": "range",
        "minimum": {"value_mode": "fixed", "value": 0},
        field: [],
    }

    with pytest.raises(ValueError, match="non-empty ordered sequence"):
        validate_quality_test(test, grid=_grid())


@pytest.mark.parametrize(
    ("field", "values"), [("sources", ["entsoe", 123]), ("contexts", ["ALB", None])]
)
def test_validate_quality_test_rejects_invalid_selector_entries(field, values):
    """Reject source or context selectors containing invalid entries."""
    test = {
        "name": "selected_range",
        "method": "range",
        "minimum": {"value_mode": "fixed", "value": 0},
        field: values,
    }

    with pytest.raises(ValueError, match="only non-empty strings"):
        validate_quality_test(test, grid=_grid())


@pytest.mark.parametrize(
    ("field", "values"),
    [("sources", ["entsoe", "entsoe"]), ("contexts", ["ALB", "ALB"])],
)
def test_validate_quality_test_rejects_duplicate_selector_entries(field, values):
    """Reject duplicate entries in source or context selectors."""
    test = {
        "name": "selected_range",
        "method": "range",
        "minimum": {"value_mode": "fixed", "value": 0},
        field: values,
    }

    with pytest.raises(ValueError, match="entries must be unique"):
        validate_quality_test(test, grid=_grid())


def test_validate_range_test_rejects_failed_period_inclusion():
    """Reject reference-data controls for methods that do not use them."""
    with pytest.raises(ValueError, match="only supported when at least one"):
        validate_quality_test(
            {
                "name": "non_negative",
                "method": "range",
                "minimum": {"value_mode": "fixed", "value": 0},
                "include_failed_periods_from": ["earlier_test"],
            },
            grid=_grid(),
        )


def test_validate_value_run_normalizes_duration_and_default_tolerance():
    """Normalize value-run duration and supply zero tolerance by default."""
    result = validate_quality_test(
        {
            "name": "zero_run",
            "method": "value_run",
            "value": {"value_mode": "fixed", "value": 0},
            "minimum_duration": "3h",
        },
        grid=_grid(),
    )

    assert result["value"] == {"value_mode": "fixed", "value": 0}
    assert result["minimum_duration"] == pd.Timedelta("3h")
    assert result["tolerance"] == {"value_mode": "fixed", "value": 0.0}


def test_validate_value_run_accepts_positive_tolerance():
    """Accept a finite non-negative value-run tolerance."""
    result = validate_quality_test(
        {
            "name": "near_zero_run",
            "method": "value_run",
            "value": {"value_mode": "fixed", "value": 0},
            "minimum_duration": "3h",
            "tolerance": {"value_mode": "fixed", "value": 0.1},
        },
        grid=_grid(),
    )

    assert result["tolerance"] == {"value_mode": "fixed", "value": 0.1}


def test_validate_value_run_rejects_zero_duration():
    """Reject a zero minimum value-run duration."""
    with pytest.raises(ValueError, match="must be greater than zero"):
        validate_quality_test(
            {
                "name": "zero_run",
                "method": "value_run",
                "value": {"value_mode": "fixed", "value": 0},
                "minimum_duration": "0h",
            },
            grid=_grid(),
        )


def test_validate_value_run_rejects_off_grid_duration():
    """Require minimum duration to align with the configured grid."""
    with pytest.raises(ValueError, match="integer multiple"):
        validate_quality_test(
            {
                "name": "zero_run",
                "method": "value_run",
                "value": {"value_mode": "fixed", "value": 0},
                "minimum_duration": "90min",
            },
            grid=_grid(),
        )


def test_validate_value_run_rejects_negative_tolerance():
    """Reject negative value-run tolerance."""
    with pytest.raises(ValueError, match="greater than or equal to zero"):
        validate_quality_test(
            {
                "name": "zero_run",
                "method": "value_run",
                "value": {"value_mode": "fixed", "value": 0},
                "minimum_duration": "3h",
                "tolerance": {"value_mode": "fixed", "value": -0.1},
            },
            grid=_grid(),
        )


def test_validate_value_run_rejects_non_numeric_value():
    """Require the target value to be numeric and finite."""
    with pytest.raises(ValueError, match="finite real number"):
        validate_quality_test(
            {
                "name": "bad_value",
                "method": "value_run",
                "value": {"value_mode": "fixed", "value": "zero"},
                "minimum_duration": "3h",
            },
            grid=_grid(),
        )


def test_validate_value_run_rejects_failed_period_inclusion():
    """Do not allow reference controls on a non-reference-aware method."""
    with pytest.raises(ValueError, match="only supported when at least one"):
        validate_quality_test(
            {
                "name": "zero_run",
                "method": "value_run",
                "value": {"value_mode": "fixed", "value": 0},
                "minimum_duration": "3h",
                "include_failed_periods_from": ["earlier_test"],
            },
            grid=_grid(),
        )


def test_validate_flatline_normalizes_duration_and_default_tolerance():
    """Normalize flatline duration and supply zero tolerance by default."""
    result = validate_quality_test(
        {"name": "constant_values", "method": "flatline", "minimum_duration": "3h"},
        grid=_grid(),
    )

    assert result["minimum_duration"] == pd.Timedelta("3h")
    assert result["tolerance"] == {"value_mode": "fixed", "value": 0.0}


def test_validate_flatline_accepts_positive_tolerance():
    """Accept a finite non-negative flatline tolerance."""
    result = validate_quality_test(
        {
            "name": "nearly_constant",
            "method": "flatline",
            "minimum_duration": "3h",
            "tolerance": {"value_mode": "fixed", "value": 0.1},
        },
        grid=_grid(),
    )

    assert result["tolerance"] == {"value_mode": "fixed", "value": 0.1}


def test_validate_flatline_rejects_single_step_duration():
    """Require a flatline to contain at least two observations."""
    with pytest.raises(ValueError, match="at least two grid steps"):
        validate_quality_test(
            {"name": "constant_values", "method": "flatline", "minimum_duration": "1h"},
            grid=_grid(),
        )


def test_validate_flatline_rejects_zero_duration():
    """Reject a zero flatline duration."""
    with pytest.raises(ValueError, match="must be greater than zero"):
        validate_quality_test(
            {"name": "constant_values", "method": "flatline", "minimum_duration": "0h"},
            grid=_grid(),
        )


def test_validate_flatline_rejects_off_grid_duration():
    """Require flatline duration to align with the configured grid."""
    with pytest.raises(ValueError, match="integer multiple"):
        validate_quality_test(
            {
                "name": "constant_values",
                "method": "flatline",
                "minimum_duration": "150min",
            },
            grid=_grid(),
        )


def test_validate_flatline_rejects_negative_tolerance():
    """Reject a negative flatline tolerance."""
    with pytest.raises(ValueError, match="greater than or equal to zero"):
        validate_quality_test(
            {
                "name": "constant_values",
                "method": "flatline",
                "minimum_duration": "3h",
                "tolerance": {"value_mode": "fixed", "value": -0.1},
            },
            grid=_grid(),
        )


def test_validate_flatline_rejects_failed_period_inclusion():
    """Do not allow reference controls on a non-reference-aware method."""
    with pytest.raises(ValueError, match="only supported when at least one"):
        validate_quality_test(
            {
                "name": "constant_values",
                "method": "flatline",
                "minimum_duration": "3h",
                "include_failed_periods_from": ["earlier_test"],
            },
            grid=_grid(),
        )


def test_validate_low_variability_normalizes_parameters():
    """Normalize low-variability window and range threshold."""
    result = validate_quality_test(
        {
            "name": "stable_values",
            "method": "low_variability",
            "window_duration": "3h",
            "maximum_range": {"value_mode": "fixed", "value": 5},
        },
        grid=_grid(),
    )

    assert result["window_duration"] == pd.Timedelta("3h")
    assert result["maximum_range"] == {"value_mode": "fixed", "value": 5}


def test_validate_low_variability_accepts_zero_maximum_range():
    """Allow zero range as an exact-constancy criterion."""
    result = validate_quality_test(
        {
            "name": "exactly_constant",
            "method": "low_variability",
            "window_duration": "3h",
            "maximum_range": {"value_mode": "fixed", "value": 0},
        },
        grid=_grid(),
    )

    assert result["maximum_range"] == {"value_mode": "fixed", "value": 0}


def test_validate_low_variability_rejects_single_step_window():
    """Require variability windows to contain at least two observations."""
    with pytest.raises(ValueError, match="at least two grid steps"):
        validate_quality_test(
            {
                "name": "stable_values",
                "method": "low_variability",
                "window_duration": "1h",
                "maximum_range": {"value_mode": "fixed", "value": 5},
            },
            grid=_grid(),
        )


def test_validate_low_variability_rejects_zero_window():
    """Reject a zero-duration variability window."""
    with pytest.raises(ValueError, match="must be greater than zero"):
        validate_quality_test(
            {
                "name": "stable_values",
                "method": "low_variability",
                "window_duration": "0h",
                "maximum_range": {"value_mode": "fixed", "value": 5},
            },
            grid=_grid(),
        )


def test_validate_low_variability_rejects_off_grid_window():
    """Require variability windows to align with the grid."""
    with pytest.raises(ValueError, match="integer multiple"):
        validate_quality_test(
            {
                "name": "stable_values",
                "method": "low_variability",
                "window_duration": "150min",
                "maximum_range": {"value_mode": "fixed", "value": 5},
            },
            grid=_grid(),
        )


def test_validate_low_variability_rejects_negative_maximum_range():
    """Reject a negative variability threshold."""
    with pytest.raises(ValueError, match="greater than or equal to zero"):
        validate_quality_test(
            {
                "name": "stable_values",
                "method": "low_variability",
                "window_duration": "3h",
                "maximum_range": {"value_mode": "fixed", "value": -1},
            },
            grid=_grid(),
        )


def test_validate_low_variability_requires_maximum_range():
    """Require an explicit variability threshold."""
    with pytest.raises(ValueError, match="Missing keys"):
        validate_quality_test(
            {
                "name": "stable_values",
                "method": "low_variability",
                "window_duration": "3h",
            },
            grid=_grid(),
        )


def test_validate_low_variability_rejects_failed_period_inclusion():
    """Do not allow prior-failure controls on this direct method."""
    with pytest.raises(ValueError, match="only supported when at least one"):
        validate_quality_test(
            {
                "name": "stable_values",
                "method": "low_variability",
                "window_duration": "3h",
                "maximum_range": {"value_mode": "fixed", "value": 5},
                "include_failed_periods_from": ["earlier_test"],
            },
            grid=_grid(),
        )


def test_validate_repeated_pattern_normalizes_parameters():
    """Normalize repeated-pattern configuration."""
    result = validate_quality_test(
        {
            "name": "repeated_day",
            "method": "repeated_pattern",
            "pattern_duration": "24h",
            "minimum_matches": 3,
        },
        grid=_grid(),
    )

    assert result["pattern_duration"] == pd.Timedelta("24h")
    assert result["minimum_matches"] == 3
    assert result["tolerance"] == {"value_mode": "fixed", "value": 0.0}


def test_validate_repeated_pattern_accepts_tolerance():
    """Accept a finite non-negative pointwise tolerance."""
    result = validate_quality_test(
        {
            "name": "repeated_day",
            "method": "repeated_pattern",
            "pattern_duration": "24h",
            "minimum_matches": 2,
            "tolerance": {"value_mode": "fixed", "value": 0.1},
        },
        grid=_grid(),
    )

    assert result["tolerance"] == {"value_mode": "fixed", "value": 0.1}


def test_validate_repeated_pattern_rejects_single_step_pattern():
    """Require a repeated pattern to span at least two observations."""
    with pytest.raises(ValueError, match="at least two grid steps"):
        validate_quality_test(
            {
                "name": "repeated_value",
                "method": "repeated_pattern",
                "pattern_duration": "1h",
                "minimum_matches": 2,
            },
            grid=_grid(),
        )


def test_validate_repeated_pattern_rejects_off_grid_duration():
    """Require pattern duration to align with the grid."""
    with pytest.raises(ValueError, match="integer multiple"):
        validate_quality_test(
            {
                "name": "repeated_pattern",
                "method": "repeated_pattern",
                "pattern_duration": "150min",
                "minimum_matches": 2,
            },
            grid=_grid(),
        )


@pytest.mark.parametrize("minimum_matches", [0, 1, -1])
def test_validate_repeated_pattern_requires_at_least_two_matches(minimum_matches):
    """Require at least two occurrences of a repeated pattern."""
    with pytest.raises(ValueError, match="greater than or equal to 2"):
        validate_quality_test(
            {
                "name": "repeated_pattern",
                "method": "repeated_pattern",
                "pattern_duration": "2h",
                "minimum_matches": minimum_matches,
            },
            grid=_grid(),
        )


@pytest.mark.parametrize("minimum_matches", [2.0, "2", True])
def test_validate_repeated_pattern_requires_integer_match_count(minimum_matches):
    """Require minimum_matches to be an integer."""
    with pytest.raises(ValueError, match="must be an integer"):
        validate_quality_test(
            {
                "name": "repeated_pattern",
                "method": "repeated_pattern",
                "pattern_duration": "2h",
                "minimum_matches": minimum_matches,
            },
            grid=_grid(),
        )


def test_validate_repeated_pattern_rejects_negative_tolerance():
    """Reject a negative repeated-pattern tolerance."""
    with pytest.raises(ValueError, match="greater than or equal to zero"):
        validate_quality_test(
            {
                "name": "repeated_pattern",
                "method": "repeated_pattern",
                "pattern_duration": "2h",
                "minimum_matches": 2,
                "tolerance": {"value_mode": "fixed", "value": -0.1},
            },
            grid=_grid(),
        )


def test_validate_repeated_pattern_rejects_failed_period_inclusion():
    """Do not allow prior-failure controls on this direct method."""
    with pytest.raises(ValueError, match="only supported when at least one"):
        validate_quality_test(
            {
                "name": "repeated_pattern",
                "method": "repeated_pattern",
                "pattern_duration": "2h",
                "minimum_matches": 2,
                "include_failed_periods_from": ["earlier_test"],
            },
            grid=_grid(),
        )


def test_validate_rate_of_change_normalizes_fixed_mode():
    """Normalize a fixed rate-of-change test."""
    result = validate_quality_test(
        {
            "name": "large_change",
            "method": "rate_of_change",
            "difference_mode": "fixed",
            "threshold": {"value_mode": "fixed", "value": 20},
        },
        grid=_grid(),
    )

    assert result["difference_mode"] == "fixed"
    assert result["threshold"] == {"value_mode": "fixed", "value": 20}
    assert "reference_magnitude_threshold" not in result


def test_validate_rate_of_change_supplies_default_relative_reference_threshold():
    """Supply a zero reference magnitude floor for relative mode."""
    result = validate_quality_test(
        {
            "name": "large_relative_change",
            "method": "rate_of_change",
            "difference_mode": "relative",
            "threshold": {"value_mode": "fixed", "value": 0.2},
        },
        grid=_grid(),
    )

    assert result["difference_mode"] == "relative"
    assert result["threshold"] == {"value_mode": "fixed", "value": 0.2}
    assert result["reference_magnitude_threshold"] == {
        "value_mode": "fixed",
        "value": 0.0,
    }


def test_validate_rate_of_change_accepts_relative_reference_threshold():
    """Accept a finite non-negative reference floor in relative mode."""
    result = validate_quality_test(
        {
            "name": "large_relative_change",
            "method": "rate_of_change",
            "difference_mode": "relative",
            "threshold": {"value_mode": "fixed", "value": 0.2},
            "reference_magnitude_threshold": {"value_mode": "fixed", "value": 10},
        },
        grid=_grid(),
    )

    assert result["reference_magnitude_threshold"] == {
        "value_mode": "fixed",
        "value": 10,
    }


def test_validate_rate_of_change_requires_difference_mode():
    """Require an explicit fixed or relative difference mode."""
    with pytest.raises(ValueError, match="Missing keys"):
        validate_quality_test(
            {
                "name": "large_change",
                "method": "rate_of_change",
                "threshold": {"value_mode": "fixed", "value": 20},
            },
            grid=_grid(),
        )


def test_validate_rate_of_change_rejects_unknown_difference_mode():
    """Reject unsupported rate-of-change difference modes."""
    with pytest.raises(ValueError, match="difference_mode"):
        validate_quality_test(
            {
                "name": "large_change",
                "method": "rate_of_change",
                "difference_mode": "absolute",
                "threshold": {"value_mode": "fixed", "value": 20},
            },
            grid=_grid(),
        )


def test_validate_rate_of_change_rejects_zero_threshold():
    """Require a strictly positive rate-of-change threshold."""
    with pytest.raises(ValueError, match="must be greater than zero"):
        validate_quality_test(
            {
                "name": "large_change",
                "method": "rate_of_change",
                "difference_mode": "fixed",
                "threshold": {"value_mode": "fixed", "value": 0},
            },
            grid=_grid(),
        )


def test_validate_rate_of_change_rejects_negative_threshold():
    """Reject a negative rate-of-change threshold."""
    with pytest.raises(ValueError, match="greater than zero"):
        validate_quality_test(
            {
                "name": "large_change",
                "method": "rate_of_change",
                "difference_mode": "fixed",
                "threshold": {"value_mode": "fixed", "value": -1},
            },
            grid=_grid(),
        )


def test_validate_rate_of_change_rejects_reference_threshold_in_fixed_mode():
    """Do not accept a relative-only reference floor in fixed mode."""
    with pytest.raises(ValueError, match="only supported"):
        validate_quality_test(
            {
                "name": "large_change",
                "method": "rate_of_change",
                "difference_mode": "fixed",
                "threshold": {"value_mode": "fixed", "value": 20},
                "reference_magnitude_threshold": {"value_mode": "fixed", "value": 10},
            },
            grid=_grid(),
        )


def test_validate_rate_of_change_rejects_negative_relative_reference_threshold():
    """Reject a negative relative reference magnitude floor."""
    with pytest.raises(ValueError, match="greater than or equal to zero"):
        validate_quality_test(
            {
                "name": "large_relative_change",
                "method": "rate_of_change",
                "difference_mode": "relative",
                "threshold": {"value_mode": "fixed", "value": 0.2},
                "reference_magnitude_threshold": {"value_mode": "fixed", "value": -1},
            },
            grid=_grid(),
        )


def test_validate_rate_of_change_rejects_failed_period_inclusion():
    """Do not allow prior-failure controls on this direct method."""
    with pytest.raises(ValueError, match="only supported when at least one"):
        validate_quality_test(
            {
                "name": "large_change",
                "method": "rate_of_change",
                "difference_mode": "fixed",
                "threshold": {"value_mode": "fixed", "value": 20},
                "include_failed_periods_from": ["earlier_test"],
            },
            grid=_grid(),
        )


def test_validate_level_shift_normalizes_parameters():
    """Normalize level-shift window duration and threshold."""
    result = validate_quality_test(
        {
            "name": "level_change",
            "method": "level_shift",
            "window_duration": "6h",
            "threshold": {"value_mode": "fixed", "value": 50},
        },
        grid=_grid(),
    )

    assert result["window_duration"] == pd.Timedelta("6h")
    assert result["threshold"] == {"value_mode": "fixed", "value": 50}


def test_validate_level_shift_rejects_single_step_window():
    """Require level-shift windows to contain at least two observations."""
    with pytest.raises(ValueError, match="must span at least two grid steps"):
        validate_quality_test(
            {
                "name": "level_change",
                "method": "level_shift",
                "window_duration": "1h",
                "threshold": {"value_mode": "fixed", "value": 50},
            },
            grid=_grid(),
        )


def test_validate_level_shift_rejects_zero_window():
    """Reject a zero-duration level-shift window."""
    with pytest.raises(ValueError, match="must be greater than zero"):
        validate_quality_test(
            {
                "name": "level_change",
                "method": "level_shift",
                "window_duration": "0h",
                "threshold": {"value_mode": "fixed", "value": 50},
            },
            grid=_grid(),
        )


def test_validate_level_shift_rejects_off_grid_window():
    """Require level-shift windows to align with the grid."""
    with pytest.raises(ValueError, match="integer multiple"):
        validate_quality_test(
            {
                "name": "level_change",
                "method": "level_shift",
                "window_duration": "150min",
                "threshold": {"value_mode": "fixed", "value": 50},
            },
            grid=_grid(),
        )


def test_validate_level_shift_rejects_zero_threshold():
    """Require a strictly positive level-shift threshold."""
    with pytest.raises(ValueError, match="must be greater than zero"):
        validate_quality_test(
            {
                "name": "level_change",
                "method": "level_shift",
                "window_duration": "6h",
                "threshold": {"value_mode": "fixed", "value": 0},
            },
            grid=_grid(),
        )


def test_validate_level_shift_rejects_negative_threshold():
    """Reject a negative level-shift threshold."""
    with pytest.raises(ValueError, match="greater than zero"):
        validate_quality_test(
            {
                "name": "level_change",
                "method": "level_shift",
                "window_duration": "6h",
                "threshold": {"value_mode": "fixed", "value": -1},
            },
            grid=_grid(),
        )


def test_validate_level_shift_rejects_reference_history_option():
    """Do not allow prior-failure controls on this direct method."""
    with pytest.raises(ValueError, match="only supported when at least one"):
        validate_quality_test(
            {
                "name": "level_change",
                "method": "level_shift",
                "window_duration": "6h",
                "threshold": {"value_mode": "fixed", "value": 50},
                "include_failed_periods_from": ["earlier_test"],
            },
            grid=_grid(),
        )


def test_validate_contextual_level_accepts_robust_criterion():
    """Allow contextual-level testing using robust deviation only."""
    result = validate_quality_test(
        {
            "name": "unusual_level",
            "method": "contextual_level",
            "reference_orders": [{"period": "7D", "radius": 4}],
            "robust_deviation_threshold": 6,
        },
        grid=_grid(),
    )

    assert result["robust_deviation_threshold"] == 6
    assert "maximum_predictive_probability" not in result


def test_validate_contextual_level_accepts_predictive_criterion():
    """Allow contextual-level testing using predictive probability only."""
    result = validate_quality_test(
        {
            "name": "unusual_level",
            "method": "contextual_level",
            "reference_orders": [{"period": "7D", "radius": 4}],
            "maximum_predictive_probability": 0.1,
        },
        grid=_grid(),
    )

    assert result["maximum_predictive_probability"] == 0.1
    assert "robust_deviation_threshold" not in result


def test_validate_contextual_level_accepts_both_criteria():
    """Allow both contextual-level criteria to be configured."""
    result = validate_quality_test(
        {
            "name": "unusual_level",
            "method": "contextual_level",
            "reference_orders": [
                {"period": "7D", "radius": 4},
                {"period": "1y", "radius": 2},
            ],
            "robust_deviation_threshold": 6,
            "maximum_predictive_probability": 0.1,
        },
        grid=_grid(),
    )

    assert len(result["reference_orders"]) == 2
    assert result["robust_deviation_threshold"] == 6
    assert result["maximum_predictive_probability"] == 0.1


def test_validate_contextual_level_requires_a_criterion():
    """Require at least one contextual-level failure criterion."""
    with pytest.raises(ValueError, match="must configure at least one"):
        validate_quality_test(
            {
                "name": "unusual_level",
                "method": "contextual_level",
                "reference_orders": [{"period": "7D", "radius": 4}],
            },
            grid=_grid(),
        )


@pytest.mark.parametrize("threshold", [0, -1])
def test_validate_contextual_level_requires_positive_robust_threshold(threshold):
    """Require a strictly positive robust-deviation threshold."""
    with pytest.raises(ValueError, match="greater than zero"):
        validate_quality_test(
            {
                "name": "unusual_level",
                "method": "contextual_level",
                "reference_orders": [{"period": "7D", "radius": 4}],
                "robust_deviation_threshold": threshold,
            },
            grid=_grid(),
        )


@pytest.mark.parametrize("probability", [0, -0.1, 1, 1.1])
def test_validate_contextual_level_requires_probability_between_zero_and_one(
    probability,
):
    """Require predictive probability strictly between zero and one."""
    with pytest.raises(ValueError, match="greater than zero and less than one"):
        validate_quality_test(
            {
                "name": "unusual_level",
                "method": "contextual_level",
                "reference_orders": [{"period": "7D", "radius": 4}],
                "maximum_predictive_probability": probability,
            },
            grid=_grid(),
        )


def test_validate_contextual_level_accepts_failed_period_inclusion():
    """Allow contextual-level references to retain named prior failures."""
    result = validate_quality_tests(
        [
            {
                "name": "negative",
                "method": "range",
                "minimum": {"value_mode": "fixed", "value": 0},
            },
            {
                "name": "unusual_level",
                "method": "contextual_level",
                "reference_orders": [{"period": "7D", "radius": 4}],
                "robust_deviation_threshold": 6,
                "include_failed_periods_from": ["negative"],
            },
        ],
        grid=_grid(),
    )

    assert result[1]["include_failed_periods_from"] == ["negative"]


def test_validate_contextual_profile_normalizes_configuration():
    """Normalize contextual-profile duration, offset, lattice, and criteria."""
    result = validate_quality_test(
        {
            "name": "unusual_daily_shape",
            "method": "contextual_profile",
            "profile_duration": "4h",
            "profile_offset": "1h",
            "reference_orders": [{"period": "8h", "radius": 2}],
            "robust_deviation_threshold": 6,
            "maximum_predictive_probability": 0.1,
        },
        grid=_grid(),
    )

    assert result["profile_duration"] == pd.Timedelta("4h")
    assert result["profile_offset"] == pd.Timedelta("1h")
    assert result["robust_deviation_threshold"] == 6
    assert result["maximum_predictive_probability"] == 0.1
    assert len(result["reference_orders"]) == 1


def test_validate_contextual_profile_defaults_zero_offset():
    """Align profiles to the grid start when no offset is configured."""
    result = validate_quality_test(
        {
            "name": "unusual_shape",
            "method": "contextual_profile",
            "profile_duration": "4h",
            "reference_orders": [{"period": "8h", "radius": 2}],
            "robust_deviation_threshold": 6,
        },
        grid=_grid(),
    )

    assert result["profile_offset"] == pd.Timedelta(0)


def test_validate_contextual_profile_accepts_predictive_only():
    """Allow the rank-based predictive criterion without robust evidence."""
    result = validate_quality_test(
        {
            "name": "unusual_shape",
            "method": "contextual_profile",
            "profile_duration": "4h",
            "reference_orders": [{"period": "8h", "radius": 2}],
            "maximum_predictive_probability": 0.1,
        },
        grid=_grid(),
    )

    assert "robust_deviation_threshold" not in result
    assert result["maximum_predictive_probability"] == 0.1


def test_validate_contextual_profile_requires_criterion():
    """Require at least one contextual-profile anomaly criterion."""
    with pytest.raises(ValueError, match="at least one"):
        validate_quality_test(
            {
                "name": "unusual_shape",
                "method": "contextual_profile",
                "profile_duration": "4h",
                "reference_orders": [{"period": "8h", "radius": 2}],
            },
            grid=_grid(),
        )


def test_validate_contextual_profile_requires_two_step_duration():
    """Require enough values for a profile shape."""
    with pytest.raises(ValueError, match="at least two grid steps"):
        validate_quality_test(
            {
                "name": "unusual_shape",
                "method": "contextual_profile",
                "profile_duration": "1h",
                "reference_orders": [{"period": "8h", "radius": 2}],
                "robust_deviation_threshold": 6,
            },
            grid=_grid(),
        )


def test_validate_contextual_profile_rejects_off_grid_duration():
    """Require profile duration to contain complete grid steps."""
    with pytest.raises(ValueError, match="integer multiple"):
        validate_quality_test(
            {
                "name": "unusual_shape",
                "method": "contextual_profile",
                "profile_duration": "150min",
                "reference_orders": [{"period": "8h", "radius": 2}],
                "robust_deviation_threshold": 6,
            },
            grid=_grid(),
        )


def test_validate_contextual_profile_rejects_negative_offset():
    """Require a canonical non-negative profile offset."""
    with pytest.raises(ValueError, match="greater than or equal to zero"):
        validate_quality_test(
            {
                "name": "unusual_shape",
                "method": "contextual_profile",
                "profile_duration": "4h",
                "profile_offset": "-1h",
                "reference_orders": [{"period": "8h", "radius": 2}],
                "robust_deviation_threshold": 6,
            },
            grid=_grid(),
        )


def test_validate_contextual_profile_rejects_off_grid_offset():
    """Require profile offset to align with the configured frequency."""
    with pytest.raises(ValueError, match="integer multiple"):
        validate_quality_test(
            {
                "name": "unusual_shape",
                "method": "contextual_profile",
                "profile_duration": "4h",
                "profile_offset": "30min",
                "reference_orders": [{"period": "8h", "radius": 2}],
                "robust_deviation_threshold": 6,
            },
            grid=_grid(),
        )


def test_validate_contextual_profile_requires_offset_below_duration():
    """Keep equivalent profile phases in one canonical offset range."""
    with pytest.raises(ValueError, match="less than 'profile_duration'"):
        validate_quality_test(
            {
                "name": "unusual_shape",
                "method": "contextual_profile",
                "profile_duration": "4h",
                "profile_offset": "4h",
                "reference_orders": [{"period": "8h", "radius": 2}],
                "robust_deviation_threshold": 6,
            },
            grid=_grid(),
        )


@pytest.mark.parametrize("threshold", [0, -1])
def test_validate_contextual_profile_requires_positive_robust_threshold(threshold):
    """Require a positive robust profile-deviation threshold."""
    with pytest.raises(ValueError, match="greater than zero"):
        validate_quality_test(
            {
                "name": "unusual_shape",
                "method": "contextual_profile",
                "profile_duration": "4h",
                "reference_orders": [{"period": "8h", "radius": 2}],
                "robust_deviation_threshold": threshold,
            },
            grid=_grid(),
        )


@pytest.mark.parametrize("probability", [0, 1, -0.1, 1.1])
def test_validate_contextual_profile_requires_probability_strictly_between_zero_and_one(
    probability,
):
    """Require a proper predictive-probability threshold."""
    with pytest.raises(ValueError, match="greater than zero and less than one"):
        validate_quality_test(
            {
                "name": "unusual_shape",
                "method": "contextual_profile",
                "profile_duration": "4h",
                "reference_orders": [{"period": "8h", "radius": 2}],
                "maximum_predictive_probability": probability,
            },
            grid=_grid(),
        )


def test_validate_contextual_profile_accepts_prior_failure_inclusion():
    """Allow reference-aware profile tests to restore named prior failures."""
    result = validate_quality_tests(
        [
            {
                "name": "plausible_range",
                "method": "range",
                "minimum": {"value_mode": "fixed", "value": 0},
            },
            {
                "name": "unusual_shape",
                "method": "contextual_profile",
                "profile_duration": "4h",
                "reference_orders": [{"period": "8h", "radius": 2}],
                "robust_deviation_threshold": 6,
                "include_failed_periods_from": ["plausible_range"],
            },
        ],
        grid=_grid(),
    )

    assert result[1]["include_failed_periods_from"] == ["plausible_range"]
