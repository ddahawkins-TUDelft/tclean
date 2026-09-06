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
        start="2026-01-01T00:00:00Z",
        end="2026-01-03T00:00:00Z",
        frequency=frequency,
    )


def test_validate_range_test_accepts_both_bounds():
    """Accept a range test with minimum and maximum bounds."""
    result = validate_quality_test(
        {
            "name": "plausible_values",
            "method": "range",
            "minimum": 0,
            "maximum": 1000,
        },
        grid=_grid(),
    )

    assert result == {
        "name": "plausible_values",
        "method": "range",
        "minimum": 0,
        "maximum": 1000,
    }


def test_validate_range_test_accepts_minimum_only():
    """Accept a range test with only a lower bound."""
    result = validate_quality_test(
        {
            "name": "non_negative",
            "method": "range",
            "minimum": 0,
        },
        grid=_grid(),
    )

    assert result["minimum"] == 0
    assert "maximum" not in result


def test_validate_range_test_accepts_maximum_only():
    """Accept a range test with only an upper bound."""
    result = validate_quality_test(
        {
            "name": "upper_bound",
            "method": "range",
            "maximum": 1000.5,
        },
        grid=_grid(),
    )

    assert result["maximum"] == 1000.5
    assert "minimum" not in result


def test_validate_range_test_accepts_equal_bounds():
    """Allow a range test whose minimum equals its maximum."""
    result = validate_quality_test(
        {
            "name": "exact_value",
            "method": "range",
            "minimum": 5,
            "maximum": 5,
        },
        grid=_grid(),
    )

    assert result["minimum"] == 5
    assert result["maximum"] == 5


def test_validate_range_test_normalizes_selectors():
    """Normalize source and context selectors to lists."""
    result = validate_quality_test(
        {
            "name": "selected_range",
            "method": "range",
            "sources": ("entsoe", "opsd"),
            "contexts": ("ALB", "BIH"),
            "minimum": 0,
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
            "minimum": 0,
        },
        {
            "name": "second",
            "method": "range",
            "maximum": 1000,
        },
    ]

    result = validate_quality_tests(tests, grid=_grid())

    assert [test["name"] for test in result] == [
        "first",
        "second",
    ]


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
            "minimum": 0,
        },
        {
            "name": "duplicate",
            "method": "range",
            "maximum": 1000,
        },
    ]

    with pytest.raises(ValueError, match="must be unique"):
        validate_quality_tests(tests, grid=_grid())


def test_validate_quality_tests_rejects_non_sequence():
    """Reject unordered or non-sequence quality-test collections."""
    tests = {
        "name": "non_negative",
        "method": "range",
        "minimum": 0,
    }

    with pytest.raises(TypeError, match="ordered sequence"):
        validate_quality_tests(tests, grid=_grid())


def test_validate_quality_test_rejects_non_mapping():
    """Reject individual tests that are not mappings."""
    with pytest.raises(TypeError, match="must be a mapping"):
        validate_quality_test(
            ["non_negative", "range", 0],
            grid=_grid(),
        )


def test_validate_quality_test_rejects_blank_name():
    """Reject blank data-quality test names."""
    with pytest.raises(ValueError, match="'name' must be a non-empty string"):
        validate_quality_test(
            {
                "name": "   ",
                "method": "range",
                "minimum": 0,
            },
            grid=_grid(),
        )


def test_validate_quality_test_rejects_blank_method():
    """Reject blank data-quality method names."""
    with pytest.raises(ValueError, match="'method' must be a non-empty string"):
        validate_quality_test(
            {
                "name": "non_negative",
                "method": "",
                "minimum": 0,
            },
            grid=_grid(),
        )


def test_validate_quality_test_rejects_unknown_method():
    """Reject unsupported data-quality methods."""
    with pytest.raises(
        ValueError,
        match="Unsupported data-quality method",
    ):
        validate_quality_test(
            {
                "name": "unknown",
                "method": "something_else",
                "minimum": 0,
            },
            grid=_grid(),
        )


def test_validate_range_test_rejects_unknown_argument():
    """Reject arguments unsupported by the range method."""
    with pytest.raises(ValueError, match="unknown keys"):
        validate_quality_test(
            {
                "name": "non_negative",
                "method": "range",
                "minimum": 0,
                "tolerance": 0.1,
            },
            grid=_grid(),
        )


def test_validate_range_test_requires_a_bound():
    """Require at least one range boundary."""
    with pytest.raises(
        ValueError,
        match="at least one of 'minimum' or 'maximum'",
    ):
        validate_quality_test(
            {
                "name": "unbounded",
                "method": "range",
            },
            grid=_grid(),
        )


def test_validate_range_test_rejects_reversed_bounds():
    """Reject a minimum greater than the maximum."""
    with pytest.raises(
        ValueError,
        match="less than or equal",
    ):
        validate_quality_test(
            {
                "name": "invalid_range",
                "method": "range",
                "minimum": 10,
                "maximum": 5,
            },
            grid=_grid(),
        )


@pytest.mark.parametrize(
    "value",
    [
        "zero",
        None,
        True,
        pd.NA,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_validate_range_test_rejects_invalid_minimum(value):
    """Reject non-numeric or non-finite lower bounds."""
    with pytest.raises(
        ValueError,
        match="finite real number",
    ):
        validate_quality_test(
            {
                "name": "invalid_minimum",
                "method": "range",
                "minimum": value,
            },
            grid=_grid(),
        )


@pytest.mark.parametrize(
    "value",
    [
        "one_thousand",
        None,
        False,
        pd.NA,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_validate_range_test_rejects_invalid_maximum(value):
    """Reject non-numeric or non-finite upper bounds."""
    with pytest.raises(
        ValueError,
        match="finite real number",
    ):
        validate_quality_test(
            {
                "name": "invalid_maximum",
                "method": "range",
                "maximum": value,
            },
            grid=_grid(),
        )


@pytest.mark.parametrize(
    "field",
    ["sources", "contexts"],
)
def test_validate_quality_test_rejects_string_selector(field):
    """Reject a string where an ordered selector sequence is required."""
    test = {
        "name": "selected_range",
        "method": "range",
        "minimum": 0,
        field: "entsoe",
    }

    with pytest.raises(
        ValueError,
        match="non-empty ordered sequence",
    ):
        validate_quality_test(test, grid=_grid())


@pytest.mark.parametrize(
    "field",
    ["sources", "contexts"],
)
def test_validate_quality_test_rejects_empty_selector(field):
    """Reject an explicitly empty source or context selector."""
    test = {
        "name": "selected_range",
        "method": "range",
        "minimum": 0,
        field: [],
    }

    with pytest.raises(
        ValueError,
        match="non-empty ordered sequence",
    ):
        validate_quality_test(test, grid=_grid())


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("sources", ["entsoe", 123]),
        ("contexts", ["ALB", None]),
    ],
)
def test_validate_quality_test_rejects_invalid_selector_entries(
    field,
    values,
):
    """Reject source or context selectors containing invalid entries."""
    test = {
        "name": "selected_range",
        "method": "range",
        "minimum": 0,
        field: values,
    }

    with pytest.raises(
        ValueError,
        match="only non-empty strings",
    ):
        validate_quality_test(test, grid=_grid())


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("sources", ["entsoe", "entsoe"]),
        ("contexts", ["ALB", "ALB"]),
    ],
)
def test_validate_quality_test_rejects_duplicate_selector_entries(
    field,
    values,
):
    """Reject duplicate entries in source or context selectors."""
    test = {
        "name": "selected_range",
        "method": "range",
        "minimum": 0,
        field: values,
    }

    with pytest.raises(ValueError, match="entries must be unique"):
        validate_quality_test(test, grid=_grid())


def test_validate_range_test_rejects_failed_period_inclusion():
    """Reject reference-data controls for methods that do not use them."""
    with pytest.raises(ValueError, match="unknown keys"):
        validate_quality_test(
            {
                "name": "non_negative",
                "method": "range",
                "minimum": 0,
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
            "value": 0,
            "minimum_duration": "3h",
        },
        grid=_grid(),
    )

    assert result["value"] == 0
    assert result["minimum_duration"] == pd.Timedelta("3h")
    assert result["tolerance"] == 0.0


def test_validate_value_run_accepts_positive_tolerance():
    """Accept a finite non-negative value-run tolerance."""
    result = validate_quality_test(
        {
            "name": "near_zero_run",
            "method": "value_run",
            "value": 0,
            "minimum_duration": "3h",
            "tolerance": 0.1,
        },
        grid=_grid(),
    )

    assert result["tolerance"] == 0.1


def test_validate_value_run_rejects_zero_duration():
    """Reject a zero minimum value-run duration."""
    with pytest.raises(
        ValueError,
        match="must be greater than zero",
    ):
        validate_quality_test(
            {
                "name": "zero_run",
                "method": "value_run",
                "value": 0,
                "minimum_duration": "0h",
            },
            grid=_grid(),
        )


def test_validate_value_run_rejects_off_grid_duration():
    """Require minimum duration to align with the configured grid."""
    with pytest.raises(
        ValueError,
        match="integer multiple",
    ):
        validate_quality_test(
            {
                "name": "zero_run",
                "method": "value_run",
                "value": 0,
                "minimum_duration": "90min",
            },
            grid=_grid(),
        )


def test_validate_value_run_rejects_negative_tolerance():
    """Reject negative value-run tolerance."""
    with pytest.raises(
        ValueError,
        match="greater than or equal to zero",
    ):
        validate_quality_test(
            {
                "name": "zero_run",
                "method": "value_run",
                "value": 0,
                "minimum_duration": "3h",
                "tolerance": -0.1,
            },
            grid=_grid(),
        )


def test_validate_value_run_rejects_non_numeric_value():
    """Require the target value to be numeric and finite."""
    with pytest.raises(
        ValueError,
        match="finite real number",
    ):
        validate_quality_test(
            {
                "name": "bad_value",
                "method": "value_run",
                "value": "zero",
                "minimum_duration": "3h",
            },
            grid=_grid(),
        )


def test_validate_value_run_rejects_failed_period_inclusion():
    """Do not allow reference controls on a non-reference-aware method."""
    with pytest.raises(ValueError, match="unknown keys"):
        validate_quality_test(
            {
                "name": "zero_run",
                "method": "value_run",
                "value": 0,
                "minimum_duration": "3h",
                "include_failed_periods_from": ["earlier_test"],
            },
            grid=_grid(),
        )


def test_validate_flatline_normalizes_duration_and_default_tolerance():
    """Normalize flatline duration and supply zero tolerance by default."""
    result = validate_quality_test(
        {
            "name": "constant_values",
            "method": "flatline",
            "minimum_duration": "3h",
        },
        grid=_grid(),
    )

    assert result["minimum_duration"] == pd.Timedelta("3h")
    assert result["tolerance"] == 0.0


def test_validate_flatline_accepts_positive_tolerance():
    """Accept a finite non-negative flatline tolerance."""
    result = validate_quality_test(
        {
            "name": "nearly_constant",
            "method": "flatline",
            "minimum_duration": "3h",
            "tolerance": 0.1,
        },
        grid=_grid(),
    )

    assert result["tolerance"] == 0.1


def test_validate_flatline_rejects_single_step_duration():
    """Require a flatline to contain at least two observations."""
    with pytest.raises(
        ValueError,
        match="at least two grid steps",
    ):
        validate_quality_test(
            {
                "name": "constant_values",
                "method": "flatline",
                "minimum_duration": "1h",
            },
            grid=_grid(),
        )


def test_validate_flatline_rejects_zero_duration():
    """Reject a zero flatline duration."""
    with pytest.raises(
        ValueError,
        match="must be greater than zero",
    ):
        validate_quality_test(
            {
                "name": "constant_values",
                "method": "flatline",
                "minimum_duration": "0h",
            },
            grid=_grid(),
        )


def test_validate_flatline_rejects_off_grid_duration():
    """Require flatline duration to align with the configured grid."""
    with pytest.raises(
        ValueError,
        match="integer multiple",
    ):
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
    with pytest.raises(
        ValueError,
        match="greater than or equal to zero",
    ):
        validate_quality_test(
            {
                "name": "constant_values",
                "method": "flatline",
                "minimum_duration": "3h",
                "tolerance": -0.1,
            },
            grid=_grid(),
        )


def test_validate_flatline_rejects_failed_period_inclusion():
    """Do not allow reference controls on a non-reference-aware method."""
    with pytest.raises(ValueError, match="unknown keys"):
        validate_quality_test(
            {
                "name": "constant_values",
                "method": "flatline",
                "minimum_duration": "3h",
                "include_failed_periods_from": ["earlier_test"],
            },
            grid=_grid(),
        )

