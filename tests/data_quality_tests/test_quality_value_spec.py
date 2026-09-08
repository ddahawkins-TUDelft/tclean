"""Tests for configurable data-quality value specifications."""

import math

import pandas as pd
import pytest
from _method_helpers import method_context

from tclean import TimeGrid
from tclean.data_quality._value_spec import (
    normalize_value_spec,
    require_fixed_value_spec,
    resolve_context_value,
    resolve_value_spec,
)


def _series(values):
    """Build a simple float series for value-resolution tests."""
    return pd.Series(values, dtype=float)


def test_normalize_fixed_value():
    """Normalize an explicitly fixed scalar value."""
    result = normalize_value_spec(
        {"value_mode": "fixed", "value": 5000}, field="threshold"
    )

    assert result == {"value_mode": "fixed", "value": 5000.0}


@pytest.mark.parametrize(
    "value_mode",
    [
        "mean",
        "median",
        "standard_deviation",
        "mean_absolute_increment",
        "median_absolute_increment",
    ],
)
def test_normalize_simple_derived_modes_default_multiplier(value_mode):
    """Default multiplier to one for simple derived modes."""
    result = normalize_value_spec({"value_mode": value_mode}, field="threshold")

    assert result == {"value_mode": value_mode, "multiplier": 1.0}


def test_normalize_simple_derived_mode_accepts_multiplier():
    """Normalize an explicit multiplier for a derived value."""
    result = normalize_value_spec(
        {"value_mode": "median", "multiplier": 0.1}, field="threshold"
    )

    assert result == {"value_mode": "median", "multiplier": 0.1}


def test_normalize_quantile():
    """Normalize a quantile-derived value."""
    result = normalize_value_spec(
        {"value_mode": "quantile", "quantile": 0.95, "multiplier": 0.1},
        field="threshold",
    )

    assert result == {"value_mode": "quantile", "quantile": 0.95, "multiplier": 0.1}


def test_normalize_quantile_defaults_multiplier():
    """Default quantile multiplier to one."""
    result = normalize_value_spec(
        {"value_mode": "quantile", "quantile": 0.95}, field="threshold"
    )

    assert result["multiplier"] == 1.0


def test_normalize_quantile_range():
    """Normalize a quantile-range-derived value."""
    result = normalize_value_spec(
        {
            "value_mode": "quantile_range",
            "lower_quantile": 0.05,
            "upper_quantile": 0.95,
            "multiplier": 0.2,
        },
        field="threshold",
    )

    assert result == {
        "value_mode": "quantile_range",
        "lower_quantile": 0.05,
        "upper_quantile": 0.95,
        "multiplier": 0.2,
    }


def test_normalize_requires_mapping():
    """Reject bare numeric values now that value_mode is explicit."""
    with pytest.raises(ValueError, match="value specification"):
        normalize_value_spec(5000, field="threshold")


def test_normalize_requires_value_mode():
    """Require an explicit value mode."""
    with pytest.raises(ValueError, match="value_mode"):
        normalize_value_spec({"value": 5000}, field="threshold")


def test_normalize_rejects_unknown_value_mode():
    """Reject unsupported value modes."""
    with pytest.raises(ValueError, match="value_mode"):
        normalize_value_spec({"value_mode": "range"}, field="threshold")


def test_fixed_requires_value():
    """Require a value for fixed mode."""
    with pytest.raises(ValueError, match="Missing keys"):
        normalize_value_spec({"value_mode": "fixed"}, field="threshold")


@pytest.mark.parametrize(
    "extra",
    [
        {"multiplier": 2},
        {"quantile": 0.95},
        {"lower_quantile": 0.05},
        {"upper_quantile": 0.95},
    ],
)
def test_fixed_rejects_derived_fields(extra):
    """Reject derived-value arguments in fixed mode."""
    spec = {"value_mode": "fixed", "value": 5000, **extra}

    with pytest.raises(ValueError, match="unknown keys"):
        normalize_value_spec(spec, field="threshold")


@pytest.mark.parametrize(
    "value_mode",
    [
        "mean",
        "median",
        "standard_deviation",
        "mean_absolute_increment",
        "median_absolute_increment",
    ],
)
def test_simple_derived_modes_reject_irrelevant_fields(value_mode):
    """Reject fields that do not belong to simple derived modes."""
    with pytest.raises(ValueError, match="unknown keys"):
        normalize_value_spec(
            {"value_mode": value_mode, "quantile": 0.95}, field="threshold"
        )


def test_quantile_requires_quantile():
    """Require the quantile argument for quantile mode."""
    with pytest.raises(ValueError, match="Missing keys"):
        normalize_value_spec({"value_mode": "quantile"}, field="threshold")


@pytest.mark.parametrize("quantile", [-0.01, 1.01])
def test_quantile_rejects_values_outside_unit_interval(quantile):
    """Require quantiles between zero and one inclusively."""
    with pytest.raises(ValueError, match="between zero and one"):
        normalize_value_spec(
            {"value_mode": "quantile", "quantile": quantile}, field="threshold"
        )


@pytest.mark.parametrize("quantile", [0.0, 1.0])
def test_quantile_accepts_interval_boundaries(quantile):
    """Allow quantiles at zero and one."""
    result = normalize_value_spec(
        {"value_mode": "quantile", "quantile": quantile}, field="threshold"
    )

    assert result["quantile"] == quantile


def test_quantile_rejects_range_fields():
    """Reject quantile-range arguments in single-quantile mode."""
    with pytest.raises(ValueError, match="unknown keys"):
        normalize_value_spec(
            {"value_mode": "quantile", "quantile": 0.95, "lower_quantile": 0.05},
            field="threshold",
        )


@pytest.mark.parametrize(
    "spec",
    [
        {"value_mode": "quantile_range", "upper_quantile": 0.95},
        {"value_mode": "quantile_range", "lower_quantile": 0.05},
    ],
)
def test_quantile_range_requires_both_bounds(spec):
    """Require both lower and upper quantiles."""
    with pytest.raises(ValueError, match="Missing keys"):
        normalize_value_spec(spec, field="threshold")


@pytest.mark.parametrize(("lower", "upper"), [(-0.01, 0.95), (0.05, 1.01)])
def test_quantile_range_rejects_bounds_outside_unit_interval(lower, upper):
    """Require quantile-range bounds inside the unit interval."""
    with pytest.raises(ValueError, match="between zero and one"):
        normalize_value_spec(
            {
                "value_mode": "quantile_range",
                "lower_quantile": lower,
                "upper_quantile": upper,
            },
            field="threshold",
        )


@pytest.mark.parametrize(("lower", "upper"), [(0.5, 0.5), (0.75, 0.25)])
def test_quantile_range_requires_increasing_bounds(lower, upper):
    """Require the lower quantile to precede the upper quantile."""
    with pytest.raises(ValueError, match="less than"):
        normalize_value_spec(
            {
                "value_mode": "quantile_range",
                "lower_quantile": lower,
                "upper_quantile": upper,
            },
            field="threshold",
        )


def test_quantile_range_rejects_single_quantile_field():
    """Reject the single-quantile argument in quantile-range mode."""
    with pytest.raises(ValueError, match="unknown keys"):
        normalize_value_spec(
            {
                "value_mode": "quantile_range",
                "lower_quantile": 0.05,
                "upper_quantile": 0.95,
                "quantile": 0.5,
            },
            field="threshold",
        )


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ({"value_mode": "fixed", "value": float("nan")}, "finite"),
        ({"value_mode": "fixed", "value": float("inf")}, "finite"),
        ({"value_mode": "median", "multiplier": float("nan")}, "finite"),
        ({"value_mode": "quantile", "quantile": float("inf")}, "finite"),
    ],
)
def test_normalize_rejects_nonfinite_numbers(spec, match):
    """Reject non-finite numeric configuration values."""
    with pytest.raises(ValueError, match=match):
        normalize_value_spec(spec, field="threshold")


def test_resolve_fixed_without_observed_data():
    """Resolve a fixed value without requiring focal observations."""
    spec = normalize_value_spec({"value_mode": "fixed", "value": 25}, field="threshold")

    result = resolve_value_spec(spec, data=_series([None, None]))

    assert result.evaluable
    assert result.value == 25.0
    assert result.property_value is None
    assert result.eligible_observations is None
    assert result.eligible_increments is None


def test_resolve_mean():
    """Resolve the mean of eligible observations."""
    result = resolve_value_spec(
        normalize_value_spec(
            {"value_mode": "mean", "multiplier": 2}, field="threshold"
        ),
        data=_series([1, 2, None, 5]),
    )

    assert result.evaluable
    assert result.property_value == pytest.approx(8 / 3)
    assert result.value == pytest.approx(16 / 3)
    assert result.eligible_observations == 3


def test_resolve_median():
    """Resolve the median of eligible observations."""
    result = resolve_value_spec(
        normalize_value_spec({"value_mode": "median"}, field="threshold"),
        data=_series([1, 100, 3, None]),
    )

    assert result.evaluable
    assert result.property_value == 3.0
    assert result.value == 3.0
    assert result.eligible_observations == 3


def test_resolve_quantile():
    """Resolve a selected quantile."""
    result = resolve_value_spec(
        normalize_value_spec(
            {"value_mode": "quantile", "quantile": 0.5}, field="threshold"
        ),
        data=_series([0, 10, 20, 30, 40]),
    )

    assert result.evaluable
    assert result.property_value == 20.0
    assert result.value == 20.0


def test_resolve_quantile_one_is_maximum():
    """Allow the maximum to be expressed as the 100th quantile."""
    result = resolve_value_spec(
        normalize_value_spec(
            {"value_mode": "quantile", "quantile": 1.0}, field="threshold"
        ),
        data=_series([4, 10, 7, None]),
    )

    assert result.value == 10.0


def test_resolve_quantile_zero_is_minimum():
    """Allow the minimum to be expressed as the zeroth quantile."""
    result = resolve_value_spec(
        normalize_value_spec(
            {"value_mode": "quantile", "quantile": 0.0}, field="threshold"
        ),
        data=_series([4, 10, 7, None]),
    )

    assert result.value == 4.0


def test_resolve_quantile_range():
    """Resolve the difference between upper and lower quantiles."""
    result = resolve_value_spec(
        normalize_value_spec(
            {
                "value_mode": "quantile_range",
                "lower_quantile": 0.25,
                "upper_quantile": 0.75,
            },
            field="threshold",
        ),
        data=_series([0, 10, 20, 30, 40]),
    )

    assert result.evaluable
    assert result.property_value == 20.0
    assert result.value == 20.0


def test_resolve_quantile_range_can_reproduce_iqr():
    """Represent the interquartile range through quantile_range."""
    data = _series([0, 10, 20, 30, 40])

    result = resolve_value_spec(
        normalize_value_spec(
            {
                "value_mode": "quantile_range",
                "lower_quantile": 0.25,
                "upper_quantile": 0.75,
            },
            field="threshold",
        ),
        data=data,
    )

    expected = data.quantile(0.75) - data.quantile(0.25)

    assert result.value == pytest.approx(expected)


def test_resolve_quantile_range_can_reproduce_full_range():
    """Represent the raw range through the zeroth and 100th quantiles."""
    result = resolve_value_spec(
        normalize_value_spec(
            {
                "value_mode": "quantile_range",
                "lower_quantile": 0.0,
                "upper_quantile": 1.0,
            },
            field="threshold",
        ),
        data=_series([4, 10, 7]),
    )

    assert result.value == 6.0


def test_resolve_standard_deviation_uses_population_definition():
    """Use population standard deviation with ddof zero."""
    data = _series([1, 2, 3, 4])

    result = resolve_value_spec(
        normalize_value_spec({"value_mode": "standard_deviation"}, field="threshold"),
        data=data,
    )

    assert result.evaluable
    assert result.property_value == pytest.approx(data.std(ddof=0))
    assert result.value == pytest.approx(data.std(ddof=0))


def test_resolve_mean_absolute_increment():
    """Resolve the mean absolute adjacent-grid increment."""
    result = resolve_value_spec(
        normalize_value_spec(
            {"value_mode": "mean_absolute_increment"}, field="threshold"
        ),
        data=_series([10, 15, 25, 20]),
    )

    assert result.evaluable
    assert result.property_value == pytest.approx((5 + 10 + 5) / 3)
    assert result.value == pytest.approx((5 + 10 + 5) / 3)
    assert result.eligible_observations == 4
    assert result.eligible_increments == 3


def test_resolve_median_absolute_increment():
    """Resolve the median absolute adjacent-grid increment."""
    result = resolve_value_spec(
        normalize_value_spec(
            {"value_mode": "median_absolute_increment"}, field="threshold"
        ),
        data=_series([10, 12, 20, 21]),
    )

    assert result.evaluable
    assert result.property_value == 2.0
    assert result.value == 2.0
    assert result.eligible_increments == 3


def test_absolute_increments_do_not_bridge_missing_values():
    """Do not construct artificial increments across missing observations."""
    result = resolve_value_spec(
        normalize_value_spec(
            {"value_mode": "mean_absolute_increment"}, field="threshold"
        ),
        data=_series([10, 15, None, 20, 28]),
    )

    assert result.evaluable
    assert result.property_value == pytest.approx((5 + 8) / 2)
    assert result.eligible_observations == 4
    assert result.eligible_increments == 2


def test_all_missing_data_is_not_evaluable_for_derived_mode():
    """Report an unresolved derived value when no eligible data exist."""
    result = resolve_value_spec(
        normalize_value_spec({"value_mode": "median"}, field="threshold"),
        data=_series([None, None, None]),
    )

    assert not result.evaluable
    assert result.value is None
    assert result.property_value is None
    assert result.eligible_observations == 0
    assert result.reason == "no_eligible_observations"


def test_increment_mode_requires_at_least_one_adjacent_pair():
    """Report unresolved increment statistics when no adjacent pair exists."""
    result = resolve_value_spec(
        normalize_value_spec(
            {"value_mode": "median_absolute_increment"}, field="threshold"
        ),
        data=_series([10, None, 20, None, 30]),
    )

    assert not result.evaluable
    assert result.value is None
    assert result.property_value is None
    assert result.eligible_observations == 3
    assert result.eligible_increments == 0
    assert result.reason == "no_eligible_increments"


def test_zero_property_value_is_a_successful_generic_resolution():
    """Allow generic resolution to return zero for a constant series."""
    result = resolve_value_spec(
        normalize_value_spec({"value_mode": "standard_deviation"}, field="threshold"),
        data=_series([5, 5, 5, 5]),
    )

    assert result.evaluable
    assert result.value == 0.0
    assert result.property_value == 0.0
    assert result.reason is None


def test_multiplier_is_applied_after_property_calculation():
    """Apply the configured multiplier to the calculated property."""
    result = resolve_value_spec(
        normalize_value_spec(
            {"value_mode": "median", "multiplier": 0.1}, field="threshold"
        ),
        data=_series([100, 200, 300]),
    )

    assert result.property_value == 200.0
    assert result.value == 20.0


def test_resolution_values_are_finite_when_evaluable():
    """Ensure evaluable resolutions expose finite values."""
    result = resolve_value_spec(
        normalize_value_spec({"value_mode": "mean"}, field="threshold"),
        data=_series([1, 2, 3]),
    )

    assert result.evaluable
    assert math.isfinite(result.value)
    assert math.isfinite(result.property_value)


def _grid() -> TimeGrid:
    """Return a four-hour grid for context-resolution tests."""
    return TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-01T04:00:00Z", frequency="1h"
    )


def test_resolve_context_value_is_independent_per_source_and_context():
    """Resolve derived values independently for each focal source-context."""
    index = _grid().target_index
    sources = {
        "primary": pd.DataFrame(
            {"A": [10, 20, 30, 40], "B": [100, 200, 300, 400]}, index=index
        ),
        "secondary": pd.DataFrame(
            {"A": [1, 2, 3, 4], "B": [1000, 2000, 3000, 4000]}, index=index
        ),
    }
    test = {"name": "derived", "method": "range", "maximum": {"value_mode": "median"}}

    primary = method_context(
        sources["primary"],
        source_name="primary",
        sources=sources,
        test=test,
        grid=_grid(),
    )
    secondary = method_context(
        sources["secondary"],
        source_name="secondary",
        sources=sources,
        test=test,
        grid=_grid(),
    )

    assert resolve_context_value(primary, field="maximum", context_name="A").value == 25
    assert (
        resolve_context_value(primary, field="maximum", context_name="B").value == 250
    )
    assert (
        resolve_context_value(secondary, field="maximum", context_name="A").value == 2.5
    )
    assert (
        resolve_context_value(secondary, field="maximum", context_name="B").value
        == 2500
    )


def test_resolve_context_value_excludes_preceding_failures():
    """Derive values from failure-filtered focal evidence by default."""
    index = _grid().target_index
    data = pd.DataFrame({"A": [1, 100, 3, 5]}, index=index)
    preceding_failures = (
        {
            "context": "A",
            "source": "primary",
            "start": index[1],
            "end": index[2],
            "test_name": "earlier",
            "method": "range",
            "details": {},
        },
    )
    context = method_context(
        data,
        test={
            "name": "derived",
            "method": "range",
            "maximum": {"value_mode": "median"},
        },
        grid=_grid(),
        preceding_failures=preceding_failures,
    )

    result = resolve_context_value(context, field="maximum", context_name="A")

    assert result.value == 3
    assert result.eligible_observations == 3


def test_resolve_context_value_can_reinclude_named_preceding_failures():
    """Restore named failed periods when deriving a configured value."""
    index = _grid().target_index
    data = pd.DataFrame({"A": [1, 100, 3, 5]}, index=index)
    preceding_failures = (
        {
            "context": "A",
            "source": "primary",
            "start": index[1],
            "end": index[2],
            "test_name": "earlier",
            "method": "range",
            "details": {},
        },
    )
    context = method_context(
        data,
        test={
            "name": "derived",
            "method": "range",
            "maximum": {"value_mode": "median"},
            "include_failed_periods_from": ["earlier"],
        },
        grid=_grid(),
        preceding_failures=preceding_failures,
    )

    result = resolve_context_value(context, field="maximum", context_name="A")

    assert result.value == 4
    assert result.eligible_observations == 4


def test_require_fixed_value_spec_rejects_derived_dimensionless_value():
    """Reject a derived mode where a consumer requires a dimensionless literal."""
    with pytest.raises(ValueError, match="dimensionless"):
        require_fixed_value_spec(
            normalize_value_spec(
                {"value_mode": "median", "multiplier": 0.1}, field="threshold"
            ),
            field="threshold",
        )
