"""Repeated-value data-quality testing."""

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tclean.data_quality._method import MethodContext, MethodResult, MethodSpec
from tclean.data_quality._validation_helpers import (
    normalize_common_selectors,
    positive_timedelta,
    validate_keys,
)
from tclean.data_quality._value_spec import (
    build_value_issues,
    fixed_value_spec,
    normalize_value_reference_controls,
    normalize_value_spec,
    resolve_context_value,
    value_resolution_details,
    value_resolution_problem,
)
from tclean.time_grid import TimeGrid


def validate(test: Mapping[str, Any], *, grid: TimeGrid) -> dict[str, Any]:
    """Validate and normalize a value-run quality test."""
    validate_keys(
        test,
        required={"name", "method", "value", "minimum_duration"},
        optional={"sources", "contexts", "tolerance", "include_failed_periods_from"},
    )

    normalized = normalize_common_selectors(test)
    normalized["value"] = normalize_value_spec(test["value"], field="value")
    normalized["minimum_duration"] = positive_timedelta(
        test["minimum_duration"], field="minimum_duration", grid=grid
    )
    normalized["tolerance"] = (
        normalize_value_spec(test["tolerance"], field="tolerance")
        if "tolerance" in test
        else fixed_value_spec(0.0, field="tolerance")
    )

    tolerance = normalized["tolerance"]
    if tolerance["value_mode"] == "fixed" and tolerance["value"] < 0:
        raise ValueError(
            "Fixed 'tolerance.value' must be greater than or equal to zero."
        )

    normalize_value_reference_controls(test, normalized, fields=("value", "tolerance"))

    return normalized


def _resolved_values(context: MethodContext, *, context_name: str):
    """Resolve the target value and tolerance for one focal context."""
    value = resolve_context_value(context, field="value", context_name=context_name)
    tolerance = resolve_context_value(
        context, field="tolerance", context_name=context_name
    )

    return value, tolerance


def evaluate(context: MethodContext) -> MethodResult:
    """Flag sufficiently long runs near a configured value."""
    data = context.target_data
    test = context.test
    grid = context.grid
    minimum_steps = int(test["minimum_duration"] / grid.frequency)

    failures = pd.DataFrame(False, index=data.index, columns=data.columns, dtype=bool)
    issues = []

    for context_name in data.columns:
        values = data[context_name]
        value, tolerance = _resolved_values(context, context_name=context_name)

        problems = [
            problem
            for problem in (
                value_resolution_problem(value, field="value", spec=test["value"]),
                value_resolution_problem(
                    tolerance, field="tolerance", spec=test["tolerance"], minimum=0.0
                ),
            )
            if problem is not None
        ]

        context_issues = build_value_issues(
            context,
            context_name=context_name,
            problems=problems,
            candidates=values.notna(),
        )
        if context_issues:
            issues.extend(context_issues)
            continue

        if value.value is None or tolerance.value is None:
            raise RuntimeError("Evaluable value-run parameters have no resolved value.")

        matching = values.notna() & values.sub(value.value).abs().le(tolerance.value)
        run_ids = matching.ne(matching.shift()).cumsum()
        run_lengths = matching.groupby(run_ids).transform("sum")
        failures[context_name] = matching & run_lengths.ge(minimum_steps)

    return MethodResult(mask=failures, issues=tuple(issues))


def build_details(
    context: MethodContext,
    result: MethodResult,
    *,
    context_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Build structured diagnostics for one failed value run."""
    del result

    data = context.target_data[context_name]
    test = context.test
    value, tolerance = _resolved_values(context, context_name=context_name)

    problems = [
        problem
        for problem in (
            value_resolution_problem(value, field="value", spec=test["value"]),
            value_resolution_problem(
                tolerance, field="tolerance", spec=test["tolerance"], minimum=0.0
            ),
        )
        if problem is not None
    ]

    if problems or value.value is None or tolerance.value is None:
        raise RuntimeError(
            "Cannot build value-run failure details with unusable resolved values."
        )

    failed_values = data.loc[(data.index >= start) & (data.index < end)].dropna()
    deviations = (failed_values - value.value).abs()

    return {
        "value": value.value,
        "value_resolution": value_resolution_details(
            value, field="value", spec=test["value"]
        ),
        "tolerance": tolerance.value,
        "tolerance_resolution": value_resolution_details(
            tolerance, field="tolerance", spec=test["tolerance"]
        ),
        "duration": end - start,
        "minimum_duration": test["minimum_duration"],
        "observed_minimum": float(failed_values.min()),
        "observed_maximum": float(failed_values.max()),
        "maximum_absolute_deviation": float(deviations.max()),
    }


METHOD = MethodSpec(
    name="value_run", validate=validate, evaluate=evaluate, build_details=build_details
)
