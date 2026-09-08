"""Flatline data-quality testing."""

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
    """Validate and normalize a flatline quality test."""
    validate_keys(
        test,
        required={"name", "method", "minimum_duration"},
        optional={"sources", "contexts", "tolerance", "include_failed_periods_from"},
    )

    normalized = normalize_common_selectors(test)

    minimum_duration = positive_timedelta(
        test["minimum_duration"], field="minimum_duration", grid=grid
    )
    if minimum_duration < 2 * grid.frequency:
        raise ValueError(
            "'minimum_duration' for a flatline test must span at least two grid steps."
        )

    normalized["minimum_duration"] = minimum_duration
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

    normalize_value_reference_controls(test, normalized, fields=("tolerance",))

    return normalized


def evaluate(context: MethodContext) -> MethodResult:
    """Flag sufficiently long runs of effectively unchanged values."""
    data = context.target_data
    test = context.test
    grid = context.grid
    minimum_steps = int(test["minimum_duration"] / grid.frequency)

    failures = pd.DataFrame(False, index=data.index, columns=data.columns, dtype=bool)
    issues = []

    for context_name in data.columns:
        values = data[context_name]
        tolerance = resolve_context_value(
            context, field="tolerance", context_name=context_name
        )
        problem = value_resolution_problem(
            tolerance, field="tolerance", spec=test["tolerance"], minimum=0.0
        )

        context_issues = build_value_issues(
            context,
            context_name=context_name,
            problems=[] if problem is None else [problem],
            candidates=values.notna(),
        )
        if context_issues:
            issues.extend(context_issues)
            continue

        if tolerance.value is None:
            raise RuntimeError("Evaluable flatline tolerance has no resolved value.")

        observed = values.notna()
        stable_with_previous = (
            observed
            & observed.shift(1, fill_value=False)
            & values.diff().abs().le(tolerance.value)
        )
        run_starts = ~stable_with_previous
        run_ids = run_starts.cumsum()
        run_lengths = observed.groupby(run_ids).transform("sum")
        failures[context_name] = observed & run_lengths.ge(minimum_steps)

    return MethodResult(mask=failures, issues=tuple(issues))


def build_details(
    context: MethodContext,
    result: MethodResult,
    *,
    context_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Build structured diagnostics for one failed flatline period."""
    del result

    data = context.target_data[context_name]
    test = context.test
    tolerance = resolve_context_value(
        context, field="tolerance", context_name=context_name
    )
    problem = value_resolution_problem(
        tolerance, field="tolerance", spec=test["tolerance"], minimum=0.0
    )

    if problem is not None or tolerance.value is None:
        raise RuntimeError(
            "Cannot build flatline failure details with an unusable tolerance."
        )

    failed_values = data.loc[(data.index >= start) & (data.index < end)].dropna()
    step_changes = failed_values.diff().abs().dropna()

    return {
        "duration": end - start,
        "minimum_duration": test["minimum_duration"],
        "tolerance": tolerance.value,
        "tolerance_resolution": value_resolution_details(
            tolerance, field="tolerance", spec=test["tolerance"]
        ),
        "observed_minimum": float(failed_values.min()),
        "observed_maximum": float(failed_values.max()),
        "observed_range": float(failed_values.max() - failed_values.min()),
        "maximum_step_change": (
            float(step_changes.max()) if not step_changes.empty else 0.0
        ),
    }


METHOD = MethodSpec(
    name="flatline", validate=validate, evaluate=evaluate, build_details=build_details
)
