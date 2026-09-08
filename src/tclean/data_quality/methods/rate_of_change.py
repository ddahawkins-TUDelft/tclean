"""Rate-of-change data-quality testing."""

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tclean.data_quality._method import MethodContext, MethodResult, MethodSpec
from tclean.data_quality._validation_helpers import (
    normalize_common_selectors,
    string_choice,
    validate_keys,
)
from tclean.data_quality._value_spec import (
    build_value_issues,
    fixed_value_spec,
    normalize_value_reference_controls,
    normalize_value_spec,
    require_fixed_value_spec,
    resolve_context_value,
    value_resolution_details,
    value_resolution_problem,
)
from tclean.time_grid import TimeGrid


def validate(test: Mapping[str, Any], *, grid: TimeGrid) -> dict[str, Any]:
    """Validate and normalize a rate-of-change quality test."""
    del grid

    validate_keys(
        test,
        required={"name", "method", "difference_mode", "threshold"},
        optional={
            "sources",
            "contexts",
            "reference_magnitude_threshold",
            "include_failed_periods_from",
        },
    )

    normalized = normalize_common_selectors(test)
    difference_mode = string_choice(
        test["difference_mode"], field="difference_mode", choices=("fixed", "relative")
    )
    threshold = normalize_value_spec(test["threshold"], field="threshold")

    if difference_mode == "relative":
        require_fixed_value_spec(threshold, field="threshold")

    if threshold["value_mode"] == "fixed" and threshold["value"] <= 0:
        raise ValueError("Fixed 'threshold.value' must be greater than zero.")

    normalized["difference_mode"] = difference_mode
    normalized["threshold"] = threshold

    if difference_mode == "fixed":
        if "reference_magnitude_threshold" in test:
            raise ValueError(
                "'reference_magnitude_threshold' is only supported when "
                "'difference_mode' is 'relative'."
            )
    else:
        normalized["reference_magnitude_threshold"] = (
            normalize_value_spec(
                test["reference_magnitude_threshold"],
                field="reference_magnitude_threshold",
            )
            if "reference_magnitude_threshold" in test
            else fixed_value_spec(0.0, field="reference_magnitude_threshold")
        )

        reference_threshold = normalized["reference_magnitude_threshold"]
        if (
            reference_threshold["value_mode"] == "fixed"
            and reference_threshold["value"] < 0
        ):
            raise ValueError(
                "Fixed 'reference_magnitude_threshold.value' must be "
                "greater than or equal to zero."
            )

    normalize_value_reference_controls(
        test, normalized, fields=("threshold", "reference_magnitude_threshold")
    )

    return normalized


def _analyse(
    data: pd.Series,
    *,
    difference_mode: str,
    threshold: float,
    reference_magnitude_threshold: float = 0.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate changes, configured differences, and failures."""
    previous = data.shift(1)
    change = (data - previous).abs()
    observed = data.notna() & previous.notna()

    if difference_mode == "fixed":
        difference = change.where(observed)
        eligible = observed
    else:
        reference_magnitude = previous.abs()
        eligible = observed & (reference_magnitude > reference_magnitude_threshold)
        difference = change.div(reference_magnitude.where(eligible))

    failures = eligible & (difference > threshold)
    return previous, change, difference, failures.astype(bool)


def _resolved_parameters(context: MethodContext, *, context_name: str):
    """Resolve rate-of-change scalar parameters for one focal context."""
    threshold = resolve_context_value(
        context, field="threshold", context_name=context_name
    )
    reference_threshold = (
        resolve_context_value(
            context, field="reference_magnitude_threshold", context_name=context_name
        )
        if context.test["difference_mode"] == "relative"
        else None
    )

    return threshold, reference_threshold


def evaluate(context: MethodContext) -> MethodResult:
    """Flag rate-of-change differences exceeding the configured threshold."""
    data = context.target_data
    test = context.test
    failures = pd.DataFrame(False, index=data.index, columns=data.columns, dtype=bool)
    issues = []

    for context_name in data.columns:
        values = data[context_name]
        threshold, reference_threshold = _resolved_parameters(
            context, context_name=context_name
        )

        problems = []
        threshold_problem = value_resolution_problem(
            threshold,
            field="threshold",
            spec=test["threshold"],
            minimum=0.0,
            strict_minimum=True,
        )
        if threshold_problem is not None:
            problems.append(threshold_problem)

        if reference_threshold is not None:
            reference_problem = value_resolution_problem(
                reference_threshold,
                field="reference_magnitude_threshold",
                spec=test["reference_magnitude_threshold"],
                minimum=0.0,
            )
            if reference_problem is not None:
                problems.append(reference_problem)

        candidates = values.notna() & values.shift(1).notna()
        context_issues = build_value_issues(
            context, context_name=context_name, problems=problems, candidates=candidates
        )
        if context_issues:
            issues.extend(context_issues)
            continue

        if threshold.value is None:
            raise RuntimeError(
                "Evaluable rate-of-change threshold has no resolved value."
            )

        reference_value = 0.0
        if reference_threshold is not None:
            if reference_threshold.value is None:
                raise RuntimeError(
                    "Evaluable reference magnitude threshold has no resolved value."
                )
            reference_value = reference_threshold.value

        _, _, _, context_failures = _analyse(
            values,
            difference_mode=test["difference_mode"],
            threshold=threshold.value,
            reference_magnitude_threshold=reference_value,
        )
        failures[context_name] = context_failures

    return MethodResult(mask=failures, issues=tuple(issues))


def build_details(
    context: MethodContext,
    result: MethodResult,
    *,
    context_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Build structured diagnostics for rate-of-change failures."""
    del result

    data = context.target_data[context_name]
    test = context.test
    grid = context.grid
    threshold, reference_threshold = _resolved_parameters(
        context, context_name=context_name
    )

    threshold_problem = value_resolution_problem(
        threshold,
        field="threshold",
        spec=test["threshold"],
        minimum=0.0,
        strict_minimum=True,
    )
    reference_problem = (
        value_resolution_problem(
            reference_threshold,
            field="reference_magnitude_threshold",
            spec=test["reference_magnitude_threshold"],
            minimum=0.0,
        )
        if reference_threshold is not None
        else None
    )

    if threshold_problem is not None or reference_problem is not None:
        raise RuntimeError(
            "Cannot build rate-of-change failure details with unusable resolved values."
        )
    if threshold.value is None:
        raise RuntimeError("Evaluable rate-of-change threshold has no resolved value.")

    reference_value = 0.0
    if reference_threshold is not None:
        if reference_threshold.value is None:
            raise RuntimeError(
                "Evaluable reference magnitude threshold has no resolved value."
            )
        reference_value = reference_threshold.value

    previous, change, difference, failures = _analyse(
        data,
        difference_mode=test["difference_mode"],
        threshold=threshold.value,
        reference_magnitude_threshold=reference_value,
    )
    period_failures = failures & (data.index >= start) & (data.index < end)

    transitions = []
    for timestamp in data.index[period_failures]:
        transition = {
            "timestamp": pd.Timestamp(timestamp),
            "previous_timestamp": pd.Timestamp(timestamp) - grid.frequency,
            "previous_value": float(previous.loc[timestamp]),
            "value": float(data.loc[timestamp]),
            "change": float(change.loc[timestamp]),
        }
        if test["difference_mode"] == "relative":
            transition["relative_change"] = float(difference.loc[timestamp])
        transitions.append(transition)

    details: dict[str, Any] = {
        "difference_mode": test["difference_mode"],
        "threshold": threshold.value,
        "threshold_resolution": value_resolution_details(
            threshold, field="threshold", spec=test["threshold"]
        ),
        "transition_count": len(transitions),
        "transitions": transitions,
    }

    if reference_threshold is not None:
        details["reference_magnitude_threshold"] = reference_value
        details["reference_magnitude_threshold_resolution"] = value_resolution_details(
            reference_threshold,
            field="reference_magnitude_threshold",
            spec=test["reference_magnitude_threshold"],
        )

    return details


METHOD = MethodSpec(
    name="rate_of_change",
    validate=validate,
    evaluate=evaluate,
    build_details=build_details,
)
