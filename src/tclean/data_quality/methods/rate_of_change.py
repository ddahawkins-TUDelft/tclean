"""Rate-of-change data-quality testing."""

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tclean.data_quality._method import (
    MethodContext,
    MethodIssue,
    MethodResult,
    MethodSpec,
)
from tclean.data_quality._periods import failure_mask_to_periods
from tclean.data_quality._validation_helpers import (
    nonnegative_real,
    normalize_common_selectors,
    normalize_string_sequence,
    string_choice,
    validate_keys,
)
from tclean.data_quality._value_spec import (
    ValueResolution,
    normalize_value_spec,
    resolve_context_value,
    value_resolution_details,
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
        test["difference_mode"],
        field="difference_mode",
        choices=("fixed", "relative"),
    )

    threshold = normalize_value_spec(
        test["threshold"],
        field="threshold",
    )

    if threshold["value_mode"] == "fixed" and threshold["value"] <= 0:
        raise ValueError(
            "'threshold.value' for a rate-of-change test "
            "must be greater than zero."
        )

    normalized["difference_mode"] = difference_mode
    normalized["threshold"] = threshold

    if "include_failed_periods_from" in test:
        if threshold["value_mode"] == "fixed":
            raise ValueError(
                "'include_failed_periods_from' is only supported "
                "for a derived rate-of-change threshold."
            )

        normalized["include_failed_periods_from"] = normalize_string_sequence(
            test["include_failed_periods_from"],
            field="include_failed_periods_from",
        )

    if difference_mode == "fixed":
        if "reference_magnitude_threshold" in test:
            raise ValueError(
                "'reference_magnitude_threshold' is only supported when "
                "'difference_mode' is 'relative'."
            )
    else:
        normalized["reference_magnitude_threshold"] = nonnegative_real(
            test.get("reference_magnitude_threshold", 0.0),
            field="reference_magnitude_threshold",
        )

    return normalized


def _threshold_issue_reason(
    resolution: ValueResolution,
) -> str | None:
    """Return why a resolved rate-of-change threshold cannot be used."""
    if not resolution.evaluable:
        return resolution.reason

    if resolution.value is None or resolution.value <= 0:
        return "resolved_value_not_positive"

    return None


def _candidate_transitions(
    data: pd.Series,
    *,
    difference_mode: str,
    reference_magnitude_threshold: float,
) -> pd.Series:
    """Return transitions that could otherwise be evaluated."""
    previous = data.shift(1)
    candidates = data.notna() & previous.notna()

    if difference_mode == "relative":
        candidates &= previous.abs() > reference_magnitude_threshold

    return candidates.astype(bool)


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

        eligible = observed & (
            reference_magnitude > reference_magnitude_threshold
        )

        difference = change.div(
            reference_magnitude.where(eligible)
        )

    failures = eligible & (difference > threshold)

    return previous, change, difference, failures.astype(bool)


def evaluate(context: MethodContext) -> MethodResult:
    """Flag rate-of-change differences exceeding the configured threshold."""
    data = context.target_data
    test = context.test

    failures = pd.DataFrame(
        False,
        index=data.index,
        columns=data.columns,
        dtype=bool,
    )
    issues: list[MethodIssue] = []

    for context_name in data.columns:
        focal = data[context_name]

        threshold_resolution = resolve_context_value(
            context,
            field="threshold",
            context_name=context_name,
        )

        issue_reason = _threshold_issue_reason(
            threshold_resolution
        )

        if issue_reason is not None:
            candidates = _candidate_transitions(
                focal,
                difference_mode=test["difference_mode"],
                reference_magnitude_threshold=test.get(
                    "reference_magnitude_threshold",
                    0.0,
                ),
            )

            for start, end in failure_mask_to_periods(
                candidates,
                grid=context.grid,
            ):
                issues.append(
                    MethodIssue(
                        context=context_name,
                        start=start,
                        end=end,
                        severity="not_evaluable",
                        code="derived_value_not_evaluable",
                        details=value_resolution_details(
                            threshold_resolution,
                            field="threshold",
                            spec=test["threshold"],
                            reason=issue_reason,
                        ),
                    )
                )

            continue

        threshold = threshold_resolution.value

        if threshold is None:
            raise RuntimeError(
                "Evaluable rate-of-change threshold has no resolved value."
            )

        _, _, _, context_failures = _analyse(
            focal,
            difference_mode=test["difference_mode"],
            threshold=threshold,
            reference_magnitude_threshold=test.get(
                "reference_magnitude_threshold",
                0.0,
            ),
        )

        failures[context_name] = context_failures

    return MethodResult(
        mask=failures,
        issues=tuple(issues),
    )


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

    threshold_resolution = resolve_context_value(
        context,
        field="threshold",
        context_name=context_name,
    )

    issue_reason = _threshold_issue_reason(
        threshold_resolution
    )

    if issue_reason is not None or threshold_resolution.value is None:
        raise RuntimeError(
            "Cannot build rate-of-change failure details "
            "with an unusable resolved threshold."
        )

    threshold = threshold_resolution.value

    previous, change, difference, failures = _analyse(
        data,
        difference_mode=test["difference_mode"],
        threshold=threshold,
        reference_magnitude_threshold=test.get(
            "reference_magnitude_threshold",
            0.0,
        ),
    )

    period_failures = (
        failures
        & (data.index >= start)
        & (data.index < end)
    )

    transitions = []

    for timestamp in data.index[period_failures]:
        transition = {
            "timestamp": pd.Timestamp(timestamp),
            "previous_timestamp": (
                pd.Timestamp(timestamp) - grid.frequency
            ),
            "previous_value": float(previous.loc[timestamp]),
            "value": float(data.loc[timestamp]),
            "change": float(change.loc[timestamp]),
        }

        if test["difference_mode"] == "relative":
            transition["relative_change"] = float(
                difference.loc[timestamp]
            )

        transitions.append(transition)

    details: dict[str, Any] = {
        "difference_mode": test["difference_mode"],
        "threshold": threshold,
        "threshold_resolution": value_resolution_details(
            threshold_resolution,
            field="threshold",
            spec=test["threshold"],
        ),
        "transition_count": len(transitions),
        "transitions": transitions,
    }

    if test["difference_mode"] == "relative":
        details["reference_magnitude_threshold"] = (
            test["reference_magnitude_threshold"]
        )

    return details


METHOD = MethodSpec(
    name="rate_of_change",
    validate=validate,
    evaluate=evaluate,
    build_details=build_details,
)
