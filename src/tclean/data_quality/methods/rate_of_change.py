"""Rate-of-change data-quality testing."""

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tclean.data_quality._method import MethodContext, MethodResult, MethodSpec
from tclean.data_quality._validation_helpers import (
    nonnegative_real,
    normalize_common_selectors,
    string_choice,
    validate_keys,
)
from tclean.time_grid import TimeGrid


def validate(test: Mapping[str, Any], *, grid: TimeGrid) -> dict[str, Any]:
    """Validate and normalize a rate-of-change quality test."""
    del grid

    validate_keys(
        test,
        required={"name", "method", "difference_mode", "threshold"},
        optional={"sources", "contexts", "reference_magnitude_threshold"},
    )

    normalized = normalize_common_selectors(test)

    difference_mode = string_choice(
        test["difference_mode"], field="difference_mode", choices=("fixed", "relative")
    )

    threshold = nonnegative_real(test["threshold"], field="threshold")

    if threshold == 0:
        raise ValueError(
            "'threshold' for a rate-of-change test must be greater than zero."
        )

    normalized["difference_mode"] = difference_mode
    normalized["threshold"] = threshold

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


def evaluate(context: MethodContext) -> MethodResult:
    """Flag rate-of-change differences exceeding the configured threshold."""
    data = context.target_data
    test = context.test

    failures = pd.DataFrame(False, index=data.index, columns=data.columns, dtype=bool)

    for context_name in data.columns:
        _, _, _, context_failures = _analyse(
            data[context_name],
            difference_mode=test["difference_mode"],
            threshold=test["threshold"],
            reference_magnitude_threshold=test.get(
                "reference_magnitude_threshold", 0.0
            ),
        )

        failures[context_name] = context_failures

    return MethodResult(mask=failures)


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

    previous, change, difference, failures = _analyse(
        data,
        difference_mode=test["difference_mode"],
        threshold=test["threshold"],
        reference_magnitude_threshold=test.get("reference_magnitude_threshold", 0.0),
    )

    period_failures = failures & (data.index >= start) & (data.index < end)

    transitions = []

    for timestamp in data.index[period_failures]:
        transition = {
            "timestamp": pd.Timestamp(timestamp),
            "previous_timestamp": (pd.Timestamp(timestamp) - grid.frequency),
            "previous_value": float(previous.loc[timestamp]),
            "value": float(data.loc[timestamp]),
            "change": float(change.loc[timestamp]),
        }

        if test["difference_mode"] == "relative":
            transition["relative_change"] = float(difference.loc[timestamp])

        transitions.append(transition)

    details = {
        "difference_mode": test["difference_mode"],
        "threshold": test["threshold"],
        "transition_count": len(transitions),
        "transitions": transitions,
    }

    if test["difference_mode"] == "relative":
        details["reference_magnitude_threshold"] = test["reference_magnitude_threshold"]

    return details


METHOD = MethodSpec(
    name="rate_of_change",
    validate=validate,
    evaluate=evaluate,
    build_details=build_details,
)
