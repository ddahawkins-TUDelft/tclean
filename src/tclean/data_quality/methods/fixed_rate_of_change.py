"""Fixed rate-of-change data-quality testing."""

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tclean.data_quality._method import MethodContext, MethodResult, MethodSpec
from tclean.data_quality._validation_helpers import (
    nonnegative_real,
    normalize_common_selectors,
    validate_keys,
)
from tclean.time_grid import TimeGrid


def validate(test: Mapping[str, Any], *, grid: TimeGrid) -> dict[str, Any]:
    """Validate and normalize a fixed rate-of-change quality test."""
    del grid

    validate_keys(
        test, required={"name", "method", "threshold"}, optional={"sources", "contexts"}
    )

    normalized = normalize_common_selectors(test)

    threshold = nonnegative_real(test["threshold"], field="threshold")

    if threshold == 0:
        raise ValueError(
            "'threshold' for a fixed rate-of-change test must be greater than zero."
        )

    normalized["threshold"] = threshold

    return normalized


def _analyse(
    data: pd.Series, *, threshold: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate preceding values, changes, and failures."""
    previous = data.shift(1)
    change = (data - previous).abs()

    failures = data.notna() & previous.notna() & (change > threshold)

    return previous, change, failures.astype(bool)


def evaluate(context: MethodContext) -> MethodResult:
    """Flag changes whose magnitude exceeds a fixed threshold."""
    data = context.target_data
    test = context.test

    failures = pd.DataFrame(False, index=data.index, columns=data.columns, dtype=bool)

    for context in data.columns:
        _, _, context_failures = _analyse(data[context], threshold=test["threshold"])

        failures[context] = context_failures

    return MethodResult(mask=failures)


def build_details(
    context: MethodContext,
    result: MethodResult,
    *,
    context_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Build structured diagnostics for fixed rate-of-change failures."""
    del result

    data = context.target_data[context_name]
    test = context.test
    grid = context.grid

    previous, change, failures = _analyse(data, threshold=test["threshold"])

    period_failures = failures & (data.index >= start) & (data.index < end)

    transitions = []

    for timestamp in data.index[period_failures]:
        transitions.append(
            {
                "timestamp": pd.Timestamp(timestamp),
                "previous_timestamp": (pd.Timestamp(timestamp) - grid.frequency),
                "previous_value": float(previous.loc[timestamp]),
                "value": float(data.loc[timestamp]),
                "change": float(change.loc[timestamp]),
            }
        )

    return {
        "threshold": test["threshold"],
        "transition_count": len(transitions),
        "transitions": transitions,
    }


METHOD = MethodSpec(
    name="fixed_rate_of_change",
    validate=validate,
    evaluate=evaluate,
    build_details=build_details,
)
