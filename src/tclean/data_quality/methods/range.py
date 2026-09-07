"""Range-based data-quality testing."""

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tclean.data_quality._method import MethodContext, MethodResult, MethodSpec
from tclean.data_quality._validation_helpers import (
    finite_real,
    normalize_common_selectors,
    validate_keys,
)
from tclean.time_grid import TimeGrid


def validate(test: Mapping[str, Any], *, grid: TimeGrid) -> dict[str, Any]:
    """Validate and normalize a range quality test."""
    del grid  # Range thresholds do not depend on the temporal grid.

    validate_keys(
        test,
        required={"name", "method"},
        optional={"sources", "contexts", "minimum", "maximum"},
    )

    if "minimum" not in test and "maximum" not in test:
        raise ValueError(
            "Range quality tests require at least one of 'minimum' or 'maximum'."
        )

    normalized = normalize_common_selectors(test)

    if "minimum" in test:
        normalized["minimum"] = finite_real(test["minimum"], field="minimum")

    if "maximum" in test:
        normalized["maximum"] = finite_real(test["maximum"], field="maximum")

    if (
        "minimum" in normalized
        and "maximum" in normalized
        and normalized["minimum"] > normalized["maximum"]
    ):
        raise ValueError("'minimum' must be less than or equal to 'maximum'.")

    return normalized


def evaluate(context: MethodContext) -> MethodResult:
    """Evaluate whether observed values fall outside configured bounds.

    Missing observations do not fail the test.
    """
    data = context.target_data
    test = context.test

    failures = pd.DataFrame(False, index=data.index, columns=data.columns, dtype=bool)

    observed = data.notna()

    if "minimum" in test:
        failures |= observed & data.lt(test["minimum"])

    if "maximum" in test:
        failures |= observed & data.gt(test["maximum"])

    return MethodResult(mask=failures)


def build_details(
    context: MethodContext,
    result: MethodResult,
    *,
    context_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Build structured diagnostics for one failed range period."""
    del result

    data = context.target_data[context_name]
    test = context.test

    failed_values = data.loc[(data.index >= start) & (data.index < end)].dropna()

    details: dict[str, Any] = {
        "observed_minimum": float(failed_values.min()),
        "observed_maximum": float(failed_values.max()),
    }

    if "minimum" in test:
        minimum = test["minimum"]

        details["minimum"] = minimum
        details["maximum_below_minimum"] = max(
            0.0, float(minimum - failed_values.min())
        )

    if "maximum" in test:
        maximum = test["maximum"]

        details["maximum"] = maximum
        details["maximum_above_maximum"] = max(
            0.0, float(failed_values.max() - maximum)
        )

    return details


METHOD = MethodSpec(
    name="range", validate=validate, evaluate=evaluate, build_details=build_details
)
