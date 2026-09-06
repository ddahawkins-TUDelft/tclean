"""Range-based data-quality testing."""

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tclean.data_quality._validation_helpers import (
    finite_real,
    normalize_common_selectors,
    validate_keys,
)
from tclean.time_grid import TimeGrid

METHOD_NAME = "range"


def validate(
    test: Mapping[str, Any],
    *,
    grid: TimeGrid,
) -> dict[str, Any]:
    """Validate and normalize a range quality test."""
    del grid  # Range thresholds do not depend on the temporal grid.

    validate_keys(
        test,
        required={"name", "method"},
        optional={
            "sources",
            "contexts",
            "minimum",
            "maximum",
        },
    )

    if "minimum" not in test and "maximum" not in test:
        raise ValueError(
            "Range quality tests require at least one of "
            "'minimum' or 'maximum'."
        )

    normalized = normalize_common_selectors(test)

    if "minimum" in test:
        normalized["minimum"] = finite_real(
            test["minimum"],
            field="minimum",
        )

    if "maximum" in test:
        normalized["maximum"] = finite_real(
            test["maximum"],
            field="maximum",
        )

    if (
        "minimum" in normalized
        and "maximum" in normalized
        and normalized["minimum"] > normalized["maximum"]
    ):
        raise ValueError(
            "'minimum' must be less than or equal to 'maximum'."
        )

    return normalized


def evaluate(
    data: pd.DataFrame,
    *,
    test: Mapping[str, Any],
) -> pd.DataFrame:
    """Evaluate whether observed values fall outside configured bounds.

    Missing observations do not fail the test.

    Args:
        data: Time-series values indexed by timestamp and with contexts
            represented by columns.
        test: Validated range-test configuration.

    Returns:
        Boolean DataFrame aligned exactly to ``data`` where ``True`` marks
        observations that fail the configured range test.
    """
    failures = pd.DataFrame(
        False,
        index=data.index,
        columns=data.columns,
        dtype=bool,
    )

    observed = data.notna()

    if "minimum" in test:
        failures |= observed & data.lt(test["minimum"])

    if "maximum" in test:
        failures |= observed & data.gt(test["maximum"])

    return failures


def build_details(
    data: pd.Series,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    test: Mapping[str, Any],
) -> dict[str, Any]:
    """Build structured diagnostics for one failed range period."""
    failed_values = data.loc[(data.index >= start) & (data.index < end)].dropna()

    details: dict[str, Any] = {
        "observed_minimum": float(failed_values.min()),
        "observed_maximum": float(failed_values.max()),
    }

    if "minimum" in test:
        minimum = test["minimum"]

        details["minimum"] = minimum
        details["maximum_below_minimum"] = max(
            0.0,
            float(minimum - failed_values.min()),
        )

    if "maximum" in test:
        maximum = test["maximum"]

        details["maximum"] = maximum
        details["maximum_above_maximum"] = max(
            0.0,
            float(failed_values.max() - maximum),
        )

    return details

