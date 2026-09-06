"""Repeated-value data-quality testing."""

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tclean.data_quality._validation_helpers import (
    finite_real,
    nonnegative_real,
    normalize_common_selectors,
    positive_timedelta,
    validate_keys,
)
from tclean.time_grid import TimeGrid

METHOD_NAME = "value_run"


def validate(
    test: Mapping[str, Any],
    *,
    grid: TimeGrid,
) -> dict[str, Any]:
    """Validate and normalize a value-run quality test."""
    validate_keys(
        test,
        required={
            "name",
            "method",
            "value",
            "minimum_duration",
        },
        optional={
            "sources",
            "contexts",
            "tolerance",
        },
    )

    normalized = normalize_common_selectors(test)

    normalized["value"] = finite_real(
        test["value"],
        field="value",
    )

    normalized["minimum_duration"] = positive_timedelta(
        test["minimum_duration"],
        field="minimum_duration",
        grid=grid,
    )

    normalized["tolerance"] = nonnegative_real(
        test.get("tolerance", 0.0),
        field="tolerance",
    )

    return normalized


def evaluate(
    data: pd.DataFrame,
    *,
    test: Mapping[str, Any],
    grid: TimeGrid,
) -> pd.DataFrame:
    """Flag sufficiently long runs near a configured value.

    Missing observations break a run and do not themselves fail.

    Args:
        data: Time-series values indexed by timestamp with contexts as columns.
        test: Validated value-run test configuration.
        grid: Temporal grid defining the observation frequency.

    Returns:
        Boolean DataFrame aligned exactly to ``data``. All observations
        belonging to qualifying runs are marked ``True``.
    """
    matches = (
        data.notna()
        & data.sub(test["value"]).abs().le(test["tolerance"])
    )

    minimum_steps = int(
        test["minimum_duration"] / grid.frequency
    )

    failures = pd.DataFrame(
        False,
        index=data.index,
        columns=data.columns,
        dtype=bool,
    )

    for context in data.columns:
        matching = matches[context]

        run_ids = matching.ne(matching.shift()).cumsum()
        run_lengths = matching.groupby(run_ids).transform("sum")

        failures[context] = (
            matching
            & run_lengths.ge(minimum_steps)
        )

    return failures


def build_details(
    data: pd.Series,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    test: Mapping[str, Any],
    grid: TimeGrid,
) -> dict[str, Any]:
    """Build structured diagnostics for one failed value run."""
    del grid

    failed_values = data.loc[
        (data.index >= start) & (data.index < end)
    ].dropna()

    deviations = (failed_values - test["value"]).abs()

    return {
        "value": test["value"],
        "tolerance": test["tolerance"],
        "duration": end - start,
        "minimum_duration": test["minimum_duration"],
        "observed_minimum": float(failed_values.min()),
        "observed_maximum": float(failed_values.max()),
        "maximum_absolute_deviation": float(deviations.max()),
    }
