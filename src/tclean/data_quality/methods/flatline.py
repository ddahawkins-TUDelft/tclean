"""Flatline data-quality testing."""

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tclean.data_quality._validation_helpers import (
    nonnegative_real,
    normalize_common_selectors,
    positive_timedelta,
    validate_keys,
)
from tclean.time_grid import TimeGrid

METHOD_NAME = "flatline"


def validate(test: Mapping[str, Any], *, grid: TimeGrid) -> dict[str, Any]:
    """Validate and normalize a flatline quality test."""
    validate_keys(
        test,
        required={"name", "method", "minimum_duration"},
        optional={"sources", "contexts", "tolerance"},
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

    normalized["tolerance"] = nonnegative_real(
        test.get("tolerance", 0.0), field="tolerance"
    )

    return normalized


def evaluate(
    data: pd.DataFrame, *, test: Mapping[str, Any], grid: TimeGrid
) -> pd.DataFrame:
    """Flag sufficiently long runs of effectively unchanged values.

    Consecutive observed values belong to the same flatline run when their
    absolute difference is less than or equal to the configured tolerance.
    Missing observations break a run and do not themselves fail.

    Args:
        data: Time-series values indexed by timestamp with contexts as columns.
        test: Validated flatline-test configuration.
        grid: Temporal grid defining the observation frequency.

    Returns:
        Boolean DataFrame aligned exactly to ``data``. Every observation in a
        qualifying flatline run is marked ``True``.
    """
    minimum_steps = int(test["minimum_duration"] / grid.frequency)

    failures = pd.DataFrame(False, index=data.index, columns=data.columns, dtype=bool)

    for context in data.columns:
        values = data[context]
        observed = values.notna()

        stable_with_previous = (
            observed
            & observed.shift(1, fill_value=False)
            & values.diff().abs().le(test["tolerance"])
        )

        run_starts = ~stable_with_previous
        run_ids = run_starts.cumsum()

        run_lengths = observed.groupby(run_ids).transform("sum")

        failures[context] = observed & run_lengths.ge(minimum_steps)

    return failures


def build_details(
    data: pd.Series,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    test: Mapping[str, Any],
    grid: TimeGrid,
) -> dict[str, Any]:
    """Build structured diagnostics for one failed flatline period."""
    del grid

    failed_values = data.loc[(data.index >= start) & (data.index < end)].dropna()

    step_changes = failed_values.diff().abs().dropna()

    return {
        "duration": end - start,
        "minimum_duration": test["minimum_duration"],
        "tolerance": test["tolerance"],
        "observed_minimum": float(failed_values.min()),
        "observed_maximum": float(failed_values.max()),
        "observed_range": float(failed_values.max() - failed_values.min()),
        "maximum_step_change": (
            float(step_changes.max()) if not step_changes.empty else 0.0
        ),
    }
