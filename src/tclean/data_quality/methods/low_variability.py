"""Low-variability data-quality testing."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from tclean.data_quality._method import MethodContext, MethodResult, MethodSpec
from tclean.data_quality._validation_helpers import (
    nonnegative_real,
    normalize_common_selectors,
    positive_timedelta,
    validate_keys,
)
from tclean.time_grid import TimeGrid


def validate(test: Mapping[str, Any], *, grid: TimeGrid) -> dict[str, Any]:
    """Validate and normalize a low-variability quality test."""
    validate_keys(
        test,
        required={"name", "method", "window_duration", "maximum_range"},
        optional={"sources", "contexts"},
    )

    normalized = normalize_common_selectors(test)

    window_duration = positive_timedelta(
        test["window_duration"], field="window_duration", grid=grid
    )

    if window_duration < 2 * grid.frequency:
        raise ValueError(
            "'window_duration' for a low-variability test must span "
            "at least two grid steps."
        )

    normalized["window_duration"] = window_duration

    normalized["maximum_range"] = nonnegative_real(
        test["maximum_range"], field="maximum_range"
    )

    return normalized


def _window_ranges(
    values: pd.Series, *, window_steps: int
) -> tuple[pd.Series, pd.Series]:
    """Return rolling ranges and whether each window is complete."""
    rolling = values.rolling(window=window_steps, min_periods=window_steps)

    complete = rolling.count().eq(window_steps)
    ranges = rolling.max() - rolling.min()

    return ranges, complete


def _qualifying_windows(
    ranges: pd.Series, complete: pd.Series, *, maximum_range: float
) -> pd.Series:
    """Return complete windows whose range does not exceed the threshold.

    A very small relative tolerance is used only to avoid binary
    floating-point representation affecting inclusive threshold comparisons.
    """
    at_threshold = np.isclose(
        ranges.to_numpy(dtype=float),
        float(maximum_range),
        rtol=1e-12,
        atol=0.0,
        equal_nan=False,
    )

    return complete & (
        ranges.le(maximum_range)
        | pd.Series(at_threshold, index=ranges.index, dtype=bool)
    )


def evaluate(context: MethodContext) -> MethodResult:
    """Flag observations belonging to qualifying low-variability windows.

    A window qualifies when every expected observation is present and the
    range between its maximum and minimum values is less than or equal to the
    configured maximum range. Overlapping qualifying windows are combined.
    """
    data = context.target_data
    test = context.test
    grid = context.grid

    window_steps = int(test["window_duration"] / grid.frequency)

    failures = pd.DataFrame(False, index=data.index, columns=data.columns, dtype=bool)

    for context in data.columns:
        values = data[context]

        ranges, complete = _window_ranges(values, window_steps=window_steps)

        qualifying_ends = _qualifying_windows(
            ranges, complete, maximum_range=test["maximum_range"]
        )
        difference = np.zeros(len(values) + 1, dtype=int)

        for end_position in np.flatnonzero(qualifying_ends.to_numpy(dtype=bool)):
            start_position = end_position - window_steps + 1

            difference[start_position] += 1
            difference[end_position + 1] -= 1

        failures[context] = np.cumsum(difference[:-1]) > 0

    return MethodResult(mask=failures)


def build_details(
    context: MethodContext,
    result: MethodResult,
    *,
    context_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Build structured diagnostics for one low-variability period."""
    del result

    data = context.target_data[context_name]
    test = context.test
    grid = context.grid

    window_steps = int(test["window_duration"] / grid.frequency)

    ranges, complete = _window_ranges(data, window_steps=window_steps)

    qualifying = _qualifying_windows(
        ranges, complete, maximum_range=test["maximum_range"]
    )

    window_starts = (
        pd.Series(ranges.index, index=ranges.index)
        - (window_steps - 1) * grid.frequency
    )

    contributing = (
        qualifying & window_starts.ge(start) & ranges.index.to_series().lt(end)
    )

    contributing_ranges = ranges.loc[contributing]

    period_values = data.loc[(data.index >= start) & (data.index < end)].dropna()

    return {
        "window_duration": test["window_duration"],
        "maximum_range": test["maximum_range"],
        "qualifying_window_count": int(contributing.sum()),
        "minimum_window_range": float(contributing_ranges.min()),
        "maximum_window_range": float(contributing_ranges.max()),
        "period_observed_minimum": float(period_values.min()),
        "period_observed_maximum": float(period_values.max()),
        "period_observed_range": float(period_values.max() - period_values.min()),
    }


METHOD = MethodSpec(
    name="low_variability",
    validate=validate,
    evaluate=evaluate,
    build_details=build_details,
)
