"""Repeated-value data-quality testing."""

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tclean.data_quality._method import MethodContext, MethodResult, MethodSpec
from tclean.data_quality._validation_helpers import (
    finite_real,
    nonnegative_real,
    normalize_common_selectors,
    positive_timedelta,
    validate_keys,
)
from tclean.time_grid import TimeGrid


def validate(test: Mapping[str, Any], *, grid: TimeGrid) -> dict[str, Any]:
    """Validate and normalize a value-run quality test."""
    validate_keys(
        test,
        required={"name", "method", "value", "minimum_duration"},
        optional={"sources", "contexts", "tolerance"},
    )

    normalized = normalize_common_selectors(test)

    normalized["value"] = finite_real(test["value"], field="value")

    normalized["minimum_duration"] = positive_timedelta(
        test["minimum_duration"], field="minimum_duration", grid=grid
    )

    normalized["tolerance"] = nonnegative_real(
        test.get("tolerance", 0.0), field="tolerance"
    )

    return normalized


def evaluate(context: MethodContext) -> MethodResult:
    """Flag sufficiently long runs near a configured value.

    Missing observations break a run and do not themselves fail.
    """
    data = context.target_data
    test = context.test
    grid = context.grid

    matches = data.notna() & data.sub(test["value"]).abs().le(test["tolerance"])

    minimum_steps = int(test["minimum_duration"] / grid.frequency)

    failures = pd.DataFrame(False, index=data.index, columns=data.columns, dtype=bool)

    for context in data.columns:
        matching = matches[context]

        run_ids = matching.ne(matching.shift()).cumsum()
        run_lengths = matching.groupby(run_ids).transform("sum")

        failures[context] = matching & run_lengths.ge(minimum_steps)

    return MethodResult(mask=failures)


def build_details(
    context: MethodContext,
    result: MethodResult,
    *,
    context_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Build structured diagnostics for one failed value run."""
    del result

    data = context.target_data[context_name]
    test = context.test

    failed_values = data.loc[(data.index >= start) & (data.index < end)].dropna()

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


METHOD = MethodSpec(
    name="value_run", validate=validate, evaluate=evaluate, build_details=build_details
)
