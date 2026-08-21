"""Coordinate deterministic basic electricity-demand cleaning methods."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from tclean.basic.methods.average_periods import METHOD_NAME as AVERAGE_PERIODS
from tclean.basic.methods.average_periods import apply_average_periods
from tclean.basic.methods.copy_periods import METHOD_NAME as COPY_PERIODS
from tclean.basic.methods.copy_periods import apply_copy_periods
from tclean.basic.methods.linear_interpolation import (
    METHOD_NAME as LINEAR_INTERPOLATION,
)
from tclean.basic.methods.linear_interpolation import apply_linear_interpolation
from tclean.validation import infer_regular_timestep, validate_load

logger = logging.getLogger(__name__)


def fill_basic_gaps(
    load: pd.DataFrame,
    *,
    cleaning_method: pd.DataFrame,
    rules: Sequence[Mapping[str, Any]],
    enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply configured basic cleaning rules and record provenance.

    Parameters
    ----------
    load:
        Hourly demand data indexed by timestamp, with one column per region.
    cleaning_method:
        Per-cell cleaning-method provenance for the observed input values.
        Missing input values should contain ``pd.NA``.
    rules:
        Ordered basic cleaning rules.
    enabled:
        Whether basic gap filling should be applied.

    Returns:
    -------
    filled:
        Demand data after applying the configured rules.
    cleaning_method:
        Per-cell provenance containing the observed-source identifier,
        configured cleaning-rule name, or ``"missing"``.
    """
    load = validate_load(load)

    _validate_cleaning_method(load=load, cleaning_method=cleaning_method)

    filled = load.copy()
    cleaning_method = cleaning_method.copy()

    if not enabled:
        logger.info("Basic gap filling is disabled.")
        cleaning_method = cleaning_method.fillna("missing")
        return filled, cleaning_method

    original_gap_duration = calculate_missing_run_durations(load)

    for rule in rules:
        method = str(rule["method"])
        rule_name = str(rule["name"])

        if method == LINEAR_INTERPOLATION:
            filled, newly_filled = apply_linear_interpolation(
                filled,
                max_gap=rule["max_gap"],
                original_gap_duration=original_gap_duration,
            )

        elif method == AVERAGE_PERIODS:
            filled, newly_filled = apply_average_periods(
                filled,
                max_gap=rule["max_gap"],
                source_offsets=rule["source_offsets"],
                original_gap_duration=original_gap_duration,
            )

        elif method == COPY_PERIODS:
            filled, newly_filled = apply_copy_periods(
                filled,
                max_gap=rule["max_gap"],
                source_offset=rule["source_offset"],
                require_complete_source=rule.get("require_complete_source", True),
                original_gap_duration=original_gap_duration,
            )

        else:
            raise ValueError(f"Unsupported gap-filling method: {method!r}")

        cleaning_method = cleaning_method.mask(newly_filled, rule_name)

        _log_rule_results(rule_name=rule_name, method=method, newly_filled=newly_filled)

    cleaning_method = cleaning_method.fillna("missing")

    unresolved = int(filled.isna().to_numpy().sum())
    logger.info("Gap filling completed with %s unresolved values.", unresolved)

    return filled, cleaning_method


def calculate_missing_run_durations(load: pd.DataFrame) -> pd.DataFrame:
    """Return the original duration of each contiguous missing run.

    Observed values receive a duration of zero.

    Parameters
    ----------
    load:
        Regularly indexed electricity-demand data.

    Returns:
    -------
    pandas.DataFrame
        Per-cell durations of the original contiguous missing runs.
    """
    load = validate_load(load)
    timestep = infer_regular_timestep(load.index)
    durations = pd.DataFrame(pd.Timedelta(0), index=load.index, columns=load.columns)

    for column in load.columns:
        missing = load[column].isna()
        group_ids = missing.ne(missing.shift()).cumsum()

        run_lengths = missing.groupby(group_ids).transform("sum").where(missing, 0)

        durations[column] = run_lengths * timestep

    return durations


def _log_rule_results(
    *, rule_name: str, method: str, newly_filled: pd.DataFrame
) -> None:
    """Log the number of values filled by one cleaning rule."""
    total = int(newly_filled.to_numpy().sum())

    logger.info(
        "Gap-filling rule '%s' using method '%s' filled %s values.",
        rule_name,
        method,
        total,
    )

    for region, count in newly_filled.sum().items():
        count = int(count)

        if count:
            logger.info(
                "%s: %s values filled using rule '%s'.", region, count, rule_name
            )


def _validate_cleaning_method(
    *, load: pd.DataFrame, cleaning_method: pd.DataFrame
) -> None:
    """Validate that provenance data align with the demand data."""
    if not cleaning_method.index.equals(load.index):
        raise ValueError(
            "Cleaning-method provenance must use the same index as the load data."
        )

    if not cleaning_method.columns.equals(load.columns):
        raise ValueError(
            "Cleaning-method provenance must use the same columns as the load data."
        )
