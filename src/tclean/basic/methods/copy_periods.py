"""Gap filling using values from another period."""

from __future__ import annotations

import pandas as pd

METHOD_NAME = "copy_periods"


def apply_copy_periods(
    load: pd.DataFrame,
    *,
    max_gap: str | pd.Timedelta,
    source_offset: str | pd.Timedelta,
    original_gap_duration: pd.DataFrame,
    require_complete_source: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill eligible gaps using values from another period.

    For each target timestamp ``t``, the source value is taken from
    ``t + source_offset``. A negative offset therefore copies from an
    earlier period.

    Parameters
    ----------
    load:
        Demand data indexed by timestamp.
    max_gap:
        Maximum original gap duration eligible for filling.
    source_offset:
        Temporal offset used to locate source values.
    original_gap_duration:
        Per-cell duration of the original missing run.
    require_complete_source:
        Whether every source value for an eligible gap must be available.

    Returns
    -------
    filled:
        Demand data after applying the copy-period rule.
    newly_filled:
        Boolean DataFrame identifying values filled by this rule.
    """
    max_gap = pd.Timedelta(max_gap)
    source_offset = pd.Timedelta(source_offset)

    eligible = (
        load.isna()
        & original_gap_duration.gt(pd.Timedelta(0))
        & original_gap_duration.le(max_gap)
    )

    source = _values_at_offset(load, source_offset=source_offset)

    if require_complete_source:
        eligible = _require_complete_source_for_each_gap(
            eligible=eligible,
            source=source,
        )
    else:
        eligible &= source.notna()

    filled = load.mask(eligible, source)
    newly_filled = load.isna() & filled.notna()

    return filled, newly_filled


def _values_at_offset(
    load: pd.DataFrame,
    *,
    source_offset: pd.Timedelta,
) -> pd.DataFrame:
    """Align values at a temporal offset to each target timestamp."""
    source_timestamps = load.index + source_offset

    source = load.reindex(source_timestamps)
    source.index = load.index

    return source


def _require_complete_source_for_each_gap(
    *,
    eligible: pd.DataFrame,
    source: pd.DataFrame,
) -> pd.DataFrame:
    """Keep a gap eligible only when every source value exists."""
    result = pd.DataFrame(
        False,
        index=eligible.index,
        columns=eligible.columns,
    )

    for column in eligible.columns:
        eligible_column = eligible[column]
        gap_ids = eligible_column.ne(
            eligible_column.shift(fill_value=False)
        ).cumsum()

        for _, gap_mask in eligible_column.groupby(gap_ids):
            gap_index = gap_mask.index[gap_mask]

            if gap_index.empty:
                continue

            if source.loc[gap_index, column].notna().all():
                result.loc[gap_index, column] = True

    return result