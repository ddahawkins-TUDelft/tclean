"""Gap filling using averages from corresponding periods."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

METHOD_NAME = "average_periods"


def apply_average_periods(
    data: pd.DataFrame,
    *,
    max_gap: str | pd.Timedelta,
    source_offsets: Sequence[str | pd.Timedelta],
    original_gap_duration: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill eligible gaps using the mean of complete source periods.

    For each target timestamp ``t``, source values are taken from
    ``t + source_offset`` for each configured source offset.

    Parameters
    ----------
    data:
        Data indexed by timestamp.
    max_gap:
        Maximum original gap duration eligible for filling.
    source_offsets:
        Temporal offsets used to locate corresponding source values.
    original_gap_duration:
        Per-cell duration of the original missing run.

    Returns:
    -------
    filled:
        Data after applying the averaging rule.
    newly_filled:
        Boolean DataFrame identifying values filled by this rule.
    """
    max_gap = pd.Timedelta(max_gap)
    offsets = tuple(pd.Timedelta(offset) for offset in source_offsets)

    eligible = (
        data.isna()
        & original_gap_duration.gt(pd.Timedelta(0))
        & original_gap_duration.le(max_gap)
    )

    sources = [_values_at_offset(data, source_offset=offset) for offset in offsets]

    candidate = _mean_complete_sources(sources=sources)
    eligible &= candidate.notna()

    filled = data.mask(eligible, candidate)
    newly_filled = data.isna() & filled.notna()

    return filled, newly_filled


def _values_at_offset(
    data: pd.DataFrame, *, source_offset: pd.Timedelta
) -> pd.DataFrame:
    """Align values at a temporal offset to each target timestamp."""
    source_timestamps = data.index + source_offset

    source = data.reindex(source_timestamps)
    source.index = data.index

    return source


def _mean_complete_sources(*, sources: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Calculate a mean only where every configured source exists."""
    source_sum = sources[0].copy()
    complete = sources[0].notna()

    for source in sources[1:]:
        source_sum = source_sum + source
        complete &= source.notna()

    candidate = source_sum / len(sources)

    return candidate.where(complete)
