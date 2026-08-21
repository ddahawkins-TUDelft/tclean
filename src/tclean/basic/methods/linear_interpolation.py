"""Linear interpolation for bounded gaps in demand time series."""

from __future__ import annotations

import pandas as pd

METHOD_NAME = "linear_interpolation"


def apply_linear_interpolation(
    load: pd.DataFrame,
    *,
    max_gap: str | pd.Timedelta,
    original_gap_duration: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill eligible bounded gaps using time-based linear interpolation.

    Only complete original gaps whose duration is less than or equal to
    ``max_gap`` are eligible.

    Parameters
    ----------
    load:
        Demand data indexed by timestamp.
    max_gap:
        Maximum original gap duration eligible for interpolation.
    original_gap_duration:
        Per-cell duration of the original missing run.

    Returns:
    -------
    filled:
        Demand data after applying interpolation.
    newly_filled:
        Boolean DataFrame identifying values filled by this rule.
    """
    max_gap = pd.Timedelta(max_gap)

    eligible = (
        load.isna()
        & original_gap_duration.gt(pd.Timedelta(0))
        & original_gap_duration.le(max_gap)
    )

    interpolated = load.interpolate(method="time", limit_area="inside")
    filled = load.mask(eligible, interpolated)
    newly_filled = load.isna() & filled.notna()

    return filled, newly_filled
