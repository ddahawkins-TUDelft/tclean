"""Helpers for converting quality-failure masks to periods."""

import pandas as pd
from pandas.api.types import is_bool_dtype

from tclean.time_grid import TimeGrid


def failure_mask_to_periods(
    mask: pd.Series, *, grid: TimeGrid
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Convert failed grid timestamps into contiguous half-open periods.

    Each ``True`` value represents a failing observation at the corresponding
    timestamp. Consecutive failures are merged into one ``[start, end)`` period.

    Args:
        mask: Boolean failure mask indexed exactly by the target grid.
        grid: Temporal grid defining the timestamp sequence and exclusive end.

    Returns:
        Ordered ``(start, end)`` pairs for contiguous failure periods.

    Raises:
        TypeError: If ``mask`` is not a pandas Series or is not Boolean.
        ValueError: If the mask contains missing values or does not use the
            complete target grid index.
    """
    if not isinstance(mask, pd.Series):
        raise TypeError("Failure mask must be a pandas Series.")

    if not is_bool_dtype(mask.dtype):
        raise TypeError("Failure mask must contain boolean values.")

    if mask.isna().any():
        raise ValueError("Failure mask must not contain missing values.")

    target_index = pd.DatetimeIndex(grid.target_index)

    if not mask.index.equals(target_index):
        raise ValueError("Failure mask index must exactly match the target time grid.")

    failed_positions = [
        position for position, failed in enumerate(mask.to_numpy(dtype=bool)) if failed
    ]

    if not failed_positions:
        return []

    periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    run_start = failed_positions[0]
    run_end = failed_positions[0]

    for position in failed_positions[1:]:
        if position == run_end + 1:
            run_end = position
            continue

        periods.append((target_index[run_start], target_index[run_end + 1]))

        run_start = position
        run_end = position

    if run_end + 1 < len(target_index):
        end = target_index[run_end + 1]
    else:
        end = pd.Timestamp(grid.end)

    periods.append((target_index[run_start], end))

    return periods
