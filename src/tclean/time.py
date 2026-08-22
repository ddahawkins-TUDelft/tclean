"""Utilities for constructing canonical time indexes."""

import pandas as pd

from tclean.validation import validate_temporal_range


def build_time_index(
    start: object, end: object, *, frequency: pd.Timedelta
) -> pd.DatetimeIndex:
    """Build an end-exclusive regular timestamp index.

    Args:
        start: Inclusive start timestamp.
        end: Exclusive end timestamp.
        frequency: Fixed interval between consecutive timestamps.

    Returns:
        Regular UTC timestamp index.
    """
    start, end = validate_temporal_range(start, end)

    duration = end - start

    if duration % frequency != pd.Timedelta(0):
        raise ValueError(
            "Temporal range must contain an integer "
            "number of configured intervals. "
            f"Duration: {duration}; "
            f"frequency: {frequency}."
        )

    return pd.date_range(
        start=start, end=end, freq=frequency, inclusive="left", name="timestamp"
    )
