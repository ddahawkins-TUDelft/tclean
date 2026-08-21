"""Utilities for constructing canonical time indexes."""

import pandas as pd

from tclean.validation import validate_temporal_range


def build_hourly_index(*, start: object, end: object) -> pd.DatetimeIndex:
    """Build an end-exclusive canonical hourly timestamp index.

    Args:
        start: Start of the requested temporal range.
        end: End of the requested temporal range.

    Returns:
        An hourly UTC datetime index from start up to, but excluding, end.

    Raises:
        pandera.errors.SchemaErrors: If the temporal range is invalid.
    """
    start_timestamp, end_timestamp = validate_temporal_range(start=start, end=end)

    return pd.date_range(
        start=start_timestamp,
        end=end_timestamp,
        freq="h",
        inclusive="left",
        name="timestamp",
    )
