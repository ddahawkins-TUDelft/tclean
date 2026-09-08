"""Validation interfaces for shared canonical T-Clean data."""

import pandas as pd

from tclean._schemas import TIME_SERIES_SCHEMA, TIMESTAMP_INDEX_SCHEMA
from tclean.time_grid import TimeGrid


def validate_time_series(data: pd.DataFrame, *, grid: TimeGrid) -> pd.DataFrame:
    """Validate canonical time-series data.

    Args:
        data: Time-series data with timestamps on the index and one or
            more contexts in the columns.
        grid: Temporal grid against which timestamps are validated.

    Returns:
        Validated canonical time-series data.

    Raises:
        pandera.errors.SchemaErrors: If the data violate the canonical
            T-Clean time-series contract.
        ValueError: If timestamps do not form a complete configured grid.
    """
    validated = TIME_SERIES_SCHEMA.validate(data, lazy=True)

    index = _require_datetime_index(validated.index, field="Time-series index")

    grid.validate_complete_index(index)

    return validated


def validate_timestamp_index(
    index: pd.DatetimeIndex, *, grid: TimeGrid
) -> pd.DatetimeIndex:
    """Validate a canonical regular timestamp index.

    Args:
        index: Timestamp index to validate.
        grid: Temporal grid against which timestamps are validated.

    Returns:
        Validated timestamp index.

    Raises:
        TypeError: If index is not a pandas DatetimeIndex.
        ValueError: If timestamps do not form a complete configured grid.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("'index' must be a pandas DatetimeIndex.")

    frame = pd.DataFrame(index=index)
    validated = TIMESTAMP_INDEX_SCHEMA.validate(frame, lazy=True)

    validated_index = _require_datetime_index(validated.index, field="Timestamp index")

    grid.validate_complete_index(validated_index)

    return validated_index


def _validate_periods_against_grid(periods: pd.DataFrame, *, grid: TimeGrid) -> None:
    """Validate period boundaries against a configured time grid."""
    for start, end in periods[["start", "end"]].itertuples(index=False, name=None):
        grid.validate_period(start=pd.Timestamp(start), end=pd.Timestamp(end))


def _require_datetime_index(index: pd.Index, *, field: str) -> pd.DatetimeIndex:
    """Narrow an already validated index to a DatetimeIndex."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{field} must be a pandas DatetimeIndex.")

    return index
