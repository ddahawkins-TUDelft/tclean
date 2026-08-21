"""Validation interfaces for canonical T-Clean data."""

import pandas as pd

from tclean._schemas import (
    DEMAND_SCHEMA,
    HOURLY_TIMESTAMP_INDEX_SCHEMA,
    SOURCE_PERIODS_SCHEMA,
    TEMPORAL_RANGE_SCHEMA,
)


def validate_load(load: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize canonical electricity-demand data.

    Args:
        load: Electricity-demand data with timestamps as the index and
            one demand column per region.

    Returns:
        Validated demand data with a UTC ``timestamp`` index and
        floating-point demand columns.

    Raises:
        pandera.errors.SchemaErrors: If the demand data violate the
            canonical T-Clean demand contract.
    """
    return DEMAND_SCHEMA.validate(load, lazy=True)


def validate_temporal_range(
    *, start: object, end: object
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate and normalize a temporal range.

    Args:
        start: Start of the temporal range.
        end: End of the temporal range.

    Returns:
        Validated UTC start and end timestamps.

    Raises:
        pandera.errors.SchemaErrors: If the temporal range violates
            the T-Clean temporal contract.
    """
    temporal_range = pd.DataFrame({"start": [start], "end": [end]})

    validated = TEMPORAL_RANGE_SCHEMA.validate(temporal_range, lazy=True)

    return (
        pd.Timestamp(validated.loc[0, "start"]),
        pd.Timestamp(validated.loc[0, "end"]),
    )


def infer_regular_timestep(index: pd.Index) -> pd.Timedelta:
    """Return the timestep of an already validated datetime index.

    Args:
        index: Validated regular datetime index.

    Returns:
        The temporal difference between consecutive timestamps.
    """
    return index[1] - index[0]


def validate_hourly_timestamp_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Validate and normalize a canonical hourly timestamp index.

    Args:
        index: Timestamp index to validate.

    Returns:
        Validated UTC hourly timestamp index named ``timestamp``.

    Raises:
        pandera.errors.SchemaErrors: If the index violates the canonical
            hourly timestamp contract.
    """
    frame = pd.DataFrame(index=index)

    validated = HOURLY_TIMESTAMP_INDEX_SCHEMA.validate(frame, lazy=True)

    return validated.index


def validate_source_periods(source_periods: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize source-period definitions.

    Args:
        source_periods: Source definitions containing country, start,
            end, and weight columns.

    Returns:
        Validated source-period definitions with UTC timestamps and
        floating-point positive weights.

    Raises:
        pandera.errors.SchemaErrors: If the source-period definitions
            violate the T-Clean source-period contract.
    """
    return SOURCE_PERIODS_SCHEMA.validate(source_periods, lazy=True)
