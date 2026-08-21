"""Validation interfaces for canonical T-Clean data."""

import pandas as pd

from tclean._schemas import DEMAND_SCHEMA, TEMPORAL_RANGE_SCHEMA


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
