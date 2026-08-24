"""Read externally supplied time-series profiles."""

from pathlib import Path

import pandas as pd

from tclean._schemas import EXTERNAL_PROFILE_SCHEMA
from tclean.time_grid import TimeGrid
from tclean.validation import validate_advanced_source

METHOD_NAME = "external_profile"


def read_external_profile(
    path: str | Path,
    *,
    grid: TimeGrid,
) -> pd.Series:
    """Read and validate an external time-series profile.

    Args:
        path: CSV file containing timestamp and value columns.
        grid: Temporal grid against which source timestamps are validated.

    Returns:
        Validated numeric time series indexed by UTC timestamps.

    Raises:
        pandera.errors.SchemaErrors: If the external profile violates the
            T-Clean external-profile contract.
        ValueError: If timestamps do not lie on the configured time grid.
    """
    data = pd.read_csv(path)

    validated = EXTERNAL_PROFILE_SCHEMA.validate(
        data,
        lazy=True,
    )

    validated = validated.sort_values(
        "timestamp",
        kind="stable",
    )

    source = pd.Series(
        validated["value"].to_numpy(),
        index=pd.DatetimeIndex(
            validated["timestamp"],
            name="timestamp",
        ),
        name="value",
    )

    return validate_advanced_source(
        source,
        grid=grid,
    )
