"""data and validate user-supplied external profiles."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tclean._schemas import EXTERNAL_PROFILE_SCHEMA
from tclean.validation import validate_advanced_source

METHOD_NAME = "external_profile"


def read_external_profile(path: str | Path, *, frequency: pd.Timedelta) -> pd.Series:
    """Data and validate an external profile.

    The CSV is read once and validated against the strict T-Clean
    external-profile schema.

    Args:
        path: Path to the external profile CSV file.
        frequency: pd.Timedelta of time series frequency.

    Returns:
        Validated time series indexed by sorted UTC timestamps named
        ``timestamp``.

    Raises:
        pandera.errors.SchemaErrors: If the CSV violates the external-profile
            contract.
        ValueError: If timestamps are incompatible with the configured
            frequency.
    """
    profile = pd.read_csv(path)

    validated = EXTERNAL_PROFILE_SCHEMA.validate(profile, lazy=True)

    series = pd.Series(
        validated["value"].to_numpy(),
        index=pd.DatetimeIndex(validated["timestamp"], name="timestamp"),
        dtype=float,
        name="value",
    ).sort_index()

    return validate_advanced_source(series, frequency=frequency)
