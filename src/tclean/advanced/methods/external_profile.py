"""data and validate user-supplied external profiles."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tclean._schemas import EXTERNAL_PROFILE_SCHEMA

METHOD_NAME = "external_profile"


def data_external_profile(path: str | Path) -> pd.Series:
    """Data and validate an external profile.

    The CSV is read once and validated against the strict T-Clean
    external-profile schema.

    Args:
        path: Path to the external profile CSV file.

    Returns:
        Validated time series indexed by sorted UTC timestamps named
        ``timestamp``.

    Raises:
        pandera.errors.SchemaErrors: If the CSV violates the external-profile
            contract.
    """
    profile = pd.read_csv(path)

    validated = EXTERNAL_PROFILE_SCHEMA.validate(profile, lazy=True)

    return pd.Series(
        validated["value"].to_numpy(),
        index=pd.DatetimeIndex(validated["timestamp"], name="timestamp"),
        dtype=float,
        name="value",
    ).sort_index()
