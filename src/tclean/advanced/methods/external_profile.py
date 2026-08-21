"""Load and validate user-supplied external demand profiles."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tclean._schemas import EXTERNAL_PROFILE_SCHEMA

METHOD_NAME = "external_profile"


def load_external_profile(path: str | Path) -> pd.Series:
    """Load and validate an external electricity-demand profile.

    The CSV is read once and validated against the strict T-Clean
    external-profile schema.

    Args:
        path: Path to the external profile CSV file.

    Returns:
        Validated demand indexed by sorted UTC timestamps named
        ``timestamp``.

    Raises:
        pandera.errors.SchemaErrors: If the CSV violates the external-profile
            contract.
    """
    profile = pd.read_csv(path)

    validated = EXTERNAL_PROFILE_SCHEMA.validate(profile, lazy=True)

    return pd.Series(
        validated["demand"].to_numpy(),
        index=pd.DatetimeIndex(validated["timestamp"], name="timestamp"),
        dtype=float,
        name="demand",
    ).sort_index()
