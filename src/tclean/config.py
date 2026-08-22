"""Global configuration for T-Clean."""

import re
from dataclasses import dataclass

import pandas as pd


def _normalize_frequency(value: str | pd.Timedelta) -> pd.Timedelta:
    """Validate and normalize a fixed time-series frequency."""
    if pd.isna(value):
        raise ValueError("'frequency' must not be missing.")

    if isinstance(value, str):
        value = value.strip()

        if not value:
            raise ValueError("'frequency' must not be empty.")

        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value):
            raise ValueError("'frequency' must include an explicit duration unit.")

    elif not isinstance(value, pd.Timedelta):
        raise TypeError("'frequency' must be a duration string or pandas Timedelta.")

    try:
        frequency = pd.Timedelta(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("'frequency' must be a valid fixed duration.") from exc

    if frequency <= pd.Timedelta(0):
        raise ValueError("'frequency' must be greater than zero.")

    return frequency


@dataclass(frozen=True, init=False)
class TCleanConfig:
    """Global configuration for T-Clean.

    Attributes:
        frequency: Fixed time interval between consecutive observations.
    """

    frequency: pd.Timedelta

    def __init__(self, *, frequency: str | pd.Timedelta) -> None:
        """Create a T-Clean configuration.

        Args:
            frequency: Fixed interval between consecutive observations,
                such as ``"30min"``, ``"1h"``, or ``"2h"``.
        """
        object.__setattr__(self, "frequency", _normalize_frequency(frequency))
