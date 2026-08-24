"""Low-level temporal normalization for T-Clean."""

import re

import pandas as pd

from tclean._schemas import TEMPORAL_RANGE_SCHEMA


def normalize_fixed_duration(value: str | pd.Timedelta, *, field: str) -> pd.Timedelta:
    """Validate and normalize a fixed duration.

    Args:
        value: Duration string or pandas Timedelta.
        field: Field name used in validation messages.

    Returns:
        Validated pandas Timedelta.

    Raises:
        TypeError: If the value is not a supported duration type.
        ValueError: If the value is missing, empty, or invalid.
    """
    if pd.isna(value):
        raise ValueError(f"'{field}' must not be missing.")

    if isinstance(value, str):
        value = value.strip()

        if not value:
            raise ValueError(f"'{field}' must not be empty.")

        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value):
            raise ValueError(f"'{field}' must include an explicit duration unit.")

    elif not isinstance(value, pd.Timedelta):
        raise TypeError(f"'{field}' must be a duration string or pandas Timedelta.")

    try:
        duration = pd.Timedelta(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field}' must be a valid fixed duration.") from exc

    if pd.isna(duration):
        raise ValueError(f"'{field}' must not be missing.")

    return duration


def normalize_temporal_range(
    *, start: object, end: object
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate and normalize a temporal range.

    Args:
        start: Inclusive start of the temporal range.
        end: Exclusive end of the temporal range.

    Returns:
        Validated UTC start and end timestamps.
    """
    temporal_range = pd.DataFrame({"start": [start], "end": [end]})

    validated = TEMPORAL_RANGE_SCHEMA.validate(temporal_range, lazy=True)

    return (
        pd.Timestamp(validated.loc[0, "start"]),
        pd.Timestamp(validated.loc[0, "end"]),
    )
