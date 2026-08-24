"""Fixed-frequency temporal grid for T-Clean."""

import re
from dataclasses import dataclass

import pandas as pd

from tclean.validation import validate_temporal_range


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

    if pd.isna(frequency):
        raise ValueError("'frequency' must not be missing.")

    if frequency <= pd.Timedelta(0):
        raise ValueError("'frequency' must be greater than zero.")

    return frequency


@dataclass(frozen=True, init=False)
class TimeGrid:
    """Fixed-frequency temporal grid and requested output window.

    Attributes:
        start: Inclusive start of the requested output period.
        end: Exclusive end of the requested output period.
        frequency: Fixed interval between consecutive grid timestamps.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    frequency: pd.Timedelta

    def __init__(
        self,
        *,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        frequency: str | pd.Timedelta,
    ) -> None:
        """Create and validate a fixed-frequency time grid.

        Args:
            start: Inclusive start of the requested output period.
            end: Exclusive end of the requested output period.
            frequency: Fixed interval between consecutive observations.

        Raises:
            ValueError: If the temporal range or frequency is invalid, or
                if the output period does not contain a whole number of
                configured time steps.
        """
        normalized_start, normalized_end = validate_temporal_range(start=start, end=end)
        normalized_frequency = _normalize_frequency(frequency)

        duration = normalized_end - normalized_start

        if duration % normalized_frequency != pd.Timedelta(0):
            raise ValueError(
                "The interval from 'start' to 'end' must contain an integer "
                "number of configured time steps."
            )

        object.__setattr__(self, "start", normalized_start)
        object.__setattr__(self, "end", normalized_end)
        object.__setattr__(self, "frequency", normalized_frequency)

    @property
    def duration(self) -> pd.Timedelta:
        """Return the duration of the requested output period."""
        return self.end - self.start

    @property
    def n_periods(self) -> int:
        """Return the number of configured periods in the output window."""
        return int(self.duration / self.frequency)

    @property
    def index(self) -> pd.DatetimeIndex:
        """Return the complete canonical index for the output window."""
        return pd.date_range(
            start=self.start,
            end=self.end,
            freq=self.frequency,
            inclusive="left",
            name="timestamp",
        )

    def is_aligned(self, timestamp: pd.Timestamp) -> bool:
        """Return whether a timestamp lies on this grid."""
        timestamp = pd.Timestamp(timestamp)

        if timestamp.tz is None:
            raise ValueError("Timestamp must be timezone-aware.")

        timestamp = timestamp.tz_convert("UTC")

        return (timestamp - self.start) % self.frequency == pd.Timedelta(0)

    def validate_aligned_index(
        self, index: pd.DatetimeIndex, *, require_complete: bool
    ) -> None:
        """Validate timestamp alignment and spacing against the grid.

        Args:
            index: Timestamp index to validate.
            require_complete: Whether consecutive timestamps must be exactly
                one configured interval apart. If False, gaps are permitted
                provided all timestamps remain on the configured grid.

        Raises:
            ValueError: If timestamps are not aligned or spacing is invalid.
        """
        if len(index) == 0:
            return

        misaligned = [
            timestamp for timestamp in index if not self.is_aligned(timestamp)
        ]

        if misaligned:
            raise ValueError(
                "Timestamps must align with the configured time grid. "
                f"First misaligned timestamp: {misaligned[0]}."
            )

        if len(index) < 2:
            return

        differences = index.to_series().diff().dropna()

        if require_complete:
            valid = differences.eq(self.frequency).all()
        else:
            valid = (differences % self.frequency).eq(pd.Timedelta(0)).all()

        if not valid:
            requirement = (
                "exactly one configured interval apart"
                if require_complete
                else "integer multiples of the configured interval"
            )

            raise ValueError(
                f"Timestamps must be {requirement}. "
                f"Configured frequency: {self.frequency}."
            )

    def crop(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """Crop time-indexed data to the requested output window."""
        return data.loc[(data.index >= self.start) & (data.index < self.end)]
