"""Fixed-frequency temporal grid for T-Clean."""

from dataclasses import dataclass

import pandas as pd

from tclean._temporal import normalize_fixed_duration, normalize_temporal_range


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
        normalized_start, normalized_end = normalize_temporal_range(
            start=start, end=end
        )

        normalized_frequency = normalize_fixed_duration(frequency, field="frequency")

        if normalized_frequency <= pd.Timedelta(0):
            raise ValueError("'frequency' must be greater than zero.")

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
    def target_index(self) -> pd.DatetimeIndex:
        """Return the complete end-exclusive index for the output window."""
        return pd.date_range(
            start=self.start,
            end=self.end,
            freq=self.frequency,
            inclusive="left",
            name="timestamp",
        )

    def is_aligned(self, timestamp: pd.Timestamp) -> bool:
        """Return whether a timestamp lies on the configured time grid.

        Args:
            timestamp: Timestamp to test.

        Returns:
            Whether the timestamp lies on the grid.

        Raises:
            ValueError: If the timestamp is timezone-naive.
        """
        timestamp = pd.Timestamp(timestamp)

        if timestamp.tz is None:
            raise ValueError("Timestamp must be timezone-aware.")

        timestamp = timestamp.tz_convert("UTC")

        return (timestamp - self.start) % self.frequency == pd.Timedelta(0)

    def validate_complete_index(self, index: pd.DatetimeIndex) -> None:
        """Validate a complete consecutive index against the grid.

        Args:
            index: Timestamp index to validate.

        Raises:
            TypeError: If index is not a DatetimeIndex.
            ValueError: If timestamps are unsorted, misaligned, or not
                exactly one configured interval apart.
        """
        self._validate_index(index, require_complete=True)

    def validate_sparse_index(self, index: pd.DatetimeIndex) -> None:
        """Validate a potentially sparse index against the grid.

        Args:
            index: Timestamp index to validate.

        Raises:
            TypeError: If index is not a DatetimeIndex.
            ValueError: If timestamps are unsorted or misaligned.
        """
        self._validate_index(index, require_complete=False)

    def validate_duration_multiple(self, duration: pd.Timedelta, *, field: str) -> None:
        """Validate that a duration contains whole configured intervals.

        Args:
            duration: Duration to validate.
            field: Field name used in validation messages.

        Raises:
            ValueError: If the duration is not an integer multiple of the
                configured frequency.
        """
        if duration % self.frequency != pd.Timedelta(0):
            raise ValueError(
                f"'{field}' must be an integer multiple of the configured "
                f"frequency ({self.frequency})."
            )

    def validate_period(self, *, start: pd.Timestamp, end: pd.Timestamp) -> None:
        """Validate period boundaries against the configured grid.

        The period may lie outside the requested output window.

        Args:
            start: Inclusive period start.
            end: Exclusive period end.

        Raises:
            ValueError: If the period is invalid or either boundary does
                not align with the configured grid.
        """
        normalized_start, normalized_end = normalize_temporal_range(
            start=start, end=end
        )

        if not self.is_aligned(normalized_start):
            raise ValueError(
                f"Period start '{normalized_start}' does not align with "
                "the configured time grid."
            )

        if not self.is_aligned(normalized_end):
            raise ValueError(
                f"Period end '{normalized_end}' does not align with "
                "the configured time grid."
            )

    def validate_target_coverage(self, index: pd.DatetimeIndex) -> None:
        """Validate that an index covers the complete requested output window.

        Data may extend before or after the target period.

        Args:
            index: Timestamp index to validate.

        Raises:
            TypeError: If index is not a DatetimeIndex.
            ValueError: If any requested target timestamp is absent.
        """
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError("Time-series index must be a pandas DatetimeIndex.")

        missing = self.target_index.difference(index)

        if not missing.empty:
            raise ValueError(
                "Time-series data do not cover the complete requested "
                f"output window. First missing timestamp: {missing[0]}."
            )

    def crop(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """Crop time-indexed data to the requested output window."""
        return data.loc[(data.index >= self.start) & (data.index < self.end)]

    def _validate_index(
        self,
        index: pd.DatetimeIndex,
        *,
        require_complete: bool,
    ) -> None:
        """Validate timestamp alignment and spacing against the grid."""
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError(
                "Time-series index must be a pandas DatetimeIndex."
            )

        if not index.is_monotonic_increasing:
            raise ValueError(
                "Timestamps must be sorted in ascending order."
            )

        if len(index) == 0:
            return

        if len(index) >= 2:
            differences = index.to_series().diff().dropna()

            if require_complete:
                valid = differences.eq(self.frequency).all()

                if not valid:
                    raise ValueError(
                        "Timestamps must be exactly one configured interval apart. "
                        f"Configured frequency: {self.frequency}."
                    )

            else:
                valid = (
                    differences % self.frequency
                ).eq(pd.Timedelta(0)).all()

                if not valid:
                    raise ValueError(
                        "Timestamp differences must be integer multiples of the "
                        "configured interval. "
                        f"Configured frequency: {self.frequency}."
                    )

        misaligned = [
            timestamp
            for timestamp in index
            if not self.is_aligned(timestamp)
        ]

        if misaligned:
            raise ValueError(
                "Timestamps must align with the configured time grid. "
                f"First misaligned timestamp: {misaligned[0]}."
            )
