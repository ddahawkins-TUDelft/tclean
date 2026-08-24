"""Global configuration for T-Clean."""

from dataclasses import dataclass

import pandas as pd

from tclean.time_grid import TimeGrid


@dataclass(frozen=True, init=False)
class TCleanConfig:
    """Global configuration for T-Clean.

    Attributes:
        grid: Fixed-frequency temporal grid and requested output window.
    """

    grid: TimeGrid

    def __init__(
        self,
        *,
        frequency: str | pd.Timedelta,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> None:
        """Create a T-Clean configuration."""
        grid = TimeGrid(start=start, end=end, frequency=frequency)

        object.__setattr__(self, "grid", grid)

    @property
    def start(self) -> pd.Timestamp:
        """Inclusive start of the requested output period."""
        return self.grid.start

    @property
    def end(self) -> pd.Timestamp:
        """Exclusive end of the requested output period."""
        return self.grid.end

    @property
    def frequency(self) -> pd.Timedelta:
        """Fixed interval between consecutive observations."""
        return self.grid.frequency
