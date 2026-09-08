"""Build reports describing unresolved gaps."""

import pandas as pd

from tclean.time_grid import TimeGrid
from tclean.validation import validate_time_series


def build_gap_report(
    data: pd.DataFrame, *, grid: TimeGrid, enabled: bool
) -> pd.DataFrame:
    """Describe contiguous unresolved gaps in cleaned data.

    An empty report with the expected columns is returned when reporting
    is disabled or when no unresolved gaps remain.

    Args:
        data: Time-series data indexed by timestamp.
        grid: Temporal grid describing the time-series frequency.
        enabled: Whether unresolved-gap reporting should be performed.

    Returns:
        One row per contiguous unresolved gap, including its context,
        temporal extent, duration, and whether it touches either boundary
        of the supplied time series.
    """
    columns = [
        "context",
        "gap_start",
        "gap_end",
        "gap_duration",
        "touches_start_boundary",
        "touches_end_boundary",
    ]

    if not enabled:
        return pd.DataFrame(columns=columns)

    data = validate_time_series(data, grid=grid)

    gaps: list[dict[str, object]] = []

    for context in data.columns:
        missing = data[context].isna()

        if not missing.any():
            continue

        group_ids = missing.ne(missing.shift()).cumsum()

        for _, group in missing.groupby(group_ids):
            if not bool(group.iloc[0]):
                continue

            gap_index = group.index

            gap_start = gap_index[0]
            gap_end = gap_index[-1] + grid.frequency

            gaps.append(
                {
                    "context": context,
                    "gap_start": gap_start,
                    "gap_end": gap_end,
                    "gap_duration": len(gap_index) * grid.frequency,
                    "touches_start_boundary": gap_index[0] == data.index[0],
                    "touches_end_boundary": gap_index[-1] == data.index[-1],
                }
            )

    if not gaps:
        return pd.DataFrame(columns=columns)

    report = pd.DataFrame(gaps, columns=columns)

    return report.sort_values(["context", "gap_start"], kind="stable").reset_index(
        drop=True
    )
