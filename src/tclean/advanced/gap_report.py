"""Build reports for unresolved gaps in time series data."""

from __future__ import annotations

from typing import Any

import pandas as pd

from tclean.validation import validate_time_series


def build_gap_report(
    data: pd.DataFrame, *, frequency: pd.Timedelta, enabled: bool
) -> pd.DataFrame:
    """Describe contiguous unresolved gaps in cleaned data.

    An empty report with the expected columns is returned when reporting
    is disabled or when no unresolved gaps remain.

    Parameters
    ----------
    data:
        Hourly data indexed by timestamp.
    frequency:
        pd.Timedelta of time series frequency.
    enabled:
        Whether unresolved-gap reporting should be performed.

    Returns:
    -------
    pandas.DataFrame
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

    data = validate_time_series(data, frequency=frequency)

    records: list[dict[str, Any]] = []

    first_timestamp = data.index[0]
    last_timestamp = data.index[-1]

    for context in data.columns:
        missing = data[context].isna()

        if not missing.any():
            continue

        group_ids = missing.ne(missing.shift(fill_value=False)).cumsum()

        for _, group in missing.groupby(group_ids):
            if not bool(group.iloc[0]):
                continue

            timestamps = group.index

            records.append(
                {
                    "context": context,
                    "gap_start": timestamps[0],
                    "gap_end": timestamps[-1] + frequency,
                    "gap_duration": len(timestamps) * frequency,
                    "touches_start_boundary": (timestamps[0] == first_timestamp),
                    "touches_end_boundary": (timestamps[-1] == last_timestamp),
                }
            )

    report = pd.DataFrame.from_records(records, columns=columns)

    if report.empty:
        return report

    return report.sort_values(["context", "gap_start"]).reset_index(drop=True)
