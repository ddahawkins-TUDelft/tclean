"""Build reports for unresolved gaps in electricity-demand data."""

from __future__ import annotations

from typing import Any

import pandas as pd

from tclean.validation import validate_load


def build_gap_report(load: pd.DataFrame, *, enabled: bool) -> pd.DataFrame:
    """Describe contiguous unresolved gaps in cleaned demand data.

    An empty report with the expected columns is returned when reporting
    is disabled or when no unresolved gaps remain.

    Parameters
    ----------
    load:
        Hourly electricity-demand data indexed by timestamp.
    enabled:
        Whether unresolved-gap reporting should be performed.

    Returns:
    -------
    pandas.DataFrame
        One row per contiguous unresolved gap, including its country,
        temporal extent, duration, and whether it touches either boundary
        of the supplied demand series.
    """
    columns = [
        "country",
        "gap_start",
        "gap_end",
        "gap_hours",
        "touches_start_boundary",
        "touches_end_boundary",
    ]

    if not enabled:
        return pd.DataFrame(columns=columns)

    load = validate_load(load)

    records: list[dict[str, Any]] = []

    first_timestamp = load.index[0]
    last_timestamp = load.index[-1]

    for country in load.columns:
        missing = load[country].isna()

        if not missing.any():
            continue

        group_ids = missing.ne(missing.shift(fill_value=False)).cumsum()

        for _, group in missing.groupby(group_ids):
            if not bool(group.iloc[0]):
                continue

            timestamps = group.index

            records.append(
                {
                    "country": country,
                    "gap_start": timestamps[0],
                    "gap_end": timestamps[-1] + pd.Timedelta(hours=1),
                    "gap_hours": len(timestamps),
                    "touches_start_boundary": (timestamps[0] == first_timestamp),
                    "touches_end_boundary": (timestamps[-1] == last_timestamp),
                }
            )

    report = pd.DataFrame.from_records(records, columns=columns)

    if report.empty:
        return report

    return report.sort_values(["country", "gap_start"]).reset_index(drop=True)
