"""Pandera schemas for shared T-Clean data contracts."""

import pandas as pd
import pandera.pandas as pa
from pandera.engines.pandas_engine import DateTime


def _has_data_columns(data: pd.DataFrame) -> bool:
    """Check that time-series data contain at least one data column."""
    return data.shape[1] > 0


def _timestamps_are_sorted(data: pd.DataFrame) -> bool:
    """Check that timestamps are monotonically increasing."""
    return data.index.is_monotonic_increasing


TIME_SERIES_SCHEMA = pa.DataFrameSchema(
    {r".+": pa.Column(float, nullable=True, coerce=True, regex=True)},
    index=pa.Index(
        DateTime(tz="UTC", to_datetime_kwargs={"utc": True}),
        name="timestamp",
        nullable=False,
        unique=True,
        coerce=True,
    ),
    checks=[
        pa.Check(
            _has_data_columns, error="Timeseries must contain at least one data column."
        ),
        pa.Check(
            _timestamps_are_sorted,
            error="Timestamps must be sorted in increasing order.",
        ),
    ],
    strict=True,
    unique_column_names=True,
    name="time_series",
)


TEMPORAL_RANGE_SCHEMA = pa.DataFrameSchema(
    {
        "start": pa.Column(
            DateTime(tz="UTC", to_datetime_kwargs={"utc": True}),
            nullable=False,
            coerce=True,
        ),
        "end": pa.Column(
            DateTime(tz="UTC", to_datetime_kwargs={"utc": True}),
            nullable=False,
            coerce=True,
        ),
    },
    checks=[
        pa.Check(
            lambda data: bool((data["end"] > data["start"]).all()),
            error="Temporal range end must be later than start.",
        )
    ],
    strict=True,
    ordered=True,
    unique_column_names=True,
    name="temporal_range",
)


TIMESTAMP_INDEX_SCHEMA = pa.DataFrameSchema(
    {},
    index=pa.Index(
        DateTime(tz="UTC", to_datetime_kwargs={"utc": True}),
        name="timestamp",
        nullable=False,
        unique=True,
        coerce=True,
    ),
    checks=[
        pa.Check(
            _timestamps_are_sorted,
            error="Timestamps must be sorted in increasing order.",
        )
    ],
    strict=True,
    name="timestamp_index",
)
