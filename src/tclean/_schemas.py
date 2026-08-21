"""Pandera schemas defining T-Clean data contracts."""

import pandas as pd
import pandera.pandas as pa
from pandera.engines.pandas_engine import DateTime


def _has_demand_columns(data: pd.DataFrame) -> bool:
    """Check that demand data contain at least one demand column."""
    return data.shape[1] > 0


def _has_multiple_timestamps(data: pd.DataFrame) -> bool:
    """Check that demand data contain enough timestamps to infer a timestep."""
    return len(data.index) >= 2


def _timestamps_are_sorted(data: pd.DataFrame) -> bool:
    """Check that timestamps are monotonically increasing."""
    return data.index.is_monotonic_increasing


def _timestamps_are_hourly(data: pd.DataFrame) -> bool:
    """Check that timestamps form a complete hourly sequence."""
    differences = data.index.to_series().diff().dropna()

    return bool(not differences.empty and differences.eq(pd.Timedelta(hours=1)).all())


def _external_timestamps_are_hourly(series: pd.Series) -> pd.Series:
    """Check that external-profile timestamps align to whole hours."""
    return series.dt.minute.eq(0) & series.dt.second.eq(0) & series.dt.microsecond.eq(0)


DEMAND_SCHEMA = pa.DataFrameSchema(
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
            _has_demand_columns,
            error="Demand data must contain at least one demand column.",
        ),
        pa.Check(
            _has_multiple_timestamps,
            error="Demand data must contain at least two timestamps.",
        ),
        pa.Check(
            _timestamps_are_sorted,
            error="Demand timestamps must be sorted in increasing order.",
        ),
        pa.Check(
            _timestamps_are_hourly,
            error="Demand data must use a complete hourly time index.",
        ),
    ],
    strict=True,
    unique_column_names=True,
    name="demand",
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


EXTERNAL_PROFILE_SCHEMA = pa.DataFrameSchema(
    {
        "timestamp": pa.Column(
            DateTime(tz="UTC", to_datetime_kwargs={"utc": True}),
            checks=pa.Check(
                _external_timestamps_are_hourly,
                error="Timestamps must be aligned to whole hours.",
            ),
            nullable=False,
            unique=True,
            coerce=True,
        ),
        "demand": pa.Column(float, nullable=False, coerce=True),
    },
    strict=True,
    ordered=True,
    unique_column_names=True,
    name="external_profile",
)
