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


def _timestamps_are_on_hour(series: pd.Series) -> pd.Series:
    """Check that timestamps are aligned exactly to whole hours."""
    return series.dt.minute.eq(0) & series.dt.second.eq(0) & series.dt.microsecond.eq(0)


def _source_period_ends_after_start(data: pd.DataFrame) -> pd.Series:
    """Check that every source period ends after it starts."""
    return data["end"] > data["start"]


def _has_source_periods(data: pd.DataFrame) -> bool:
    """Check that at least one source period is configured."""
    return len(data) > 0


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
                _timestamps_are_on_hour,
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

HOURLY_TIMESTAMP_INDEX_SCHEMA = pa.DataFrameSchema(
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
            _has_multiple_timestamps,
            error="Hourly timestamp index must contain at least two timestamps.",
        ),
        pa.Check(
            _timestamps_are_sorted,
            error="Hourly timestamps must be sorted in increasing order.",
        ),
        pa.Check(
            _timestamps_are_hourly,
            error="Timestamp index must form a complete hourly sequence.",
        ),
    ],
    strict=True,
    name="hourly_timestamp_index",
)

SOURCE_PERIODS_SCHEMA = pa.DataFrameSchema(
    {
        "country": pa.Column(str, nullable=False),
        "start": pa.Column(
            DateTime(tz="UTC", to_datetime_kwargs={"utc": True}),
            checks=pa.Check(
                _timestamps_are_on_hour,
                error="Source-period start timestamps must be aligned to whole hours.",
            ),
            nullable=False,
            coerce=True,
        ),
        "end": pa.Column(
            DateTime(tz="UTC", to_datetime_kwargs={"utc": True}),
            checks=pa.Check(
                _timestamps_are_on_hour,
                error="Source-period end timestamps must be aligned to whole hours.",
            ),
            nullable=False,
            coerce=True,
        ),
        "weight": pa.Column(float, checks=pa.Check.gt(0), nullable=False, coerce=True),
    },
    checks=[
        pa.Check(
            _has_source_periods, error="At least one source period must be configured."
        ),
        pa.Check(
            _source_period_ends_after_start,
            error="Each source period must end later than it starts.",
        ),
    ],
    strict=True,
    ordered=True,
    unique_column_names=True,
    name="source_periods",
)

CLEANING_METHOD_SCHEMA = pa.DataFrameSchema(
    {r".+": pa.Column("string", nullable=True, coerce=True, regex=True)},
    strict=True,
    unique_column_names=True,
    name="cleaning_method",
)


def _advanced_method_is_supported(series: pd.Series) -> pd.Series:
    """Check that advanced-fill methods are supported."""
    return series.isin(["construct_from_sources", "external_profile", "leave_missing"])


def _advanced_scope_is_supported(series: pd.Series) -> pd.Series:
    """Check that advanced-fill scopes are supported."""
    return series.isin(["fill_gaps", "overwrite"])


ADVANCED_FILL_RULES_SCHEMA = pa.DataFrameSchema(
    {
        "rule_name": pa.Column(str, nullable=False),
        "method": pa.Column(
            str,
            checks=pa.Check(
                _advanced_method_is_supported, error="Unsupported advanced-fill method."
            ),
            nullable=False,
        ),
        "country": pa.Column(str, nullable=False),
        "start": pa.Column(
            DateTime(tz="UTC", to_datetime_kwargs={"utc": True}),
            checks=pa.Check(
                _timestamps_are_on_hour,
                error="Advanced-fill start timestamps must align to whole hours.",
            ),
            nullable=False,
            coerce=True,
        ),
        "end": pa.Column(
            DateTime(tz="UTC", to_datetime_kwargs={"utc": True}),
            checks=pa.Check(
                _timestamps_are_on_hour,
                error="Advanced-fill end timestamps must align to whole hours.",
            ),
            nullable=False,
            coerce=True,
        ),
        "scope": pa.Column(
            str,
            checks=pa.Check(
                _advanced_scope_is_supported, error="Unsupported advanced-fill scope."
            ),
            nullable=False,
        ),
    },
    checks=[
        pa.Check(
            lambda data: data["rule_name"].is_unique,
            error="Advanced-fill rule names must be unique.",
        ),
        pa.Check(
            lambda data: data["end"] > data["start"],
            error="Advanced-fill rules must end later than they start.",
        ),
    ],
    strict=True,
    ordered=True,
    unique_column_names=True,
    name="advanced_fill_rules",
)
