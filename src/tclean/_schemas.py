"""Pandera schemas defining T-Clean data contracts."""

import pandas as pd
import pandera.pandas as pa
from pandera.engines.pandas_engine import DateTime


def _has_data_columns(data: pd.DataFrame) -> bool:
    """Check that time series data contain at least one data column."""
    return data.shape[1] > 0


def _has_multiple_timestamps(data: pd.DataFrame) -> bool:
    """Check that time series data contain enough timestamps to infer a timestep."""
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


def _has_rows(data: pd.DataFrame) -> bool:
    """Check that a tabular contract contains at least one row."""
    return not data.empty


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
            _has_multiple_timestamps,
            error="Timeseries must contain at least two timestamps.",
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
        "value": pa.Column(float, nullable=False, coerce=True),
    },
    strict=True,
    ordered=True,
    unique_column_names=True,
    name="external_profile",
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

SOURCE_PERIODS_SCHEMA = pa.DataFrameSchema(
    {
        "context": pa.Column(str, nullable=False),
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
        pa.Check(_has_rows, error="At least one source period must be configured."),
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

PROVENANCE_SCHEMA = pa.DataFrameSchema(
    {r".+": pa.Column("string", nullable=True, coerce=True, regex=True)},
    strict=True,
    unique_column_names=True,
    name="provenance",
)


def _advanced_method_is_supported(series: pd.Series) -> pd.Series:
    """Check that advanced-fill methods are supported."""
    return series.isin(["construct_from_sources", "external_profile", "leave_missing"])


def _advanced_scope_is_supported(series: pd.Series) -> pd.Series:
    """Check that advanced-fill scopes are supported."""
    return series.isin(["fill_gaps", "overwrite"])


def _advanced_rule_sources_match_methods(data: pd.DataFrame) -> bool:
    """Check that advanced-rule source usage matches the method."""
    source_required = data["method"].isin(
        ["external_profile", "construct_from_sources"]
    )

    leave_missing = data["method"].eq("leave_missing")

    return bool(
        data.loc[source_required, "source"].notna().all()
        and data.loc[leave_missing, "source"].isna().all()
    )


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
        "source": pa.Column(str, nullable=True),
        "context": pa.Column(str, nullable=False),
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
        pa.Check(
            _advanced_rule_sources_match_methods,
            error=(
                "Advanced rules using external_profile or "
                "construct_from_sources must define a source, while "
                "leave_missing rules must not define one."
            ),
        ),
    ],
    strict=True,
    ordered=True,
    unique_column_names=True,
    name="advanced_fill_rules",
)

AUXILIARY_REQUIREMENTS_SCHEMA = pa.DataFrameSchema(
    {
        "context": pa.Column(str, nullable=False),
        "start": pa.Column(
            DateTime(tz="UTC", to_datetime_kwargs={"utc": True}),
            checks=pa.Check(
                _timestamps_are_on_hour,
                error="Auxiliary requirement start timestamps must align to whole hours.",
            ),
            nullable=False,
            coerce=True,
        ),
        "end": pa.Column(
            DateTime(tz="UTC", to_datetime_kwargs={"utc": True}),
            checks=pa.Check(
                _timestamps_are_on_hour,
                error="Auxiliary requirement end timestamps must align to whole hours.",
            ),
            nullable=False,
            coerce=True,
        ),
    },
    checks=[
        pa.Check(
            lambda data: data["end"] > data["start"],
            error="Auxiliary requirements must end later than they start.",
        )
    ],
    strict=True,
    ordered=True,
    unique_column_names=True,
    name="auxiliary_requirements",
)


def _source_capabilities_do_not_mix_wildcard_and_explicit(data: pd.DataFrame) -> bool:
    """Check that each source uses wildcard or explicit context coverage."""
    for _, group in data.groupby("source"):
        has_wildcard = group["context"].isna().any()
        has_explicit = group["context"].notna().any()

        if has_wildcard and has_explicit:
            return False

    return True


SOURCE_CAPABILITIES_SCHEMA = pa.DataFrameSchema(
    {
        "source": pa.Column(str, nullable=False),
        "context": pa.Column(str, nullable=True),
    },
    checks=[
        pa.Check(_has_rows, error="At least one source capability must be configured."),
        pa.Check(
            lambda data: ~data.duplicated(subset=["source", "context"]).any(),
            error="Source capabilities must be unique.",
        ),
        pa.Check(
            _source_capabilities_do_not_mix_wildcard_and_explicit,
            error=(
                "A source must use either wildcard context coverage "
                "or explicit context coverage, not both."
            ),
        ),
    ],
    strict=True,
    ordered=True,
    unique_column_names=True,
    name="source_capabilities",
)

AUXILIARY_SOURCE_REQUESTS_SCHEMA = pa.DataFrameSchema(
    {
        "source": pa.Column(str, nullable=False),
        "context": pa.Column(str, nullable=False),
        "start": pa.Column(
            DateTime(tz="UTC", to_datetime_kwargs={"utc": True}),
            checks=pa.Check(
                _timestamps_are_on_hour,
                error="Source-request start timestamps must align to whole hours.",
            ),
            nullable=False,
            coerce=True,
        ),
        "end": pa.Column(
            DateTime(tz="UTC", to_datetime_kwargs={"utc": True}),
            checks=pa.Check(
                _timestamps_are_on_hour,
                error="Source-request end timestamps must align to whole hours.",
            ),
            nullable=False,
            coerce=True,
        ),
    },
    checks=[
        pa.Check(
            lambda data: data["end"] > data["start"],
            error="Source requests must end later than they start.",
        ),
        pa.Check(
            lambda data: ~data.duplicated().any(),
            error="Auxiliary source requests must be unique.",
        ),
    ],
    strict=True,
    ordered=True,
    unique_column_names=True,
    name="auxiliary_source_requests",
)

ADVANCED_SOURCE_SCHEMA = pa.SeriesSchema(
    float,
    index=pa.Index(
        DateTime(tz="UTC", to_datetime_kwargs={"utc": True}),
        name="timestamp",
        nullable=False,
        coerce=True,
        unique=True,
        checks=pa.Check(
            _timestamps_are_on_hour,
            error="Advanced source timestamps must align to whole hours.",
        ),
    ),
    nullable=False,
    coerce=True,
)
