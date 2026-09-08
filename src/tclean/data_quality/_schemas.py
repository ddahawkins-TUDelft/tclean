"""Pandera schemas for data-quality result contracts."""

import pandas as pd
import pandera.pandas as pa
from pandera.engines.pandas_engine import DateTime


def _is_nonblank_string(series: pd.Series) -> pd.Series:
    """Check that present string values contain non-whitespace characters."""
    return series.isna() | series.astype("string").str.strip().ne("")


def _is_mapping(series: pd.Series) -> pd.Series:
    """Check that details values are dictionaries."""
    return series.map(lambda value: isinstance(value, dict))


def _quality_period_ends_after_start(data: pd.DataFrame) -> pd.Series:
    """Check that quality-event periods end after they start."""
    return data["end"] > data["start"]


def _quality_issue_severity_is_supported(series: pd.Series) -> pd.Series:
    """Check that quality-evaluation issue severities are supported."""
    return series.isin(["warning", "not_evaluable"])


QUALITY_FAILURES_SCHEMA = pa.DataFrameSchema(
    {
        "context": pa.Column(
            str,
            checks=pa.Check(
                _is_nonblank_string, error="Quality-failure context must not be blank."
            ),
            nullable=False,
        ),
        "source": pa.Column(
            str,
            checks=pa.Check(
                _is_nonblank_string, error="Quality-failure source must not be blank."
            ),
            nullable=False,
        ),
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
        "test_name": pa.Column(
            str,
            checks=pa.Check(
                _is_nonblank_string,
                error="Quality-failure test name must not be blank.",
            ),
            nullable=False,
        ),
        "method": pa.Column(
            str,
            checks=pa.Check(
                _is_nonblank_string, error="Quality-failure method must not be blank."
            ),
            nullable=False,
        ),
        "details": pa.Column(
            object,
            checks=pa.Check(
                _is_mapping, error="Quality-failure details must be dictionaries."
            ),
            nullable=False,
        ),
    },
    checks=[
        pa.Check(
            _quality_period_ends_after_start,
            error="Quality-failure periods must end later than they start.",
        ),
        pa.Check(
            lambda data: (
                ~data.duplicated(
                    subset=["context", "source", "start", "end", "test_name", "method"]
                ).any()
            ),
            error="Quality-failure events must be unique.",
        ),
    ],
    strict=True,
    ordered=True,
    unique_column_names=True,
    name="quality_failures",
)


QUALITY_ISSUES_SCHEMA = pa.DataFrameSchema(
    {
        "context": pa.Column(
            str,
            checks=pa.Check(
                _is_nonblank_string, error="Quality issue context must not be blank."
            ),
            nullable=False,
        ),
        "source": pa.Column(
            str,
            checks=pa.Check(
                _is_nonblank_string, error="Quality issue source must not be blank."
            ),
            nullable=False,
        ),
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
        "test_name": pa.Column(
            str,
            checks=pa.Check(
                _is_nonblank_string, error="Quality issue test name must not be blank."
            ),
            nullable=False,
        ),
        "method": pa.Column(
            str,
            checks=pa.Check(
                _is_nonblank_string, error="Quality issue method must not be blank."
            ),
            nullable=False,
        ),
        "severity": pa.Column(
            str,
            checks=pa.Check(
                _quality_issue_severity_is_supported,
                error="Unsupported quality issue severity.",
            ),
            nullable=False,
        ),
        "code": pa.Column(
            str,
            checks=pa.Check(
                _is_nonblank_string, error="Quality issue code must not be blank."
            ),
            nullable=False,
        ),
        "details": pa.Column(
            object,
            checks=pa.Check(
                _is_mapping, error="Quality-issue details must be dictionaries."
            ),
            nullable=False,
        ),
    },
    checks=[
        pa.Check(
            _quality_period_ends_after_start,
            error="Quality-issue periods must end later than they start.",
        ),
        pa.Check(
            lambda data: (
                ~data.duplicated(
                    subset=[
                        "context",
                        "source",
                        "start",
                        "end",
                        "test_name",
                        "method",
                        "severity",
                        "code",
                    ]
                ).any()
            ),
            error="Quality-evaluation issues must be unique.",
        ),
    ],
    strict=True,
    ordered=True,
    unique_column_names=True,
    name="quality_issues",
)
