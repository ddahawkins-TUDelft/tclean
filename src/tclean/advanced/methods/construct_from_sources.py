"""Construct profiles from configured source periods."""

import pandas as pd

from tclean.validation import (
    validate_source_periods,
    validate_time_series,
    validate_timestamp_index,
)

METHOD_NAME = "construct_from_sources"


def _align_leap_day(
    source_data: pd.Series,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """Align source values with the target calendar around February 29."""
    source_values = source_data.loc[
        (source_data.index >= start) & (source_data.index < end)
    ]

    source_has_leap_day = (
        (source_values.index.month == 2) & (source_values.index.day == 29)
    ).any()

    target_has_leap_day = ((target_index.month == 2) & (target_index.day == 29)).any()

    if source_has_leap_day and not target_has_leap_day:
        leap_day = (source_values.index.month == 2) & (source_values.index.day == 29)

        return source_values.loc[~leap_day]

    if target_has_leap_day and not source_has_leap_day:
        feb_28 = source_data.loc[
            (source_data.index.year == start.year)
            & (source_data.index.month == 2)
            & (source_data.index.day == 28)
        ]

        march_1 = source_data.loc[
            (source_data.index.year == start.year)
            & (source_data.index.month == 3)
            & (source_data.index.day == 1)
        ]

        leap_values = (feb_28.to_numpy(dtype=float) + march_1.to_numpy(dtype=float)) / 2

        insertion_point = (source_values.index.month < 3).sum()

        values = source_values.to_numpy(dtype=float)

        return pd.Series(
            [*values[:insertion_point], *leap_values, *values[insertion_point:]],
            dtype=float,
        )

    return source_values


def _scale_to_reference_total(
    profile: pd.Series, *, source_data: pd.DataFrame, reference_sources: pd.DataFrame
) -> pd.Series:
    """Scale a profile to weighted mean quantity of reference periods."""
    weighted_value = 0.0
    total_weight = 0.0

    for source in reference_sources.itertuples(index=False):
        if source.context not in source_data.columns:
            raise ValueError(
                "source_data data do not contain requested scaling context "
                f"{source.context!r}."
            )

        source_values = source_data.loc[
            (source_data.index >= source.start) & (source_data.index < source.end),
            source.context,
        ]

        if source_values.empty:
            raise ValueError(
                "Scaling source period contains no values. "
                f"Source {source.context!r}: "
                f"{source.start} to {source.end}."
            )

        if source_values.isna().any():
            raise ValueError(
                "Scaling source period contains missing values. "
                f"Source {source.context!r}: "
                f"{source.start} to {source.end}."
            )

        weighted_value += float(source_values.sum()) * source.weight
        total_weight += source.weight

    target_value = weighted_value / total_weight
    profile_value = float(profile.sum())

    if profile_value == 0:
        raise ValueError(
            "Cannot match value for a constructed profile with zero total value."
        )

    return profile * (target_value / profile_value)


def construct_from_sources(
    source_data: pd.DataFrame,
    *,
    target_index: pd.DatetimeIndex,
    sources: pd.DataFrame,
    scaling_sources: pd.DataFrame | None = None,
    frequency: pd.Timedelta,
) -> pd.Series:
    """Construct a target profile from weighted source periods.

    Args:
        source_data: Canonical source_data data.
        target_index: Timestamps for the constructed profile.
        sources: Explicit weighted source-period definitions.
        scaling_sources: Optional explicit weighted reference periods
            whose mean value the constructed profile should match.
        frequency: pd. Timestamp of time series frequency.

    Returns:
        Constructed floating-point profile indexed by
        ``target_index``.

    Raises:
        ValueError: If a source period does not match the target length,
            contains missing values, refers to an unavailable context,
            or value matching cannot be performed.
        pandera.errors.SchemaErrors: If any input violates its
            T-Clean data contract.
    """
    source_data = validate_time_series(source_data, frequency=frequency)

    target_index = validate_timestamp_index(target_index, frequency=frequency)

    sources = validate_source_periods(sources, frequency=frequency)

    if scaling_sources is not None:
        scaling_sources = validate_source_periods(scaling_sources, frequency=frequency)

    weighted_sources: list[pd.Series] = []
    weights: list[float] = []

    for source in sources.itertuples(index=False):
        if source.context not in source_data.columns:
            raise ValueError(
                f"source_data data do not contain requested context {source.context!r}."
            )

        source_values = _align_leap_day(
            source_data[source.context],
            start=source.start,
            end=source.end,
            target_index=target_index,
        )

        if len(source_values) != len(target_index):
            raise ValueError(
                "source_data source period must contain "
                "the same number of values as the target "
                f"period. Source {source.context!r} contains "
                f"{len(source_values)} values; target "
                f"contains {len(target_index)}."
            )

        if source_values.isna().any():
            raise ValueError(
                "source_data source period contains missing values. "
                f"Source {source.context!r}: "
                f"{source.start} to {source.end}."
            )

        remapped = pd.Series(source_values.to_numpy(), index=target_index, dtype=float)

        weighted_sources.append(remapped * source.weight)
        weights.append(source.weight)

    weighted_sum = sum(weighted_sources[1:], weighted_sources[0].copy())

    profile = weighted_sum / sum(weights)

    if scaling_sources is not None:
        profile = _scale_to_reference_total(
            profile, source_data=source_data, reference_sources=scaling_sources
        )

    return profile
