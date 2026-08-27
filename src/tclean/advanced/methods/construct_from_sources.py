"""Construct profiles from configured source periods."""

import pandas as pd

from tclean.time_grid import TimeGrid
from tclean.validation import (
    validate_source_periods,
    validate_time_series,
    validate_timestamp_index,
)

METHOD_NAME = "construct_from_sources"


def _validate_leap_day_frequency(frequency: pd.Timedelta) -> None:
    """Require whole calendar days to contain an integer number of periods."""
    one_day = pd.Timedelta(days=1)

    if one_day % frequency != pd.Timedelta(0):
        raise ValueError(
            "Leap-day alignment requires the configured frequency "
            "to divide one calendar day exactly."
        )


def _align_leap_day(
    source_data: pd.Series,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    target_index: pd.DatetimeIndex,
    frequency: pd.Timedelta,
) -> pd.Series:
    """Align source values with the target calendar around February 29."""
    source_values = source_data.loc[
        (source_data.index >= start) & (source_data.index < end)
    ]

    source_has_leap_day = (
        (source_values.index.month == 2) & (source_values.index.day == 29)
    ).any()

    target_has_leap_day = ((target_index.month == 2) & (target_index.day == 29)).any()

    if source_has_leap_day != target_has_leap_day:
        _validate_leap_day_frequency(frequency)

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
            ((source_data.index >= source.start) & (source_data.index < source.end)),
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


def _normalise_mean(profile: pd.Series) -> pd.Series:
    """Scale a profile so that its arithmetic mean equals one."""
    profile_mean = float(profile.mean())

    if profile_mean == 0:
        raise ValueError("Cannot normalise a constructed profile with zero mean.")

    return profile / profile_mean


def _normalise_max(profile: pd.Series) -> pd.Series:
    """Scale a profile so that its maximum equals one."""
    profile_max = float(profile.max())

    if profile_max <= 0:
        raise ValueError(
            "Cannot normalise a constructed profile whose maximum is not positive."
        )

    return profile / profile_max


def _apply_scaling(
    profile: pd.Series,
    *,
    method: str,
    source_data: pd.DataFrame,
    scaling_sources: pd.DataFrame | None,
) -> pd.Series:
    """Scale a constructed profile using the requested method."""
    if method == "match_total":
        if scaling_sources is None:
            raise ValueError("Scaling method 'match_total' requires scaling_sources.")

        return _scale_to_reference_total(
            profile, source_data=source_data, reference_sources=scaling_sources
        )

    if method == "normalise_mean":
        if scaling_sources is not None:
            raise ValueError(
                "Scaling method 'normalise_mean' does not use scaling_sources."
            )

        return _normalise_mean(profile)

    if method == "normalise_max":
        if scaling_sources is not None:
            raise ValueError(
                "Scaling method 'normalise_max' does not use scaling_sources."
            )

        return _normalise_max(profile)

    raise ValueError(f"Unsupported scaling method: {method!r}.")


def construct_from_sources(
    source_data: pd.DataFrame,
    *,
    target_index: pd.DatetimeIndex,
    sources: pd.DataFrame,
    scaling_method: str | None = None,
    scaling_sources: pd.DataFrame | None = None,
    grid: TimeGrid,
) -> pd.Series:
    """Construct a target profile from weighted source periods.

    Args:
        source_data: Canonical supporting time-series data.
        target_index: Timestamps for the constructed profile.
        sources: Explicit weighted source-period definitions.
        scaling_method: Optional method used to scale the constructed profile.
        scaling_sources: Optional weighted reference periods used by
            scaling methods that require external reference data.
        grid: Temporal grid against which all temporal inputs are
            validated.

    Returns:
        Constructed floating-point profile indexed by ``target_index``.

    Raises:
        ValueError: If a source period does not match the target length,
            contains missing values, refers to an unavailable context,
            or value matching cannot be performed.
        pandera.errors.SchemaErrors: If any input violates its
            T-Clean data contract.
    """
    source_data = validate_time_series(source_data, grid=grid)

    target_index = validate_timestamp_index(target_index, grid=grid)

    sources = validate_source_periods(sources, grid=grid)

    if scaling_sources is not None:
        scaling_sources = validate_source_periods(scaling_sources, grid=grid)

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
            frequency=grid.frequency,
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

    if scaling_method is not None:
        profile = _apply_scaling(
            profile,
            method=scaling_method,
            source_data=source_data,
            scaling_sources=scaling_sources,
        )
    elif scaling_sources is not None:
        raise ValueError("scaling_sources were supplied without a scaling_method.")

    return profile
