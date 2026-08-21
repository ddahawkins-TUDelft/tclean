"""Construct demand profiles from configured source periods."""

import pandas as pd

from tclean.validation import (
    validate_hourly_timestamp_index,
    validate_source_periods,
    validate_time_series,
)

METHOD_NAME = "construct_from_sources"


def _align_leap_day(
    auxiliary: pd.Series,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """Align source values with the target calendar around February 29."""
    source_values = auxiliary.loc[(auxiliary.index >= start) & (auxiliary.index < end)]

    source_has_leap_day = (
        (source_values.index.month == 2) & (source_values.index.day == 29)
    ).any()

    target_has_leap_day = ((target_index.month == 2) & (target_index.day == 29)).any()

    if source_has_leap_day and not target_has_leap_day:
        leap_day = (source_values.index.month == 2) & (source_values.index.day == 29)

        return source_values.loc[~leap_day]

    if target_has_leap_day and not source_has_leap_day:
        feb_28 = auxiliary.loc[
            (auxiliary.index.year == start.year)
            & (auxiliary.index.month == 2)
            & (auxiliary.index.day == 28)
        ]

        march_1 = auxiliary.loc[
            (auxiliary.index.year == start.year)
            & (auxiliary.index.month == 3)
            & (auxiliary.index.day == 1)
        ]

        leap_values = (feb_28.to_numpy(dtype=float) + march_1.to_numpy(dtype=float)) / 2

        insertion_point = (source_values.index.month < 3).sum()

        values = source_values.to_numpy(dtype=float)

        return pd.Series(
            [*values[:insertion_point], *leap_values, *values[insertion_point:]],
            dtype=float,
        )

    return source_values


def _match_energy(
    profile: pd.Series, *, auxiliary: pd.DataFrame, reference_sources: pd.DataFrame
) -> pd.Series:
    """Scale a profile to weighted mean energy of reference periods."""
    weighted_energy = 0.0
    total_weight = 0.0

    for source in reference_sources.itertuples(index=False):
        if source.context not in auxiliary.columns:
            raise ValueError(
                "Auxiliary data do not contain requested scaling context "
                f"{source.context!r}."
            )

        source_values = auxiliary.loc[
            (auxiliary.index >= source.start) & (auxiliary.index < source.end),
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

        weighted_energy += float(source_values.sum()) * source.weight
        total_weight += source.weight

    target_energy = weighted_energy / total_weight
    profile_energy = float(profile.sum())

    if profile_energy == 0:
        raise ValueError(
            "Cannot match energy for a constructed profile with zero total energy."
        )

    return profile * (target_energy / profile_energy)


def construct_from_sources(
    auxiliary: pd.DataFrame,
    *,
    target_index: pd.DatetimeIndex,
    sources: pd.DataFrame,
    scaling_sources: pd.DataFrame | None = None,
) -> pd.Series:
    """Construct a target demand profile from weighted source periods.

    Args:
        auxiliary: Canonical hourly auxiliary demand data.
        target_index: Hourly timestamps for the constructed profile.
        sources: Explicit weighted source-period definitions.
        scaling_sources: Optional explicit weighted reference periods
            whose mean energy the constructed profile should match.

    Returns:
        Constructed floating-point demand profile indexed by
        ``target_index``.

    Raises:
        ValueError: If a source period does not match the target length,
            contains missing values, refers to an unavailable context,
            or energy matching cannot be performed.
        pandera.errors.SchemaErrors: If any input violates its
            T-Clean data contract.
    """
    auxiliary = validate_time_series(auxiliary)

    target_index = validate_hourly_timestamp_index(target_index)

    sources = validate_source_periods(sources)

    if scaling_sources is not None:
        scaling_sources = validate_source_periods(scaling_sources)

    weighted_sources: list[pd.Series] = []
    weights: list[float] = []

    for source in sources.itertuples(index=False):
        if source.context not in auxiliary.columns:
            raise ValueError(
                f"Auxiliary data do not contain requested context {source.context!r}."
            )

        source_values = _align_leap_day(
            auxiliary[source.context],
            start=source.start,
            end=source.end,
            target_index=target_index,
        )

        if len(source_values) != len(target_index):
            raise ValueError(
                "Auxiliary source period must contain "
                "the same number of values as the target "
                f"period. Source {source.context!r} contains "
                f"{len(source_values)} values; target "
                f"contains {len(target_index)}."
            )

        if source_values.isna().any():
            raise ValueError(
                "Auxiliary source period contains missing values. "
                f"Source {source.context!r}: "
                f"{source.start} to {source.end}."
            )

        remapped = pd.Series(source_values.to_numpy(), index=target_index, dtype=float)

        weighted_sources.append(remapped * source.weight)
        weights.append(source.weight)

    weighted_sum = sum(weighted_sources[1:], weighted_sources[0].copy())

    profile = weighted_sum / sum(weights)

    if scaling_sources is not None:
        profile = _match_energy(
            profile, auxiliary=auxiliary, reference_sources=scaling_sources
        )

    return profile
