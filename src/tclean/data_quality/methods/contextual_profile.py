"""Contextual-profile data-quality testing."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tclean.data_quality._method import (
    MethodContext,
    MethodIssue,
    MethodResult,
    MethodSpec,
)
from tclean.data_quality._progress import ProgressTracker
from tclean.data_quality._validation_helpers import (
    finite_real,
    nonnegative_timedelta,
    normalize_common_selectors,
    normalize_string_sequence,
    positive_timedelta,
    validate_keys,
)
from tclean.data_quality.methods._lattice import (
    normalize_reference_orders,
    reference_timestamps,
)
from tclean.data_quality.methods._robust import robust_location_scale
from tclean.time_grid import TimeGrid

logger = logging.getLogger(__name__)

_DISTANCE_ZERO_ATOL = 1e-12


@dataclass(frozen=True)
class _Profile:
    """One complete normalized profile."""

    start: pd.Timestamp
    end: pd.Timestamp
    values: np.ndarray


_ProfileCache = dict[pd.Timestamp, _Profile | None]


@dataclass(frozen=True)
class _RobustProfileDeviation:
    """Robust contextual-profile evidence for one target profile."""

    reference_profiles: int
    target_distance: float
    reference_distance_median: float
    reference_distance_mad: float
    robust_scale: float
    deviation: float
    robust_deviation: float | None
    failed: bool


@dataclass(frozen=True)
class _PredictiveProfileProbability:
    """Rank-based predictive evidence for one target profile."""

    reference_profiles: int
    target_distance: float
    profiles_at_least_as_nonconforming: int
    comparison_profiles: int
    predictive_probability: float
    failed: bool


@dataclass(frozen=True)
class _ContextualProfileEvidence:
    """Contextual-profile evidence for one target profile."""

    robust: _RobustProfileDeviation | None
    predictive: _PredictiveProfileProbability | None
    failed_criteria: tuple[str, ...]

    @property
    def failed(self) -> bool:
        """Return whether any configured criterion failed."""
        return bool(self.failed_criteria)

    @property
    def evaluable(self) -> bool:
        """Return whether at least one configured criterion was evaluable."""
        return self.robust is not None or self.predictive is not None


def validate(test: Mapping[str, Any], *, grid: TimeGrid) -> dict[str, Any]:
    """Validate and normalize a contextual-profile quality test."""
    validate_keys(
        test,
        required={"name", "method", "profile_duration", "reference_orders"},
        optional={
            "sources",
            "contexts",
            "profile_offset",
            "include_failed_periods_from",
            "robust_deviation_threshold",
            "maximum_predictive_probability",
        },
    )

    normalized = normalize_common_selectors(test)

    profile_duration = positive_timedelta(
        test["profile_duration"], field="profile_duration", grid=grid
    )

    if profile_duration < 2 * grid.frequency:
        raise ValueError(
            "'profile_duration' for a contextual-profile test must span "
            "at least two grid steps."
        )

    profile_offset = nonnegative_timedelta(
        test.get("profile_offset", pd.Timedelta(0)), field="profile_offset", grid=grid
    )

    if profile_offset >= profile_duration:
        raise ValueError("'profile_offset' must be less than 'profile_duration'.")

    normalized["profile_duration"] = profile_duration
    normalized["profile_offset"] = profile_offset
    normalized["reference_orders"] = normalize_reference_orders(
        test["reference_orders"], grid=grid
    )

    has_robust = "robust_deviation_threshold" in test
    has_predictive = "maximum_predictive_probability" in test

    if not has_robust and not has_predictive:
        raise ValueError(
            "A contextual-profile test must configure at least one of "
            "'robust_deviation_threshold' or "
            "'maximum_predictive_probability'."
        )

    if has_robust:
        robust_threshold = finite_real(
            test["robust_deviation_threshold"], field="robust_deviation_threshold"
        )

        if robust_threshold <= 0:
            raise ValueError("'robust_deviation_threshold' must be greater than zero.")

        normalized["robust_deviation_threshold"] = robust_threshold

    if has_predictive:
        predictive_probability = finite_real(
            test["maximum_predictive_probability"],
            field="maximum_predictive_probability",
        )

        if not 0 < predictive_probability < 1:
            raise ValueError(
                "'maximum_predictive_probability' must be greater than "
                "zero and less than one."
            )

        normalized["maximum_predictive_probability"] = predictive_probability

    if "include_failed_periods_from" in test:
        normalized["include_failed_periods_from"] = normalize_string_sequence(
            test["include_failed_periods_from"], field="include_failed_periods_from"
        )

    return normalized


def _target_profile_starts(
    index: pd.DatetimeIndex, *, test: Mapping[str, Any], grid: TimeGrid
) -> pd.DatetimeIndex:
    """Return complete target-profile starts within data and grid horizons."""
    if index.empty:
        return pd.DatetimeIndex([])

    duration = test["profile_duration"]
    offset = test["profile_offset"]

    coverage_start = max(grid.start, pd.Timestamp(index[0]))
    coverage_end = min(grid.end, pd.Timestamp(index[-1]) + grid.frequency)

    anchor = grid.start + offset
    latest_start = coverage_end - duration

    if latest_start < anchor:
        return pd.DatetimeIndex([])

    starts = pd.date_range(start=anchor, end=latest_start, freq=duration)
    starts = starts[starts >= coverage_start]

    return pd.DatetimeIndex(starts)


def _profile_values(
    data: pd.Series, *, start: pd.Timestamp, duration: pd.Timedelta, grid: TimeGrid
) -> pd.Series | None:
    """Return one complete observed raw profile, or None when unavailable."""
    end = start + duration
    expected = grid.index_for_period(start=start, end=end)

    if not expected.isin(data.index).all():
        return None

    values = data.reindex(expected)

    if values.isna().any():
        return None

    return values


def _normalize_profile(values: np.ndarray) -> np.ndarray:
    """Center and scale one profile to shape only."""
    centered = values.astype(float, copy=False) - float(np.mean(values))
    scale = float(np.sqrt(np.mean(centered**2)))

    if scale == 0:
        return np.zeros_like(centered, dtype=float)

    return centered / scale


def _build_profile(
    data: pd.Series, *, start: pd.Timestamp, duration: pd.Timedelta, grid: TimeGrid
) -> _Profile | None:
    """Build one complete normalized profile."""
    values = _profile_values(data, start=start, duration=duration, grid=grid)

    if values is None:
        return None

    return _Profile(
        start=start,
        end=start + duration,
        values=_normalize_profile(values.to_numpy(dtype=float)),
    )


def _cached_profile(
    data: pd.Series,
    *,
    start: pd.Timestamp,
    duration: pd.Timedelta,
    grid: TimeGrid,
    cache: _ProfileCache,
) -> _Profile | None:
    """Return one normalized profile, building and caching it if needed."""
    start = pd.Timestamp(start)

    if start not in cache:
        cache[start] = _build_profile(data, start=start, duration=duration, grid=grid)

    return cache[start]


def _profiles_overlap(
    first: _Profile, second_start: pd.Timestamp, second_end: pd.Timestamp
) -> bool:
    """Return whether one profile overlaps another half-open profile period."""
    return first.start < second_end and first.end > second_start


def _reference_profiles(
    reference: pd.Series,
    *,
    target: _Profile,
    test: Mapping[str, Any],
    grid: TimeGrid,
    cache: _ProfileCache,
) -> list[_Profile]:
    """Build complete non-overlapping contextual reference profiles."""
    starts = reference_timestamps(
        target.start, orders=test["reference_orders"], available_index=reference.index
    )

    profiles: list[_Profile] = []

    for start in starts:
        profile = _cached_profile(
            reference,
            start=pd.Timestamp(start),
            duration=test["profile_duration"],
            grid=grid,
            cache=cache,
        )

        if profile is None:
            continue

        if _profiles_overlap(profile, target.start, target.end):
            continue

        profiles.append(profile)

    return profiles


def _profile_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return aligned RMSE between two normalized profiles."""
    distance = float(np.sqrt(np.mean((first - second) ** 2)))

    if np.isclose(distance, 0.0, rtol=0.0, atol=_DISTANCE_ZERO_ATOL):
        return 0.0

    return distance


def _distance_matrix(profiles: Sequence[_Profile]) -> np.ndarray:
    """Return an efficient pairwise RMSE matrix for normalized profiles."""
    if not profiles:
        return np.empty((0, 0), dtype=float)

    values = np.stack([profile.values for profile in profiles], axis=0)
    n_steps = values.shape[1]

    mean_squares = np.mean(values**2, axis=1)
    cross_products = values @ values.T / n_steps

    squared = mean_squares[:, None] + mean_squares[None, :] - 2 * cross_products

    np.maximum(squared, 0.0, out=squared)

    distances = np.sqrt(squared)

    distances[np.isclose(distances, 0.0, rtol=0.0, atol=_DISTANCE_ZERO_ATOL)] = 0.0

    np.fill_diagonal(distances, 0.0)

    return distances


def _peer_distance_scores(distances: np.ndarray) -> np.ndarray:
    """Return each profile's median distance to all other profiles."""
    count = distances.shape[0]

    if count < 2:
        return np.empty(0, dtype=float)

    peers = distances[~np.eye(count, dtype=bool)].reshape(count, count - 1)

    return np.median(peers, axis=1)


def _target_distance(target: _Profile, references: Sequence[_Profile]) -> float:
    """Return target median distance to its contextual reference profiles."""
    distances = np.array(
        [_profile_distance(target.values, profile.values) for profile in references],
        dtype=float,
    )

    return float(np.median(distances))


def _robust_profile_deviation(
    target: _Profile, references: Sequence[_Profile], *, threshold: float
) -> _RobustProfileDeviation | None:
    """Evaluate a target profile against robust peer-distance evidence."""
    reference_profiles = len(references)

    if reference_profiles < 3:
        return None

    distances = _distance_matrix(references)
    reference_scores = _peer_distance_scores(distances)
    robust_reference = robust_location_scale(reference_scores)

    target_distance = _target_distance(target, references)
    deviation = float(target_distance - robust_reference.median)

    if robust_reference.scale == 0:
        if deviation <= 0:
            robust_deviation = 0.0
            failed = False
        else:
            robust_deviation = None
            failed = True
    else:
        robust_deviation = float(max(deviation, 0.0) / robust_reference.scale)
        failed = robust_deviation > threshold and not np.isclose(
            robust_deviation, threshold, rtol=1e-12, atol=0.0
        )

    return _RobustProfileDeviation(
        reference_profiles=reference_profiles,
        target_distance=target_distance,
        reference_distance_median=robust_reference.median,
        reference_distance_mad=robust_reference.mad,
        robust_scale=robust_reference.scale,
        deviation=deviation,
        robust_deviation=robust_deviation,
        failed=failed,
    )


def _predictive_profile_probability(
    target: _Profile, references: Sequence[_Profile], *, maximum_probability: float
) -> _PredictiveProfileProbability | None:
    """Evaluate a target with a rank-based contextual predictive probability."""
    reference_profiles = len(references)

    if reference_profiles == 0:
        return None

    profiles = [*references, target]
    distances = _distance_matrix(profiles)
    scores = _peer_distance_scores(distances)

    target_distance = float(scores[-1])
    at_least_as_nonconforming = (scores > target_distance) | np.isclose(
        scores, target_distance, rtol=1e-12, atol=0.0
    )

    profiles_at_least_as_nonconforming = int(
        np.count_nonzero(at_least_as_nonconforming)
    )
    comparison_profiles = len(profiles)
    predictive_probability = float(
        profiles_at_least_as_nonconforming / comparison_profiles
    )

    return _PredictiveProfileProbability(
        reference_profiles=reference_profiles,
        target_distance=target_distance,
        profiles_at_least_as_nonconforming=profiles_at_least_as_nonconforming,
        comparison_profiles=comparison_profiles,
        predictive_probability=predictive_probability,
        failed=predictive_probability <= maximum_probability,
    )


def _contextual_profile_evidence(
    target: _Profile, references: Sequence[_Profile], *, test: Mapping[str, Any]
) -> _ContextualProfileEvidence:
    """Evaluate configured contextual-profile criteria for one target."""
    robust = None
    predictive = None
    failed_criteria: list[str] = []

    if "robust_deviation_threshold" in test:
        robust = _robust_profile_deviation(
            target, references, threshold=test["robust_deviation_threshold"]
        )

        if robust is not None and robust.failed:
            failed_criteria.append("robust_deviation")

    if "maximum_predictive_probability" in test:
        predictive = _predictive_profile_probability(
            target,
            references,
            maximum_probability=test["maximum_predictive_probability"],
        )

        if predictive is not None and predictive.failed:
            failed_criteria.append("predictive_probability")

    return _ContextualProfileEvidence(
        robust=robust, predictive=predictive, failed_criteria=tuple(failed_criteria)
    )


def _configured_criteria(test: Mapping[str, Any]) -> list[str]:
    """Return configured contextual-profile criteria."""
    criteria: list[str] = []

    if "robust_deviation_threshold" in test:
        criteria.append("robust_deviation")

    if "maximum_predictive_probability" in test:
        criteria.append("predictive_probability")

    return criteria


def _unavailable_criteria(
    evidence: _ContextualProfileEvidence, *, test: Mapping[str, Any]
) -> list[str]:
    """Return configured criteria that could not be evaluated."""
    unavailable: list[str] = []

    if "robust_deviation_threshold" in test and evidence.robust is None:
        unavailable.append("robust_deviation")

    if "maximum_predictive_probability" in test and evidence.predictive is None:
        unavailable.append("predictive_probability")

    return unavailable


def evaluate(context: MethodContext) -> MethodResult:
    """Evaluate contextual profiles and report evaluation limitations."""
    data = context.target_data
    reference = context.reference_data(context.source_name)
    test = context.test
    grid = context.grid

    progress = ProgressTracker(
            total=len(data.columns),
            label=f"{test['name']} [{test['method']}]",
            logger=logger,
        )

    mask = pd.DataFrame(False, index=data.index, columns=data.columns, dtype=bool)
    issues: list[MethodIssue] = []

    starts = _target_profile_starts(data.index, test=test, grid=grid)
    duration = test["profile_duration"]


    for context_name in data.columns:

        values = data[context_name]
        reference_values = reference[context_name]
        reference_profile_cache: _ProfileCache = {}

        for start in starts:
            start = pd.Timestamp(start)
            end = start + duration

            target_values = _profile_values(
                values, start=start, duration=duration, grid=grid
            )

            if target_values is None:
                expected = grid.index_for_period(start=start, end=end)
                observed = values.reindex(expected)
                missing_observations = int(observed.isna().sum())

                issues.append(
                    MethodIssue(
                        context=context_name,
                        start=start,
                        end=end,
                        severity="not_evaluable",
                        code="incomplete_target_profile",
                        details={
                            "profile_observations": len(expected),
                            "missing_observations": missing_observations,
                        },
                    )
                )
                continue

            target = _Profile(
                start=start,
                end=end,
                values=_normalize_profile(target_values.to_numpy(dtype=float)),
            )

            references = _reference_profiles(
                reference_values,
                target=target,
                test=test,
                grid=grid,
                cache=reference_profile_cache,
            )

            evidence = _contextual_profile_evidence(target, references, test=test)
            reference_profiles = len(references)

            if not evidence.evaluable:
                issues.append(
                    MethodIssue(
                        context=context_name,
                        start=start,
                        end=end,
                        severity="not_evaluable",
                        code="insufficient_reference_profiles",
                        details={
                            "reference_profiles": reference_profiles,
                            "configured_criteria": _configured_criteria(test),
                        },
                    )
                )
                continue

            if evidence.failed:
                profile_index = grid.index_for_period(start=start, end=end)
                mask.loc[profile_index, context_name] = True

            for criterion in _unavailable_criteria(evidence, test=test):
                issues.append(
                    MethodIssue(
                        context=context_name,
                        start=start,
                        end=end,
                        severity="warning",
                        code="criterion_not_evaluable",
                        details={
                            "criterion": criterion,
                            "reference_profiles": reference_profiles,
                        },
                    )
                )
        progress.complete(context_name)

    return MethodResult(mask=mask, issues=tuple(issues))


def _profile_details(
    target: _Profile,
    *,
    references: Sequence[_Profile],
    evidence: _ContextualProfileEvidence,
    test: Mapping[str, Any],
) -> dict[str, Any]:
    """Build evidence for one failed contextual profile."""
    details: dict[str, Any] = {
        "profile_start": target.start.isoformat(),
        "profile_end": target.end.isoformat(),
        "reference_profiles": len(references),
        "failed_criteria": list(evidence.failed_criteria),
    }

    if evidence.robust is not None:
        robust = evidence.robust
        details.update(
            {
                "target_profile_distance": robust.target_distance,
                "reference_distance_median": robust.reference_distance_median,
                "reference_distance_mad": robust.reference_distance_mad,
                "robust_scale": robust.robust_scale,
                "robust_signed_deviation": robust.deviation,
                "robust_deviation": robust.robust_deviation,
                "robust_deviation_threshold": float(test["robust_deviation_threshold"]),
                "robust_failed": robust.failed,
            }
        )

    if evidence.predictive is not None:
        predictive = evidence.predictive
        details.setdefault("target_profile_distance", predictive.target_distance)
        details.update(
            {
                "profiles_at_least_as_nonconforming": (
                    predictive.profiles_at_least_as_nonconforming
                ),
                "comparison_profiles": predictive.comparison_profiles,
                "predictive_probability": predictive.predictive_probability,
                "maximum_predictive_probability": float(
                    test["maximum_predictive_probability"]
                ),
                "predictive_failed": predictive.failed,
            }
        )

    return details


def build_details(
    context: MethodContext,
    result: MethodResult,
    *,
    context_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Build evidence for one contextual-profile failure period."""
    del result

    data = context.target_data[context_name]
    reference = context.reference_data(context.source_name)[context_name]
    test = context.test
    grid = context.grid

    failed_profiles: list[dict[str, Any]] = []
    failed_criteria: list[str] = []

    reference_profile_cache: _ProfileCache = {}

    duration = test["profile_duration"]
    starts = _target_profile_starts(data.index, test=test, grid=grid)

    for profile_start in starts:
        profile_start = pd.Timestamp(profile_start)
        profile_end = profile_start + duration

        if profile_start < start or profile_end > end:
            continue

        target = _build_profile(data, start=profile_start, duration=duration, grid=grid)

        if target is None:
            continue

        references = _reference_profiles(
            reference,
            target=target,
            test=test,
            grid=grid,
            cache=reference_profile_cache,
        )
        evidence = _contextual_profile_evidence(target, references, test=test)

        if not evidence.failed:
            continue

        failed_profiles.append(
            _profile_details(
                target, references=references, evidence=evidence, test=test
            )
        )

        for criterion in evidence.failed_criteria:
            if criterion not in failed_criteria:
                failed_criteria.append(criterion)

    return {
        "failed_profiles": len(failed_profiles),
        "failed_criteria": failed_criteria,
        "profiles": failed_profiles,
    }


METHOD = MethodSpec(
    name="contextual_profile",
    validate=validate,
    evaluate=evaluate,
    build_details=build_details,
)
