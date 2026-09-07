"""Contextual-level data-quality testing."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from tclean.data_quality._validation_helpers import (
    finite_real,
    normalize_common_selectors,
    normalize_string_sequence,
    validate_keys,
)
from tclean.data_quality.methods._lattice import (
    normalize_reference_orders,
    reference_timestamps,
)
from tclean.data_quality.methods._robust import robust_location_scale
from tclean.time_grid import TimeGrid

METHOD_NAME = "contextual_level"
USES_REFERENCE_DATA = True


@dataclass(frozen=True)
class _RobustDeviation:
    """Robust contextual-deviation evidence for one target."""

    reference_observations: int
    reference_median: float
    reference_mad: float
    robust_scale: float
    deviation: float
    robust_deviation: float | None
    failed: bool


@dataclass(frozen=True)
class _PredictiveProbability:
    """Predictive contextual-level evidence for one target."""

    reference_observations: int
    reference_mean: float
    reference_standard_deviation: float
    prediction_standard_error: float
    deviation: float
    degrees_of_freedom: int
    t_statistic: float | None
    predictive_probability: float
    failed: bool


@dataclass(frozen=True)
class _ContextualLevelEvidence:
    """Contextual-level evidence for one target."""

    robust: _RobustDeviation | None
    predictive: _PredictiveProbability | None
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
    """Validate and normalize a contextual-level quality test."""
    validate_keys(
        test,
        required={"name", "method", "reference_orders"},
        optional={
            "sources",
            "contexts",
            "include_failed_periods_from",
            "robust_deviation_threshold",
            "maximum_predictive_probability",
        },
    )

    normalized = normalize_common_selectors(test)

    normalized["reference_orders"] = normalize_reference_orders(
        test["reference_orders"], grid=grid
    )

    has_robust = "robust_deviation_threshold" in test
    has_predictive = "maximum_predictive_probability" in test

    if not has_robust and not has_predictive:
        raise ValueError(
            "A contextual-level test must configure at least one of "
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


def _robust_deviation(
    target: float, reference: pd.Series, *, threshold: float
) -> _RobustDeviation | None:
    """Evaluate one target against a robust contextual reference."""
    reference = reference.dropna()

    if reference.empty:
        return None

    values = reference.to_numpy(dtype=float)

    robust_reference = robust_location_scale(values)

    reference_median = robust_reference.median
    reference_mad = robust_reference.mad
    robust_scale = robust_reference.scale

    deviation = float(target - reference_median)

    if robust_scale == 0:
        if deviation == 0:
            robust_deviation = 0.0
            failed = False
        else:
            # The target differs from a reference population
            # with no observed robust variation.
            robust_deviation = None
            failed = True

        return _RobustDeviation(
            reference_observations=len(reference),
            reference_median=reference_median,
            reference_mad=reference_mad,
            robust_scale=robust_scale,
            deviation=deviation,
            robust_deviation=robust_deviation,
            failed=failed,
        )

    robust_deviation = float(abs(deviation) / robust_scale)

    failed = robust_deviation > threshold and not np.isclose(
        robust_deviation, threshold, rtol=1e-12, atol=0.0
    )

    return _RobustDeviation(
        reference_observations=len(reference),
        reference_median=reference_median,
        reference_mad=reference_mad,
        robust_scale=robust_scale,
        deviation=deviation,
        robust_deviation=robust_deviation,
        failed=failed,
    )


def _predictive_probability(
    target: float, reference: pd.Series, *, maximum_probability: float
) -> _PredictiveProbability | None:
    """Evaluate one target using a Student-t predictive probability."""
    reference = reference.dropna()

    reference_observations = len(reference)

    if reference_observations < 2:
        return None

    values = reference.to_numpy(dtype=float)

    reference_mean = float(np.mean(values))

    reference_standard_deviation = float(np.std(values, ddof=1))

    deviation = float(target - reference_mean)

    degrees_of_freedom = reference_observations - 1

    prediction_standard_error = float(
        reference_standard_deviation * np.sqrt(1 + 1 / reference_observations)
    )

    if prediction_standard_error == 0:
        if deviation == 0:
            t_statistic = 0.0
            predictive_probability = 1.0
        else:
            t_statistic = None
            predictive_probability = 0.0

    else:
        t_statistic = float(deviation / prediction_standard_error)

        predictive_probability = float(
            2 * student_t.sf(abs(t_statistic), df=degrees_of_freedom)
        )

    failed = predictive_probability <= maximum_probability

    return _PredictiveProbability(
        reference_observations=reference_observations,
        reference_mean=reference_mean,
        reference_standard_deviation=(reference_standard_deviation),
        prediction_standard_error=(prediction_standard_error),
        deviation=deviation,
        degrees_of_freedom=degrees_of_freedom,
        t_statistic=t_statistic,
        predictive_probability=(predictive_probability),
        failed=failed,
    )


def _contextual_level_evidence(
    target: float, reference: pd.Series, *, test: Mapping[str, Any]
) -> _ContextualLevelEvidence:
    """Evaluate configured contextual-level criteria for one target."""
    robust = None
    predictive = None
    failed_criteria: list[str] = []

    if "robust_deviation_threshold" in test:
        robust = _robust_deviation(
            target, reference, threshold=test["robust_deviation_threshold"]
        )

        if robust is not None and robust.failed:
            failed_criteria.append("robust_deviation")

    if "maximum_predictive_probability" in test:
        predictive = _predictive_probability(
            target,
            reference,
            maximum_probability=(test["maximum_predictive_probability"]),
        )

        if predictive is not None and predictive.failed:
            failed_criteria.append("predictive_probability")

    return _ContextualLevelEvidence(
        robust=robust, predictive=predictive, failed_criteria=tuple(failed_criteria)
    )


def _reference_values(
    reference: pd.Series, *, target: pd.Timestamp, test: Mapping[str, Any]
) -> pd.Series:
    """Return contextual reference values for one target timestamp."""
    timestamps = reference_timestamps(
        target, orders=test["reference_orders"], available_index=reference.index
    )

    return reference.loc[timestamps]


def _reference_observation_count(reference: pd.Series) -> int:
    """Return the number of observed contextual reference values."""
    return int(reference.notna().sum())


def evaluate_with_issues(
    data: pd.DataFrame,
    *,
    reference: pd.DataFrame,
    test: Mapping[str, Any],
    grid: TimeGrid,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Evaluate contextual levels and report evaluation limitations."""
    mask = pd.DataFrame(False, index=data.index, columns=data.columns, dtype=bool)

    issues: list[dict[str, Any]] = []

    for context in data.columns:
        values = data[context]
        reference_values = reference[context]

        for timestamp, target in values.items():
            if pd.isna(target):
                continue

            contextual_reference = _reference_values(
                reference_values, target=timestamp, test=test
            )

            evidence = _contextual_level_evidence(
                float(target), contextual_reference, test=test
            )

            reference_observations = _reference_observation_count(contextual_reference)

            if not evidence.evaluable:
                issues.append(
                    {
                        "context": context,
                        "start": timestamp,
                        "end": timestamp + grid.frequency,
                        "severity": "not_evaluable",
                        "code": "insufficient_reference_data",
                        "details": {
                            "reference_observations": (reference_observations),
                            "configured_criteria": (_configured_criteria(test)),
                        },
                    }
                )

                continue

            mask.loc[timestamp, context] = evidence.failed

            unavailable_criteria = _unavailable_criteria(evidence, test=test)

            for criterion in unavailable_criteria:
                issues.append(
                    {
                        "context": context,
                        "start": timestamp,
                        "end": timestamp + grid.frequency,
                        "severity": "warning",
                        "code": "criterion_not_evaluable",
                        "details": {
                            "criterion": criterion,
                            "reference_observations": (reference_observations),
                        },
                    }
                )

    return mask, issues


def _configured_criteria(test: Mapping[str, Any]) -> list[str]:
    """Return configured contextual-level criteria."""
    criteria: list[str] = []

    if "robust_deviation_threshold" in test:
        criteria.append("robust_deviation")

    if "maximum_predictive_probability" in test:
        criteria.append("predictive_probability")

    return criteria


def _unavailable_criteria(
    evidence: _ContextualLevelEvidence, *, test: Mapping[str, Any]
) -> list[str]:
    """Return configured criteria that could not be evaluated."""
    unavailable: list[str] = []

    if "robust_deviation_threshold" in test and evidence.robust is None:
        unavailable.append("robust_deviation")

    if "maximum_predictive_probability" in test and evidence.predictive is None:
        unavailable.append("predictive_probability")

    return unavailable


def evaluate(
    data: pd.DataFrame,
    *,
    reference: pd.DataFrame,
    test: Mapping[str, Any],
    grid: TimeGrid,
) -> pd.DataFrame:
    """Evaluate contextual-level failures."""
    mask, _ = evaluate_with_issues(data, reference=reference, test=test, grid=grid)

    return mask


def _observation_details(
    target: float,
    *,
    timestamp: pd.Timestamp,
    reference: pd.Series,
    evidence: _ContextualLevelEvidence,
    test: Mapping[str, Any],
) -> dict[str, Any]:
    """Build contextual-level evidence for one failed observation."""
    details: dict[str, Any] = {
        "timestamp": timestamp.isoformat(),
        "value": float(target),
        "reference_observations": (_reference_observation_count(reference)),
        "failed_criteria": list(evidence.failed_criteria),
    }

    if evidence.robust is not None:
        robust = evidence.robust

        details.update(
            {
                "reference_median": (robust.reference_median),
                "reference_mad": (robust.reference_mad),
                "robust_scale": (robust.robust_scale),
                "robust_signed_deviation": (robust.deviation),
                "robust_deviation": (robust.robust_deviation),
                "robust_deviation_threshold": float(test["robust_deviation_threshold"]),
                "robust_failed": (robust.failed),
            }
        )

    if evidence.predictive is not None:
        predictive = evidence.predictive

        details.update(
            {
                "reference_mean": (predictive.reference_mean),
                "reference_standard_deviation": (
                    predictive.reference_standard_deviation
                ),
                "prediction_standard_error": (predictive.prediction_standard_error),
                "predictive_signed_deviation": (predictive.deviation),
                "degrees_of_freedom": (predictive.degrees_of_freedom),
                "t_statistic": (predictive.t_statistic),
                "predictive_probability": (predictive.predictive_probability),
                "maximum_predictive_probability": float(
                    test["maximum_predictive_probability"]
                ),
                "predictive_failed": (predictive.failed),
            }
        )

    return details


def build_details(
    data: pd.Series,
    *,
    reference: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    test: Mapping[str, Any],
    grid: TimeGrid,
) -> dict[str, Any]:
    """Build evidence for one contextual-level failure period."""
    failed_observations: list[dict[str, Any]] = []

    failed_criteria: list[str] = []

    period = data.loc[(data.index >= start) & (data.index < end)]

    for timestamp, target in period.items():
        if pd.isna(target):
            continue

        contextual_reference = _reference_values(reference, target=timestamp, test=test)

        evidence = _contextual_level_evidence(
            float(target), contextual_reference, test=test
        )

        if not evidence.failed:
            continue

        failed_observations.append(
            _observation_details(
                float(target),
                timestamp=timestamp,
                reference=contextual_reference,
                evidence=evidence,
                test=test,
            )
        )

        for criterion in evidence.failed_criteria:
            if criterion not in failed_criteria:
                failed_criteria.append(criterion)

    return {
        "failed_observations": len(failed_observations),
        "failed_criteria": failed_criteria,
        "observations": failed_observations,
    }
