"""Contextual-level data-quality testing."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from tclean.data_quality._method import (
    MethodContext,
    MethodIssue,
    MethodResult,
    MethodSpec,
)
from tclean.data_quality._parallel import ordered_thread_map
from tclean.data_quality._progress import ProgressTracker
from tclean.data_quality._validation_helpers import (
    finite_real,
    normalize_common_selectors,
    normalize_string_sequence,
    validate_keys,
)
from tclean.data_quality.methods._lattice import (
    ReferenceLattice,
    build_reference_lattice,
    normalize_reference_orders,
    reference_timestamps,
)
from tclean.data_quality.methods._robust import robust_location_scale
from tclean.time_grid import TimeGrid

logger = logging.getLogger(__name__)

_ReferenceValues = pd.Series | np.ndarray


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


@dataclass(frozen=True)
class _ContextEvaluation:
    """Evaluation output for one contextual-level context."""

    context_name: str
    mask: np.ndarray
    issues: tuple[MethodIssue, ...]
    failure_details: dict[pd.Timestamp, dict[str, Any]]


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


def _observed_reference_values(reference: _ReferenceValues) -> np.ndarray:
    """Return observed contextual-reference values as a float array."""
    if isinstance(reference, pd.Series):
        values = reference.to_numpy(dtype=float)
    else:
        values = np.asarray(reference, dtype=float)

    return values[~pd.isna(values)]


def _robust_deviation_from_values(
    target: float, reference: np.ndarray, *, threshold: float
) -> _RobustDeviation | None:
    """Evaluate one target against already-filtered robust reference values."""
    if reference.size == 0:
        return None

    robust_reference = robust_location_scale(reference)

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


def _robust_deviation(
    target: float, reference: _ReferenceValues, *, threshold: float
) -> _RobustDeviation | None:
    """Evaluate one target against a robust contextual reference."""
    return _robust_deviation_from_values(
        target, _observed_reference_values(reference), threshold=threshold
    )


def _predictive_probability_from_values(
    target: float, reference: np.ndarray, *, maximum_probability: float
) -> _PredictiveProbability | None:
    """Evaluate one target using already-filtered predictive reference values."""
    reference_observations = len(reference)

    if reference_observations < 2:
        return None

    reference_mean = float(np.mean(reference))
    reference_standard_deviation = float(np.std(reference, ddof=1))
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
        reference_standard_deviation=reference_standard_deviation,
        prediction_standard_error=prediction_standard_error,
        deviation=deviation,
        degrees_of_freedom=degrees_of_freedom,
        t_statistic=t_statistic,
        predictive_probability=predictive_probability,
        failed=failed,
    )


def _predictive_probability(
    target: float, reference: _ReferenceValues, *, maximum_probability: float
) -> _PredictiveProbability | None:
    """Evaluate one target using a Student-t predictive probability."""
    return _predictive_probability_from_values(
        target,
        _observed_reference_values(reference),
        maximum_probability=maximum_probability,
    )


def _contextual_level_evidence(
    target: float, reference: _ReferenceValues, *, test: Mapping[str, Any]
) -> _ContextualLevelEvidence:
    """Evaluate configured contextual-level criteria for one target."""
    values = _observed_reference_values(reference)

    robust = None
    predictive = None
    failed_criteria: list[str] = []

    if "robust_deviation_threshold" in test:
        robust = _robust_deviation_from_values(
            target, values, threshold=test["robust_deviation_threshold"]
        )

        if robust is not None and robust.failed:
            failed_criteria.append("robust_deviation")

    if "maximum_predictive_probability" in test:
        predictive = _predictive_probability_from_values(
            target, values, maximum_probability=test["maximum_predictive_probability"]
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


def _reference_observation_count(reference: _ReferenceValues) -> int:
    """Return the number of observed contextual reference values."""
    return int(_observed_reference_values(reference).size)


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


def _observation_details(
    target: float,
    *,
    timestamp: pd.Timestamp,
    reference: _ReferenceValues,
    evidence: _ContextualLevelEvidence,
    test: Mapping[str, Any],
) -> dict[str, Any]:
    """Build contextual-level evidence for one failed observation."""
    details: dict[str, Any] = {
        "timestamp": timestamp.isoformat(),
        "value": float(target),
        "reference_observations": _reference_observation_count(reference),
        "failed_criteria": list(evidence.failed_criteria),
    }

    if evidence.robust is not None:
        robust = evidence.robust

        details.update(
            {
                "reference_median": robust.reference_median,
                "reference_mad": robust.reference_mad,
                "robust_scale": robust.robust_scale,
                "robust_signed_deviation": robust.deviation,
                "robust_deviation": robust.robust_deviation,
                "robust_deviation_threshold": float(test["robust_deviation_threshold"]),
                "robust_failed": robust.failed,
            }
        )

    if evidence.predictive is not None:
        predictive = evidence.predictive

        details.update(
            {
                "reference_mean": predictive.reference_mean,
                "reference_standard_deviation": predictive.reference_standard_deviation,
                "prediction_standard_error": predictive.prediction_standard_error,
                "predictive_signed_deviation": predictive.deviation,
                "degrees_of_freedom": predictive.degrees_of_freedom,
                "t_statistic": predictive.t_statistic,
                "predictive_probability": predictive.predictive_probability,
                "maximum_predictive_probability": float(
                    test["maximum_predictive_probability"]
                ),
                "predictive_failed": predictive.failed,
            }
        )

    return details


def _evaluate_context(
    context_name: str,
    *,
    data: pd.DataFrame,
    reference: pd.DataFrame,
    lattice: ReferenceLattice,
    test: Mapping[str, Any],
    grid: TimeGrid,
    progress: ProgressTracker,
) -> _ContextEvaluation:
    """Evaluate one contextual-level context independently."""
    target_values = data[context_name].to_numpy(dtype=float)
    reference_values = reference[context_name].to_numpy(dtype=float)

    context_mask = np.zeros(len(data.index), dtype=bool)
    issues: list[MethodIssue] = []
    failure_details: dict[pd.Timestamp, dict[str, Any]] = {}

    configured_criteria = _configured_criteria(test)

    for target_number, timestamp in enumerate(data.index):
        target = target_values[target_number]

        if pd.isna(target):
            continue

        reference_positions = lattice.positions_for(target_number)
        contextual_reference = reference_values[reference_positions]

        evidence = _contextual_level_evidence(
            float(target), contextual_reference, test=test
        )

        reference_observations = _reference_observation_count(contextual_reference)
        timestamp = pd.Timestamp(timestamp)

        if not evidence.evaluable:
            issues.append(
                MethodIssue(
                    context=context_name,
                    start=timestamp,
                    end=timestamp + grid.frequency,
                    severity="not_evaluable",
                    code="insufficient_reference_data",
                    details={
                        "reference_observations": reference_observations,
                        "configured_criteria": configured_criteria,
                    },
                )
            )
            continue

        context_mask[target_number] = evidence.failed

        for criterion in _unavailable_criteria(evidence, test=test):
            issues.append(
                MethodIssue(
                    context=context_name,
                    start=timestamp,
                    end=timestamp + grid.frequency,
                    severity="warning",
                    code="criterion_not_evaluable",
                    details={
                        "criterion": criterion,
                        "reference_observations": reference_observations,
                    },
                )
            )

        if evidence.failed:
            failure_details[timestamp] = _observation_details(
                float(target),
                timestamp=timestamp,
                reference=contextual_reference,
                evidence=evidence,
                test=test,
            )

    progress.complete(context_name)

    return _ContextEvaluation(
        context_name=context_name,
        mask=context_mask,
        issues=tuple(issues),
        failure_details=failure_details,
    )


def evaluate(context: MethodContext) -> MethodResult:
    """Evaluate contextual levels and report evaluation limitations."""
    data = context.target_data
    reference = context.reference_data(context.source_name)
    test = context.test
    grid = context.grid

    lattice = build_reference_lattice(
        data.index, orders=test["reference_orders"], available_index=reference.index
    )

    progress = ProgressTracker(
        total=len(data.columns),
        label=f"{test['name']} [{test['method']}]",
        logger=logger,
    )

    def evaluate_context(context_name: str) -> _ContextEvaluation:
        return _evaluate_context(
            context_name,
            data=data,
            reference=reference,
            lattice=lattice,
            test=test,
            grid=grid,
            progress=progress,
        )

    results = ordered_thread_map(
        evaluate_context, list(data.columns), threads=context.threads
    )

    mask = pd.DataFrame(False, index=data.index, columns=data.columns, dtype=bool)
    issues: list[MethodIssue] = []
    failure_details: dict[str, dict[pd.Timestamp, dict[str, Any]]] = {}

    for context_result in results:
        mask[context_result.context_name] = context_result.mask
        issues.extend(context_result.issues)
        failure_details[context_result.context_name] = context_result.failure_details

    return MethodResult(
        mask=mask,
        issues=tuple(issues),
        diagnostics={"failure_details": failure_details},
    )


def build_details(
    context: MethodContext,
    result: MethodResult,
    *,
    context_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Build evidence for one contextual-level failure period."""
    stored_details = result.diagnostics.get("failure_details", {}).get(context_name, {})

    failed_observations = [
        details
        for timestamp, details in stored_details.items()
        if start <= timestamp < end
    ]

    failed_criteria: list[str] = []

    for details in failed_observations:
        for criterion in details["failed_criteria"]:
            if criterion not in failed_criteria:
                failed_criteria.append(criterion)

    return {
        "failed_observations": len(failed_observations),
        "failed_criteria": failed_criteria,
        "observations": failed_observations,
    }


METHOD = MethodSpec(
    name="contextual_level",
    validate=validate,
    evaluate=evaluate,
    build_details=build_details,
)
