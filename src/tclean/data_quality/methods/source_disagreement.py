"""Cross-source disagreement data-quality testing."""

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tclean.data_quality._method import (
    MethodContext,
    MethodIssue,
    MethodResult,
    MethodSpec,
)
from tclean.data_quality._periods import failure_mask_to_periods
from tclean.data_quality._validation_helpers import (
    normalize_common_selectors,
    normalize_string_sequence,
    string_choice,
    validate_keys,
)
from tclean.data_quality._value_spec import (
    build_value_issues,
    normalize_value_spec,
    require_fixed_value_spec,
    resolve_context_value,
    value_resolution_details,
    value_resolution_problem,
)
from tclean.time_grid import TimeGrid


def validate(test: Mapping[str, Any], *, grid: TimeGrid) -> dict[str, Any]:
    """Validate and normalize a source-disagreement quality test."""
    del grid

    validate_keys(
        test,
        required={"name", "method", "difference_mode", "threshold"},
        optional={
            "sources",
            "contexts",
            "include_failed_periods_from",
            "peer_aggregation_mode",
        },
    )

    normalized = normalize_common_selectors(test)
    difference_mode = string_choice(
        test["difference_mode"], field="difference_mode", choices=("fixed", "relative")
    )
    normalized["peer_aggregation_mode"] = string_choice(
        test.get("peer_aggregation_mode", "median"),
        field="peer_aggregation_mode",
        choices=("mean", "median"),
    )

    threshold = normalize_value_spec(test["threshold"], field="threshold")
    if difference_mode == "relative":
        require_fixed_value_spec(threshold, field="threshold")
    if threshold["value_mode"] == "fixed" and threshold["value"] <= 0:
        raise ValueError("Fixed 'threshold.value' must be greater than zero.")

    normalized["difference_mode"] = difference_mode
    normalized["threshold"] = threshold

    if "include_failed_periods_from" in test:
        normalized["include_failed_periods_from"] = normalize_string_sequence(
            test["include_failed_periods_from"], field="include_failed_periods_from"
        )

    return normalized


def _peer_frame(context: MethodContext, *, context_name: str) -> pd.DataFrame:
    """Return eligible observations from all non-focal sources for one context."""
    peers: dict[str, pd.Series] = {}
    target_index = context.target_data.index

    for peer_name in context.source_names:
        if peer_name == context.source_name:
            continue
        if not context.available_contexts(peer_name, contexts=(context_name,)):
            continue

        peer_data = context.reference_data(peer_name, contexts=(context_name,))
        peers[peer_name] = peer_data[context_name].reindex(target_index)

    return pd.DataFrame(peers, index=target_index)


def _aggregate_peers(peers: pd.DataFrame, *, mode: str) -> pd.Series:
    """Aggregate available peer observations row-wise."""
    if mode == "median":
        return peers.median(axis=1, skipna=True)
    if mode == "mean":
        return peers.mean(axis=1, skipna=True)
    raise ValueError(f"Unsupported peer aggregation mode: {mode!r}.")


def _analyse(
    focal: pd.Series,
    peer_reference: pd.Series,
    *,
    difference_mode: str,
    threshold: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate cross-source disagreement evidence and failures."""
    observed = focal.notna() & peer_reference.notna()
    signed_difference = (focal - peer_reference).where(observed)
    absolute_difference = signed_difference.abs()

    if difference_mode == "fixed":
        difference = absolute_difference.copy()
        failures = observed & difference.gt(threshold)
    elif difference_mode == "relative":
        difference = pd.Series(float("nan"), index=focal.index, dtype=float)
        nonzero_reference = observed & peer_reference.ne(0)
        difference.loc[nonzero_reference] = (
            absolute_difference.loc[nonzero_reference]
            / peer_reference.loc[nonzero_reference].abs()
        )
        both_zero = observed & peer_reference.eq(0) & focal.eq(0)
        difference.loc[both_zero] = 0.0
        unbounded = observed & peer_reference.eq(0) & focal.ne(0)
        failures = (nonzero_reference & difference.gt(threshold)) | unbounded
    else:
        raise ValueError(f"Unsupported difference mode: {difference_mode!r}.")

    return signed_difference, absolute_difference, difference, failures.astype(bool)


def evaluate(context: MethodContext) -> MethodResult:
    """Flag focal observations that disagree with eligible peer sources."""
    data = context.target_data
    test = context.test
    mask = pd.DataFrame(False, index=data.index, columns=data.columns, dtype=bool)
    issues: list[MethodIssue] = []

    for context_name in data.columns:
        focal = data[context_name]
        peers = _peer_frame(context, context_name=context_name)
        peer_observations = peers.notna().sum(axis=1)
        peer_reference = _aggregate_peers(peers, mode=test["peer_aggregation_mode"])

        unavailable = (focal.notna() & peer_observations.eq(0)).astype(bool)
        for start, end in failure_mask_to_periods(unavailable, grid=context.grid):
            issues.append(
                MethodIssue(
                    context=context_name,
                    start=start,
                    end=end,
                    severity="not_evaluable",
                    code="insufficient_peer_data",
                    details={
                        "peer_observations": 0,
                        "candidate_peer_sources": list(peers.columns),
                        "difference_mode": test["difference_mode"],
                        "peer_aggregation_mode": test["peer_aggregation_mode"],
                    },
                )
            )

        threshold = resolve_context_value(
            context, field="threshold", context_name=context_name
        )
        problem = value_resolution_problem(
            threshold,
            field="threshold",
            spec=test["threshold"],
            minimum=0.0,
            strict_minimum=True,
        )
        context_issues = build_value_issues(
            context,
            context_name=context_name,
            problems=[] if problem is None else [problem],
            candidates=focal.notna() & peer_observations.gt(0),
        )
        if context_issues:
            issues.extend(context_issues)
            continue

        if threshold.value is None:
            raise RuntimeError(
                "Evaluable source-disagreement threshold has no resolved value."
            )

        _, _, _, failures = _analyse(
            focal,
            peer_reference,
            difference_mode=test["difference_mode"],
            threshold=threshold.value,
        )
        mask[context_name] = failures

    return MethodResult(mask=mask, issues=tuple(issues))


def build_details(
    context: MethodContext,
    result: MethodResult,
    *,
    context_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Build evidence for one source-disagreement failure period."""
    focal = context.target_data[context_name]
    test = context.test
    peers = _peer_frame(context, context_name=context_name)
    peer_reference = _aggregate_peers(peers, mode=test["peer_aggregation_mode"])

    threshold = resolve_context_value(
        context, field="threshold", context_name=context_name
    )
    problem = value_resolution_problem(
        threshold,
        field="threshold",
        spec=test["threshold"],
        minimum=0.0,
        strict_minimum=True,
    )
    if problem is not None or threshold.value is None:
        raise RuntimeError(
            "Cannot build source-disagreement details with an unusable threshold."
        )

    signed_difference, absolute_difference, difference, _ = _analyse(
        focal,
        peer_reference,
        difference_mode=test["difference_mode"],
        threshold=threshold.value,
    )
    period_failures = (
        result.mask[context_name] & (focal.index >= start) & (focal.index < end)
    )

    observations: list[dict[str, Any]] = []
    for timestamp in focal.index[period_failures]:
        peer_values = peers.loc[timestamp].dropna()
        observation: dict[str, Any] = {
            "timestamp": pd.Timestamp(timestamp).isoformat(),
            "value": float(focal.loc[timestamp]),
            "peer_observations": len(peer_values),
            "peer_sources": list(peer_values.index),
            "peer_values": {
                str(source_name): float(value)
                for source_name, value in peer_values.items()
            },
            "peer_reference_value": float(peer_reference.loc[timestamp]),
            "signed_difference": float(signed_difference.loc[timestamp]),
            "absolute_difference": float(absolute_difference.loc[timestamp]),
        }
        if test["difference_mode"] == "relative":
            relative_difference = difference.loc[timestamp]
            observation["relative_difference"] = (
                None if pd.isna(relative_difference) else float(relative_difference)
            )
        observations.append(observation)

    return {
        "difference_mode": test["difference_mode"],
        "peer_aggregation_mode": test["peer_aggregation_mode"],
        "threshold": threshold.value,
        "threshold_resolution": value_resolution_details(
            threshold, field="threshold", spec=test["threshold"]
        ),
        "failed_observations": len(observations),
        "observations": observations,
    }


METHOD = MethodSpec(
    name="source_disagreement",
    validate=validate,
    evaluate=evaluate,
    build_details=build_details,
)
