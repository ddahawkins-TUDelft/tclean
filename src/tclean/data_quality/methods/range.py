"""Range-based data-quality testing."""

from collections.abc import Mapping
from typing import Any

import pandas as pd

from tclean.data_quality._method import (
    MethodContext,
    MethodIssue,
    MethodResult,
    MethodSpec,
)
from tclean.data_quality._validation_helpers import (
    normalize_common_selectors,
    validate_keys,
)
from tclean.data_quality._value_spec import (
    ValueResolution,
    build_value_issues,
    is_derived_value_spec,
    normalize_value_reference_controls,
    normalize_value_spec,
    resolve_context_value,
    value_resolution_details,
)
from tclean.time_grid import TimeGrid


def validate(test: Mapping[str, Any], *, grid: TimeGrid) -> dict[str, Any]:
    """Validate and normalize a range quality test."""
    del grid

    validate_keys(
        test,
        required={"name", "method"},
        optional={
            "sources",
            "contexts",
            "minimum",
            "maximum",
            "include_failed_periods_from",
        },
    )

    if "minimum" not in test and "maximum" not in test:
        raise ValueError(
            "Range quality tests require at least one of 'minimum' or 'maximum'."
        )

    normalized = normalize_common_selectors(test)

    if "minimum" in test:
        normalized["minimum"] = normalize_value_spec(test["minimum"], field="minimum")

    if "maximum" in test:
        normalized["maximum"] = normalize_value_spec(test["maximum"], field="maximum")

    if "minimum" in normalized and "maximum" in normalized:
        minimum = normalized["minimum"]
        maximum = normalized["maximum"]

        if (
            not is_derived_value_spec(minimum)
            and not is_derived_value_spec(maximum)
            and minimum["value"] > maximum["value"]
        ):
            raise ValueError(
                "Fixed 'minimum' must be less than or equal to fixed 'maximum'."
            )

    normalize_value_reference_controls(test, normalized, fields=("minimum", "maximum"))

    return normalized


def _resolved_bounds(
    context: MethodContext, *, context_name: str
) -> tuple[ValueResolution | None, ValueResolution | None]:
    """Resolve configured bounds for one focal source-context."""
    minimum = (
        resolve_context_value(context, field="minimum", context_name=context_name)
        if "minimum" in context.test
        else None
    )
    maximum = (
        resolve_context_value(context, field="maximum", context_name=context_name)
        if "maximum" in context.test
        else None
    )

    return minimum, maximum


def _resolution_problems(
    context: MethodContext,
    *,
    minimum: ValueResolution | None,
    maximum: ValueResolution | None,
) -> list[dict[str, Any]]:
    """Return diagnostics preventing range evaluation."""
    problems: list[dict[str, Any]] = []

    for field, resolution in (("minimum", minimum), ("maximum", maximum)):
        if resolution is not None and not resolution.evaluable:
            problems.append(
                value_resolution_details(
                    resolution, field=field, spec=context.test[field]
                )
            )

    if (
        not problems
        and minimum is not None
        and maximum is not None
        and minimum.value is not None
        and maximum.value is not None
        and minimum.value > maximum.value
    ):
        problems.append(
            {
                "reason": "resolved_bounds_inverted",
                "minimum": minimum.value,
                "maximum": maximum.value,
                "minimum_resolution": value_resolution_details(
                    minimum, field="minimum", spec=context.test["minimum"]
                ),
                "maximum_resolution": value_resolution_details(
                    maximum, field="maximum", spec=context.test["maximum"]
                ),
            }
        )

    return problems


def evaluate(context: MethodContext) -> MethodResult:
    """Evaluate whether observed values fall outside configured bounds."""
    data = context.target_data
    failures = pd.DataFrame(False, index=data.index, columns=data.columns, dtype=bool)
    issues: list[MethodIssue] = []

    for context_name in data.columns:
        values = data[context_name]
        minimum, maximum = _resolved_bounds(context, context_name=context_name)
        context_issues = build_value_issues(
            context,
            context_name=context_name,
            problems=_resolution_problems(context, minimum=minimum, maximum=maximum),
            candidates=values.notna(),
        )

        if context_issues:
            issues.extend(context_issues)
            continue

        observed = values.notna()

        if minimum is not None and minimum.value is not None:
            failures[context_name] |= observed & values.lt(minimum.value)

        if maximum is not None and maximum.value is not None:
            failures[context_name] |= observed & values.gt(maximum.value)

    return MethodResult(mask=failures, issues=tuple(issues))


def build_details(
    context: MethodContext,
    result: MethodResult,
    *,
    context_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Build structured diagnostics for one failed range period."""
    del result

    data = context.target_data[context_name]
    minimum, maximum = _resolved_bounds(context, context_name=context_name)

    if _resolution_problems(context, minimum=minimum, maximum=maximum):
        raise RuntimeError(
            "Cannot build range failure details with unusable resolved bounds."
        )

    failed_values = data.loc[(data.index >= start) & (data.index < end)].dropna()

    details: dict[str, Any] = {
        "observed_minimum": float(failed_values.min()),
        "observed_maximum": float(failed_values.max()),
    }

    if minimum is not None and minimum.value is not None:
        details["minimum"] = minimum.value
        details["minimum_resolution"] = value_resolution_details(
            minimum, field="minimum", spec=context.test["minimum"]
        )
        details["maximum_below_minimum"] = max(
            0.0, float(minimum.value - failed_values.min())
        )

    if maximum is not None and maximum.value is not None:
        details["maximum"] = maximum.value
        details["maximum_resolution"] = value_resolution_details(
            maximum, field="maximum", spec=context.test["maximum"]
        )
        details["maximum_above_maximum"] = max(
            0.0, float(failed_values.max() - maximum.value)
        )

    return details


METHOD = MethodSpec(
    name="range", validate=validate, evaluate=evaluate, build_details=build_details
)
