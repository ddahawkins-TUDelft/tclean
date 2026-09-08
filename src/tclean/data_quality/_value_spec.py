"""Validation and resolution of configurable scalar values."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

import pandas as pd

from tclean.data_quality._method import MethodContext, MethodIssue
from tclean.data_quality._periods import failure_mask_to_periods
from tclean.data_quality._validation_helpers import (
    finite_real,
    normalize_string_sequence,
    string_choice,
)

_VALUE_MODES = (
    "fixed",
    "mean",
    "median",
    "quantile",
    "quantile_range",
    "standard_deviation",
    "mean_absolute_increment",
    "median_absolute_increment",
)


@dataclass(frozen=True)
class ValueResolution:
    """Result of resolving one configured scalar value."""

    value: float | None
    property_value: float | None
    eligible_observations: int | None
    eligible_increments: int | None = None
    reason: str | None = None

    @property
    def evaluable(self) -> bool:
        """Return whether the specification resolved to a finite value."""
        return self.value is not None and self.reason is None


def _validate_spec_keys(
    spec: Mapping[str, Any],
    *,
    field: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    """Require exactly the supported keys for one value specification."""
    optional = optional or set()

    keys = set(spec)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)

    if missing or unknown:
        raise ValueError(
            f"Invalid value specification for {field!r}. "
            f"Missing keys: {missing!r}; unknown keys: {unknown!r}."
        )


def _normalize_quantile(value: object, *, field: str) -> float:
    """Normalize a quantile bounded inclusively between zero and one."""
    normalized = float(finite_real(value, field=field))

    if not 0 <= normalized <= 1:
        raise ValueError(f"{field!r} must be between zero and one.")

    return normalized


def normalize_value_spec(value: object, *, field: str) -> dict[str, Any]:
    """Validate and normalize one configurable scalar value specification."""
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{field!r} must be a value specification containing 'value_mode'."
        )

    if "value_mode" not in value:
        raise ValueError(
            f"Invalid value specification for {field!r}. Missing keys: ['value_mode']."
        )

    value_mode = string_choice(
        value["value_mode"], field=f"{field}.value_mode", choices=_VALUE_MODES
    )

    if value_mode == "fixed":
        _validate_spec_keys(value, field=field, required={"value_mode", "value"})

        return {
            "value_mode": value_mode,
            "value": float(finite_real(value["value"], field=f"{field}.value")),
        }

    if value_mode in {
        "mean",
        "median",
        "standard_deviation",
        "mean_absolute_increment",
        "median_absolute_increment",
    }:
        _validate_spec_keys(
            value, field=field, required={"value_mode"}, optional={"multiplier"}
        )

        return {
            "value_mode": value_mode,
            "multiplier": float(
                finite_real(value.get("multiplier", 1.0), field=f"{field}.multiplier")
            ),
        }

    if value_mode == "quantile":
        _validate_spec_keys(
            value,
            field=field,
            required={"value_mode", "quantile"},
            optional={"multiplier"},
        )

        return {
            "value_mode": value_mode,
            "quantile": _normalize_quantile(
                value["quantile"], field=f"{field}.quantile"
            ),
            "multiplier": float(
                finite_real(value.get("multiplier", 1.0), field=f"{field}.multiplier")
            ),
        }

    if value_mode == "quantile_range":
        _validate_spec_keys(
            value,
            field=field,
            required={"value_mode", "lower_quantile", "upper_quantile"},
            optional={"multiplier"},
        )

        lower_quantile = _normalize_quantile(
            value["lower_quantile"], field=f"{field}.lower_quantile"
        )
        upper_quantile = _normalize_quantile(
            value["upper_quantile"], field=f"{field}.upper_quantile"
        )

        if lower_quantile >= upper_quantile:
            raise ValueError(
                f"{field!r} requires 'lower_quantile' to be less than 'upper_quantile'."
            )

        return {
            "value_mode": value_mode,
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
            "multiplier": float(
                finite_real(value.get("multiplier", 1.0), field=f"{field}.multiplier")
            ),
        }

    raise AssertionError(f"Unhandled value mode: {value_mode!r}.")


def fixed_value_spec(value: object, *, field: str) -> dict[str, Any]:
    """Build and normalize an internal fixed-value specification."""
    return normalize_value_spec({"value_mode": "fixed", "value": value}, field=field)


def is_derived_value_spec(spec: Mapping[str, Any]) -> bool:
    """Return whether a normalized value specification depends on focal data."""
    return spec["value_mode"] != "fixed"


def require_fixed_value_spec(spec: Mapping[str, Any], *, field: str) -> None:
    """Require a normalized specification to use a literal fixed value."""
    if is_derived_value_spec(spec):
        raise ValueError(
            f"{field!r} must use 'value_mode: fixed' because it is dimensionless."
        )


def normalize_value_reference_controls(
    test: Mapping[str, Any], normalized: dict[str, Any], *, fields: Sequence[str]
) -> None:
    """Normalize prior-failure controls used only by derived value fields."""
    if "include_failed_periods_from" not in test:
        return

    derived_fields = [
        field
        for field in fields
        if field in normalized and is_derived_value_spec(normalized[field])
    ]

    if not derived_fields:
        raise ValueError(
            "'include_failed_periods_from' is only supported when at least one "
            "configured value uses a derived 'value_mode'."
        )

    normalized["include_failed_periods_from"] = normalize_string_sequence(
        test["include_failed_periods_from"], field="include_failed_periods_from"
    )


def _unresolved(
    *,
    property_value: float | None,
    eligible_observations: int,
    eligible_increments: int | None = None,
    reason: str,
) -> ValueResolution:
    """Build one unresolved value result."""
    return ValueResolution(
        value=None,
        property_value=property_value,
        eligible_observations=eligible_observations,
        eligible_increments=eligible_increments,
        reason=reason,
    )


def resolve_value_spec(
    spec: Mapping[str, Any], *, data: pd.Series | None = None
) -> ValueResolution:
    """Resolve a normalized value specification against eligible focal data."""
    value_mode = spec["value_mode"]

    if value_mode == "fixed":
        return ValueResolution(
            value=float(spec["value"]), property_value=None, eligible_observations=None
        )

    if data is None:
        raise ValueError(
            "Derived value specifications require focal source-context data."
        )

    eligible_observations = int(data.notna().sum())

    if eligible_observations == 0:
        return _unresolved(
            property_value=None,
            eligible_observations=0,
            reason="no_eligible_observations",
        )

    eligible_increments: int | None = None

    if value_mode == "mean":
        property_value = float(data.mean())
    elif value_mode == "median":
        property_value = float(data.median())
    elif value_mode == "quantile":
        property_value = float(data.quantile(spec["quantile"]))
    elif value_mode == "quantile_range":
        lower = float(data.quantile(spec["lower_quantile"]))
        upper = float(data.quantile(spec["upper_quantile"]))
        property_value = upper - lower
    elif value_mode == "standard_deviation":
        property_value = float(data.std(ddof=0))
    elif value_mode in {"mean_absolute_increment", "median_absolute_increment"}:
        absolute_increments = data.diff().abs().dropna()
        eligible_increments = len(absolute_increments)

        if eligible_increments == 0:
            return _unresolved(
                property_value=None,
                eligible_observations=eligible_observations,
                eligible_increments=0,
                reason="no_eligible_increments",
            )

        if value_mode == "mean_absolute_increment":
            property_value = float(absolute_increments.mean())
        else:
            property_value = float(absolute_increments.median())
    else:
        raise ValueError(f"Unsupported normalized value mode: {value_mode!r}.")

    if not isfinite(property_value):
        return _unresolved(
            property_value=None,
            eligible_observations=eligible_observations,
            eligible_increments=eligible_increments,
            reason="nonfinite_property_value",
        )

    resolved_value = property_value * float(spec.get("multiplier", 1.0))

    if not isfinite(resolved_value):
        return _unresolved(
            property_value=property_value,
            eligible_observations=eligible_observations,
            eligible_increments=eligible_increments,
            reason="nonfinite_resolved_value",
        )

    return ValueResolution(
        value=resolved_value,
        property_value=property_value,
        eligible_observations=eligible_observations,
        eligible_increments=eligible_increments,
    )


def resolve_context_value(
    context: MethodContext, *, field: str, context_name: str
) -> ValueResolution:
    """Resolve one configured value independently for a focal source-context."""
    spec = context.test[field]

    if not is_derived_value_spec(spec):
        return resolve_value_spec(spec)

    eligible_data = context.reference_data(
        context.source_name, contexts=(context_name,)
    )[context_name]

    return resolve_value_spec(spec, data=eligible_data)


def value_resolution_reason(
    resolution: ValueResolution,
    *,
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> str | None:
    """Return why a resolved value violates an optional lower bound."""
    if not resolution.evaluable:
        return resolution.reason

    if resolution.value is None:
        return "missing_resolved_value"

    if minimum is None:
        return None

    if strict_minimum and resolution.value <= minimum:
        if minimum == 0:
            return "resolved_value_not_positive"
        return "resolved_value_not_above_minimum"

    if not strict_minimum and resolution.value < minimum:
        if minimum == 0:
            return "resolved_value_negative"
        return "resolved_value_below_minimum"

    return None


def value_resolution_details(
    resolution: ValueResolution,
    *,
    field: str,
    spec: Mapping[str, Any],
    reason: str | None = None,
) -> dict[str, Any]:
    """Build serializable diagnostics for one configured value resolution."""
    details: dict[str, Any] = {
        "field": field,
        "value_mode": spec["value_mode"],
        "value_spec": dict(spec),
    }

    if resolution.property_value is not None:
        details["property_value"] = resolution.property_value
    if resolution.value is not None:
        details["resolved_value"] = resolution.value
    if resolution.eligible_observations is not None:
        details["eligible_observations"] = resolution.eligible_observations
    if resolution.eligible_increments is not None:
        details["eligible_increments"] = resolution.eligible_increments

    resolved_reason = reason or resolution.reason
    if resolved_reason is not None:
        details["reason"] = resolved_reason

    return details


def value_resolution_problem(
    resolution: ValueResolution,
    *,
    field: str,
    spec: Mapping[str, Any],
    minimum: float | None = None,
    strict_minimum: bool = False,
) -> dict[str, Any] | None:
    """Return diagnostics when one resolution is unusable by a consumer."""
    reason = value_resolution_reason(
        resolution, minimum=minimum, strict_minimum=strict_minimum
    )

    if reason is None:
        return None

    return value_resolution_details(resolution, field=field, spec=spec, reason=reason)


def build_value_issues(
    context: MethodContext,
    *,
    context_name: str,
    problems: Sequence[Mapping[str, Any]],
    candidates: pd.Series,
) -> list[MethodIssue]:
    """Build grouped not-evaluable issues for unusable configured values."""
    if not problems:
        return []

    issues: list[MethodIssue] = []

    for start, end in failure_mask_to_periods(
        candidates.astype(bool), grid=context.grid
    ):
        issues.append(
            MethodIssue(
                context=context_name,
                start=start,
                end=end,
                severity="not_evaluable",
                code="derived_value_not_evaluable",
                details={"values": [dict(problem) for problem in problems]},
            )
        )

    return issues
