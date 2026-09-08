"""Validation and resolution of configurable scalar values."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

import pandas as pd

from tclean.data_quality._method import MethodContext
from tclean.data_quality._validation_helpers import finite_real, string_choice

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


def _normalize_quantile(
    value: object,
    *,
    field: str,
) -> float:
    """Normalize a quantile bounded inclusively between zero and one."""
    normalized = float(finite_real(value, field=field))

    if not 0 <= normalized <= 1:
        raise ValueError(f"{field!r} must be between zero and one.")

    return normalized


def normalize_value_spec(
    value: object,
    *,
    field: str,
) -> dict[str, Any]:
    """Validate and normalize one configurable scalar value specification."""
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{field!r} must be a value specification containing 'value_mode'."
        )

    if "value_mode" not in value:
        raise ValueError(
            f"Invalid value specification for {field!r}. "
            "Missing keys: ['value_mode']."
        )

    value_mode = string_choice(
        value["value_mode"],
        field=f"{field}.value_mode",
        choices=_VALUE_MODES,
    )

    if value_mode == "fixed":
        _validate_spec_keys(
            value,
            field=field,
            required={"value_mode", "value"},
        )

        return {
            "value_mode": value_mode,
            "value": float(
                finite_real(
                    value["value"],
                    field=f"{field}.value",
                )
            ),
        }

    if value_mode in {
        "mean",
        "median",
        "standard_deviation",
        "mean_absolute_increment",
        "median_absolute_increment",
    }:
        _validate_spec_keys(
            value,
            field=field,
            required={"value_mode"},
            optional={"multiplier"},
        )

        return {
            "value_mode": value_mode,
            "multiplier": float(
                finite_real(
                    value.get("multiplier", 1.0),
                    field=f"{field}.multiplier",
                )
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
                value["quantile"],
                field=f"{field}.quantile",
            ),
            "multiplier": float(
                finite_real(
                    value.get("multiplier", 1.0),
                    field=f"{field}.multiplier",
                )
            ),
        }

    if value_mode == "quantile_range":
        _validate_spec_keys(
            value,
            field=field,
            required={
                "value_mode",
                "lower_quantile",
                "upper_quantile",
            },
            optional={"multiplier"},
        )

        lower_quantile = _normalize_quantile(
            value["lower_quantile"],
            field=f"{field}.lower_quantile",
        )
        upper_quantile = _normalize_quantile(
            value["upper_quantile"],
            field=f"{field}.upper_quantile",
        )

        if lower_quantile >= upper_quantile:
            raise ValueError(
                f"{field!r} requires 'lower_quantile' to be less than "
                "'upper_quantile'."
            )

        return {
            "value_mode": value_mode,
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
            "multiplier": float(
                finite_real(
                    value.get("multiplier", 1.0),
                    field=f"{field}.multiplier",
                )
            ),
        }

    raise AssertionError(f"Unhandled value mode: {value_mode!r}.")


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
    spec: Mapping[str, Any],
    *,
    data: pd.Series | None = None,
) -> ValueResolution:
    """Resolve a normalized value specification against eligible focal data."""
    value_mode = spec["value_mode"]

    if value_mode == "fixed":
        return ValueResolution(
            value=float(spec["value"]),
            property_value=None,
            eligible_observations=None,
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

    elif value_mode in {
        "mean_absolute_increment",
        "median_absolute_increment",
    }:
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
        raise ValueError(
            f"Unsupported normalized value mode: {value_mode!r}."
        )

    if not isfinite(property_value):
        return _unresolved(
            property_value=None,
            eligible_observations=eligible_observations,
            eligible_increments=eligible_increments,
            reason="nonfinite_property_value",
        )

    resolved_value = property_value * float(
        spec.get("multiplier", 1.0)
    )

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
    context: MethodContext,
    *,
    field: str,
    context_name: str,
) -> ValueResolution:
    """Resolve one configured value for one focal source-context."""
    spec = context.test[field]

    if spec["value_mode"] == "fixed":
        return resolve_value_spec(spec)

    eligible_data = context.reference_data(
        context.source_name,
        contexts=(context_name,),
    )[context_name]

    return resolve_value_spec(
        spec,
        data=eligible_data,
    )


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
