"""Validate and normalize basic cleaning rules."""

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from tclean.basic.methods.average_periods import METHOD_NAME as AVERAGE_PERIODS
from tclean.basic.methods.copy_periods import METHOD_NAME as COPY_PERIODS
from tclean.basic.methods.linear_interpolation import (
    METHOD_NAME as LINEAR_INTERPOLATION,
)


def _timedelta(value: object, *, field: str) -> pd.Timedelta:
    """Validate and normalize an explicit fixed duration."""
    if pd.isna(value):
        raise ValueError(f"{field!r} must not be missing.")

    if isinstance(value, str):
        value = value.strip()

        if not value:
            raise ValueError(f"{field!r} must not be empty.")

    elif not isinstance(value, pd.Timedelta):
        raise TypeError(f"{field!r} must be a duration string or pandas Timedelta.")

    try:
        delta = pd.Timedelta(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field!r} must be a valid fixed duration.") from exc

    if pd.isna(delta):
        raise ValueError(f"{field!r} must not be missing.")

    return delta


def _validate_frequency_multiple(
    delta: pd.Timedelta, *, field: str, frequency: pd.Timedelta
) -> None:
    """Require a duration to align with the configured frequency."""
    if delta % frequency != pd.Timedelta(0):
        raise ValueError(
            f"{field!r} must be an integer multiple "
            "of the configured frequency. "
            f"Value: {delta}; frequency: {frequency}."
        )


def _positive_timedelta(
    value: object, *, field: str, frequency: pd.Timedelta
) -> pd.Timedelta:
    """Convert a value to a positive frequency-aligned timedelta."""
    delta = _timedelta(value, field=field)

    if delta <= pd.Timedelta(0):
        raise ValueError(f"{field!r} must be greater than zero.")

    _validate_frequency_multiple(delta, field=field, frequency=frequency)

    return delta


def _nonzero_timedelta(
    value: object, *, field: str, frequency: pd.Timedelta
) -> pd.Timedelta:
    """Convert a value to a non-zero frequency-aligned timedelta."""
    delta = _timedelta(value, field=field)

    if delta == pd.Timedelta(0):
        raise ValueError(f"{field!r} must not be zero.")

    _validate_frequency_multiple(delta, field=field, frequency=frequency)

    return delta


def _validate_keys(
    rule: Mapping[str, Any], *, required: set[str], optional: set[str] | None = None
) -> None:
    """Require exactly the supported keys for a rule."""
    optional = optional or set()

    keys = set(rule)

    missing = sorted(required - keys)

    unknown = sorted(keys - required - optional)

    if missing or unknown:
        raise ValueError(
            "Invalid basic rule configuration. "
            f"Missing keys: {missing!r}; "
            f"unknown keys: {unknown!r}."
        )


def _validate_common_fields(rule: Mapping[str, Any]) -> None:
    """Validate fields shared by every basic rule."""
    name = rule.get("name")
    method = rule.get("method")

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Basic rule 'name' must be a non-empty string.")

    if not isinstance(method, str) or not method.strip():
        raise ValueError("Basic rule 'method' must be a non-empty string.")


def _validate_linear_interpolation_rule(
    rule: Mapping[str, Any], *, frequency: pd.Timedelta
) -> dict[str, Any]:
    """Validate and normalize a linear-interpolation rule."""
    _validate_keys(rule, required={"name", "method", "max_gap"})

    normalized = dict(rule)

    normalized["max_gap"] = _positive_timedelta(
        rule["max_gap"], field="max_gap", frequency=frequency
    )

    return normalized


def _validate_copy_periods_rule(
    rule: Mapping[str, Any], *, frequency: pd.Timedelta
) -> dict[str, Any]:
    """Validate and normalize a copy-periods rule."""
    _validate_keys(
        rule,
        required={
            "name",
            "method",
            "max_gap",
            "source_offset",
            "require_complete_source",
        },
    )

    if not isinstance(rule["require_complete_source"], bool):
        raise ValueError("'require_complete_source' must be boolean.")

    normalized = dict(rule)

    normalized["max_gap"] = _positive_timedelta(
        rule["max_gap"], field="max_gap", frequency=frequency
    )

    normalized["source_offset"] = _nonzero_timedelta(
        rule["source_offset"], field="source_offset", frequency=frequency
    )

    return normalized


def _validate_average_periods_rule(
    rule: Mapping[str, Any], *, frequency: pd.Timedelta
) -> dict[str, Any]:
    """Validate and normalize an average-periods rule."""
    _validate_keys(rule, required={"name", "method", "max_gap", "source_offsets"})

    source_offsets = rule["source_offsets"]

    if (
        isinstance(source_offsets, (str, bytes))
        or not isinstance(source_offsets, Sequence)
        or not source_offsets
    ):
        raise ValueError("'source_offsets' must be a non-empty sequence.")

    normalized = dict(rule)

    normalized["max_gap"] = _positive_timedelta(
        rule["max_gap"], field="max_gap", frequency=frequency
    )

    normalized["source_offsets"] = [
        _nonzero_timedelta(offset, field="source_offsets", frequency=frequency)
        for offset in source_offsets
    ]

    return normalized


_RULE_VALIDATORS = {
    LINEAR_INTERPOLATION: _validate_linear_interpolation_rule,
    COPY_PERIODS: _validate_copy_periods_rule,
    AVERAGE_PERIODS: _validate_average_periods_rule,
}


def validate_basic_rule(
    rule: Mapping[str, Any], frequency: pd.Timedelta
) -> dict[str, Any]:
    """Validate and normalize one basic cleaning rule.

    Args:
        rule: Basic cleaning rule configuration.
        frequency: pd.Timedelta of time series frequency.

    Returns:
        Validated and normalized rule.

    Raises:
        TypeError: If the rule is not a mapping.
        ValueError: If the rule is invalid or uses an unsupported method.
    """
    if not isinstance(rule, Mapping):
        raise TypeError("Each basic cleaning rule must be a mapping.")

    _validate_common_fields(rule)

    method = rule["method"]

    try:
        validator = _RULE_VALIDATORS[method]
    except KeyError as exc:
        raise ValueError(f"Unsupported basic cleaning method: {method!r}.") from exc

    return validator(rule, frequency=frequency)


def validate_basic_rules(
    rules: Sequence[Mapping[str, Any]], frequency: pd.Timedelta
) -> list[dict[str, Any]]:
    """Validate and normalize ordered basic cleaning rules.

    Args:
        rules: Ordered basic cleaning rule configurations.
        frequency: pd.Timedelta of time series frequency.

    Returns:
        Validated rules in their original execution order.

    Raises:
    TypeError: If rules are not supplied as an ordered sequence.
    ValueError: If a rule is invalid or rule names are duplicated.
    """
    if isinstance(rules, (str, bytes)) or not isinstance(rules, Sequence):
        raise TypeError("Basic cleaning rules must be an ordered sequence.")

    normalized = [validate_basic_rule(rule, frequency=frequency) for rule in rules]

    names = [rule["name"] for rule in normalized]

    duplicates = sorted({name for name in names if names.count(name) > 1})

    if duplicates:
        raise ValueError(
            f"Basic cleaning rule names must be unique. Duplicates: {duplicates!r}."
        )

    return normalized
