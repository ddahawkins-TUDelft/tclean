"""Shared helpers for validating data-quality test configuration."""

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Integral, Real
from typing import Any

import pandas as pd

from tclean._temporal import normalize_fixed_duration
from tclean.time_grid import TimeGrid


def validate_keys(
    test: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    """Require exactly the supported keys for a quality test."""
    optional = optional or set()

    keys = set(test)

    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)

    if missing or unknown:
        raise ValueError(
            "Invalid data-quality test configuration. "
            f"Missing keys: {missing!r}; "
            f"unknown keys: {unknown!r}."
        )


def normalize_string_sequence(
    value: object,
    *,
    field: str,
) -> list[str]:
    """Normalize a non-empty ordered sequence of unique strings."""
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or not value
    ):
        raise ValueError(f"{field!r} must be a non-empty ordered sequence.")

    normalized = list(value)

    invalid = [
        item
        for item in normalized
        if not isinstance(item, str) or not item.strip()
    ]

    if invalid:
        raise ValueError(
            f"{field!r} must contain only non-empty strings."
        )

    duplicates = sorted(
        {
            item
            for item in normalized
            if normalized.count(item) > 1
        }
    )

    if duplicates:
        raise ValueError(
            f"{field!r} entries must be unique. "
            f"Duplicates: {duplicates!r}."
        )

    return normalized


def normalize_common_selectors(
    test: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize optional source and context selectors."""
    normalized = dict(test)

    if "sources" in test:
        normalized["sources"] = normalize_string_sequence(
            test["sources"],
            field="sources",
        )

    if "contexts" in test:
        normalized["contexts"] = normalize_string_sequence(
            test["contexts"],
            field="contexts",
        )

    return normalized


def finite_real(
    value: object,
    *,
    field: str,
) -> Real:
    """Require a finite real-valued quality-test parameter."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field!r} must be a finite real number.")

    if not isfinite(float(value)):
        raise ValueError(f"{field!r} must be a finite real number.")

    return value


def nonnegative_real(
    value: object,
    *,
    field: str,
) -> Real:
    """Require a finite non-negative real-valued parameter."""
    normalized = finite_real(value, field=field)

    if normalized < 0:
        raise ValueError(
            f"{field!r} must be greater than or equal to zero."
        )

    return normalized


def positive_timedelta(
    value: object,
    *,
    field: str,
    grid: TimeGrid,
) -> pd.Timedelta:
    """Normalize a positive duration aligned with the configured grid."""
    delta = normalize_fixed_duration(value, field=field)

    if delta <= pd.Timedelta(0):
        raise ValueError(f"{field!r} must be greater than zero.")

    grid.validate_duration_multiple(delta, field=field)

    return delta


def integer_at_least(
    value: object,
    *,
    field: str,
    minimum: int,
) -> int:
    """Require an integer greater than or equal to a minimum value."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(
            f"{field!r} must be an integer greater than or equal to "
            f"{minimum}."
        )

    normalized = int(value)

    if normalized < minimum:
        raise ValueError(
            f"{field!r} must be greater than or equal to {minimum}."
        )

    return normalized
