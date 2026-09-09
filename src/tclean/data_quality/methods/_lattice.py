"""Helpers for constructing contextual reference lattices."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from tclean.data_quality._validation_helpers import (
    integer_at_least,
    positive_timedelta,
    validate_keys,
)
from tclean.time_grid import TimeGrid

_CALENDAR_PERIOD = re.compile(r"^(?P<count>[1-9]\d*)(?P<unit>mo|y)$")


@dataclass(frozen=True)
class ReferencePeriod:
    """One fixed-duration or calendar-aware reference period."""

    kind: Literal["fixed", "months", "years"]
    value: pd.Timedelta | int


@dataclass(frozen=True)
class ReferenceOrder:
    """One level in a contextual reference lattice."""

    period: ReferencePeriod
    radius: int


@dataclass(frozen=True)
class ReferenceLattice:
    """Precomputed reference positions for an ordered set of targets."""

    targets: pd.DatetimeIndex
    offsets: np.ndarray
    positions: np.ndarray

    def positions_for(self, target_number: int) -> np.ndarray:
        """Return reference-index positions for one target."""
        start = int(self.offsets[target_number])
        end = int(self.offsets[target_number + 1])

        return self.positions[start:end]


def _normalize_reference_period(
    value: Any, *, field: str, grid: TimeGrid
) -> ReferencePeriod:
    """Normalize one fixed or calendar-aware reference period."""
    if isinstance(value, str):
        stripped = value.strip().lower()
        match = _CALENDAR_PERIOD.fullmatch(stripped)

        if match is not None:
            count = int(match.group("count"))
            unit = match.group("unit")

            if unit == "y":
                return ReferencePeriod(kind="years", value=count)

            return ReferencePeriod(kind="months", value=count)

    duration = positive_timedelta(value, field=field, grid=grid)

    return ReferencePeriod(kind="fixed", value=duration)


def normalize_reference_orders(value: Any, *, grid: TimeGrid) -> list[ReferenceOrder]:
    """Validate and normalize ordered contextual reference definitions."""
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) == 0
    ):
        raise ValueError("'reference_orders' must be a non-empty ordered sequence.")

    normalized: list[ReferenceOrder] = []
    seen: set[tuple[ReferencePeriod, int]] = set()

    for position, order in enumerate(value):
        if not isinstance(order, Mapping):
            raise ValueError("Each entry in 'reference_orders' must be a mapping.")

        validate_keys(order, required={"period", "radius"}, optional=set())

        period = _normalize_reference_period(
            order["period"], field=(f"reference_orders[{position}].period"), grid=grid
        )

        radius = integer_at_least(
            order["radius"], field=(f"reference_orders[{position}].radius"), minimum=1
        )

        signature = (period, radius)

        if signature in seen:
            raise ValueError(
                "'reference_orders' must not contain duplicate order definitions."
            )

        seen.add(signature)

        normalized.append(ReferenceOrder(period=period, radius=radius))

    return normalized


def _shift_months(timestamp: pd.Timestamp, *, months: int) -> pd.Timestamp | None:
    """Shift by calendar months without coercing invalid dates."""
    total_month = timestamp.year * 12 + timestamp.month - 1 + months

    year, zero_based_month = divmod(total_month, 12)

    try:
        return timestamp.replace(year=year, month=zero_based_month + 1)
    except (ValueError, pd.errors.OutOfBoundsDatetime):
        return None


def _shift_years(timestamp: pd.Timestamp, *, years: int) -> pd.Timestamp | None:
    """Shift by calendar years without coercing invalid dates."""
    try:
        return timestamp.replace(year=timestamp.year + years)
    except (ValueError, pd.errors.OutOfBoundsDatetime):
        return None


def _shift_timestamp(
    timestamp: pd.Timestamp, *, period: ReferencePeriod, coefficient: int
) -> pd.Timestamp | None:
    """Shift a timestamp by one reference-period coefficient."""
    if coefficient == 0:
        return timestamp

    if period.kind == "years":
        return _shift_years(timestamp, years=int(period.value) * coefficient)

    if period.kind == "months":
        return _shift_months(timestamp, months=int(period.value) * coefficient)

    try:
        return timestamp + period.value * coefficient
    except (ValueError, pd.errors.OutOfBoundsDatetime):
        return None


def reference_timestamps(
    target: pd.Timestamp,
    *,
    orders: Sequence[ReferenceOrder],
    available_index: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """Build unique available reference timestamps around one target.

    Orders are configured from lower to higher temporal scale, but the
    lattice is constructed from higher to lower order. Invalid calendar
    branches are discarded before lower-order expansion.
    """
    target = pd.Timestamp(target)

    centres = {target}

    for order in reversed(orders):
        expanded: set[pd.Timestamp] = set()

        for centre in centres:
            for coefficient in range(-order.radius, order.radius + 1):
                shifted = _shift_timestamp(
                    centre, period=order.period, coefficient=coefficient
                )

                if shifted is not None:
                    expanded.add(shifted)

        centres = expanded

    centres.discard(target)

    available = [timestamp for timestamp in centres if timestamp in available_index]

    return pd.DatetimeIndex(sorted(available))


def build_reference_lattice(
    targets: pd.DatetimeIndex,
    *,
    orders: Sequence[ReferenceOrder],
    available_index: pd.DatetimeIndex,
) -> ReferenceLattice:
    """Precompute available reference positions for multiple targets."""
    targets = pd.DatetimeIndex(targets)
    available_index = pd.DatetimeIndex(available_index)

    offsets = np.zeros(len(targets) + 1, dtype=np.int64)
    position_chunks: list[np.ndarray] = []

    for target_number, target in enumerate(targets):
        timestamps = reference_timestamps(
            target, orders=orders, available_index=available_index
        )

        positions = available_index.get_indexer(timestamps)

        if np.any(positions < 0):
            raise RuntimeError(
                "Reference lattice contains timestamps outside the available index."
            )

        positions = positions.astype(np.intp, copy=False)
        position_chunks.append(positions)

        offsets[target_number + 1] = offsets[target_number] + len(positions)

    if position_chunks:
        positions = np.concatenate(position_chunks)
    else:
        positions = np.empty(0, dtype=np.intp)

    offsets.setflags(write=False)
    positions.setflags(write=False)

    return ReferenceLattice(targets=targets, offsets=offsets, positions=positions)
