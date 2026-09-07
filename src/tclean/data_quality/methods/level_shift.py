"""Level-shift data-quality testing."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from tclean.data_quality._method import MethodContext, MethodResult, MethodSpec
from tclean.data_quality._validation_helpers import (
    nonnegative_real,
    normalize_common_selectors,
    positive_timedelta,
    validate_keys,
)
from tclean.time_grid import TimeGrid


def validate(test: Mapping[str, Any], *, grid: TimeGrid) -> dict[str, Any]:
    """Validate and normalize a level-shift quality test."""
    validate_keys(
        test,
        required={"name", "method", "window_duration", "threshold"},
        optional={"sources", "contexts"},
    )

    normalized = normalize_common_selectors(test)

    window_duration = positive_timedelta(
        test["window_duration"], field="window_duration", grid=grid
    )

    if window_duration < 2 * grid.frequency:
        raise ValueError(
            "'window_duration' for a level-shift test must span "
            "at least two grid steps."
        )

    normalized["window_duration"] = window_duration

    threshold = nonnegative_real(test["threshold"], field="threshold")

    if threshold == 0:
        raise ValueError(
            "'threshold' for a level-shift test must be greater than zero."
        )

    normalized["threshold"] = threshold

    return normalized


def _shift_scores(
    values: pd.Series, *, window_steps: int
) -> tuple[pd.Series, pd.Series]:
    """Return signed median paired shifts and complete-boundary status."""
    paired_difference = values - values.shift(window_steps)

    rolling = paired_difference.rolling(window=window_steps, min_periods=window_steps)

    offset = -(window_steps - 1)

    shift_scores = rolling.median().shift(offset)
    complete = rolling.count().eq(window_steps).shift(offset, fill_value=False)

    return shift_scores, complete.astype(bool)


def _qualifying_boundaries(
    shift_scores: pd.Series, complete: pd.Series, *, threshold: float
) -> pd.Series:
    """Return complete boundaries whose shift magnitude exceeds threshold."""
    return (complete & shift_scores.abs().gt(threshold)).astype(bool)


def _event_bounds(qualifying: pd.Series) -> list[tuple[int, int]]:
    """Return inclusive positional bounds of contiguous qualifying runs."""
    positions = np.flatnonzero(qualifying.to_numpy(dtype=bool))

    if len(positions) == 0:
        return []

    events: list[tuple[int, int]] = []

    start = int(positions[0])
    previous = start

    for position in positions[1:]:
        position = int(position)

        if position != previous + 1:
            events.append((start, previous))
            start = position

        previous = position

    events.append((start, previous))

    return events


def _change_point_position(
    shift_scores: pd.Series, *, event_start: int, event_end: int
) -> int:
    """Locate the strongest boundary, using the middle of a maximum plateau."""
    event_scores = shift_scores.iloc[event_start : event_end + 1]
    absolute_scores = event_scores.abs().to_numpy(dtype=float)

    maximum = float(np.max(absolute_scores))

    maximum_positions = np.flatnonzero(
        np.isclose(absolute_scores, maximum, rtol=1e-12, atol=0.0)
    )

    middle = int(maximum_positions[len(maximum_positions) // 2])

    return event_start + middle


def _analyse(
    values: pd.Series, *, window_steps: int, threshold: float
) -> tuple[pd.Series, pd.Series, pd.Series, list[tuple[int, int]]]:
    """Calculate shift scores, candidate support, and localized failures."""
    shift_scores, complete = _shift_scores(values, window_steps=window_steps)

    qualifying = _qualifying_boundaries(shift_scores, complete, threshold=threshold)

    events = _event_bounds(qualifying)

    failures = pd.Series(False, index=values.index, dtype=bool)

    for event_start, event_end in events:
        change_point = _change_point_position(
            shift_scores, event_start=event_start, event_end=event_end
        )

        failures.iloc[change_point] = True

    return shift_scores, qualifying, failures, events


def evaluate(context: MethodContext) -> MethodResult:
    """Flag localized boundaries supported by persistent paired level shifts."""
    data = context.target_data
    test = context.test
    grid = context.grid

    window_steps = int(test["window_duration"] / grid.frequency)

    failures = pd.DataFrame(False, index=data.index, columns=data.columns, dtype=bool)

    for context in data.columns:
        _, _, context_failures, _ = _analyse(
            data[context], window_steps=window_steps, threshold=test["threshold"]
        )

        failures[context] = context_failures

    return MethodResult(mask=failures)


def build_details(
    context: MethodContext,
    result: MethodResult,
    *,
    context_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Build structured diagnostics for one localized level-shift event."""
    del result

    data = context.target_data[context_name]
    test = context.test
    grid = context.grid

    del end

    window_steps = int(test["window_duration"] / grid.frequency)

    shift_scores, _, failures, events = _analyse(
        data, window_steps=window_steps, threshold=test["threshold"]
    )

    change_point_position = int(data.index.get_loc(start))

    if not failures.iloc[change_point_position]:
        raise ValueError(
            "Level-shift details requested for a timestamp that is not "
            "a localized level-shift failure."
        )

    event_start = None
    event_end = None

    for candidate_start, candidate_end in events:
        localized = _change_point_position(
            shift_scores, event_start=candidate_start, event_end=candidate_end
        )

        if localized == change_point_position:
            event_start = candidate_start
            event_end = candidate_end
            break

    if event_start is None or event_end is None:
        raise ValueError("Could not reconstruct the qualifying level-shift event.")

    change_point = pd.Timestamp(data.index[change_point_position])

    pre_evidence_start = pd.Timestamp(data.index[event_start]) - test["window_duration"]

    post_evidence_end = pd.Timestamp(data.index[event_end]) + test["window_duration"]

    return {
        "window_duration": test["window_duration"],
        "threshold": test["threshold"],
        "change_point": change_point,
        "estimated_shift": float(shift_scores.iloc[change_point_position]),
        "qualifying_boundary_count": (event_end - event_start + 1),
        "pre_evidence_start": pre_evidence_start,
        "post_evidence_end": post_evidence_end,
    }


METHOD = MethodSpec(
    name="level_shift",
    validate=validate,
    evaluate=evaluate,
    build_details=build_details,
)
