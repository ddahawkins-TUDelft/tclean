"""Repeated-pattern data-quality testing."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tclean.data_quality._method import MethodContext, MethodResult, MethodSpec
from tclean.data_quality._validation_helpers import (
    integer_at_least,
    nonnegative_real,
    normalize_common_selectors,
    positive_timedelta,
    validate_keys,
)
from tclean.time_grid import TimeGrid


@dataclass(frozen=True)
class _PatternBlock:
    """One complete candidate pattern block."""

    start_position: int
    end_position: int
    start: pd.Timestamp
    end: pd.Timestamp
    values: np.ndarray


def validate(test: Mapping[str, Any], *, grid: TimeGrid) -> dict[str, Any]:
    """Validate and normalize a repeated-pattern quality test."""
    validate_keys(
        test,
        required={"name", "method", "pattern_duration", "minimum_matches"},
        optional={"sources", "contexts", "tolerance"},
    )

    normalized = normalize_common_selectors(test)

    pattern_duration = positive_timedelta(
        test["pattern_duration"], field="pattern_duration", grid=grid
    )

    if pattern_duration < 2 * grid.frequency:
        raise ValueError(
            "'pattern_duration' for a repeated-pattern test must span "
            "at least two grid steps."
        )

    normalized["pattern_duration"] = pattern_duration

    normalized["minimum_matches"] = integer_at_least(
        test["minimum_matches"], field="minimum_matches", minimum=2
    )

    normalized["tolerance"] = nonnegative_real(
        test.get("tolerance", 0.0), field="tolerance"
    )

    return normalized


def _complete_blocks(
    data: pd.Series, *, pattern_steps: int, grid: TimeGrid
) -> list[_PatternBlock]:
    """Build complete non-overlapping pattern blocks aligned to the grid."""
    blocks: list[_PatternBlock] = []

    latest_start = len(data) - pattern_steps

    for start_position in range(0, latest_start + 1, pattern_steps):
        end_position = start_position + pattern_steps

        period = data.iloc[start_position:end_position]

        if period.isna().any():
            continue

        if end_position < len(data):
            end = pd.Timestamp(data.index[end_position])
        else:
            end = pd.Timestamp(grid.end)

        blocks.append(
            _PatternBlock(
                start_position=start_position,
                end_position=end_position,
                start=pd.Timestamp(data.index[start_position]),
                end=end,
                values=period.to_numpy(dtype=float),
            )
        )

    return blocks


def _within_threshold(values: np.ndarray, *, threshold: float) -> np.ndarray:
    """Compare values inclusively while ignoring binary float noise."""
    return (values <= threshold) | np.isclose(
        values, threshold, rtol=1e-12, atol=0.0, equal_nan=False
    )


def _exact_matches(blocks: list[_PatternBlock]) -> dict[int, list[tuple[int, float]]]:
    """Find exact block matches efficiently by value signature."""
    matches: dict[int, list[tuple[int, float]]] = {
        index: [] for index in range(len(blocks))
    }

    groups: dict[tuple[float, ...], list[int]] = {}

    for index, block in enumerate(blocks):
        signature = tuple(float(value) for value in block.values)

        groups.setdefault(signature, []).append(index)

    for indices in groups.values():
        if len(indices) < 2:
            continue

        for index in indices:
            matches[index] = [
                (other_index, 0.0) for other_index in indices if other_index != index
            ]

    return matches


def _approximate_matches(
    blocks: list[_PatternBlock], *, tolerance: float
) -> dict[int, list[tuple[int, float]]]:
    """Find directly matching block pairs within an absolute tolerance."""
    matches: dict[int, list[tuple[int, float]]] = {
        index: [] for index in range(len(blocks))
    }

    if len(blocks) < 2:
        return matches

    values = np.stack([block.values for block in blocks], axis=0)

    first_values = values[:, 0]

    for index in range(len(blocks) - 1):
        first_differences = np.abs(first_values[index + 1 :] - first_values[index])

        possible = _within_threshold(first_differences, threshold=tolerance)

        candidate_offsets = np.flatnonzero(possible)

        if candidate_offsets.size == 0:
            continue

        candidate_indices = candidate_offsets + index + 1

        maximum_differences = np.max(
            np.abs(values[candidate_indices] - values[index]), axis=1
        )

        qualifying = _within_threshold(maximum_differences, threshold=tolerance)

        for candidate_index, maximum_difference in zip(
            candidate_indices[qualifying], maximum_differences[qualifying], strict=True
        ):
            other_index = int(candidate_index)
            difference = float(maximum_difference)

            matches[index].append((other_index, difference))
            matches[other_index].append((index, difference))

    return matches


def _analyse(
    data: pd.Series, *, test: Mapping[str, Any], grid: TimeGrid
) -> tuple[list[_PatternBlock], dict[int, list[tuple[int, float]]]]:
    """Build candidate blocks and find their direct matches."""
    pattern_steps = int(test["pattern_duration"] / grid.frequency)

    blocks = _complete_blocks(data, pattern_steps=pattern_steps, grid=grid)

    tolerance = float(test["tolerance"])

    if tolerance == 0.0:
        matches = _exact_matches(blocks)
    else:
        matches = _approximate_matches(blocks, tolerance=tolerance)

    return blocks, matches


def _qualifying_indices(
    matches: Mapping[int, list[tuple[int, float]]], *, minimum_matches: int
) -> list[int]:
    """Return blocks with enough direct matches to constitute a failure."""
    return [
        index
        for index, block_matches in matches.items()
        if len(block_matches) + 1 >= minimum_matches
    ]


def evaluate(context: MethodContext) -> MethodResult:
    """Flag complete blocks that reproduce patterns found elsewhere.

    Candidate blocks are non-overlapping and grid-aligned. A block fails when
    it directly matches enough other complete blocks to reach
    ``minimum_matches``, counting itself.
    """
    data = context.target_data
    test = context.test
    grid = context.grid

    failures = pd.DataFrame(False, index=data.index, columns=data.columns, dtype=bool)

    for context_position, context in enumerate(data.columns):
        blocks, matches = _analyse(data[context], test=test, grid=grid)

        qualifying = _qualifying_indices(
            matches, minimum_matches=test["minimum_matches"]
        )

        for block_index in qualifying:
            block = blocks[block_index]

            failures.iloc[
                block.start_position : block.end_position, context_position
            ] = True

    return MethodResult(mask=failures)


def build_details(
    context: MethodContext,
    result: MethodResult,
    *,
    context_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Build structured diagnostics for repeated-pattern failures."""
    del result

    data = context.target_data[context_name]
    test = context.test
    grid = context.grid

    blocks, matches = _analyse(data, test=test, grid=grid)

    qualifying = set(
        _qualifying_indices(matches, minimum_matches=test["minimum_matches"])
    )

    failed_blocks = []

    for block_index in sorted(qualifying):
        block = blocks[block_index]

        if block.start < start or block.end > end:
            continue

        block_matches = []

        for other_index, maximum_difference in sorted(
            matches[block_index], key=lambda item: blocks[item[0]].start
        ):
            other = blocks[other_index]

            block_matches.append(
                {
                    "start": other.start,
                    "end": other.end,
                    "maximum_absolute_difference": (maximum_difference),
                }
            )

        failed_blocks.append(
            {
                "start": block.start,
                "end": block.end,
                "match_count": len(block_matches) + 1,
                "matches": block_matches,
            }
        )

    return {
        "pattern_duration": test["pattern_duration"],
        "minimum_matches": test["minimum_matches"],
        "tolerance": test["tolerance"],
        "matched_block_count": len(failed_blocks),
        "matched_blocks": failed_blocks,
    }


METHOD = MethodSpec(
    name="repeated_pattern",
    validate=validate,
    evaluate=evaluate,
    build_details=build_details,
)
