"""Tests for repeated-pattern data-quality evaluation."""

import pandas as pd
import pytest

from tclean import TimeGrid
from tclean.data_quality.methods.repeated_pattern import (
    build_details,
    evaluate,
)


def _grid() -> TimeGrid:
    """Return a twelve-hour test grid."""
    return TimeGrid(
        start="2026-01-01T00:00:00Z",
        end="2026-01-01T12:00:00Z",
        frequency="1h",
    )


def _data(values: list[float | None]) -> pd.DataFrame:
    """Build single-context test data."""
    return pd.DataFrame(
        {"A": values},
        index=pd.DatetimeIndex(_grid().target_index),
    )


def _test(
    *,
    minimum_matches: int = 3,
    tolerance: float = 0.0,
) -> dict:
    """Build a validated-style repeated-pattern test."""
    return {
        "name": "repeated_pattern",
        "method": "repeated_pattern",
        "pattern_duration": pd.Timedelta("2h"),
        "minimum_matches": minimum_matches,
        "tolerance": tolerance,
    }


def test_evaluate_flags_all_nonadjacent_matching_blocks():
    """Flag every occurrence without designating an original block."""
    data = _data(
        [
            1, 2,
            9, 8,
            1, 2,
            7, 6,
            1, 2,
            5, 4,
        ]
    )

    result = evaluate(
        data,
        test=_test(minimum_matches=3),
        grid=_grid(),
    )

    assert result["A"].tolist() == [
        True, True,
        False, False,
        True, True,
        False, False,
        True, True,
        False, False,
    ]


def test_evaluate_flags_adjacent_matching_blocks():
    """Flag adjacent occurrences of the same repeated pattern."""
    data = _data(
        [
            1, 2,
            1, 2,
            1, 2,
            9, 8,
            7, 6,
            5, 4,
        ]
    )

    result = evaluate(
        data,
        test=_test(minimum_matches=3),
        grid=_grid(),
    )

    assert result["A"].tolist() == [
        True, True,
        True, True,
        True, True,
        False, False,
        False, False,
        False, False,
    ]


def test_evaluate_does_not_flag_too_few_matches():
    """Do not flag patterns occurring fewer than minimum_matches times."""
    data = _data(
        [
            1, 2,
            9, 8,
            1, 2,
            7, 6,
            5, 4,
            3, 2,
        ]
    )

    result = evaluate(
        data,
        test=_test(minimum_matches=3),
        grid=_grid(),
    )

    assert not result["A"].any()


def test_evaluate_distinguishes_different_pattern_shapes():
    """Do not match blocks merely because they have similar ranges."""
    data = _data(
        [
            1, 3,
            3, 1,
            1, 3,
            3, 1,
            5, 6,
            7, 8,
        ]
    )

    result = evaluate(
        data,
        test=_test(minimum_matches=3),
        grid=_grid(),
    )

    assert not result["A"].any()


def test_evaluate_uses_pointwise_tolerance():
    """Match complete blocks whose corresponding values are within tolerance."""
    data = _data(
        [
            1.00, 2.00,
            9.00, 8.00,
            1.05, 2.05,
            7.00, 6.00,
            0.98, 1.95,
            5.00, 4.00,
        ]
    )

    result = evaluate(
        data,
        test=_test(
            minimum_matches=3,
            tolerance=0.1,
        ),
        grid=_grid(),
    )

    assert result["A"].tolist() == [
        True, True,
        False, False,
        True, True,
        False, False,
        True, True,
        False, False,
    ]


def test_evaluate_treats_tolerance_as_inclusive():
    """Treat a maximum difference equal to tolerance as a match."""
    data = _data(
        [
            1.0, 2.0,
            9.0, 8.0,
            1.1, 2.1,
            7.0, 6.0,
            5.0, 4.0,
            3.0, 2.0,
        ]
    )

    result = evaluate(
        data,
        test=_test(
            minimum_matches=2,
            tolerance=0.1,
        ),
        grid=_grid(),
    )

    assert result["A"].tolist() == [
        True, True,
        False, False,
        True, True,
        False, False,
        False, False,
        False, False,
    ]


def test_evaluate_excludes_incomplete_blocks():
    """Treat blocks containing missing values as ineligible."""
    data = _data(
        [
            1, 2,
            1, None,
            9, 8,
            1, 2,
            7, 6,
            5, 4,
        ]
    )

    result = evaluate(
        data,
        test=_test(minimum_matches=2),
        grid=_grid(),
    )

    assert result["A"].tolist() == [
        True, True,
        False, False,
        False, False,
        True, True,
        False, False,
        False, False,
    ]


def test_evaluate_applies_independently_to_contexts():
    """Evaluate repeated patterns independently for each context."""
    data = pd.DataFrame(
        {
            "A": [
                1, 2,
                9, 8,
                1, 2,
                7, 6,
                1, 2,
                5, 4,
            ],
            "B": [
                9, 9,
                3, 4,
                8, 8,
                3, 4,
                7, 7,
                3, 4,
            ],
        },
        index=pd.DatetimeIndex(_grid().target_index),
    )

    result = evaluate(
        data,
        test=_test(minimum_matches=3),
        grid=_grid(),
    )

    assert result["A"].tolist() == [
        True, True,
        False, False,
        True, True,
        False, False,
        True, True,
        False, False,
    ]

    assert result["B"].tolist() == [
        False, False,
        True, True,
        False, False,
        True, True,
        False, False,
        True, True,
    ]


def test_build_details_points_each_block_to_its_matches():
    """Describe every failed block and the other periods it matches."""
    data = _data(
        [
            1, 2,
            9, 8,
            1, 2,
            7, 6,
            1, 2,
            5, 4,
        ]
    )

    details = build_details(
        data["A"],
        start=pd.Timestamp("2026-01-01T00:00:00Z"),
        end=pd.Timestamp("2026-01-01T02:00:00Z"),
        test=_test(minimum_matches=3),
        grid=_grid(),
    )

    assert details["pattern_duration"] == pd.Timedelta("2h")
    assert details["minimum_matches"] == 3
    assert details["tolerance"] == 0.0
    assert details["matched_block_count"] == 1

    block = details["matched_blocks"][0]

    assert block["start"] == pd.Timestamp(
        "2026-01-01T00:00:00Z"
    )
    assert block["end"] == pd.Timestamp(
        "2026-01-01T02:00:00Z"
    )
    assert block["match_count"] == 3

    assert [
        match["start"]
        for match in block["matches"]
    ] == [
        pd.Timestamp("2026-01-01T04:00:00Z"),
        pd.Timestamp("2026-01-01T08:00:00Z"),
    ]

    assert all(
        match["maximum_absolute_difference"]
        == pytest.approx(0.0)
        for match in block["matches"]
    )


def test_build_details_preserves_adjacent_constituent_blocks():
    """Preserve individual duplicate blocks when failure periods are merged."""
    data = _data(
        [
            1, 2,
            1, 2,
            1, 2,
            9, 8,
            7, 6,
            5, 4,
        ]
    )

    details = build_details(
        data["A"],
        start=pd.Timestamp("2026-01-01T00:00:00Z"),
        end=pd.Timestamp("2026-01-01T06:00:00Z"),
        test=_test(minimum_matches=3),
        grid=_grid(),
    )

    assert details["matched_block_count"] == 3

    assert [
        block["start"]
        for block in details["matched_blocks"]
    ] == [
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T02:00:00Z"),
        pd.Timestamp("2026-01-01T04:00:00Z"),
    ]
