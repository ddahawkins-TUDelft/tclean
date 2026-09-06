"""Tests for high-level data-quality evaluation."""

import pandas as pd
import pytest

from tclean import TimeGrid
from tclean.data_quality import QualityEvaluation, evaluate


def _grid() -> TimeGrid:
    """Return a six-hour test grid."""
    return TimeGrid(
        start="2026-01-01T00:00:00Z",
        end="2026-01-01T06:00:00Z",
        frequency="1h",
    )


def _source(
    values: dict[str, list[float | None]],
) -> pd.DataFrame:
    """Build canonical source data on the test grid."""
    return pd.DataFrame(
        values,
        index=pd.DatetimeIndex(
            _grid().target_index,
            name="timestamp",
        ),
    )


def test_evaluate_returns_quality_evaluation():
    """Return the canonical quality-evaluation result object."""
    sources = {
        "primary": _source(
            {
                "A": [1, 2, 3, 4, 5, 6],
            }
        )
    }

    result = evaluate(
        sources,
        tests=[
            {
                "name": "non_negative",
                "method": "range",
                "minimum": 0,
            }
        ],
        grid=_grid(),
    )

    assert isinstance(result, QualityEvaluation)
    assert result.failures.empty
    assert result.issues.empty


def test_evaluate_reports_range_failure_periods():
    """Report contiguous failed values as canonical periods."""
    sources = {
        "primary": _source(
            {
                "A": [1, -2, -3, 4, -5, 6],
            }
        )
    }

    result = evaluate(
        sources,
        tests=[
            {
                "name": "non_negative",
                "method": "range",
                "minimum": 0,
            }
        ],
        grid=_grid(),
    )

    assert len(result.failures) == 2

    first = result.failures.iloc[0]
    second = result.failures.iloc[1]

    assert first["context"] == "A"
    assert first["source"] == "primary"
    assert first["test_name"] == "non_negative"
    assert first["method"] == "range"
    assert first["start"] == pd.Timestamp(
        "2026-01-01T01:00:00Z"
    )
    assert first["end"] == pd.Timestamp(
        "2026-01-01T03:00:00Z"
    )
    assert first["details"]["observed_minimum"] == -3.0
    assert first["details"]["maximum_below_minimum"] == 3.0

    assert second["start"] == pd.Timestamp(
        "2026-01-01T04:00:00Z"
    )
    assert second["end"] == pd.Timestamp(
        "2026-01-01T05:00:00Z"
    )
    assert second["details"]["observed_minimum"] == -5.0
    assert second["details"]["maximum_below_minimum"] == 5.0


def test_evaluate_applies_test_to_all_sources():
    """Evaluate ordinary quality tests independently by source."""
    sources = {
        "primary": _source(
            {
                "A": [-1, 2, 3, 4, 5, 6],
            }
        ),
        "secondary": _source(
            {
                "A": [1, 2, 3, -4, 5, 6],
            }
        ),
    }

    result = evaluate(
        sources,
        tests=[
            {
                "name": "non_negative",
                "method": "range",
                "minimum": 0,
            }
        ],
        grid=_grid(),
    )

    assert result.failures["source"].tolist() == [
        "primary",
        "secondary",
    ]

    assert result.failures["start"].tolist() == [
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T03:00:00Z"),
    ]


def test_evaluate_respects_source_selector():
    """Evaluate only explicitly selected sources."""
    sources = {
        "primary": _source(
            {
                "A": [-1, 2, 3, 4, 5, 6],
            }
        ),
        "secondary": _source(
            {
                "A": [1, 2, 3, -4, 5, 6],
            }
        ),
    }

    result = evaluate(
        sources,
        tests=[
            {
                "name": "secondary_only",
                "method": "range",
                "sources": ["secondary"],
                "minimum": 0,
            }
        ],
        grid=_grid(),
    )

    assert len(result.failures) == 1
    assert result.failures.iloc[0]["source"] == "secondary"
    assert result.failures.iloc[0]["start"] == pd.Timestamp(
        "2026-01-01T03:00:00Z"
    )


def test_evaluate_respects_context_selector():
    """Evaluate only explicitly selected contexts."""
    sources = {
        "primary": _source(
            {
                "A": [-1, 2, 3, 4, 5, 6],
                "B": [1, -2, 3, 4, 5, 6],
            }
        )
    }

    result = evaluate(
        sources,
        tests=[
            {
                "name": "context_b_only",
                "method": "range",
                "contexts": ["B"],
                "minimum": 0,
            }
        ],
        grid=_grid(),
    )

    assert len(result.failures) == 1
    assert result.failures.iloc[0]["context"] == "B"
    assert result.failures.iloc[0]["start"] == pd.Timestamp(
        "2026-01-01T01:00:00Z"
    )


def test_evaluate_allows_contexts_to_differ_between_sources():
    """Apply a requested context wherever that context is available."""
    sources = {
        "primary": _source(
            {
                "A": [-1, 2, 3, 4, 5, 6],
                "B": [1, 2, 3, 4, 5, 6],
            }
        ),
        "secondary": _source(
            {
                "B": [1, -2, 3, 4, 5, 6],
            }
        ),
    }

    result = evaluate(
        sources,
        tests=[
            {
                "name": "context_a",
                "method": "range",
                "contexts": ["A"],
                "minimum": 0,
            }
        ],
        grid=_grid(),
    )

    assert len(result.failures) == 1
    assert result.failures.iloc[0]["source"] == "primary"
    assert result.failures.iloc[0]["context"] == "A"


def test_evaluate_rejects_unknown_source_selector():
    """Reject test selectors naming sources that were not supplied."""
    sources = {
        "primary": _source(
            {
                "A": [1, 2, 3, 4, 5, 6],
            }
        )
    }

    with pytest.raises(
        ValueError,
        match="references unknown sources",
    ):
        evaluate(
            sources,
            tests=[
                {
                    "name": "unknown_source",
                    "method": "range",
                    "sources": ["missing_source"],
                    "minimum": 0,
                }
            ],
            grid=_grid(),
        )


def test_evaluate_rejects_context_unavailable_in_selected_sources():
    """Reject requested contexts absent from every selected source."""
    sources = {
        "primary": _source(
            {
                "A": [1, 2, 3, 4, 5, 6],
            }
        ),
        "secondary": _source(
            {
                "B": [1, 2, 3, 4, 5, 6],
            }
        ),
    }

    with pytest.raises(
        ValueError,
        match="unavailable in all selected sources",
    ):
        evaluate(
            sources,
            tests=[
                {
                    "name": "missing_context",
                    "method": "range",
                    "sources": ["primary"],
                    "contexts": ["B"],
                    "minimum": 0,
                }
            ],
            grid=_grid(),
        )


def test_evaluate_preserves_test_then_source_then_context_order():
    """Return failures in deterministic evaluation order."""
    sources = {
        "primary": _source(
            {
                "A": [-1, 2, 3, 4, 5, 6],
                "B": [1, -2, 3, 4, 5, 6],
            }
        ),
        "secondary": _source(
            {
                "A": [1, 2, -3, 4, 5, 6],
                "B": [1, 2, 3, -4, 5, 6],
            }
        ),
    }

    result = evaluate(
        sources,
        tests=[
            {
                "name": "negative",
                "method": "range",
                "minimum": 0,
            },
            {
                "name": "under_two",
                "method": "range",
                "minimum": 2,
            },
        ],
        grid=_grid(),
    )

    observed_order = list(
        result.failures[
            [
                "test_name",
                "source",
                "context",
            ]
        ].itertuples(index=False, name=None)
    )

    assert observed_order[:4] == [
        ("negative", "primary", "A"),
        ("negative", "primary", "B"),
        ("negative", "secondary", "A"),
        ("negative", "secondary", "B"),
    ]

    assert all(
        test_name == "under_two"
        for test_name, _, _ in observed_order[4:]
    )


def test_evaluate_accepts_empty_test_plan():
    """Return empty canonical results when no tests are configured."""
    sources = {
        "primary": _source(
            {
                "A": [1, 2, 3, 4, 5, 6],
            }
        )
    }

    result = evaluate(
        sources,
        tests=[],
        grid=_grid(),
    )

    assert result.failures.empty
    assert result.issues.empty

    assert result.failures.columns.tolist() == [
        "context",
        "source",
        "start",
        "end",
        "test_name",
        "method",
        "details",
    ]

    assert result.issues.columns.tolist() == [
        "context",
        "source",
        "start",
        "end",
        "test_name",
        "method",
        "severity",
        "code",
        "details",
    ]


def test_evaluate_rejects_empty_source_mapping():
    """Require at least one source for quality evaluation."""
    with pytest.raises(
        ValueError,
        match="At least one time-series source",
    ):
        evaluate(
            {},
            tests=[],
            grid=_grid(),
        )


def test_evaluate_rejects_blank_source_name():
    """Require source identifiers to be non-empty strings."""
    sources = {
        " ": _source(
            {
                "A": [1, 2, 3, 4, 5, 6],
            }
        )
    }

    with pytest.raises(
        ValueError,
        match="source names must be non-empty strings",
    ):
        evaluate(
            sources,
            tests=[],
            grid=_grid(),
        )


def test_evaluate_reports_value_run_failure():
    """Evaluate value-run tests through the public quality API."""
    sources = {
        "primary": _source(
            {
                "A": [5, 0, 0, 0, 5, 6],
            }
        )
    }

    result = evaluate(
        sources,
        tests=[
            {
                "name": "zero_run",
                "method": "value_run",
                "value": 0,
                "minimum_duration": "3h",
            }
        ],
        grid=_grid(),
    )

    assert len(result.failures) == 1

    failure = result.failures.iloc[0]

    assert failure["source"] == "primary"
    assert failure["context"] == "A"
    assert failure["test_name"] == "zero_run"
    assert failure["method"] == "value_run"
    assert failure["start"] == pd.Timestamp(
        "2026-01-01T01:00:00Z"
    )
    assert failure["end"] == pd.Timestamp(
        "2026-01-01T04:00:00Z"
    )
    assert failure["details"]["duration"] == pd.Timedelta("3h")


def test_evaluate_reports_flatline_failure():
    """Evaluate flatline tests through the public quality API."""
    sources = {
        "primary": _source(
            {
                "A": [1, 5, 5, 5, 2, 3],
            }
        )
    }

    result = evaluate(
        sources,
        tests=[
            {
                "name": "flat_values",
                "method": "flatline",
                "minimum_duration": "3h",
            }
        ],
        grid=_grid(),
    )

    assert len(result.failures) == 1

    failure = result.failures.iloc[0]

    assert failure["method"] == "flatline"
    assert failure["start"] == pd.Timestamp(
        "2026-01-01T01:00:00Z"
    )
    assert failure["end"] == pd.Timestamp(
        "2026-01-01T04:00:00Z"
    )
    assert failure["details"]["observed_range"] == 0.0


def test_evaluate_reports_low_variability_failure():
    """Evaluate low-variability tests through the public quality API."""
    sources = {
        "primary": _source(
            {
                "A": [10, 1.0, 1.1, 1.2, 10, 20],
            }
        )
    }

    result = evaluate(
        sources,
        tests=[
            {
                "name": "stable_values",
                "method": "low_variability",
                "window_duration": "3h",
                "maximum_range": 0.2,
            }
        ],
        grid=_grid(),
    )

    assert len(result.failures) == 1

    failure = result.failures.iloc[0]

    assert failure["method"] == "low_variability"
    assert failure["start"] == pd.Timestamp(
        "2026-01-01T01:00:00Z"
    )
    assert failure["end"] == pd.Timestamp(
        "2026-01-01T04:00:00Z"
    )

    assert failure["details"]["qualifying_window_count"] == 1
    assert failure["details"]["maximum_window_range"] == pytest.approx(
        0.2
    )
