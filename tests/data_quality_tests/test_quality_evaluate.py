"""Tests for high-level data-quality evaluation."""

import logging

import pandas as pd
import pytest

from tclean import TimeGrid
from tclean.data_quality import QualityEvaluation, evaluate


def _grid() -> TimeGrid:
    """Return a six-hour test grid."""
    return TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-01T06:00:00Z", frequency="1h"
    )


def _source(values: dict[str, list[float | None]]) -> pd.DataFrame:
    """Build canonical source data on the test grid."""
    return pd.DataFrame(
        values, index=pd.DatetimeIndex(_grid().target_index, name="timestamp")
    )


def test_evaluate_returns_quality_evaluation():
    """Return the canonical quality-evaluation result object."""
    sources = {"primary": _source({"A": [1, 2, 3, 4, 5, 6]})}

    result = evaluate(
        sources,
        tests=[
            {
                "name": "non_negative",
                "method": "range",
                "minimum": {"value_mode": "fixed", "value": 0},
            }
        ],
        grid=_grid(),
    )

    assert isinstance(result, QualityEvaluation)
    assert result.failures.empty
    assert result.issues.empty


def test_evaluate_reports_range_failure_periods():
    """Report contiguous failed values as canonical periods."""
    sources = {"primary": _source({"A": [1, -2, -3, 4, -5, 6]})}

    result = evaluate(
        sources,
        tests=[
            {
                "name": "non_negative",
                "method": "range",
                "minimum": {"value_mode": "fixed", "value": 0},
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
    assert first["start"] == pd.Timestamp("2026-01-01T01:00:00Z")
    assert first["end"] == pd.Timestamp("2026-01-01T03:00:00Z")
    assert first["details"]["observed_minimum"] == -3.0
    assert first["details"]["maximum_below_minimum"] == 3.0

    assert second["start"] == pd.Timestamp("2026-01-01T04:00:00Z")
    assert second["end"] == pd.Timestamp("2026-01-01T05:00:00Z")
    assert second["details"]["observed_minimum"] == -5.0
    assert second["details"]["maximum_below_minimum"] == 5.0


def test_evaluate_applies_test_to_all_sources():
    """Evaluate ordinary quality tests independently by source."""
    sources = {
        "primary": _source({"A": [-1, 2, 3, 4, 5, 6]}),
        "secondary": _source({"A": [1, 2, 3, -4, 5, 6]}),
    }

    result = evaluate(
        sources,
        tests=[
            {
                "name": "non_negative",
                "method": "range",
                "minimum": {"value_mode": "fixed", "value": 0},
            }
        ],
        grid=_grid(),
    )

    assert result.failures["source"].tolist() == ["primary", "secondary"]

    assert result.failures["start"].tolist() == [
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T03:00:00Z"),
    ]


def test_evaluate_respects_source_selector():
    """Evaluate only explicitly selected sources."""
    sources = {
        "primary": _source({"A": [-1, 2, 3, 4, 5, 6]}),
        "secondary": _source({"A": [1, 2, 3, -4, 5, 6]}),
    }

    result = evaluate(
        sources,
        tests=[
            {
                "name": "secondary_only",
                "method": "range",
                "sources": ["secondary"],
                "minimum": {"value_mode": "fixed", "value": 0},
            }
        ],
        grid=_grid(),
    )

    assert len(result.failures) == 1
    assert result.failures.iloc[0]["source"] == "secondary"
    assert result.failures.iloc[0]["start"] == pd.Timestamp("2026-01-01T03:00:00Z")


def test_evaluate_respects_context_selector():
    """Evaluate only explicitly selected contexts."""
    sources = {"primary": _source({"A": [-1, 2, 3, 4, 5, 6], "B": [1, -2, 3, 4, 5, 6]})}

    result = evaluate(
        sources,
        tests=[
            {
                "name": "context_b_only",
                "method": "range",
                "contexts": ["B"],
                "minimum": {"value_mode": "fixed", "value": 0},
            }
        ],
        grid=_grid(),
    )

    assert len(result.failures) == 1
    assert result.failures.iloc[0]["context"] == "B"
    assert result.failures.iloc[0]["start"] == pd.Timestamp("2026-01-01T01:00:00Z")


def test_evaluate_allows_contexts_to_differ_between_sources():
    """Apply a requested context wherever that context is available."""
    sources = {
        "primary": _source({"A": [-1, 2, 3, 4, 5, 6], "B": [1, 2, 3, 4, 5, 6]}),
        "secondary": _source({"B": [1, -2, 3, 4, 5, 6]}),
    }

    result = evaluate(
        sources,
        tests=[
            {
                "name": "context_a",
                "method": "range",
                "contexts": ["A"],
                "minimum": {"value_mode": "fixed", "value": 0},
            }
        ],
        grid=_grid(),
    )

    assert len(result.failures) == 1
    assert result.failures.iloc[0]["source"] == "primary"
    assert result.failures.iloc[0]["context"] == "A"


def test_evaluate_rejects_unknown_source_selector():
    """Reject test selectors naming sources that were not supplied."""
    sources = {"primary": _source({"A": [1, 2, 3, 4, 5, 6]})}

    with pytest.raises(ValueError, match="references unknown sources"):
        evaluate(
            sources,
            tests=[
                {
                    "name": "unknown_source",
                    "method": "range",
                    "sources": ["missing_source"],
                    "minimum": {"value_mode": "fixed", "value": 0},
                }
            ],
            grid=_grid(),
        )


def test_evaluate_rejects_context_unavailable_in_selected_sources():
    """Reject requested contexts absent from every selected source."""
    sources = {
        "primary": _source({"A": [1, 2, 3, 4, 5, 6]}),
        "secondary": _source({"B": [1, 2, 3, 4, 5, 6]}),
    }

    with pytest.raises(ValueError, match="unavailable in all selected sources"):
        evaluate(
            sources,
            tests=[
                {
                    "name": "missing_context",
                    "method": "range",
                    "sources": ["primary"],
                    "contexts": ["B"],
                    "minimum": {"value_mode": "fixed", "value": 0},
                }
            ],
            grid=_grid(),
        )


def test_evaluate_preserves_test_then_source_then_context_order():
    """Return failures in deterministic evaluation order."""
    sources = {
        "primary": _source({"A": [-1, 2, 3, 4, 5, 6], "B": [1, -2, 3, 4, 5, 6]}),
        "secondary": _source({"A": [1, 2, -3, 4, 5, 6], "B": [1, 2, 3, -4, 5, 6]}),
    }

    result = evaluate(
        sources,
        tests=[
            {
                "name": "negative",
                "method": "range",
                "minimum": {"value_mode": "fixed", "value": 0},
            },
            {
                "name": "under_two",
                "method": "range",
                "minimum": {"value_mode": "fixed", "value": 2},
            },
        ],
        grid=_grid(),
    )

    observed_order = list(
        result.failures[["test_name", "source", "context"]].itertuples(
            index=False, name=None
        )
    )

    assert observed_order[:4] == [
        ("negative", "primary", "A"),
        ("negative", "primary", "B"),
        ("negative", "secondary", "A"),
        ("negative", "secondary", "B"),
    ]

    assert all(test_name == "under_two" for test_name, _, _ in observed_order[4:])


def test_evaluate_accepts_empty_test_plan():
    """Return empty canonical results when no tests are configured."""
    sources = {"primary": _source({"A": [1, 2, 3, 4, 5, 6]})}

    result = evaluate(sources, tests=[], grid=_grid())

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
    with pytest.raises(ValueError, match="At least one time-series source"):
        evaluate({}, tests=[], grid=_grid())


def test_evaluate_rejects_blank_source_name():
    """Require source identifiers to be non-empty strings."""
    sources = {" ": _source({"A": [1, 2, 3, 4, 5, 6]})}

    with pytest.raises(ValueError, match="source names must be non-empty strings"):
        evaluate(sources, tests=[], grid=_grid())


def test_evaluate_reports_value_run_failure():
    """Evaluate value-run tests through the public quality API."""
    sources = {"primary": _source({"A": [5, 0, 0, 0, 5, 6]})}

    result = evaluate(
        sources,
        tests=[
            {
                "name": "zero_run",
                "method": "value_run",
                "value": {"value_mode": "fixed", "value": 0},
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
    assert failure["start"] == pd.Timestamp("2026-01-01T01:00:00Z")
    assert failure["end"] == pd.Timestamp("2026-01-01T04:00:00Z")
    assert failure["details"]["duration"] == pd.Timedelta("3h")


def test_evaluate_reports_flatline_failure():
    """Evaluate flatline tests through the public quality API."""
    sources = {"primary": _source({"A": [1, 5, 5, 5, 2, 3]})}

    result = evaluate(
        sources,
        tests=[{"name": "flat_values", "method": "flatline", "minimum_duration": "3h"}],
        grid=_grid(),
    )

    assert len(result.failures) == 1

    failure = result.failures.iloc[0]

    assert failure["method"] == "flatline"
    assert failure["start"] == pd.Timestamp("2026-01-01T01:00:00Z")
    assert failure["end"] == pd.Timestamp("2026-01-01T04:00:00Z")
    assert failure["details"]["observed_range"] == 0.0


def test_evaluate_reports_low_variability_failure():
    """Evaluate low-variability tests through the public quality API."""
    sources = {"primary": _source({"A": [10, 1.0, 1.1, 1.2, 10, 20]})}

    result = evaluate(
        sources,
        tests=[
            {
                "name": "stable_values",
                "method": "low_variability",
                "window_duration": "3h",
                "maximum_range": {"value_mode": "fixed", "value": 0.2},
            }
        ],
        grid=_grid(),
    )

    assert len(result.failures) == 1

    failure = result.failures.iloc[0]

    assert failure["method"] == "low_variability"
    assert failure["start"] == pd.Timestamp("2026-01-01T01:00:00Z")
    assert failure["end"] == pd.Timestamp("2026-01-01T04:00:00Z")

    assert failure["details"]["qualifying_window_count"] == 1
    assert failure["details"]["maximum_window_range"] == pytest.approx(0.2)


def test_evaluate_reports_repeated_pattern_failures():
    """Evaluate repeated-pattern tests through the public quality API."""
    sources = {"primary": _source({"A": [1, 2, 9, 8, 1, 2]})}

    result = evaluate(
        sources,
        tests=[
            {
                "name": "duplicate_block",
                "method": "repeated_pattern",
                "pattern_duration": "2h",
                "minimum_matches": 2,
            }
        ],
        grid=_grid(),
    )

    assert len(result.failures) == 2

    assert result.failures["start"].tolist() == [
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T04:00:00Z"),
    ]

    assert result.failures["end"].tolist() == [
        pd.Timestamp("2026-01-01T02:00:00Z"),
        pd.Timestamp("2026-01-01T06:00:00Z"),
    ]

    assert all(method == "repeated_pattern" for method in result.failures["method"])


def test_evaluate_reports_fixed_rate_of_change_failure():
    """Evaluate fixed rate-of-change mode through the public quality API."""
    sources = {"primary": _source({"A": [100, 130, 170, 175, 180, 185]})}

    result = evaluate(
        sources,
        tests=[
            {
                "name": "large_change",
                "method": "rate_of_change",
                "difference_mode": "fixed",
                "threshold": {"value_mode": "fixed", "value": 20},
            }
        ],
        grid=_grid(),
    )

    assert len(result.failures) == 1

    failure = result.failures.iloc[0]

    assert failure["source"] == "primary"
    assert failure["context"] == "A"
    assert failure["test_name"] == "large_change"
    assert failure["method"] == "rate_of_change"

    assert failure["start"] == pd.Timestamp("2026-01-01T01:00:00Z")
    assert failure["end"] == pd.Timestamp("2026-01-01T03:00:00Z")

    assert failure["details"]["difference_mode"] == "fixed"
    assert failure["details"]["threshold"] == 20
    assert failure["details"]["transition_count"] == 2

    assert [
        transition["change"] for transition in failure["details"]["transitions"]
    ] == [30.0, 40.0]


def test_evaluate_reports_relative_rate_of_change_failure():
    """Evaluate relative rate-of-change mode through the public quality API."""
    sources = {"primary": _source({"A": [100, 150, 240, 245, 250, 255]})}

    result = evaluate(
        sources,
        tests=[
            {
                "name": "large_relative_change",
                "method": "rate_of_change",
                "difference_mode": "relative",
                "threshold": {"value_mode": "fixed", "value": 0.2},
            }
        ],
        grid=_grid(),
    )

    assert len(result.failures) == 1

    failure = result.failures.iloc[0]

    assert failure["source"] == "primary"
    assert failure["context"] == "A"
    assert failure["test_name"] == "large_relative_change"
    assert failure["method"] == "rate_of_change"

    assert failure["start"] == pd.Timestamp("2026-01-01T01:00:00Z")
    assert failure["end"] == pd.Timestamp("2026-01-01T03:00:00Z")

    assert failure["details"]["difference_mode"] == "relative"
    assert failure["details"]["threshold"] == 0.2
    assert failure["details"]["reference_magnitude_threshold"] == 0.0
    assert failure["details"]["transition_count"] == 2

    assert [
        transition["relative_change"]
        for transition in failure["details"]["transitions"]
    ] == pytest.approx([0.5, 0.6])


def test_evaluate_reports_level_shift_failure():
    """Evaluate level-shift tests through the public quality API."""
    sources = {"primary": _source({"A": [100, 100, 100, 150, 150, 150]})}

    result = evaluate(
        sources,
        tests=[
            {
                "name": "persistent_change",
                "method": "level_shift",
                "window_duration": "2h",
                "threshold": {"value_mode": "fixed", "value": 40},
            }
        ],
        grid=_grid(),
    )

    assert len(result.failures) == 1

    failure = result.failures.iloc[0]

    assert failure["source"] == "primary"
    assert failure["context"] == "A"
    assert failure["test_name"] == "persistent_change"
    assert failure["method"] == "level_shift"

    assert failure["start"] == pd.Timestamp("2026-01-01T03:00:00Z")
    assert failure["end"] == pd.Timestamp("2026-01-01T04:00:00Z")

    assert failure["details"]["window_duration"] == pd.Timedelta("2h")
    assert failure["details"]["threshold"] == 40

    assert failure["details"]["change_point"] == pd.Timestamp("2026-01-01T03:00:00Z")
    assert failure["details"]["estimated_shift"] == 50.0

    assert failure["details"]["qualifying_boundary_count"] == 1

    assert failure["details"]["pre_evidence_start"] == pd.Timestamp(
        "2026-01-01T01:00:00Z"
    )
    assert failure["details"]["post_evidence_end"] == pd.Timestamp(
        "2026-01-01T05:00:00Z"
    )


def _contextual_grid() -> TimeGrid:
    """Return a grid spanning weekly contextual references."""
    return TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-30T00:00:00Z", frequency="1h"
    )


def _contextual_source(values: dict[str, float]) -> pd.DataFrame:
    """Build sparse contextual source data."""
    grid = _contextual_grid()

    series = pd.Series(float("nan"), index=grid.target_index, dtype=float)

    for timestamp, value in values.items():
        series.loc[pd.Timestamp(timestamp)] = value

    return series.to_frame("A")


def test_evaluate_reports_contextual_level_failure():
    """Evaluate contextual level through the public evaluator."""
    grid = _contextual_grid()

    sources = {
        "primary": _contextual_source(
            {
                "2026-01-08T12:00:00Z": 100,
                "2026-01-15T12:00:00Z": 100,
                "2026-01-22T12:00:00Z": 120,
                "2026-01-29T12:00:00Z": 100,
            }
        )
    }

    result = evaluate(
        sources,
        tests=[
            {
                "name": "unusual_level",
                "method": "contextual_level",
                "reference_orders": [{"period": "7D", "radius": 2}],
                "robust_deviation_threshold": 6,
            }
        ],
        grid=grid,
    )

    target = pd.Timestamp("2026-01-22T12:00:00Z")

    failures = result.failures.loc[result.failures["start"] == target]

    assert len(failures) == 1

    failure = failures.iloc[0]

    assert failure["source"] == "primary"
    assert failure["context"] == "A"
    assert failure["test_name"] == "unusual_level"
    assert failure["method"] == "contextual_level"

    assert failure["details"]["failed_criteria"] == ["robust_deviation"]


def test_evaluate_reports_contextual_level_not_evaluable_issue():
    """Report contextual evaluation inability through the public evaluator."""
    grid = _contextual_grid()

    sources = {
        "primary": _contextual_source(
            {"2026-01-15T12:00:00Z": 100, "2026-01-22T12:00:00Z": 110}
        )
    }

    result = evaluate(
        sources,
        tests=[
            {
                "name": "unusual_level",
                "method": "contextual_level",
                "reference_orders": [{"period": "7D", "radius": 1}],
                "maximum_predictive_probability": 0.05,
            }
        ],
        grid=grid,
    )

    target = pd.Timestamp("2026-01-22T12:00:00Z")

    issues = result.issues.loc[result.issues["start"] == target]

    assert len(issues) == 1

    issue = issues.iloc[0]

    assert issue["context"] == "A"
    assert issue["source"] == "primary"
    assert issue["test_name"] == "unusual_level"
    assert issue["method"] == "contextual_level"
    assert issue["severity"] == "not_evaluable"
    assert issue["code"] == ("insufficient_reference_data")
    assert issue["details"]["reference_observations"] == 1


def test_contextual_level_excludes_preceding_failures_from_reference():
    """Exclude earlier quality failures from contextual references."""
    grid = _contextual_grid()

    sources = {
        "primary": _contextual_source(
            {
                "2026-01-15T12:00:00Z": 200,
                "2026-01-22T12:00:00Z": 110,
                "2026-01-29T12:00:00Z": 100,
            }
        )
    }

    result = evaluate(
        sources,
        tests=[
            {
                "name": "implausibly_high",
                "method": "range",
                "maximum": {"value_mode": "fixed", "value": 150},
            },
            {
                "name": "unusual_level",
                "method": "contextual_level",
                "reference_orders": [{"period": "7D", "radius": 1}],
                "robust_deviation_threshold": 6,
            },
        ],
        grid=grid,
    )

    target = pd.Timestamp("2026-01-22T12:00:00Z")

    contextual_failures = result.failures.loc[
        (result.failures["test_name"] == "unusual_level")
        & (result.failures["start"] == target)
    ]

    assert len(contextual_failures) == 1


def test_contextual_level_can_include_named_preceding_failures():
    """Allow explicitly named earlier failures back into the reference."""
    grid = _contextual_grid()

    sources = {
        "primary": _contextual_source(
            {
                "2026-01-15T12:00:00Z": 200,
                "2026-01-22T12:00:00Z": 110,
                "2026-01-29T12:00:00Z": 100,
            }
        )
    }

    result = evaluate(
        sources,
        tests=[
            {
                "name": "implausibly_high",
                "method": "range",
                "maximum": {"value_mode": "fixed", "value": 150},
            },
            {
                "name": "unusual_level",
                "method": "contextual_level",
                "reference_orders": [{"period": "7D", "radius": 1}],
                "robust_deviation_threshold": 6,
                "include_failed_periods_from": ["implausibly_high"],
            },
        ],
        grid=grid,
    )

    target = pd.Timestamp("2026-01-22T12:00:00Z")

    contextual_failures = result.failures.loc[
        (result.failures["test_name"] == "unusual_level")
        & (result.failures["start"] == target)
    ]

    assert contextual_failures.empty


def _profile_grid() -> TimeGrid:
    """Return an hourly grid for contextual-profile integration tests."""
    return TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-02T16:00:00Z", frequency="1h"
    )


def _profile_source() -> pd.DataFrame:
    """Return repeated four-hour profile shapes across the integration grid."""
    grid = _profile_grid()
    pattern = [0.0, 1.0, 3.0, 1.0]
    values = pattern * (len(grid.target_index) // len(pattern))
    return pd.DataFrame({"A": values}, index=grid.target_index)


def test_evaluate_reports_contextual_profile_whole_block_failure():
    """Report a contextual shape anomaly as its complete profile period."""
    grid = _profile_grid()
    source = _profile_source()
    target = pd.Timestamp("2026-01-01T20:00:00Z")
    source.loc[target : target + pd.Timedelta("3h"), "A"] = [3, 0, 0, 3]

    result = evaluate(
        {"primary": source},
        tests=[
            {
                "name": "unusual_shape",
                "method": "contextual_profile",
                "profile_duration": "4h",
                "reference_orders": [{"period": "8h", "radius": 2}],
                "robust_deviation_threshold": 6,
            }
        ],
        grid=grid,
    )

    failures = result.failures.loc[
        (result.failures["test_name"] == "unusual_shape")
        & (result.failures["start"] == target)
    ]

    assert len(failures) == 1

    failure = failures.iloc[0]
    assert failure["source"] == "primary"
    assert failure["context"] == "A"
    assert failure["method"] == "contextual_profile"
    assert failure["end"] == target + pd.Timedelta("4h")
    assert failure["details"]["failed_profiles"] == 1
    assert failure["details"]["failed_criteria"] == ["robust_deviation"]


def test_evaluate_decorates_contextual_profile_issue():
    """Decorate method-level incomplete-profile issues through the public API."""
    grid = _profile_grid()
    source = _profile_source()
    target = pd.Timestamp("2026-01-01T20:00:00Z")
    source.loc[target + pd.Timedelta("1h"), "A"] = None

    result = evaluate(
        {"primary": source},
        tests=[
            {
                "name": "unusual_shape",
                "method": "contextual_profile",
                "profile_duration": "4h",
                "reference_orders": [{"period": "8h", "radius": 2}],
                "robust_deviation_threshold": 6,
            }
        ],
        grid=grid,
    )

    issues = result.issues.loc[
        (result.issues["test_name"] == "unusual_shape")
        & (result.issues["start"] == target)
    ]

    assert len(issues) == 1

    issue = issues.iloc[0]
    assert issue["source"] == "primary"
    assert issue["context"] == "A"
    assert issue["method"] == "contextual_profile"
    assert issue["severity"] == "not_evaluable"
    assert issue["code"] == "incomplete_target_profile"
    assert issue["end"] == target + pd.Timedelta("4h")


def test_contextual_profile_excludes_preceding_failures_from_reference_profiles():
    """Drop a reference profile containing observations failed by an earlier test."""
    grid = _profile_grid()
    source = _profile_source()
    scaled_reference = pd.Timestamp("2026-01-01T12:00:00Z")
    target = pd.Timestamp("2026-01-01T20:00:00Z")

    source.loc[scaled_reference : scaled_reference + pd.Timedelta("3h"), "A"] = [
        0,
        100,
        300,
        100,
    ]
    source.loc[target : target + pd.Timedelta("3h"), "A"] = [3, 0, 0, 3]

    result = evaluate(
        {"primary": source},
        tests=[
            {
                "name": "implausibly_high",
                "method": "range",
                "maximum": {"value_mode": "fixed", "value": 50},
            },
            {
                "name": "unusual_shape",
                "method": "contextual_profile",
                "profile_duration": "4h",
                "reference_orders": [{"period": "8h", "radius": 2}],
                "robust_deviation_threshold": 6,
            },
        ],
        grid=grid,
    )

    failure = result.failures.loc[
        (result.failures["test_name"] == "unusual_shape")
        & (result.failures["start"] == target)
    ].iloc[0]

    profile = failure["details"]["profiles"][0]
    assert profile["reference_profiles"] == 3


def test_contextual_profile_can_restore_named_preceding_failures():
    """Restore a failed reference profile when explicitly configured to do so."""
    grid = _profile_grid()
    source = _profile_source()
    scaled_reference = pd.Timestamp("2026-01-01T12:00:00Z")
    target = pd.Timestamp("2026-01-01T20:00:00Z")

    source.loc[scaled_reference : scaled_reference + pd.Timedelta("3h"), "A"] = [
        0,
        100,
        300,
        100,
    ]
    source.loc[target : target + pd.Timedelta("3h"), "A"] = [3, 0, 0, 3]

    result = evaluate(
        {"primary": source},
        tests=[
            {
                "name": "implausibly_high",
                "method": "range",
                "maximum": {"value_mode": "fixed", "value": 50},
            },
            {
                "name": "unusual_shape",
                "method": "contextual_profile",
                "profile_duration": "4h",
                "reference_orders": [{"period": "8h", "radius": 2}],
                "robust_deviation_threshold": 6,
                "include_failed_periods_from": ["implausibly_high"],
            },
        ],
        grid=grid,
    )

    failure = result.failures.loc[
        (result.failures["test_name"] == "unusual_shape")
        & (result.failures["start"] == target)
    ].iloc[0]

    profile = failure["details"]["profiles"][0]
    assert profile["reference_profiles"] == 4


def test_public_evaluate_uses_unselected_sources_as_peers():
    """Use all supplied sources as peers while source selectors choose focal sources."""
    sources = {
        "primary": _source({"A": [150, 150, 150, 150, 150, 150]}),
        "peer_one": _source({"A": [100, 100, 100, 100, 100, 100]}),
        "peer_two": _source({"A": [102, 102, 102, 102, 102, 102]}),
    }

    result = evaluate(
        sources,
        tests=[
            {
                "name": "disagreement",
                "method": "source_disagreement",
                "sources": ["primary"],
                "difference_mode": "fixed",
                "threshold": {"value_mode": "fixed", "value": 20},
            }
        ],
        grid=_grid(),
    )

    assert len(result.failures) == 1

    failure = result.failures.iloc[0]

    assert failure["source"] == "primary"
    assert failure["context"] == "A"
    assert failure["method"] == "source_disagreement"
    assert failure["start"] == _grid().target_index[0]
    assert failure["end"] == pd.Timestamp(_grid().end)


def test_evaluate_resolves_derived_values_per_source_and_context():
    """Resolve one configured value independently for every focal source-context."""
    sources = {
        "primary": _source(
            {"A": [1, 1, 1, 1, 1, 10], "B": [100, 100, 100, 100, 100, 1000]}
        ),
        "secondary": _source(
            {"A": [2, 2, 2, 2, 2, 20], "B": [200, 200, 200, 200, 200, 2000]}
        ),
    }

    result = evaluate(
        sources,
        tests=[
            {
                "name": "derived_maximum",
                "method": "range",
                "maximum": {"value_mode": "median"},
            }
        ],
        grid=_grid(),
    )

    failures = result.failures[result.failures["test_name"] == "derived_maximum"]

    assert len(failures) == 4
    assert set(zip(failures["source"], failures["context"], strict=True)) == {
        ("primary", "A"),
        ("primary", "B"),
        ("secondary", "A"),
        ("secondary", "B"),
    }
    assert failures["start"].eq(pd.Timestamp("2026-01-01T05:00:00Z")).all()


def test_evaluate_derived_value_excludes_preceding_failures_by_default():
    """Exclude preceding failures from later derived-value calibration data."""
    sources = {"primary": _source({"A": [1, 2, 100, 4, 5, 6]})}

    result = evaluate(
        sources,
        tests=[
            {
                "name": "implausibly_high",
                "method": "range",
                "maximum": {"value_mode": "fixed", "value": 10},
            },
            {
                "name": "derived_maximum",
                "method": "range",
                "maximum": {"value_mode": "quantile", "quantile": 1.0},
            },
        ],
        grid=_grid(),
    )

    derived = result.failures[result.failures["test_name"] == "derived_maximum"]

    assert len(derived) == 1
    assert derived.iloc[0]["start"] == pd.Timestamp("2026-01-01T02:00:00Z")


def test_evaluate_derived_value_can_reinclude_preceding_failures():
    """Allow named failures to contribute to later derived-value calibration."""
    sources = {"primary": _source({"A": [1, 2, 100, 4, 5, 6]})}

    result = evaluate(
        sources,
        tests=[
            {
                "name": "implausibly_high",
                "method": "range",
                "maximum": {"value_mode": "fixed", "value": 10},
            },
            {
                "name": "derived_maximum",
                "method": "range",
                "maximum": {"value_mode": "quantile", "quantile": 1.0},
                "include_failed_periods_from": ["implausibly_high"],
            },
        ],
        grid=_grid(),
    )

    derived = result.failures[result.failures["test_name"] == "derived_maximum"]

    assert derived.empty


def test_evaluate_accepts_thread_count():
    """Accept an explicit thread budget without changing evaluation results."""
    sources = {"primary": _source({"A": [1, -2, 3, 4, 5, 6]})}

    result = evaluate(
        sources,
        tests=[
            {
                "name": "non_negative",
                "method": "range",
                "minimum": {"value_mode": "fixed", "value": 0},
            }
        ],
        grid=_grid(),
        threads=2,
    )

    assert len(result.failures) == 1


@pytest.mark.parametrize("threads", [0, -1])
def test_evaluate_requires_positive_thread_count(threads):
    """Reject thread counts below one."""
    sources = {"primary": _source({"A": [1, 2, 3, 4, 5, 6]})}

    with pytest.raises(ValueError, match="threads must be at least 1"):
        evaluate(
            sources,
            tests=[
                {
                    "name": "non_negative",
                    "method": "range",
                    "minimum": {"value_mode": "fixed", "value": 0},
                }
            ],
            grid=_grid(),
            threads=threads,
        )


@pytest.mark.parametrize("threads", [1.5, "2", True])
def test_evaluate_requires_integer_thread_count(threads):
    """Reject non-integer thread counts."""
    sources = {"primary": _source({"A": [1, 2, 3, 4, 5, 6]})}

    with pytest.raises(TypeError, match="threads must be an integer"):
        evaluate(
            sources,
            tests=[
                {
                    "name": "non_negative",
                    "method": "range",
                    "minimum": {"value_mode": "fixed", "value": 0},
                }
            ],
            grid=_grid(),
            threads=threads,
        )


def test_evaluate_logs_single_info_summary(caplog):
    """Report one concise INFO summary when evaluation starts."""
    sources = {"primary": _source({"A": [1, 2, 3, 4, 5, 6]})}

    with caplog.at_level(logging.INFO, logger="tclean.data_quality"):
        evaluate(
            sources,
            tests=[
                {
                    "name": "non_negative",
                    "method": "range",
                    "minimum": {"value_mode": "fixed", "value": 0},
                }
            ],
            grid=_grid(),
            threads=3,
        )

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name.startswith("tclean.data_quality")
        and record.levelno == logging.INFO
    ]

    assert messages == ["Tclean data-quality evaluation: tests=1 | threads=3"]


def test_evaluate_logs_debug_test_timing(caplog):
    """Report test lifecycle details at DEBUG level."""
    sources = {"primary": _source({"A": [1, 2, 3, 4, 5, 6]})}

    with caplog.at_level(logging.DEBUG, logger="tclean.data_quality"):
        evaluate(
            sources,
            tests=[
                {
                    "name": "non_negative",
                    "method": "range",
                    "minimum": {"value_mode": "fixed", "value": 0},
                }
            ],
            grid=_grid(),
        )

    messages = [record.getMessage() for record in caplog.records]

    assert any(
        "Starting data-quality test 1/1: non_negative [range]" in message
        for message in messages
    )
    assert any(
        "Completed data-quality test 1/1: non_negative [range] in" in message
        for message in messages
    )
