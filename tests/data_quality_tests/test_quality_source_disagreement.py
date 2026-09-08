"""Tests for cross-source disagreement data-quality evaluation."""

import pandas as pd
import pytest
from _method_helpers import method_context

from tclean import TimeGrid
from tclean.data_quality.methods.source_disagreement import (
    _aggregate_peers,
    _analyse,
    build_details,
    evaluate,
)


def _grid(hours: int = 4) -> TimeGrid:
    """Return an hourly UTC test grid."""
    return TimeGrid(
        start="2026-01-01T00:00:00Z",
        end=pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(hours=hours),
        frequency="1h",
    )


def _source(values: list[float | None], *, context: str = "A") -> pd.DataFrame:
    """Build one source on a grid matching the supplied values."""
    grid = _grid(len(values))
    return pd.DataFrame({context: values}, index=grid.target_index)


def _test(
    *,
    difference_mode: str = "fixed",
    peer_aggregation_mode: str = "median",
    threshold: float = 10,
    **extra,
) -> dict:
    """Build one normalized source-disagreement test configuration."""
    return {
        "name": "disagreement",
        "method": "source_disagreement",
        "difference_mode": difference_mode,
        "peer_aggregation_mode": peer_aggregation_mode,
        "threshold": {"value_mode": "fixed", "value": threshold},
        **extra,
    }


def _context(sources: dict[str, pd.DataFrame], *, test: dict, preceding_failures=()):
    """Build a focused method context for the primary source."""
    return method_context(
        sources["primary"],
        source_name="primary",
        sources=sources,
        test=test,
        grid=_grid(len(sources["primary"])),
        preceding_failures=preceding_failures,
    )


def test_aggregate_peers_supports_median_and_mean():
    """Aggregate peers using the configured row-wise centre."""
    peers = pd.DataFrame({"one": [100.0], "two": [100.0], "outlier": [1000.0]})

    median = _aggregate_peers(peers, mode="median")
    mean = _aggregate_peers(peers, mode="mean")

    assert median.iloc[0] == 100
    assert mean.iloc[0] == 400


def test_fixed_difference_uses_strict_threshold():
    """Pass a fixed disagreement exactly equal to the threshold."""
    index = _grid(2).target_index
    focal = pd.Series([120.0, 121.0], index=index)
    peer = pd.Series([100.0, 100.0], index=index)

    _, _, difference, failures = _analyse(
        focal, peer, difference_mode="fixed", threshold=20
    )

    assert difference.tolist() == [20, 21]
    assert failures.tolist() == [False, True]


def test_relative_difference_uses_strict_threshold():
    """Pass a relative disagreement exactly equal to the threshold."""
    index = _grid(2).target_index
    focal = pd.Series([120.0, 121.0], index=index)
    peer = pd.Series([100.0, 100.0], index=index)

    _, _, difference, failures = _analyse(
        focal, peer, difference_mode="relative", threshold=0.2
    )

    assert difference.iloc[0] == pytest.approx(0.2)
    assert difference.iloc[1] == pytest.approx(0.21)
    assert failures.tolist() == [False, True]


def test_relative_zero_peer_and_zero_focal_passes():
    """Treat two zero values as zero relative disagreement."""
    index = _grid(1).target_index
    focal = pd.Series([0.0], index=index)
    peer = pd.Series([0.0], index=index)

    _, _, difference, failures = _analyse(
        focal, peer, difference_mode="relative", threshold=0.1
    )

    assert difference.iloc[0] == 0
    assert not failures.iloc[0]


def test_relative_zero_peer_and_nonzero_focal_fails():
    """Treat a nonzero focal value against zero peers as unbounded disagreement."""
    index = _grid(1).target_index
    focal = pd.Series([5.0], index=index)
    peer = pd.Series([0.0], index=index)

    _, _, difference, failures = _analyse(
        focal, peer, difference_mode="relative", threshold=10
    )

    assert pd.isna(difference.iloc[0])
    assert failures.iloc[0]


def test_median_aggregation_is_robust_to_one_bad_peer():
    """Keep a focal source aligned with the peer majority when using the median."""
    sources = {
        "primary": _source([100]),
        "peer_one": _source([100]),
        "peer_two": _source([100]),
        "bad_peer": _source([1000]),
    }

    result = evaluate(_context(sources, test=_test(threshold=200)))

    assert not result.mask["A"].iloc[0]


def test_mean_aggregation_can_be_shifted_by_one_bad_peer():
    """Use the arithmetic peer mean when explicitly configured."""
    sources = {
        "primary": _source([100]),
        "peer_one": _source([100]),
        "peer_two": _source([100]),
        "bad_peer": _source([1000]),
    }

    result = evaluate(
        _context(sources, test=_test(peer_aggregation_mode="mean", threshold=200))
    )

    assert result.mask["A"].iloc[0]


def test_one_peer_is_enough_to_evaluate():
    """Evaluate disagreement when exactly one comparator is observed."""
    sources = {"primary": _source([150]), "peer": _source([100])}

    result = evaluate(_context(sources, test=_test(threshold=20)))

    assert result.mask["A"].iloc[0]
    assert result.issues == ()


def test_missing_peer_values_are_omitted_when_other_peers_exist():
    """Ignore unavailable individual peers without making the target unevaluable."""
    sources = {
        "primary": _source([100, 150]),
        "peer_one": _source([100, 100]),
        "peer_two": _source([None, None]),
    }

    result = evaluate(_context(sources, test=_test(threshold=20)))

    assert result.mask["A"].tolist() == [False, True]
    assert result.issues == ()


def test_peer_without_focal_context_is_not_a_comparator():
    """Ignore supplied sources that do not contain the focal context."""
    sources = {
        "primary": _source([100, 100]),
        "other": _source([100, 100], context="B"),
    }

    result = evaluate(_context(sources, test=_test(threshold=20)))

    assert not result.mask.to_numpy().any()
    assert len(result.issues) == 1
    assert result.issues[0].details["candidate_peer_sources"] == []


def test_no_peer_data_is_grouped_into_one_not_evaluable_issue():
    """Group adjacent focal observations with no eligible peer evidence."""
    sources = {
        "primary": _source([100, 101, 102, 103]),
        "peer": _source([None, None, None, None]),
    }

    result = evaluate(_context(sources, test=_test(threshold=20)))

    assert not result.mask.to_numpy().any()
    assert len(result.issues) == 1

    issue = result.issues[0]
    assert issue.severity == "not_evaluable"
    assert issue.code == "insufficient_peer_data"
    assert issue.start == _grid().target_index[0]
    assert issue.end == pd.Timestamp(_grid().end)
    assert issue.details["candidate_peer_sources"] == ["peer"]


def test_missing_focal_value_does_not_create_peer_issue():
    """Leave focal missingness to missing-data quality checks."""
    sources = {"primary": _source([None]), "peer": _source([None])}

    result = evaluate(_context(sources, test=_test(threshold=20)))

    assert not result.mask["A"].iloc[0]
    assert result.issues == ()


def test_preceding_peer_failure_is_excluded_from_comparison():
    """Mask a peer observation that failed an earlier quality test."""
    sources = {
        "primary": _source([100]),
        "good_peer": _source([100]),
        "bad_peer": _source([1000]),
    }
    timestamp = _grid(1).target_index[0]
    preceding_failure = {
        "context": "A",
        "source": "bad_peer",
        "start": timestamp,
        "end": timestamp + _grid(1).frequency,
        "test_name": "earlier",
        "method": "range",
        "details": {},
    }

    result = evaluate(
        _context(
            sources,
            test=_test(peer_aggregation_mode="mean", threshold=200),
            preceding_failures=[preceding_failure],
        )
    )

    assert not result.mask["A"].iloc[0]


def test_failed_peer_observation_can_be_explicitly_restored():
    """Restore named prior failures when configured for peer evidence."""
    sources = {
        "primary": _source([100]),
        "good_peer": _source([100]),
        "bad_peer": _source([1000]),
    }
    timestamp = _grid(1).target_index[0]
    preceding_failure = {
        "context": "A",
        "source": "bad_peer",
        "start": timestamp,
        "end": timestamp + _grid(1).frequency,
        "test_name": "earlier",
        "method": "range",
        "details": {},
    }

    result = evaluate(
        _context(
            sources,
            test=_test(
                peer_aggregation_mode="mean",
                threshold=200,
                include_failed_periods_from=["earlier"],
            ),
            preceding_failures=[preceding_failure],
        )
    )

    assert result.mask["A"].iloc[0]


def test_build_details_records_contributing_peer_values():
    """Report the peer evidence used for each failed focal observation."""
    sources = {
        "primary": _source([150]),
        "peer_one": _source([100]),
        "peer_two": _source([102]),
    }
    context = _context(sources, test=_test(threshold=20))
    result = evaluate(context)
    timestamp = _grid(1).target_index[0]

    details = build_details(
        context,
        result,
        context_name="A",
        start=timestamp,
        end=timestamp + _grid(1).frequency,
    )

    assert details["difference_mode"] == "fixed"
    assert details["peer_aggregation_mode"] == "median"
    assert details["threshold"] == 20
    assert details["failed_observations"] == 1

    observation = details["observations"][0]
    assert observation["peer_sources"] == ["peer_one", "peer_two"]
    assert observation["peer_values"] == {"peer_one": 100.0, "peer_two": 102.0}
    assert observation["peer_observations"] == 2
    assert observation["peer_reference_value"] == 101
    assert observation["signed_difference"] == 49
    assert observation["absolute_difference"] == 49


def test_relative_zero_reference_details_use_none_instead_of_infinity():
    """Represent unbounded relative disagreement without non-finite details."""
    sources = {"primary": _source([5]), "peer": _source([0])}
    context = _context(sources, test=_test(difference_mode="relative", threshold=10))
    result = evaluate(context)
    timestamp = _grid(1).target_index[0]

    details = build_details(
        context,
        result,
        context_name="A",
        start=timestamp,
        end=timestamp + _grid(1).frequency,
    )

    assert details["observations"][0]["relative_difference"] is None


def test_evaluate_fixed_supports_derived_focal_threshold():
    """Derive a fixed disagreement threshold from the focal source only."""
    sources = {
        "primary": _source([150, 150, 150, 150]),
        "peer_one": _source([100, 100, 100, 100]),
        "peer_two": _source([102, 102, 102, 102]),
    }
    test = _test()
    test["threshold"] = {"value_mode": "median", "multiplier": 0.1}

    result = evaluate(_context(sources, test=test))

    assert result.mask["A"].tolist() == [True, True, True, True]
