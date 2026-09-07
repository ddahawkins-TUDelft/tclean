"""Tests for contextual-profile data-quality testing."""

import numpy as np
import pandas as pd
import pytest

from tclean import TimeGrid
from tclean.data_quality.methods.contextual_profile import (
    _contextual_profile_evidence,
    _distance_matrix,
    _normalize_profile,
    _peer_distance_scores,
    _predictive_profile_probability,
    _Profile,
    _profile_distance,
    _reference_profiles,
    _robust_profile_deviation,
    _target_profile_starts,
    build_details,
    evaluate_with_issues,
)
from tclean.data_quality.methods.contextual_profile import (
    validate as validate_contextual_profile,
)


def _grid(*, hours: int = 96) -> TimeGrid:
    """Return an hourly grid long enough for contextual profiles."""
    return TimeGrid(
        start="2026-01-01T00:00:00Z",
        end=pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(hours=hours),
        frequency="1h",
    )


def _validated_test(grid: TimeGrid, **overrides):
    """Return a validated four-hour contextual-profile test."""
    test = {
        "name": "unusual_shape",
        "method": "contextual_profile",
        "profile_duration": "4h",
        "reference_orders": [{"period": "8h", "radius": 3}],
        "robust_deviation_threshold": 6,
    }
    test.update(overrides)
    return validate_contextual_profile(test, grid=grid)


def _profile(values, *, start="2026-01-01T00:00:00Z") -> _Profile:
    """Build one normalized four-value profile for statistical tests."""
    start = pd.Timestamp(start)
    array = np.asarray(values, dtype=float)
    return _Profile(
        start=start,
        end=start + pd.Timedelta(hours=len(array)),
        values=_normalize_profile(array),
    )


def _patterned_series(grid: TimeGrid) -> pd.Series:
    """Return complete data made from repeated four-hour profile shapes."""
    pattern = np.array([0.0, 1.0, 3.0, 1.0])
    repeats = len(grid.target_index) // len(pattern)
    values = np.tile(pattern, repeats)
    return pd.Series(values, index=grid.target_index, dtype=float)


def test_normalize_profile_removes_level_and_scale():
    """Treat affine-scaled profiles as the same shape."""
    first = _normalize_profile(np.array([0, 1, 3, 1], dtype=float))
    second = _normalize_profile(np.array([10, 12, 16, 12], dtype=float))

    assert second == pytest.approx(first)


def test_normalize_profile_maps_constant_shape_to_zero():
    """Represent a constant profile as a zero shape vector."""
    result = _normalize_profile(np.array([5, 5, 5, 5], dtype=float))

    assert result == pytest.approx(np.zeros(4))


def test_profile_distance_is_zero_for_same_shape():
    """Give translated and rescaled copies zero aligned shape distance."""
    first = _profile([0, 1, 3, 1])
    second = _profile([100, 110, 130, 110])

    assert _profile_distance(first.values, second.values) == 0.0


def test_profile_distance_preserves_temporal_alignment():
    """Treat a shifted peak as a different profile shape."""
    first = _profile([0, 1, 3, 1])
    shifted = _profile([1, 3, 1, 0])

    assert _profile_distance(first.values, shifted.values) > 0


def test_distance_matrix_matches_direct_profile_distance():
    """Calculate pairwise shape distances without a 3-D difference array."""
    profiles = [_profile([0, 1, 3, 1]), _profile([1, 3, 1, 0]), _profile([3, 1, 0, 1])]

    result = _distance_matrix(profiles)

    assert result.shape == (3, 3)
    assert np.diag(result) == pytest.approx(np.zeros(3))
    assert result[0, 1] == pytest.approx(
        _profile_distance(profiles[0].values, profiles[1].values)
    )
    assert result == pytest.approx(result.T)


def test_peer_distance_scores_exclude_self_distance():
    """Use only distances to other profiles in each peer score."""
    distances = np.array([[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]])

    result = _peer_distance_scores(distances)

    assert result == pytest.approx([2.0, 1.5, 2.5])


def test_target_profile_starts_apply_offset_and_ignore_horizon_fragments():
    """Build only complete blocks at the configured profile offset."""
    grid = _grid(hours=14)
    test = _validated_test(grid, profile_offset="1h")

    starts = _target_profile_starts(grid.target_index, test=test, grid=grid)

    assert list(starts) == [
        pd.Timestamp("2026-01-01T01:00:00Z"),
        pd.Timestamp("2026-01-01T05:00:00Z"),
        pd.Timestamp("2026-01-01T09:00:00Z"),
    ]


def test_target_profile_starts_respect_shorter_data_horizon():
    """Ignore a configured block that would extend beyond available data."""
    grid = _grid(hours=16)
    test = _validated_test(grid, profile_offset="1h")
    shorter_index = grid.target_index[:12]

    starts = _target_profile_starts(shorter_index, test=test, grid=grid)

    assert list(starts) == [
        pd.Timestamp("2026-01-01T01:00:00Z"),
        pd.Timestamp("2026-01-01T05:00:00Z"),
    ]


def test_reference_profiles_discard_incomplete_profile():
    """Drop a reference block containing a missing observation."""
    grid = _grid(hours=48)
    test = _validated_test(grid, reference_orders=[{"period": "8h", "radius": 2}])
    reference = _patterned_series(grid)
    reference.loc[pd.Timestamp("2026-01-01T13:00:00Z")] = np.nan

    target = _profile([3, 0, 0, 3], start="2026-01-01T20:00:00Z")

    profiles = _reference_profiles(reference, target=target, test=test, grid=grid)

    starts = {profile.start for profile in profiles}
    assert pd.Timestamp("2026-01-01T12:00:00Z") not in starts
    assert pd.Timestamp("2026-01-01T04:00:00Z") in starts
    assert pd.Timestamp("2026-01-02T04:00:00Z") in starts


def test_reference_profiles_discard_overlap_with_target():
    """Do not use a contextual reference block that overlaps the target."""
    grid = _grid(hours=24)
    test = _validated_test(
        grid, profile_duration="8h", reference_orders=[{"period": "4h", "radius": 2}]
    )
    reference = _patterned_series(grid)
    target = _profile([0, 1, 3, 1, 0, 1, 3, 1], start="2026-01-01T08:00:00Z")

    profiles = _reference_profiles(reference, target=target, test=test, grid=grid)

    assert all(
        profile.end <= target.start or profile.start >= target.end
        for profile in profiles
    )


def test_robust_profile_deviation_requires_three_references():
    """Require a nondegenerate peer-distance baseline for robust evidence."""
    target = _profile([3, 0, 0, 3])
    references = [_profile([0, 1, 3, 1]), _profile([10, 12, 16, 12])]

    result = _robust_profile_deviation(target, references, threshold=6)

    assert result is None


def test_robust_profile_deviation_fails_different_shape_zero_mad():
    """Fail a different shape when reference peer distances have no spread."""
    target = _profile([3, 0, 0, 3])
    references = [
        _profile([0, 1, 3, 1]),
        _profile([10, 12, 16, 12]),
        _profile([100, 110, 130, 110]),
    ]

    result = _robust_profile_deviation(target, references, threshold=6)

    assert result is not None
    assert result.reference_profiles == 3
    assert result.reference_distance_median == pytest.approx(0.0, abs=1e-12)
    assert result.reference_distance_mad == pytest.approx(0.0, abs=1e-12)
    assert result.robust_scale == pytest.approx(0.0, abs=1e-12)
    assert result.robust_deviation is None
    assert result.failed is True


def test_robust_profile_deviation_passes_same_shape_zero_mad():
    """Pass a target matching a zero-spread contextual shape population."""
    target = _profile([20, 22, 26, 22])
    references = [
        _profile([0, 1, 3, 1]),
        _profile([10, 12, 16, 12]),
        _profile([100, 110, 130, 110]),
    ]

    result = _robust_profile_deviation(target, references, threshold=6)

    assert result is not None
    assert result.target_distance == pytest.approx(0.0, abs=1e-12)
    assert result.failed is False


def test_predictive_probability_is_unavailable_without_references():
    """Return no rank probability when no contextual profile exists."""
    result = _predictive_profile_probability(
        _profile([3, 0, 0, 3]), [], maximum_probability=0.1
    )

    assert result is None


def test_predictive_probability_fails_unique_outlier_at_threshold():
    """Use the target's conservative contextual nonconformity rank."""
    target = _profile([3, 0, 0, 3])
    references = [_profile([0, 1, 3, 1]) for _ in range(9)]

    result = _predictive_profile_probability(
        target, references, maximum_probability=0.1
    )

    assert result is not None
    assert result.reference_profiles == 9
    assert result.profiles_at_least_as_nonconforming == 1
    assert result.comparison_profiles == 10
    assert result.predictive_probability == pytest.approx(0.1)
    assert result.failed is True


def test_predictive_probability_small_reference_set_has_coarse_resolution():
    """Let sparse reference evidence naturally limit predictive failures."""
    target = _profile([3, 0, 0, 3])
    references = [_profile([0, 1, 3, 1]) for _ in range(8)]

    result = _predictive_profile_probability(
        target, references, maximum_probability=0.1
    )

    assert result is not None
    assert result.predictive_probability == pytest.approx(1 / 9)
    assert result.failed is False


def test_predictive_probability_counts_ties_conservatively():
    """Count equal nonconformity scores as at least as extreme as target."""
    target = _profile([20, 22, 26, 22])
    references = [_profile([0, 1, 3, 1]) for _ in range(9)]

    result = _predictive_profile_probability(
        target, references, maximum_probability=0.1
    )

    assert result is not None
    assert result.profiles_at_least_as_nonconforming == 10
    assert result.predictive_probability == 1.0
    assert result.failed is False


def test_contextual_profile_evidence_uses_available_predictive_criterion():
    """Remain evaluable when robust evidence lacks three references."""
    target = _profile([3, 0, 0, 3])
    references = [_profile([0, 1, 3, 1]), _profile([10, 12, 16, 12])]

    result = _contextual_profile_evidence(
        target,
        references,
        test={"robust_deviation_threshold": 6, "maximum_predictive_probability": 0.1},
    )

    assert result.robust is None
    assert result.predictive is not None
    assert result.evaluable is True


def test_contextual_profile_evidence_records_both_failed_criteria():
    """Combine robust and predictive profile evidence with OR semantics."""
    target = _profile([3, 0, 0, 3])
    references = [_profile([0, 1, 3, 1]) for _ in range(9)]

    result = _contextual_profile_evidence(
        target,
        references,
        test={"robust_deviation_threshold": 6, "maximum_predictive_probability": 0.1},
    )

    assert result.failed is True
    assert result.failed_criteria == ("robust_deviation", "predictive_probability")


def test_contextual_profile_flags_entire_failed_block():
    """Attribute a contextual shape anomaly to its whole profile period."""
    grid = _grid(hours=48)
    series = _patterned_series(grid)
    target = pd.Timestamp("2026-01-01T20:00:00Z")
    series.loc[target : target + pd.Timedelta("3h")] = [3, 0, 0, 3]

    data = series.to_frame("A")
    test = _validated_test(grid, reference_orders=[{"period": "8h", "radius": 2}])

    mask, issues = evaluate_with_issues(
        data, reference=data.copy(), test=test, grid=grid
    )

    expected = grid.index_for_period(start=target, end=target + pd.Timedelta("4h"))
    assert mask.loc[expected, "A"].all()
    assert not [issue for issue in issues if issue["start"] == target]


def test_contextual_profile_is_shape_only_at_runtime():
    """Ignore target level and amplitude when the profile shape is unchanged."""
    grid = _grid(hours=48)
    series = _patterned_series(grid)
    target = pd.Timestamp("2026-01-01T20:00:00Z")
    series.loc[target : target + pd.Timedelta("3h")] = [100, 110, 130, 110]

    data = series.to_frame("A")
    test = _validated_test(grid, reference_orders=[{"period": "8h", "radius": 2}])

    mask, _ = evaluate_with_issues(data, reference=data.copy(), test=test, grid=grid)

    expected = grid.index_for_period(start=target, end=target + pd.Timedelta("4h"))
    assert not mask.loc[expected, "A"].any()


def test_contextual_profile_reports_incomplete_target_profile():
    """Report missing values that prevent evaluation of a complete target block."""
    grid = _grid(hours=48)
    series = _patterned_series(grid)
    target = pd.Timestamp("2026-01-01T20:00:00Z")
    series.loc[target + pd.Timedelta("1h")] = np.nan

    data = series.to_frame("A")
    test = _validated_test(grid)

    mask, issues = evaluate_with_issues(
        data, reference=data.copy(), test=test, grid=grid
    )

    target_issues = [issue for issue in issues if issue["start"] == target]

    assert not mask.loc[target, "A"]
    assert len(target_issues) == 1
    assert target_issues[0]["severity"] == "not_evaluable"
    assert target_issues[0]["code"] == "incomplete_target_profile"
    assert target_issues[0]["end"] == target + pd.Timedelta("4h")
    assert target_issues[0]["details"] == {
        "profile_observations": 4,
        "missing_observations": 1,
    }


def test_contextual_profile_reports_insufficient_robust_references():
    """Report robust profile evaluation that has fewer than three references."""
    grid = _grid(hours=16)
    series = _patterned_series(grid)
    target = pd.Timestamp("2026-01-01T08:00:00Z")

    data = series.to_frame("A")
    test = _validated_test(grid, reference_orders=[{"period": "8h", "radius": 1}])

    mask, issues = evaluate_with_issues(
        data, reference=data.copy(), test=test, grid=grid
    )

    target_issues = [issue for issue in issues if issue["start"] == target]

    assert not mask.loc[target, "A"]
    assert len(target_issues) == 1
    assert target_issues[0]["severity"] == "not_evaluable"
    assert target_issues[0]["code"] == "insufficient_reference_profiles"
    assert target_issues[0]["details"]["reference_profiles"] == 1
    assert target_issues[0]["details"]["configured_criteria"] == ["robust_deviation"]


def test_contextual_profile_warns_when_robust_criterion_is_unavailable():
    """Use predictive evidence while warning that robust evidence is unavailable."""
    grid = _grid(hours=24)
    series = _patterned_series(grid)
    target = pd.Timestamp("2026-01-01T08:00:00Z")

    data = series.to_frame("A")
    test = _validated_test(
        grid,
        reference_orders=[{"period": "8h", "radius": 1}],
        maximum_predictive_probability=0.1,
    )

    _, issues = evaluate_with_issues(data, reference=data.copy(), test=test, grid=grid)

    target_issues = [issue for issue in issues if issue["start"] == target]

    assert len(target_issues) == 1
    assert target_issues[0]["severity"] == "warning"
    assert target_issues[0]["code"] == "criterion_not_evaluable"
    assert target_issues[0]["details"] == {
        "criterion": "robust_deviation",
        "reference_profiles": 2,
    }


def test_contextual_profile_ignores_trailing_partial_profile():
    """Do not emit an issue for a block extending beyond the series horizon."""
    grid = _grid(hours=14)
    series = pd.Series(1.0, index=grid.target_index, dtype=float)
    data = series.to_frame("A")
    test = _validated_test(grid, profile_offset="1h")

    _, issues = evaluate_with_issues(data, reference=data.copy(), test=test, grid=grid)

    assert not [
        issue
        for issue in issues
        if issue["start"] == pd.Timestamp("2026-01-01T13:00:00Z")
    ]


def test_contextual_profile_build_details_reports_failed_profile_evidence():
    """Build auditable evidence for a failed contextual profile."""
    grid = _grid(hours=48)
    series = _patterned_series(grid)
    target = pd.Timestamp("2026-01-01T20:00:00Z")
    end = target + pd.Timedelta("4h")
    series.loc[target : target + pd.Timedelta("3h")] = [3, 0, 0, 3]

    test = _validated_test(grid, reference_orders=[{"period": "8h", "radius": 2}])

    details = build_details(
        series, reference=series.copy(), start=target, end=end, test=test, grid=grid
    )

    assert details["failed_profiles"] == 1
    assert details["failed_criteria"] == ["robust_deviation"]

    profile = details["profiles"][0]
    assert profile["profile_start"] == target.isoformat()
    assert profile["profile_end"] == end.isoformat()
    assert profile["reference_profiles"] == 4
    assert profile["robust_failed"] is True
    assert profile["robust_deviation_threshold"] == 6.0
    assert "predictive_probability" not in profile
