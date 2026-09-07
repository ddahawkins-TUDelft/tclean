"""Tests for contextual-level data-quality testing."""

import pandas as pd
import pytest
from _method_helpers import method_details, method_evaluation_pair

from tclean import TimeGrid
from tclean.data_quality.methods.contextual_level import (
    _contextual_level_evidence,
    _predictive_probability,
    _robust_deviation,
    build_details,
    evaluate,
    validate,
)


def test_robust_deviation_passes_contextual_value():
    """Pass a target consistent with its robust reference population."""
    reference = pd.Series([98, 99, 100, 100, 101, 102], dtype=float)

    result = _robust_deviation(101, reference, threshold=6)

    assert result is not None
    assert result.reference_observations == 6
    assert result.reference_median == 100.0
    assert result.reference_mad == 1.0
    assert result.robust_scale == pytest.approx(1.4826)
    assert result.failed is False


def test_robust_deviation_fails_high_contextual_value():
    """Fail an unusually high contextual value."""
    reference = pd.Series([98, 99, 100, 100, 101, 102], dtype=float)

    result = _robust_deviation(110, reference, threshold=6)

    assert result is not None
    assert result.deviation == 10.0
    assert result.robust_deviation == pytest.approx(10 / 1.4826)
    assert result.failed is True


def test_robust_deviation_fails_low_contextual_value():
    """Fail an unusually low contextual value."""
    reference = pd.Series([98, 99, 100, 100, 101, 102], dtype=float)

    result = _robust_deviation(90, reference, threshold=6)

    assert result is not None
    assert result.deviation == -10.0
    assert result.failed is True


def test_robust_deviation_resists_extreme_reference_value():
    """Prevent one extreme reference from dominating robust scale."""
    reference = pd.Series([98, 99, 100, 100, 101, 102, 1000], dtype=float)

    result = _robust_deviation(110, reference, threshold=6)

    assert result is not None
    assert result.reference_median == 100.0
    assert result.reference_mad == 1.0
    assert result.failed is True


def test_robust_deviation_ignores_missing_reference_values():
    """Use only observed values from the contextual reference."""
    reference = pd.Series([98, None, 100, None, 102], dtype=float)

    result = _robust_deviation(100, reference, threshold=6)

    assert result is not None
    assert result.reference_observations == 3


def test_robust_deviation_is_unavailable_without_references():
    """Return no result when no contextual reference values exist."""
    reference = pd.Series([None, None], dtype=float)

    result = _robust_deviation(100, reference, threshold=6)

    assert result is None


def test_robust_deviation_passes_equal_target_with_zero_mad():
    """Pass an identical target when the robust reference scale is zero."""
    reference = pd.Series([100, 100, 100, 100], dtype=float)

    result = _robust_deviation(100, reference, threshold=6)

    assert result is not None
    assert result.reference_mad == 0.0
    assert result.robust_scale == 0.0
    assert result.robust_deviation == 0.0
    assert result.failed is False


def test_robust_deviation_fails_different_target_with_zero_mad():
    """Fail a differing target against a zero-variation reference."""
    reference = pd.Series([100, 100, 100, 100], dtype=float)

    result = _robust_deviation(101, reference, threshold=6)

    assert result is not None
    assert result.reference_mad == 0.0
    assert result.robust_scale == 0.0
    assert result.robust_deviation is None
    assert result.failed is True


def test_robust_deviation_passes_exact_threshold():
    """Use a strict robust-deviation failure threshold."""
    reference = pd.Series([98, 99, 100, 100, 101, 102], dtype=float)

    target = 100 + 6 * 1.4826

    result = _robust_deviation(target, reference, threshold=6)

    assert result is not None
    assert result.robust_deviation == pytest.approx(6)
    assert result.failed is False


def test_predictive_probability_calculates_student_t_evidence():
    """Calculate predictive evidence from a contextual reference."""
    reference = pd.Series([98, 99, 100, 100, 101, 102], dtype=float)

    result = _predictive_probability(110, reference, maximum_probability=0.05)

    assert result is not None
    assert result.reference_observations == 6
    assert result.reference_mean == 100.0
    assert result.reference_standard_deviation == pytest.approx(1.4142135623730951)
    assert result.prediction_standard_error == pytest.approx(1.527525231651947)
    assert result.deviation == 10.0
    assert result.degrees_of_freedom == 5
    assert result.t_statistic == pytest.approx(6.54653670707977)
    assert result.predictive_probability == pytest.approx(0.0012455792530624264)
    assert result.failed is True


def test_predictive_probability_passes_contextual_value():
    """Pass a target consistent with its predictive reference."""
    reference = pd.Series([98, 99, 100, 100, 101, 102], dtype=float)

    result = _predictive_probability(101, reference, maximum_probability=0.05)

    assert result is not None
    assert result.predictive_probability == pytest.approx(0.5416045607931202)
    assert result.failed is False


def test_predictive_probability_is_two_sided():
    """Treat equally extreme high and low deviations symmetrically."""
    reference = pd.Series([98, 99, 100, 100, 101, 102], dtype=float)

    high = _predictive_probability(110, reference, maximum_probability=0.05)

    low = _predictive_probability(90, reference, maximum_probability=0.05)

    assert high is not None
    assert low is not None

    assert high.t_statistic == pytest.approx(-low.t_statistic)
    assert high.predictive_probability == pytest.approx(low.predictive_probability)


def test_predictive_probability_ignores_missing_references():
    """Use only observed contextual reference values."""
    reference = pd.Series([98, None, 100, None, 102], dtype=float)

    result = _predictive_probability(100, reference, maximum_probability=0.05)

    assert result is not None
    assert result.reference_observations == 3


def test_predictive_probability_requires_variance_estimate():
    """Require two observations to estimate predictive variation."""
    reference = pd.Series([100, None], dtype=float)

    result = _predictive_probability(110, reference, maximum_probability=0.05)

    assert result is None


def test_predictive_probability_is_unavailable_without_references():
    """Return no predictive result without contextual references."""
    reference = pd.Series([None, None], dtype=float)

    result = _predictive_probability(110, reference, maximum_probability=0.05)

    assert result is None


def test_predictive_probability_passes_equal_zero_variance_target():
    """Pass an identical target against a zero-variance reference."""
    reference = pd.Series([100, 100, 100, 100], dtype=float)

    result = _predictive_probability(100, reference, maximum_probability=0.05)

    assert result is not None
    assert result.reference_standard_deviation == 0.0
    assert result.prediction_standard_error == 0.0
    assert result.t_statistic == 0.0
    assert result.predictive_probability == 1.0
    assert result.failed is False


def test_predictive_probability_fails_different_zero_variance_target():
    """Fail a differing target against a zero-variance reference."""
    reference = pd.Series([100, 100, 100, 100], dtype=float)

    result = _predictive_probability(101, reference, maximum_probability=0.05)

    assert result is not None
    assert result.reference_standard_deviation == 0.0
    assert result.prediction_standard_error == 0.0
    assert result.t_statistic is None
    assert result.predictive_probability == 0.0
    assert result.failed is True


def test_predictive_probability_fails_at_probability_threshold():
    """Include equality in the predictive-probability failure threshold."""
    reference = pd.Series([98, 99, 100, 100, 101, 102], dtype=float)

    baseline = _predictive_probability(105, reference, maximum_probability=1.0)

    assert baseline is not None

    result = _predictive_probability(
        105, reference, maximum_probability=baseline.predictive_probability
    )

    assert result is not None
    assert result.failed is True


def test_contextual_level_evidence_uses_robust_criterion_only():
    """Evaluate only the configured robust criterion."""
    reference = pd.Series([98, 99, 100, 100, 101, 102], dtype=float)

    result = _contextual_level_evidence(
        110, reference, test={"robust_deviation_threshold": 6}
    )

    assert result.robust is not None
    assert result.predictive is None
    assert result.failed is True
    assert result.evaluable is True
    assert result.failed_criteria == ("robust_deviation",)


def test_contextual_level_evidence_uses_predictive_criterion_only():
    """Evaluate only the configured predictive criterion."""
    reference = pd.Series([98, 99, 100, 100, 101, 102], dtype=float)

    result = _contextual_level_evidence(
        110, reference, test={"maximum_predictive_probability": 0.05}
    )

    assert result.robust is None
    assert result.predictive is not None
    assert result.failed is True
    assert result.evaluable is True
    assert result.failed_criteria == ("predictive_probability",)


def test_contextual_level_evidence_records_both_failed_criteria():
    """Record both criteria when both contextual checks fail."""
    reference = pd.Series([98, 99, 100, 100, 101, 102], dtype=float)

    result = _contextual_level_evidence(
        110,
        reference,
        test={"robust_deviation_threshold": 6, "maximum_predictive_probability": 0.05},
    )

    assert result.failed is True
    assert result.failed_criteria == ("robust_deviation", "predictive_probability")


def test_contextual_level_evidence_passes_when_no_criterion_fails():
    """Pass when all configured contextual criteria pass."""
    reference = pd.Series([98, 99, 100, 100, 101, 102], dtype=float)

    result = _contextual_level_evidence(
        100,
        reference,
        test={"robust_deviation_threshold": 6, "maximum_predictive_probability": 0.05},
    )

    assert result.failed is False
    assert result.evaluable is True
    assert result.failed_criteria == ()


def test_contextual_level_evidence_uses_available_criterion():
    """Remain evaluable when one configured criterion is unavailable."""
    reference = pd.Series([100], dtype=float)

    result = _contextual_level_evidence(
        110,
        reference,
        test={"robust_deviation_threshold": 6, "maximum_predictive_probability": 0.05},
    )

    assert result.robust is not None
    assert result.predictive is None

    assert result.evaluable is True
    assert result.failed is True
    assert result.failed_criteria == ("robust_deviation",)


def test_contextual_level_evidence_can_be_not_evaluable():
    """Identify when no configured criterion can be evaluated."""
    reference = pd.Series([100], dtype=float)

    result = _contextual_level_evidence(
        110, reference, test={"maximum_predictive_probability": 0.05}
    )

    assert result.predictive is None
    assert result.evaluable is False
    assert result.failed is False
    assert result.failed_criteria == ()


def _contextual_grid() -> TimeGrid:
    """Return a grid spanning contextual weekly references."""
    return TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-30T00:00:00Z", frequency="1h"
    )


def _contextual_series(values: dict[str, float]) -> pd.Series:
    """Build sparse observed values on the contextual test grid."""
    grid = _contextual_grid()

    series = pd.Series(float("nan"), index=grid.target_index, dtype=float)

    for timestamp, value in values.items():
        series.loc[pd.Timestamp(timestamp)] = value

    return series


def test_contextual_level_evaluates_lattice_reference():
    """Evaluate a target against its configured contextual lattice."""
    grid = _contextual_grid()

    series = _contextual_series(
        {
            "2026-01-08T12:00:00Z": 100,
            "2026-01-15T12:00:00Z": 100,
            "2026-01-22T12:00:00Z": 120,
            "2026-01-29T12:00:00Z": 100,
        }
    )

    data = series.to_frame("A")
    test = validate(
        {
            "name": "unusual_level",
            "method": "contextual_level",
            "reference_orders": [{"period": "7D", "radius": 2}],
            "robust_deviation_threshold": 6,
        },
        grid=grid,
    )

    mask, issues = method_evaluation_pair(evaluate, data, test=test, grid=grid)

    target = pd.Timestamp("2026-01-22T12:00:00Z")

    assert mask.loc[target, "A"]
    assert issues == ()


def test_contextual_level_ignores_missing_target():
    """Do not report missing targets as contextual failures or issues."""
    grid = _contextual_grid()

    data = pd.DataFrame(
        {"A": pd.Series(float("nan"), index=grid.target_index, dtype=float)}
    )

    test = validate(
        {
            "name": "unusual_level",
            "method": "contextual_level",
            "reference_orders": [{"period": "7D", "radius": 1}],
            "maximum_predictive_probability": 0.05,
        },
        grid=grid,
    )

    mask, issues = method_evaluation_pair(evaluate, data, test=test, grid=grid)

    assert not mask.to_numpy().any()
    assert issues == ()


def test_contextual_level_reports_insufficient_predictive_reference():
    """Report an observed target that cannot be predictively evaluated."""
    grid = _contextual_grid()

    series = _contextual_series(
        {"2026-01-15T12:00:00Z": 100, "2026-01-22T12:00:00Z": 110}
    )

    data = series.to_frame("A")
    test = validate(
        {
            "name": "unusual_level",
            "method": "contextual_level",
            "reference_orders": [{"period": "7D", "radius": 1}],
            "maximum_predictive_probability": 0.05,
        },
        grid=grid,
    )

    mask, issues = method_evaluation_pair(evaluate, data, test=test, grid=grid)

    target = pd.Timestamp("2026-01-22T12:00:00Z")

    assert not mask.loc[target, "A"]

    target_issues = [issue for issue in issues if issue.start == target]

    assert len(target_issues) == 1

    issue = target_issues[0]

    assert issue.severity == "not_evaluable"
    assert issue.code == "insufficient_reference_data"
    assert issue.details["reference_observations"] == 1
    assert issue.details["configured_criteria"] == ["predictive_probability"]


def test_contextual_level_warns_when_one_criterion_is_unavailable():
    """Use an available criterion while warning about another."""
    grid = _contextual_grid()

    series = _contextual_series(
        {"2026-01-15T12:00:00Z": 100, "2026-01-22T12:00:00Z": 110}
    )

    data = series.to_frame("A")
    test = validate(
        {
            "name": "unusual_level",
            "method": "contextual_level",
            "reference_orders": [{"period": "7D", "radius": 1}],
            "robust_deviation_threshold": 6,
            "maximum_predictive_probability": 0.05,
        },
        grid=grid,
    )

    mask, issues = method_evaluation_pair(evaluate, data, test=test, grid=grid)

    target = pd.Timestamp("2026-01-22T12:00:00Z")

    assert mask.loc[target, "A"]

    target_issues = [issue for issue in issues if issue.start == target]

    assert len(target_issues) == 1

    issue = target_issues[0]

    assert issue.severity == "warning"
    assert issue.code == "criterion_not_evaluable"
    assert issue.details == {
        "criterion": "predictive_probability",
        "reference_observations": 1,
    }


def test_contextual_level_build_details_reports_robust_evidence():
    """Report robust evidence for a contextual-level failure."""
    grid = _contextual_grid()

    series = _contextual_series(
        {
            "2026-01-08T12:00:00Z": 100,
            "2026-01-15T12:00:00Z": 100,
            "2026-01-22T12:00:00Z": 120,
            "2026-01-29T12:00:00Z": 100,
        }
    )

    test = validate(
        {
            "name": "unusual_level",
            "method": "contextual_level",
            "reference_orders": [{"period": "7D", "radius": 2}],
            "robust_deviation_threshold": 6,
        },
        grid=grid,
    )

    target = pd.Timestamp("2026-01-22T12:00:00Z")

    details = method_details(
        evaluate,
        build_details,
        series,
        start=target,
        end=target + grid.frequency,
        test=test,
        grid=grid,
    )

    assert details["failed_observations"] == 1
    assert details["failed_criteria"] == ["robust_deviation"]

    observation = details["observations"][0]

    assert observation["timestamp"] == (target.isoformat())
    assert observation["value"] == 120.0
    assert observation["reference_observations"] == 3

    assert observation["robust_deviation_threshold"] == 6.0

    assert observation["robust_failed"] is True

    assert "predictive_probability" not in observation


def test_contextual_level_build_details_reports_both_criteria():
    """Report both configured contextual criteria when evaluable."""
    grid = _contextual_grid()

    series = _contextual_series(
        {
            "2026-01-01T12:00:00Z": 98,
            "2026-01-08T12:00:00Z": 99,
            "2026-01-15T12:00:00Z": 100,
            "2026-01-22T12:00:00Z": 120,
            "2026-01-29T12:00:00Z": 101,
        }
    )

    test = validate(
        {
            "name": "unusual_level",
            "method": "contextual_level",
            "reference_orders": [{"period": "7D", "radius": 3}],
            "robust_deviation_threshold": 6,
            "maximum_predictive_probability": 0.05,
        },
        grid=grid,
    )

    target = pd.Timestamp("2026-01-22T12:00:00Z")

    details = method_details(
        evaluate,
        build_details,
        series,
        start=target,
        end=target + grid.frequency,
        test=test,
        grid=grid,
    )

    observation = details["observations"][0]

    assert "robust_deviation" in observation
    assert "predictive_probability" in observation
    assert "reference_median" in observation
    assert "reference_mean" in observation
