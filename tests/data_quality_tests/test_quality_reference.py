"""Tests for data-quality reference filtering."""

import pandas as pd

from tclean.data_quality._reference import (
    build_reference_data,
    reference_exclusion_mask,
)


def _data() -> pd.DataFrame:
    """Build canonical-style reference-filtering test data."""
    index = pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="1h")

    return pd.DataFrame(
        {"A": [10, 20, 30, 40, 50, 60], "B": [100, 200, 300, 400, 500, 600]},
        index=index,
    )


def _failure(
    *,
    source: str = "primary",
    context: str = "A",
    start: str = "2026-01-01T01:00:00Z",
    end: str = "2026-01-01T03:00:00Z",
    test_name: str = "earlier_test",
) -> dict:
    """Build a preceding quality-failure record."""
    return {
        "source": source,
        "context": context,
        "start": pd.Timestamp(start),
        "end": pd.Timestamp(end),
        "test_name": test_name,
        "method": "range",
        "details": {},
    }


def test_reference_exclusion_mask_excludes_same_source_context():
    """Exclude preceding failures from the same source and context."""
    result = reference_exclusion_mask(
        _data(), source_name="primary", contexts=["A"], preceding_failures=[_failure()]
    )

    assert result["A"].tolist() == [False, True, True, False, False, False]


def test_reference_filtering_uses_half_open_failure_periods():
    """Exclude the failure start but not its exclusive end timestamp."""
    result = reference_exclusion_mask(
        _data(),
        source_name="primary",
        contexts=["A"],
        preceding_failures=[
            _failure(start="2026-01-01T01:00:00Z", end="2026-01-01T02:00:00Z")
        ],
    )

    assert result["A"].tolist() == [False, True, False, False, False, False]


def test_reference_filtering_does_not_cross_sources():
    """Do not exclude failures belonging to another source."""
    result = reference_exclusion_mask(
        _data(),
        source_name="secondary",
        contexts=["A"],
        preceding_failures=[_failure(source="primary")],
    )

    assert not result["A"].any()


def test_reference_filtering_does_not_cross_contexts():
    """Do not exclude failures belonging to another context."""
    result = reference_exclusion_mask(
        _data(),
        source_name="primary",
        contexts=["B"],
        preceding_failures=[_failure(context="A")],
    )

    assert not result["B"].any()


def test_reference_filtering_reincludes_named_test():
    """Keep failures from explicitly included preceding tests eligible."""
    result = reference_exclusion_mask(
        _data(),
        source_name="primary",
        contexts=["A"],
        preceding_failures=[_failure(test_name="abrupt_change")],
        include_failed_periods_from=["abrupt_change"],
    )

    assert not result["A"].any()


def test_reference_filtering_still_excludes_other_overlapping_failures():
    """Require every relevant failure to be included before restoring a value."""
    failures = [_failure(test_name="abrupt_change"), _failure(test_name="flatline")]

    result = reference_exclusion_mask(
        _data(),
        source_name="primary",
        contexts=["A"],
        preceding_failures=failures,
        include_failed_periods_from=["abrupt_change"],
    )

    assert result["A"].tolist() == [False, True, True, False, False, False]


def test_build_reference_data_masks_excluded_observations():
    """Represent excluded reference observations as missing values."""
    result = build_reference_data(
        _data(), source_name="primary", contexts=["A"], preceding_failures=[_failure()]
    )

    assert result["A"].tolist()[:1] == [10]
    assert pd.isna(result.iloc[1, 0])
    assert pd.isna(result.iloc[2, 0])
    assert result["A"].tolist()[3:] == [40, 50, 60]


def test_build_reference_data_does_not_modify_target_data():
    """Keep target observations unchanged while filtering references."""
    data = _data()
    original = data.copy()

    build_reference_data(
        data, source_name="primary", contexts=["A"], preceding_failures=[_failure()]
    )

    pd.testing.assert_frame_equal(data, original)
