"""Tests for unresolved electricity-demand gap reports."""

import pandas as pd

from tclean.advanced.gap_report import build_gap_report

EXPECTED_COLUMNS = [
    "country",
    "gap_start",
    "gap_end",
    "gap_hours",
    "touches_start_boundary",
    "touches_end_boundary",
]


def test_build_gap_report_returns_empty_report_when_disabled():
    """Return the expected empty report when reporting is disabled."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    data = pd.DataFrame({"ALB": [10.0, float("nan"), 12.0]}, index=index)

    result = build_gap_report(data, enabled=False)

    assert result.empty
    assert result.columns.tolist() == EXPECTED_COLUMNS


def test_build_gap_report_returns_empty_report_without_gaps():
    """Return an empty report when no unresolved gaps remain."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    data = pd.DataFrame({"ALB": [10.0, 11.0, 12.0]}, index=index)

    result = build_gap_report(data, enabled=True)

    assert result.empty
    assert result.columns.tolist() == EXPECTED_COLUMNS


def test_build_gap_report_describes_internal_gap():
    """Describe the start, end, and duration of an internal gap."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=5, freq="h", tz="UTC", name="timestamp"
    )
    data = pd.DataFrame(
        {"ALB": [10.0, float("nan"), float("nan"), 13.0, 14.0]}, index=index
    )

    result = build_gap_report(data, enabled=True)

    assert len(result) == 1

    row = result.iloc[0]

    assert row["country"] == "ALB"
    assert row["gap_start"] == index[1]
    assert row["gap_end"] == index[3]
    assert row["gap_hours"] == 2
    assert not row["touches_start_boundary"]
    assert not row["touches_end_boundary"]


def test_build_gap_report_marks_start_boundary_gap():
    """Mark a gap that begins at the start of the available series."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=4, freq="h", tz="UTC", name="timestamp"
    )
    data = pd.DataFrame({"ALB": [float("nan"), float("nan"), 12.0, 13.0]}, index=index)

    result = build_gap_report(data, enabled=True)

    row = result.iloc[0]

    assert bool(row["touches_start_boundary"])
    assert not bool(row["touches_end_boundary"])


def test_build_gap_report_marks_end_boundary_gap():
    """Mark a gap that reaches the end of the available series."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=4, freq="h", tz="UTC", name="timestamp"
    )
    data = pd.DataFrame({"ALB": [10.0, 11.0, float("nan"), float("nan")]}, index=index)

    result = build_gap_report(data, enabled=True)

    row = result.iloc[0]

    assert not bool(row["touches_start_boundary"])
    assert bool(row["touches_end_boundary"])


def test_build_gap_report_separates_multiple_gaps():
    """Return separate rows for distinct contiguous missing runs."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=6, freq="h", tz="UTC", name="timestamp"
    )
    data = pd.DataFrame(
        {"ALB": [10.0, float("nan"), 12.0, 13.0, float("nan"), 15.0]}, index=index
    )

    result = build_gap_report(data, enabled=True)

    assert len(result) == 2
    assert result["gap_start"].tolist() == [index[1], index[4]]


def test_build_gap_report_sorts_by_country_and_gap_start():
    """Sort gap-report rows consistently across countries."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=4, freq="h", tz="UTC", name="timestamp"
    )
    data = pd.DataFrame(
        {
            "GBR": [10.0, float("nan"), 12.0, 13.0],
            "ALB": [20.0, 21.0, float("nan"), 23.0],
        },
        index=index,
    )

    result = build_gap_report(data, enabled=True)

    assert result["country"].tolist() == ["ALB", "GBR"]
