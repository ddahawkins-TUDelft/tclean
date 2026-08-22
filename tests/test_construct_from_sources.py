"""Tests for constructing profiles from source periods."""

import pandas as pd
import pytest

from tclean.advanced.methods.construct_from_sources import construct_from_sources


def test_construct_from_sources_remaps_single_source():
    """Remap one complete source period onto the target timestamps."""
    source_data_index = pd.date_range(
        "2025-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    source_data = pd.DataFrame({"GBR": [10.0, 20.0, 30.0]}, index=source_data_index)

    target_index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )

    sources = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2025-01-01T03:00:00Z"],
            "weight": [1.0],
        }
    )

    result = construct_from_sources(
        source_data,
        target_index=target_index,
        sources=sources,
        frequency=pd.Timedelta("1h"),
    )

    expected = pd.Series([10.0, 20.0, 30.0], index=target_index, dtype=float)

    pd.testing.assert_series_equal(result, expected, check_index_type=False)


def test_construct_from_sources_combines_weighted_sources():
    """Combine multiple source periods according to explicit weights."""
    source_data_index = pd.date_range(
        "2025-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    source_data = pd.DataFrame(
        {"GBR": [10.0, 20.0, 30.0], "FRA": [40.0, 50.0, 60.0]}, index=source_data_index
    )

    target_index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )

    sources = pd.DataFrame(
        {
            "context": ["GBR", "FRA"],
            "start": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "end": ["2025-01-01T03:00:00Z", "2025-01-01T03:00:00Z"],
            "weight": [1.0, 3.0],
        }
    )

    result = construct_from_sources(
        source_data,
        target_index=target_index,
        sources=sources,
        frequency=pd.Timedelta("1h"),
    )

    expected = pd.Series(
        [
            (10.0 * 1.0 + 40.0 * 3.0) / 4.0,
            (20.0 * 1.0 + 50.0 * 3.0) / 4.0,
            (30.0 * 1.0 + 60.0 * 3.0) / 4.0,
        ],
        index=target_index,
        dtype=float,
    )

    pd.testing.assert_series_equal(result, expected, check_index_type=False)


def test_construct_from_sources_rejects_unknown_context():
    """Reject a source context that is absent from source_data data."""
    source_data_index = pd.date_range(
        "2025-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    source_data = pd.DataFrame({"GBR": [10.0, 20.0, 30.0]}, index=source_data_index)

    target_index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )

    sources = pd.DataFrame(
        {
            "context": ["FRA"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2025-01-01T03:00:00Z"],
            "weight": [1.0],
        }
    )

    with pytest.raises(
        ValueError, match="source_data data do not contain requested context"
    ):
        construct_from_sources(
            source_data,
            target_index=target_index,
            sources=sources,
            frequency=pd.Timedelta("1h"),
        )


def test_construct_from_sources_rejects_missing_source_values():
    """Reject a source period containing missing source_data values."""
    source_data_index = pd.date_range(
        "2025-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    source_data = pd.DataFrame(
        {"GBR": [10.0, float("nan"), 30.0]}, index=source_data_index
    )

    target_index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )

    sources = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2025-01-01T03:00:00Z"],
            "weight": [1.0],
        }
    )

    with pytest.raises(
        ValueError, match="source_data source period contains missing values"
    ):
        construct_from_sources(
            source_data,
            target_index=target_index,
            sources=sources,
            frequency=pd.Timedelta("1h"),
        )


def test_construct_from_sources_rejects_length_mismatch():
    """Reject a source period whose aligned length differs from the target."""
    source_data_index = pd.date_range(
        "2025-01-01 00:00", periods=2, freq="h", tz="UTC", name="timestamp"
    )
    source_data = pd.DataFrame({"GBR": [10.0, 20.0]}, index=source_data_index)

    target_index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )

    sources = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2025-01-01T02:00:00Z"],
            "weight": [1.0],
        }
    )

    with pytest.raises(ValueError, match="must contain the same number of values"):
        construct_from_sources(
            source_data,
            target_index=target_index,
            sources=sources,
            frequency=pd.Timedelta("1h"),
        )


def test_construct_from_sources_removes_leap_day_for_non_leap_target():
    """Remove February 29 when mapping a leap-year source to a non-leap target."""
    source_data_index = pd.date_range(
        "2024-02-28 00:00", "2024-03-01 23:00", freq="h", tz="UTC", name="timestamp"
    )
    source_data = pd.DataFrame(
        {"GBR": range(len(source_data_index))}, index=source_data_index, dtype=float
    )

    target_index = pd.date_range(
        "2025-02-28 00:00", "2025-03-01 23:00", freq="h", tz="UTC", name="timestamp"
    )

    sources = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2024-02-28T00:00:00Z"],
            "end": ["2024-03-02T00:00:00Z"],
            "weight": [1.0],
        }
    )

    result = construct_from_sources(
        source_data,
        target_index=target_index,
        sources=sources,
        frequency=pd.Timedelta("1h"),
    )

    expected_values = pd.concat(
        [source_data.loc["2024-02-28", "GBR"], source_data.loc["2024-03-01", "GBR"]]
    ).to_numpy()

    expected = pd.Series(expected_values, index=target_index, dtype=float)

    pd.testing.assert_series_equal(result, expected, check_index_type=False)


def test_construct_from_sources_inserts_leap_day_for_leap_target():
    """Insert an averaged February 29 for a leap target from a non-leap source."""
    source_data_index = pd.date_range(
        "2025-02-28 00:00", "2025-03-01 23:00", freq="h", tz="UTC", name="timestamp"
    )

    feb_28 = [10.0] * 24
    march_1 = [30.0] * 24

    source_data = pd.DataFrame({"GBR": [*feb_28, *march_1]}, index=source_data_index)

    target_index = pd.date_range(
        "2024-02-28 00:00", "2024-03-01 23:00", freq="h", tz="UTC", name="timestamp"
    )

    sources = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-02-28T00:00:00Z"],
            "end": ["2025-03-02T00:00:00Z"],
            "weight": [1.0],
        }
    )

    result = construct_from_sources(
        source_data,
        target_index=target_index,
        sources=sources,
        frequency=pd.Timedelta("1h"),
    )

    expected = pd.Series(
        [*([10.0] * 24), *([20.0] * 24), *([30.0] * 24)],
        index=target_index,
        dtype=float,
    )

    pd.testing.assert_series_equal(result, expected, check_index_type=False)


def test_construct_from_sources_matches_reference_energy():
    """Scale the constructed profile to weighted reference-period energy."""
    source_data_index = pd.date_range(
        "2025-01-01 00:00", periods=6, freq="h", tz="UTC", name="timestamp"
    )

    source_data = pd.DataFrame(
        {"GBR": [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]}, index=source_data_index
    )

    target_index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )

    sources = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2025-01-01T03:00:00Z"],
            "weight": [1.0],
        }
    )

    scaling_sources = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T03:00:00Z"],
            "end": ["2025-01-01T06:00:00Z"],
            "weight": [1.0],
        }
    )

    result = construct_from_sources(
        source_data,
        target_index=target_index,
        sources=sources,
        scaling_sources=scaling_sources,
        frequency=pd.Timedelta("1h"),
    )

    expected = pd.Series([20.0, 20.0, 20.0], index=target_index, dtype=float)

    pd.testing.assert_series_equal(result, expected, check_index_type=False)


def test_construct_from_sources_uses_weighted_reference_energy():
    """Use explicit weights when combining scaling reference energies."""
    source_data_index = pd.date_range(
        "2025-01-01 00:00", periods=6, freq="h", tz="UTC", name="timestamp"
    )

    source_data = pd.DataFrame(
        {
            "GBR": [10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
            "FRA": [10.0, 10.0, 10.0, 40.0, 40.0, 40.0],
        },
        index=source_data_index,
    )

    target_index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )

    sources = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2025-01-01T03:00:00Z"],
            "weight": [1.0],
        }
    )

    scaling_sources = pd.DataFrame(
        {
            "context": ["GBR", "FRA"],
            "start": ["2025-01-01T03:00:00Z", "2025-01-01T03:00:00Z"],
            "end": ["2025-01-01T06:00:00Z", "2025-01-01T06:00:00Z"],
            "weight": [1.0, 3.0],
        }
    )

    result = construct_from_sources(
        source_data,
        target_index=target_index,
        sources=sources,
        scaling_sources=scaling_sources,
        frequency=pd.Timedelta("1h"),
    )

    expected_energy = ((60.0 * 1.0) + (120.0 * 3.0)) / 4.0

    assert result.sum() == pytest.approx(expected_energy)


def test_construct_from_sources_rejects_unknown_scaling_context():
    """Reject a scaling context that is absent from source_data data."""
    source_data_index = pd.date_range(
        "2025-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    source_data = pd.DataFrame({"GBR": [10.0, 20.0, 30.0]}, index=source_data_index)

    target_index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )

    sources = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2025-01-01T03:00:00Z"],
            "weight": [1.0],
        }
    )

    scaling_sources = pd.DataFrame(
        {
            "context": ["FRA"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2025-01-01T03:00:00Z"],
            "weight": [1.0],
        }
    )

    with pytest.raises(
        ValueError, match="source_data data do not contain requested scaling context"
    ):
        construct_from_sources(
            source_data,
            target_index=target_index,
            sources=sources,
            scaling_sources=scaling_sources,
            frequency=pd.Timedelta("1h"),
        )


def test_construct_from_sources_rejects_missing_scaling_values():
    """Reject a scaling reference period containing missing values."""
    source_data_index = pd.date_range(
        "2025-01-01 00:00", periods=6, freq="h", tz="UTC", name="timestamp"
    )

    source_data = pd.DataFrame(
        {"GBR": [10.0, 20.0, 30.0, 40.0, float("nan"), 60.0]}, index=source_data_index
    )

    target_index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )

    sources = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2025-01-01T03:00:00Z"],
            "weight": [1.0],
        }
    )

    scaling_sources = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T03:00:00Z"],
            "end": ["2025-01-01T06:00:00Z"],
            "weight": [1.0],
        }
    )

    with pytest.raises(
        ValueError, match="Scaling source period contains missing values"
    ):
        construct_from_sources(
            source_data,
            target_index=target_index,
            sources=sources,
            scaling_sources=scaling_sources,
            frequency=pd.Timedelta("1h"),
        )


def test_construct_from_sources_rejects_empty_scaling_period():
    """Reject a scaling reference period containing no source_data values."""
    source_data_index = pd.date_range(
        "2025-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    source_data = pd.DataFrame({"GBR": [10.0, 20.0, 30.0]}, index=source_data_index)

    target_index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )

    sources = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2025-01-01T03:00:00Z"],
            "weight": [1.0],
        }
    )

    scaling_sources = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2024-01-01T00:00:00Z"],
            "end": ["2024-01-01T03:00:00Z"],
            "weight": [1.0],
        }
    )

    with pytest.raises(ValueError, match="Scaling source period contains no values"):
        construct_from_sources(
            source_data,
            target_index=target_index,
            sources=sources,
            scaling_sources=scaling_sources,
            frequency=pd.Timedelta("1h"),
        )


def test_construct_from_sources_rejects_zero_energy_profile():
    """Reject energy matching when the constructed profile has zero value."""
    source_data_index = pd.date_range(
        "2025-01-01 00:00", periods=6, freq="h", tz="UTC", name="timestamp"
    )

    source_data = pd.DataFrame(
        {"GBR": [0.0, 0.0, 0.0, 10.0, 10.0, 10.0]}, index=source_data_index
    )

    target_index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )

    sources = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2025-01-01T03:00:00Z"],
            "weight": [1.0],
        }
    )

    scaling_sources = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T03:00:00Z"],
            "end": ["2025-01-01T06:00:00Z"],
            "weight": [1.0],
        }
    )

    with pytest.raises(ValueError, match="zero total value"):
        construct_from_sources(
            source_data,
            target_index=target_index,
            sources=sources,
            scaling_sources=scaling_sources,
            frequency=pd.Timedelta("1h"),
        )
