"""Tests for constructing demand profiles from source periods."""

import pandas as pd
import pytest

from tclean.advanced.methods.construct_from_sources import construct_from_sources


def test_construct_from_sources_remaps_single_source():
    """Remap one complete source period onto the target timestamps."""
    auxiliary_index = pd.date_range(
        "2025-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    auxiliary = pd.DataFrame({"GBR": [10.0, 20.0, 30.0]}, index=auxiliary_index)

    target_index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )

    sources = pd.DataFrame(
        {
            "country": ["GBR"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2025-01-01T03:00:00Z"],
            "weight": [1.0],
        }
    )

    result = construct_from_sources(
        auxiliary, target_index=target_index, sources=sources
    )

    expected = pd.Series([10.0, 20.0, 30.0], index=target_index, dtype=float)

    pd.testing.assert_series_equal(result, expected, check_index_type=False)


def test_construct_from_sources_combines_weighted_sources():
    """Combine multiple source periods according to explicit weights."""
    auxiliary_index = pd.date_range(
        "2025-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    auxiliary = pd.DataFrame(
        {"GBR": [10.0, 20.0, 30.0], "FRA": [40.0, 50.0, 60.0]}, index=auxiliary_index
    )

    target_index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )

    sources = pd.DataFrame(
        {
            "country": ["GBR", "FRA"],
            "start": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "end": ["2025-01-01T03:00:00Z", "2025-01-01T03:00:00Z"],
            "weight": [1.0, 3.0],
        }
    )

    result = construct_from_sources(
        auxiliary, target_index=target_index, sources=sources
    )

    expected = pd.Series(
        [
            (10.0 * 1 + 40.0 * 3) / 4,
            (20.0 * 1 + 50.0 * 3) / 4,
            (30.0 * 1 + 60.0 * 3) / 4,
        ],
        index=target_index,
        dtype=float,
    )

    pd.testing.assert_series_equal(result, expected, check_index_type=False)


def test_construct_from_sources_rejects_unknown_country():
    """Reject a source country that is absent from auxiliary data."""
    auxiliary_index = pd.date_range(
        "2025-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    auxiliary = pd.DataFrame({"GBR": [10.0, 20.0, 30.0]}, index=auxiliary_index)

    target_index = pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )

    sources = pd.DataFrame(
        {
            "country": ["FRA"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2025-01-01T03:00:00Z"],
            "weight": [1.0],
        }
    )

    with pytest.raises(
        ValueError, match="Auxiliary data do not contain requested country"
    ):
        construct_from_sources(auxiliary, target_index=target_index, sources=sources)
