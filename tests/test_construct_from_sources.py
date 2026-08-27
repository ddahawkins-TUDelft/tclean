"""Tests for constructing profiles from source periods."""

import pandas as pd
import pytest

from tclean.advanced.methods.construct_from_sources import construct_from_sources
from tclean.time_grid import TimeGrid


def _grid(frequency: str = "1h") -> TimeGrid:
    return TimeGrid(
        start="2024-01-01T00:00:00Z", end="2027-01-01T00:00:00Z", frequency=frequency
    )


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
        source_data, target_index=target_index, sources=sources, grid=_grid()
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
        source_data, target_index=target_index, sources=sources, grid=_grid()
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
            grid=_grid(),
            scaling_method="match_total",
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
            source_data, target_index=target_index, sources=sources, grid=_grid()
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
            source_data, target_index=target_index, sources=sources, grid=_grid()
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
        source_data, target_index=target_index, sources=sources, grid=_grid()
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
        source_data, target_index=target_index, sources=sources, grid=_grid()
    )

    expected = pd.Series(
        [*([10.0] * 24), *([20.0] * 24), *([30.0] * 24)],
        index=target_index,
        dtype=float,
    )

    pd.testing.assert_series_equal(result, expected, check_index_type=False)


def test_construct_from_sources_matches_reference_total():
    """Scale the constructed profile to the weighted reference-period total."""
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
        scaling_method="match_total",
        scaling_sources=scaling_sources,
        grid=_grid(),
    )

    expected = pd.Series([20.0, 20.0, 20.0], index=target_index, dtype=float)

    pd.testing.assert_series_equal(result, expected, check_index_type=False)


def test_construct_from_sources_uses_weighted_reference_total():
    """Use explicit weights when combining scaling reference totals."""
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
        scaling_method="match_total",
        scaling_sources=scaling_sources,
        grid=_grid(),
    )

    expected_total = ((60.0 * 1.0) + (120.0 * 3.0)) / 4.0

    assert result.sum() == pytest.approx(expected_total)


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
            scaling_method="match_total",
            grid=_grid(),
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
            scaling_method="match_total",
            grid=_grid(),
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
            scaling_method="match_total",
            grid=_grid(),
        )


def test_construct_from_sources_rejects_zero_total_profile():
    """Reject total matching when the constructed profile has zero total."""
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
            scaling_method="match_total",
            grid=_grid(),
        )


def test_construct_from_sources_normalises_mean():
    """Scale a constructed profile so that its arithmetic mean equals one."""
    source_index = pd.date_range(
        "2025-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    source_data = pd.DataFrame({"GBR": [1.0, 2.0, 3.0]}, index=source_index)

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
        scaling_method="normalise_mean",
        grid=_grid(),
    )

    expected = pd.Series([0.5, 1.0, 1.5], index=target_index, dtype=float)

    pd.testing.assert_series_equal(result, expected, check_index_type=False)
    assert result.mean() == pytest.approx(1.0)


def test_construct_from_sources_normalises_max():
    """Scale a constructed profile so that its maximum equals one."""
    source_index = pd.date_range(
        "2025-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    source_data = pd.DataFrame({"GBR": [2.0, 4.0, 8.0]}, index=source_index)

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
        scaling_method="normalise_max",
        grid=_grid(),
    )

    expected = pd.Series([0.25, 0.5, 1.0], index=target_index, dtype=float)

    pd.testing.assert_series_equal(result, expected, check_index_type=False)
    assert result.max() == pytest.approx(1.0)


def test_construct_from_sources_normalise_max_allows_negative_values():
    """Allow negative values when the constructed profile maximum is positive."""
    source_index = pd.date_range(
        "2025-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    source_data = pd.DataFrame({"GBR": [-2.0, 4.0, 8.0]}, index=source_index)

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
        scaling_method="normalise_max",
        grid=_grid(),
    )

    expected = pd.Series([-0.25, 0.5, 1.0], index=target_index, dtype=float)

    pd.testing.assert_series_equal(result, expected, check_index_type=False)


def test_construct_from_sources_rejects_zero_mean_normalisation():
    """Reject mean normalisation when the constructed profile mean is zero."""
    source_index = pd.date_range(
        "2025-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    source_data = pd.DataFrame({"GBR": [-1.0, 0.0, 1.0]}, index=source_index)

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

    with pytest.raises(ValueError, match="zero mean"):
        construct_from_sources(
            source_data,
            target_index=target_index,
            sources=sources,
            scaling_method="normalise_mean",
            grid=_grid(),
        )


@pytest.mark.parametrize("values", [[0.0, 0.0, 0.0], [-3.0, -2.0, -1.0]])
def test_construct_from_sources_rejects_non_positive_max_normalisation(values):
    """Reject maximum normalisation when the profile maximum is not positive."""
    source_index = pd.date_range(
        "2025-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    source_data = pd.DataFrame({"GBR": values}, index=source_index)

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

    with pytest.raises(ValueError, match="maximum is not positive"):
        construct_from_sources(
            source_data,
            target_index=target_index,
            sources=sources,
            scaling_method="normalise_max",
            grid=_grid(),
        )


def test_construct_from_sources_rejects_match_total_without_scaling_sources():
    """Require reference periods when using total-matching scaling."""
    source_index = pd.date_range(
        "2025-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    source_data = pd.DataFrame({"GBR": [10.0, 20.0, 30.0]}, index=source_index)

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

    with pytest.raises(ValueError, match="requires scaling_sources"):
        construct_from_sources(
            source_data,
            target_index=target_index,
            sources=sources,
            scaling_method="match_total",
            grid=_grid(),
        )


@pytest.mark.parametrize("scaling_method", ["normalise_mean", "normalise_max"])
def test_construct_from_sources_rejects_scaling_sources_for_normalisation(
    scaling_method,
):
    """Reject reference periods for normalisation methods that do not use them."""
    source_index = pd.date_range(
        "2025-01-01 00:00", periods=6, freq="h", tz="UTC", name="timestamp"
    )
    source_data = pd.DataFrame(
        {"GBR": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]}, index=source_index
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

    with pytest.raises(ValueError, match="does not use scaling_sources"):
        construct_from_sources(
            source_data,
            target_index=target_index,
            sources=sources,
            scaling_method=scaling_method,
            scaling_sources=scaling_sources,
            grid=_grid(),
        )


def test_construct_from_sources_rejects_scaling_sources_without_method():
    """Reject reference periods when no scaling method is configured."""
    source_index = pd.date_range(
        "2025-01-01 00:00", periods=6, freq="h", tz="UTC", name="timestamp"
    )
    source_data = pd.DataFrame(
        {"GBR": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]}, index=source_index
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

    with pytest.raises(ValueError, match="without a scaling_method"):
        construct_from_sources(
            source_data,
            target_index=target_index,
            sources=sources,
            scaling_sources=scaling_sources,
            grid=_grid(),
        )


def test_construct_from_sources_rejects_unknown_scaling_method():
    """Reject unsupported constructed-profile scaling methods."""
    source_index = pd.date_range(
        "2025-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )
    source_data = pd.DataFrame({"GBR": [10.0, 20.0, 30.0]}, index=source_index)

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

    with pytest.raises(ValueError, match="Unsupported scaling method"):
        construct_from_sources(
            source_data,
            target_index=target_index,
            sources=sources,
            scaling_method="mystery_scaling",
            grid=_grid(),
        )


def test_construct_from_sources_rejects_incompatible_leap_day_frequency():
    """Reject leap-day alignment when frequency does not divide one day."""
    source_data_index = pd.date_range(
        "2025-02-28T01:00:00Z", periods=10, freq="5h", name="timestamp"
    )

    source_data = pd.DataFrame({"GBR": range(10)}, index=source_data_index, dtype=float)

    target_index = pd.date_range(
        "2024-02-28T00:00:00Z", periods=10, freq="5h", name="timestamp"
    )

    sources = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": [source_data_index[0]],
            "end": [source_data_index[-1] + pd.Timedelta("5h")],
            "weight": [1.0],
        }
    )

    grid = TimeGrid(
        start="2024-02-28T00:00:00Z", end="2024-03-04T00:00:00Z", frequency="5h"
    )

    with pytest.raises(ValueError, match="divide one calendar day exactly"):
        construct_from_sources(
            source_data, target_index=target_index, sources=sources, grid=grid
        )


def test_construct_from_sources_allows_non_daily_frequency_without_leap_adjustment():
    """Allow non-daily-divisible frequencies when leap adjustment is unnecessary."""
    source_index = pd.date_range(
        "2025-01-01T02:00:00Z", periods=3, freq="5h", name="timestamp"
    )

    source_data = pd.DataFrame({"GBR": [10.0, 20.0, 30.0]}, index=source_index)

    target_index = pd.date_range(
        "2026-01-01T02:00:00Z", periods=3, freq="5h", name="timestamp"
    )

    sources = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": [source_index[0]],
            "end": [source_index[-1] + pd.Timedelta("5h")],
            "weight": [1.0],
        }
    )

    grid = TimeGrid(
        start=target_index[0], end=target_index[-1] + pd.Timedelta("5h"), frequency="5h"
    )

    result = construct_from_sources(
        source_data, target_index=target_index, sources=sources, grid=grid
    )

    assert result.tolist() == [10.0, 20.0, 30.0]
