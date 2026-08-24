"""Tests for combining prepared sources."""

import pandas as pd
import pytest

from tclean.combine import combine_auxiliary_sources, combine_sources
from tclean.time_grid import TimeGrid


def _grid(
    frequency: str = "1h",
) -> TimeGrid:
    return TimeGrid(
        start="2026-01-01T00:00:00Z",
        end="2026-01-03T00:00:00Z",
        frequency=frequency,
    )


def _index() -> pd.DatetimeIndex:
    """Return a canonical test timestamp index."""
    return pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )


def test_combine_sources_uses_mapping_order_as_priority():
    """Use source mapping order to determine precedence."""
    index = _index()

    primary = pd.DataFrame({"GBR": [10.0, float("nan"), 30.0]}, index=index)

    secondary = pd.DataFrame({"GBR": [100.0, 200.0, 300.0]}, index=index)

    combined, data_source, cleaning_method = combine_sources(
        {"entsoe": primary, "opsd": secondary}, grid=_grid(),
    )

    assert combined["GBR"].tolist() == [10.0, 200.0, 30.0]

    assert data_source["GBR"].tolist() == ["entsoe", "opsd", "entsoe"]

    assert cleaning_method["GBR"].tolist() == [
        "observed_entsoe",
        "observed_opsd",
        "observed_entsoe",
    ]


def test_combine_sources_changes_priority_when_mapping_order_changes():
    """Treat earlier mapping entries as higher-priority sources."""
    index = _index()

    first = pd.DataFrame({"GBR": [10.0, 20.0, 30.0]}, index=index)

    second = pd.DataFrame({"GBR": [100.0, 200.0, 300.0]}, index=index)

    combined, _, _ = combine_sources(
        {"second": second, "first": first}, grid=_grid(),
    )

    assert combined["GBR"].tolist() == [100.0, 200.0, 300.0]


def test_combine_sources_rejects_empty_sources():
    """Reject combination when no sources are supplied."""
    with pytest.raises(ValueError, match="At least one time-series source"):
        combine_sources({}, grid=_grid(),)


def test_combine_auxiliary_sources_aligns_context_columns():
    """Align differing auxiliary context coverage before combination."""
    index = _index()

    entsoe = pd.DataFrame({"GBR": [10.0, 11.0, 12.0]}, index=index)

    opsd = pd.DataFrame({"FRA": [20.0, 21.0, 22.0]}, index=index)

    combined, data_source, _ = combine_auxiliary_sources(
        {"entsoe": entsoe, "opsd": opsd}, grid=_grid(),
    )

    assert combined.columns.tolist() == ["FRA", "GBR"]

    assert data_source["GBR"].tolist() == ["entsoe", "entsoe", "entsoe"]

    assert data_source["FRA"].tolist() == ["opsd", "opsd", "opsd"]


def test_combine_auxiliary_sources_skips_unavailable_configured_source():
    """Allow configured sources that supplied no auxiliary data."""
    index = _index()

    entsoe = pd.DataFrame({"GBR": [1.0, 2.0, 3.0]}, index=index)

    combined, _, _ = combine_auxiliary_sources(
        {"entsoe": entsoe}, grid=_grid(),
    )

    assert combined["GBR"].tolist() == [1.0, 2.0, 3.0]


def test_combine_auxiliary_sources_returns_empty_without_datas():
    """Return empty outputs when no auxiliary sources are available."""
    combined, data_source, cleaning_method = combine_auxiliary_sources(
        {}, grid=_grid(),
    )

    assert combined.empty
    assert data_source.empty
    assert cleaning_method.empty
