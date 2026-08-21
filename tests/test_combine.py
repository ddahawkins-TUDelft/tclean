"""Tests for combining prepared demand sources."""

import pandas as pd
import pytest

from tclean.combine import combine_auxiliary_sources, combine_sources


def _index() -> pd.DatetimeIndex:
    """Return a canonical test timestamp index."""
    return pd.date_range(
        "2026-01-01 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )


def test_combine_sources_uses_priority_order():
    """Use lower-priority values only where higher-priority data are missing."""
    index = _index()

    primary = pd.DataFrame({"GBR": [10.0, float("nan"), 30.0]}, index=index)

    secondary = pd.DataFrame({"GBR": [100.0, 200.0, 300.0]}, index=index)

    combined, data_source, cleaning_method = combine_sources(
        {"entsoe": primary, "opsd": secondary}, priority=["entsoe", "opsd"]
    )

    assert combined["GBR"].tolist() == [10.0, 200.0, 30.0]

    assert data_source["GBR"].tolist() == ["entsoe", "opsd", "entsoe"]

    assert cleaning_method["GBR"].tolist() == [
        "observed_entsoe",
        "observed_opsd",
        "observed_entsoe",
    ]


def test_combine_sources_preserves_unresolved_missing_values():
    """Leave values missing when no supplied source can provide them."""
    index = _index()

    first = pd.DataFrame({"GBR": [10.0, float("nan"), 30.0]}, index=index)

    second = pd.DataFrame({"GBR": [100.0, float("nan"), 300.0]}, index=index)

    combined, data_source, cleaning_method = combine_sources(
        {"first": first, "second": second}, priority=["first", "second"]
    )

    assert pd.isna(combined.loc[index[1], "GBR"])
    assert pd.isna(data_source.loc[index[1], "GBR"])
    assert pd.isna(cleaning_method.loc[index[1], "GBR"])


def test_combine_sources_rejects_missing_priority_source():
    """Reject a priority that omits a supplied source."""
    index = _index()

    load = pd.DataFrame({"GBR": [1.0, 2.0, 3.0]}, index=index)

    with pytest.raises(ValueError, match="must contain every supplied source"):
        combine_sources({"entsoe": load, "opsd": load.copy()}, priority=["entsoe"])


def test_combine_sources_rejects_unknown_priority_source():
    """Reject a priority containing a source that was not supplied."""
    index = _index()

    load = pd.DataFrame({"GBR": [1.0, 2.0, 3.0]}, index=index)

    with pytest.raises(ValueError, match="must contain every supplied source"):
        combine_sources({"entsoe": load}, priority=["entsoe", "opsd"])


def test_combine_sources_rejects_duplicate_priority():
    """Reject duplicate entries in source priority."""
    index = _index()

    load = pd.DataFrame({"GBR": [1.0, 2.0, 3.0]}, index=index)

    with pytest.raises(ValueError, match="must not contain duplicate"):
        combine_sources({"entsoe": load}, priority=["entsoe", "entsoe"])


def test_combine_sources_rejects_misaligned_index():
    """Reject prepared sources with different timestamp grids."""
    first_index = _index()

    second_index = pd.date_range(
        "2026-01-02 00:00", periods=3, freq="h", tz="UTC", name="timestamp"
    )

    first = pd.DataFrame({"GBR": [1.0, 2.0, 3.0]}, index=first_index)

    second = pd.DataFrame({"GBR": [4.0, 5.0, 6.0]}, index=second_index)

    with pytest.raises(ValueError, match="same timestamp index"):
        combine_sources(
            {"first": first, "second": second}, priority=["first", "second"]
        )


def test_combine_sources_rejects_misaligned_columns():
    """Reject primary sources with different target country columns."""
    index = _index()

    first = pd.DataFrame({"GBR": [1.0, 2.0, 3.0]}, index=index)

    second = pd.DataFrame({"FRA": [4.0, 5.0, 6.0]}, index=index)

    with pytest.raises(ValueError, match="same country columns"):
        combine_sources(
            {"first": first, "second": second}, priority=["first", "second"]
        )


def test_combine_auxiliary_sources_aligns_country_columns():
    """Align differing auxiliary country coverage before combination."""
    index = _index()

    entsoe = pd.DataFrame({"GBR": [10.0, 11.0, 12.0]}, index=index)

    opsd = pd.DataFrame({"FRA": [20.0, 21.0, 22.0]}, index=index)

    combined, data_source, _ = combine_auxiliary_sources(
        {"entsoe": entsoe, "opsd": opsd}, priority=["entsoe", "opsd"]
    )

    assert combined.columns.tolist() == ["FRA", "GBR"]

    assert data_source["GBR"].tolist() == ["entsoe", "entsoe", "entsoe"]

    assert data_source["FRA"].tolist() == ["opsd", "opsd", "opsd"]


def test_combine_auxiliary_sources_skips_unavailable_configured_source():
    """Allow configured sources that supplied no auxiliary data."""
    index = _index()

    entsoe = pd.DataFrame({"GBR": [1.0, 2.0, 3.0]}, index=index)

    combined, _, _ = combine_auxiliary_sources(
        {"entsoe": entsoe}, priority=["neso", "entsoe", "opsd"]
    )

    assert combined["GBR"].tolist() == [1.0, 2.0, 3.0]


def test_combine_auxiliary_sources_returns_empty_without_loads():
    """Return empty outputs when no auxiliary sources are available."""
    combined, data_source, cleaning_method = combine_auxiliary_sources(
        {}, priority=["entsoe", "opsd"]
    )

    assert combined.empty
    assert data_source.empty
    assert cleaning_method.empty
