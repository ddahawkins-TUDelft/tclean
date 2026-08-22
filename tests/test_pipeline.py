"""Tests for the high-level cleaning pipeline."""

import pandas as pd
import pytest

from tclean import TCleanConfig
from tclean.pipeline import clean

config = TCleanConfig(frequency="1h")


def _index() -> pd.DatetimeIndex:
    """Return a canonical test timestamp index."""
    return pd.date_range(
        "2026-01-01 00:00", periods=5, freq="h", tz="UTC", name="timestamp"
    )


def test_clean_combines_sources_without_cleaning_rules():
    """Combine sources when no cleaning rules are supplied."""
    index = _index()

    primary = pd.DataFrame(
        {"GBR": [10.0, float("nan"), 30.0, float("nan"), 50.0]}, index=index
    )

    secondary = pd.DataFrame({"GBR": [100.0, 200.0, 300.0, 400.0, 500.0]}, index=index)

    cleaned, data_source, cleaning_method = clean(
        {"primary": primary, "secondary": secondary}, config=config
    )

    assert cleaned["GBR"].tolist() == [10.0, 200.0, 30.0, 400.0, 50.0]

    assert data_source["GBR"].tolist() == [
        "primary",
        "secondary",
        "primary",
        "secondary",
        "primary",
    ]

    assert cleaning_method["GBR"].tolist() == [
        "observed_primary",
        "observed_secondary",
        "observed_primary",
        "observed_secondary",
        "observed_primary",
    ]


def test_clean_applies_basic_rules_after_combination():
    """Apply basic cleaning after combining primary sources."""
    index = _index()

    data = pd.DataFrame({"GBR": [10.0, float("nan"), 30.0, 40.0, 50.0]}, index=index)

    basic_rules = [
        {
            "name": "interpolate_short_gaps",
            "method": "linear_interpolation",
            "max_gap": "1h",
        }
    ]

    cleaned, _, cleaning_method = clean(
        {"primary": data}, basic_rules=basic_rules, config=config
    )

    assert cleaned.loc[index[1], "GBR"] == pytest.approx(20.0)

    assert pd.notna(cleaning_method.loc[index[1], "GBR"])


def test_clean_applies_advanced_rules_after_basic_rules():
    """Apply advanced cleaning after the basic cleaning stage."""
    index = _index()

    data = pd.DataFrame(
        {"GBR": [10.0, float("nan"), float("nan"), 40.0, 50.0]}, index=index
    )

    basic_rules = [
        {
            "name": "interpolate_short_gaps",
            "method": "linear_interpolation",
            "max_gap": "1h",
        }
    ]

    advanced_rules = pd.DataFrame(
        {
            "rule_name": ["advanced_fill"],
            "method": ["external_profile"],
            "source": ["fallback"],
            "context": ["GBR"],
            "start": ["2026-01-01T00:00:00Z"],
            "end": ["2026-01-01T05:00:00Z"],
            "scope": ["fill_gaps"],
        }
    )

    fallback = pd.Series(
        [100.0, 200.0, 300.0, 400.0, 500.0], index=index, name="fallback"
    )

    cleaned, _, cleaning_method = clean(
        {"primary": data},
        basic_rules=basic_rules,
        advanced_rules=advanced_rules,
        advanced_sources={"fallback": fallback},
        config=config,
    )

    assert cleaned.loc[index[1], "GBR"] == pytest.approx(200.0)

    assert cleaned.loc[index[2], "GBR"] == pytest.approx(300.0)

    assert cleaning_method.loc[index[1], "GBR"] == "advanced_fill"

    assert cleaning_method.loc[index[2], "GBR"] == "advanced_fill"


def test_clean_rejects_advanced_sources_without_rules():
    """Reject supplied advanced sources when no advanced rules exist."""
    index = _index()

    data = pd.DataFrame({"GBR": [10.0, 20.0, 30.0, 40.0, 50.0]}, index=index)

    source = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=index)

    with pytest.raises(ValueError, match="without advanced rules"):
        clean({"primary": data}, advanced_sources={"unused": source}, config=config)


def test_clean_allows_leave_missing_without_advanced_sources():
    """Allow source-free advanced rules such as leave_missing."""
    index = _index()

    data = pd.DataFrame({"GBR": [10.0, float("nan"), 30.0, 40.0, 50.0]}, index=index)

    advanced_rules = pd.DataFrame(
        {
            "rule_name": ["leave_gap"],
            "method": ["leave_missing"],
            "source": [None],
            "context": ["GBR"],
            "start": ["2026-01-01T00:00:00Z"],
            "end": ["2026-01-01T05:00:00Z"],
            "scope": ["fill_gaps"],
        }
    )

    cleaned, _, _ = clean(
        {"primary": data}, advanced_rules=advanced_rules, config=config
    )

    assert pd.isna(cleaned.loc[index[1], "GBR"])


def test_clean_rejects_unused_advanced_source():
    """Reject advanced sources not referenced by any rule."""
    index = _index()

    data = pd.DataFrame({"GBR": [10.0, 20.0, 30.0, 40.0, 50.0]}, index=index)

    advanced_rules = pd.DataFrame(
        {
            "rule_name": ["leave"],
            "method": ["leave_missing"],
            "source": [None],
            "context": ["GBR"],
            "start": ["2026-01-01T00:00:00Z"],
            "end": ["2026-01-01T05:00:00Z"],
            "scope": ["fill_gaps"],
        }
    )

    unused = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=index)

    with pytest.raises(ValueError, match="Advanced sources must exactly match"):
        clean(
            {"primary": data},
            advanced_rules=advanced_rules,
            advanced_sources={"unused": unused},
            config=config,
        )
