"""Tests for applying advanced auxiliary-fill rules."""

import pandas as pd
import pytest

from tclean.advanced.apply import apply_auxiliary_fill_rules


def _load() -> pd.DataFrame:
    """Return canonical test demand data."""
    index = pd.date_range(
        "2026-01-01 00:00", periods=4, freq="h", tz="UTC", name="timestamp"
    )

    return pd.DataFrame({"GBR": [10.0, float("nan"), 30.0, float("nan")]}, index=index)


def _methods(load: pd.DataFrame) -> pd.DataFrame:
    """Return provenance aligned with test demand."""
    return pd.DataFrame(
        {"GBR": ["observed", pd.NA, "observed", pd.NA]},
        index=load.index,
        dtype="string",
    )


def test_apply_constructed_profile_fills_only_gaps():
    """Fill missing values from an exactly aligned constructed profile."""
    load = _load()
    methods = _methods(load)

    rules = pd.DataFrame(
        {
            "rule_name": ["constructed"],
            "method": ["construct_from_sources"],
            "source": ["constructed"],
            "country": ["GBR"],
            "start": ["2026-01-01T00:00:00Z"],
            "end": ["2026-01-01T04:00:00Z"],
            "scope": ["fill_gaps"],
        }
    )

    profile = pd.Series([100.0, 110.0, 120.0, 130.0], index=load.index, dtype=float)

    filled, provenance = apply_auxiliary_fill_rules(
        load, methods, rules=rules, advanced_sources={"constructed": profile}
    )

    assert filled["GBR"].tolist() == [10.0, 110.0, 30.0, 130.0]

    assert provenance.loc[load.index[1], "GBR"] == "constructed"

    assert provenance.loc[load.index[3], "GBR"] == "constructed"


def test_apply_constructed_profile_overwrites_values():
    """Overwrite target values from an exactly aligned constructed profile."""
    load = _load()
    methods = _methods(load)

    rules = pd.DataFrame(
        {
            "rule_name": ["constructed"],
            "method": ["construct_from_sources"],
            "source": ["constructed"],
            "country": ["GBR"],
            "start": ["2026-01-01T00:00:00Z"],
            "end": ["2026-01-01T04:00:00Z"],
            "scope": ["overwrite"],
        }
    )

    profile = pd.Series([100.0, 110.0, 120.0, 130.0], index=load.index, dtype=float)

    filled, provenance = apply_auxiliary_fill_rules(
        load, methods, rules=rules, advanced_sources={"constructed": profile}
    )

    assert filled["GBR"].tolist() == [100.0, 110.0, 120.0, 130.0]

    assert provenance["GBR"].tolist() == [
        "constructed",
        "constructed",
        "constructed",
        "constructed",
    ]


def test_apply_constructed_profile_requires_exact_index():
    """Reject a constructed profile that does not exactly match its target."""
    load = _load()
    methods = _methods(load)

    rules = pd.DataFrame(
        {
            "rule_name": ["constructed"],
            "method": ["construct_from_sources"],
            "source": ["constructed"],
            "country": ["GBR"],
            "start": ["2026-01-01T00:00:00Z"],
            "end": ["2026-01-01T04:00:00Z"],
            "scope": ["fill_gaps"],
        }
    )

    profile = pd.Series([100.0, 110.0], index=load.index[:2], dtype=float)

    with pytest.raises(ValueError, match="must exactly match"):
        apply_auxiliary_fill_rules(
            load, methods, rules=rules, advanced_sources={"constructed": profile}
        )


def test_apply_external_profile_uses_overlap():
    """Apply only timestamps where an external profile overlaps target data."""
    load = _load()
    methods = _methods(load)

    rules = pd.DataFrame(
        {
            "rule_name": ["external"],
            "method": ["external_profile"],
            "source": ["external"],
            "country": ["GBR"],
            "start": ["2026-01-01T00:00:00Z"],
            "end": ["2026-01-01T04:00:00Z"],
            "scope": ["fill_gaps"],
        }
    )

    profile_index = pd.date_range(
        "2026-01-01 01:00", periods=2, freq="h", tz="UTC", name="timestamp"
    )

    profile = pd.Series([110.0, 120.0], index=profile_index, dtype=float)

    filled, provenance = apply_auxiliary_fill_rules(
        load, methods, rules=rules, advanced_sources={"external": profile}
    )

    assert filled.loc[load.index[1], "GBR"] == 110.0

    assert pd.isna(filled.loc[load.index[3], "GBR"])

    assert provenance.loc[load.index[1], "GBR"] == "external"


def test_apply_external_profile_can_overwrite_overlap():
    """Overwrite target values where an external profile overlaps."""
    load = _load()
    methods = _methods(load)

    rules = pd.DataFrame(
        {
            "rule_name": ["external"],
            "method": ["external_profile"],
            "source": ["external"],
            "country": ["GBR"],
            "start": ["2026-01-01T00:00:00Z"],
            "end": ["2026-01-01T04:00:00Z"],
            "scope": ["overwrite"],
        }
    )

    profile_index = pd.date_range(
        "2026-01-01 01:00", periods=2, freq="h", tz="UTC", name="timestamp"
    )

    profile = pd.Series([110.0, 120.0], index=profile_index, dtype=float)

    filled, provenance = apply_auxiliary_fill_rules(
        load, methods, rules=rules, advanced_sources={"external": profile}
    )

    assert filled.loc[load.index[1], "GBR"] == 110.0

    assert filled.loc[load.index[2], "GBR"] == 120.0

    assert provenance.loc[load.index[1], "GBR"] == "external"

    assert provenance.loc[load.index[2], "GBR"] == "external"


def test_apply_rules_rejects_missing_advanced_source():
    """Reject a rule whose referenced advanced source was not supplied."""
    load = _load()
    methods = _methods(load)

    rules = pd.DataFrame(
        {
            "rule_name": ["constructed"],
            "method": ["construct_from_sources"],
            "source": ["missing_source"],
            "country": ["GBR"],
            "start": ["2026-01-01T00:00:00Z"],
            "end": ["2026-01-01T04:00:00Z"],
            "scope": ["fill_gaps"],
        }
    )

    with pytest.raises(ValueError, match="Advanced sources must exactly match"):
        apply_auxiliary_fill_rules(load, methods, rules=rules, advanced_sources={})


def test_apply_rule_rejects_unknown_target_country():
    """Reject a rule targeting a country absent from demand data."""
    load = _load()
    methods = _methods(load)

    rules = pd.DataFrame(
        {
            "rule_name": ["external"],
            "method": ["external_profile"],
            "source": ["external"],
            "country": ["FRA"],
            "start": ["2026-01-01T00:00:00Z"],
            "end": ["2026-01-01T04:00:00Z"],
            "scope": ["fill_gaps"],
        }
    )

    profile = pd.Series([1.0, 2.0, 3.0, 4.0], index=load.index, dtype=float)

    with pytest.raises(ValueError, match="Target country 'FRA' is not present"):
        apply_auxiliary_fill_rules(
            load, methods, rules=rules, advanced_sources={"external": profile}
        )


def test_leave_missing_changes_nothing():
    """Leave demand and provenance unchanged for a leave-missing rule."""
    load = _load()
    methods = _methods(load)

    rules = pd.DataFrame(
        {
            "rule_name": ["leave"],
            "method": ["leave_missing"],
            "source": [None],
            "country": ["GBR"],
            "start": ["2026-01-01T00:00:00Z"],
            "end": ["2026-01-01T04:00:00Z"],
            "scope": ["fill_gaps"],
        }
    )

    filled, provenance = apply_auxiliary_fill_rules(
        load, methods, rules=rules, advanced_sources={}
    )

    pd.testing.assert_index_equal(filled.index, load.index, exact=False)

    pd.testing.assert_frame_equal(
        filled.reset_index(drop=True), load.reset_index(drop=True)
    )

    assert provenance.loc[load.index[0], "GBR"] == "observed"

    assert pd.isna(provenance.loc[load.index[1], "GBR"])


def test_advanced_rules_are_applied_sequentially():
    """Apply later advanced rules to the result of earlier rules."""
    load = _load()
    methods = _methods(load)

    rules = pd.DataFrame(
        {
            "rule_name": ["first", "second"],
            "method": ["external_profile", "external_profile"],
            "source": ["first", "second"],
            "country": ["GBR", "GBR"],
            "start": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "end": ["2026-01-01T04:00:00Z", "2026-01-01T04:00:00Z"],
            "scope": ["overwrite", "overwrite"],
        }
    )

    first = pd.Series([100.0, 100.0, 100.0, 100.0], index=load.index, dtype=float)

    second = pd.Series([200.0, 200.0, 200.0, 200.0], index=load.index, dtype=float)

    filled, provenance = apply_auxiliary_fill_rules(
        load, methods, rules=rules, advanced_sources={"first": first, "second": second}
    )

    assert filled["GBR"].tolist() == [200.0, 200.0, 200.0, 200.0]

    assert provenance["GBR"].tolist() == ["second", "second", "second", "second"]
