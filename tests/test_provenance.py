"""Tests for cleaning-method provenance utilities."""

import pandas as pd
import pytest

from tclean.provenance import (
    build_cleaning_method_ranks,
    combine_cleaning_rules,
    derive_cleaning_method_rank,
)


def test_build_cleaning_method_ranks_orders_sources_before_rules():
    """Rank observed sources before configured gap-filling rules."""
    ranks = build_cleaning_method_ranks(
        source_priority=["entsoe", "neso"],
        rules=[{"name": "linear"}, {"name": "copy_previous_week"}],
    )

    assert ranks == {
        "observed_entsoe": 0,
        "observed_neso": 1,
        "linear": 2,
        "copy_previous_week": 3,
        "missing": 4,
    }


def test_build_cleaning_method_ranks_handles_no_rules():
    """Place missing data after observed sources when no rules exist."""
    ranks = build_cleaning_method_ranks(source_priority=["entsoe"], rules=[])

    assert ranks == {"observed_entsoe": 0, "missing": 1}


def test_derive_cleaning_method_rank_maps_methods_to_integers():
    """Translate provenance labels into integer rank values."""
    cleaning_method = pd.DataFrame({"ALB": ["observed_entsoe", "linear", "missing"]})

    ranks = {"observed_entsoe": 0, "linear": 1, "missing": 2}

    result = derive_cleaning_method_rank(cleaning_method=cleaning_method, ranks=ranks)

    expected = pd.DataFrame({"ALB": [0, 1, 2]}, dtype="int16")

    pd.testing.assert_frame_equal(result, expected)


def test_derive_cleaning_method_rank_rejects_unknown_method():
    """Reject provenance labels that have no configured rank."""
    cleaning_method = pd.DataFrame({"ALB": ["observed_entsoe", "unknown_method"]})

    ranks = {"observed_entsoe": 0, "missing": 1}

    with pytest.raises(ValueError, match="No cleaning-method rank is defined"):
        derive_cleaning_method_rank(cleaning_method=cleaning_method, ranks=ranks)


def test_combine_cleaning_rules_preserves_rule_order():
    """Combine basic and advanced rules without changing their order."""
    basic_rules = [{"name": "linear"}, {"name": "copy_previous_week"}]
    advanced_rules = [{"name": "fill_alb_2022", "method": "construct_from_sources"}]

    result = combine_cleaning_rules(
        basic_rules=basic_rules, advanced_rules=advanced_rules
    )

    assert result == [
        {"name": "linear"},
        {"name": "copy_previous_week"},
        {"name": "fill_alb_2022", "method": "construct_from_sources"},
    ]


def test_combine_cleaning_rules_does_not_mutate_inputs():
    """Return copied rules so callers retain ownership of input mappings."""
    basic_rule = {"name": "linear"}

    result = combine_cleaning_rules(basic_rules=[basic_rule])

    result[0]["name"] = "changed"

    assert basic_rule["name"] == "linear"
