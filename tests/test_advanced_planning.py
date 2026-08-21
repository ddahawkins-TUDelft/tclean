"""Tests for auxiliary-data planning."""

import pandas as pd
import pandera.errors
import pytest

from tclean.advanced.planning import (
    build_auxiliary_source_requests,
    compile_auxiliary_requirements,
    expand_auxiliary_requirements,
    get_basic_cleaning_context,
    select_active_advanced_rules,
)
from tclean.validation import validate_source_capabilities


def test_compile_auxiliary_requirements_returns_empty_for_no_sources():
    """Return an empty canonical requirements table when no sources exist."""
    result = compile_auxiliary_requirements([])

    assert result.empty
    assert result.columns.tolist() == ["context", "start", "end"]


def test_compile_auxiliary_requirements_preserves_distinct_periods():
    """Preserve non-overlapping requirements for the same context."""
    first = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2025-01-02T00:00:00Z"],
            "weight": [1.0],
        }
    )

    second = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-02-01T00:00:00Z"],
            "end": ["2025-02-02T00:00:00Z"],
            "weight": [1.0],
        }
    )

    result = compile_auxiliary_requirements([first, second])

    assert len(result) == 2


def test_compile_auxiliary_requirements_merges_overlapping_periods():
    """Merge overlapping requirements for the same context."""
    first = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2025-01-03T00:00:00Z"],
            "weight": [1.0],
        }
    )

    second = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-02T00:00:00Z"],
            "end": ["2025-01-04T00:00:00Z"],
            "weight": [1.0],
        }
    )

    result = compile_auxiliary_requirements([first, second])

    assert len(result) == 1
    assert result.loc[0, "start"] == pd.Timestamp("2025-01-01T00:00:00Z")
    assert result.loc[0, "end"] == pd.Timestamp("2025-01-04T00:00:00Z")


def test_compile_auxiliary_requirements_merges_adjacent_periods():
    """Merge adjacent requirements without an hourly gap."""
    first = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2025-01-02T00:00:00Z"],
            "weight": [1.0],
        }
    )

    second = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-02T00:00:00Z"],
            "end": ["2025-01-03T00:00:00Z"],
            "weight": [1.0],
        }
    )

    result = compile_auxiliary_requirements([first, second])

    assert len(result) == 1
    assert result.loc[0, "start"] == pd.Timestamp("2025-01-01T00:00:00Z")
    assert result.loc[0, "end"] == pd.Timestamp("2025-01-03T00:00:00Z")


def test_compile_auxiliary_requirements_keeps_contexts_separate():
    """Do not merge periods belonging to different contexts."""
    sources = pd.DataFrame(
        {
            "context": ["GBR", "FRA"],
            "start": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "end": ["2025-01-03T00:00:00Z", "2025-01-03T00:00:00Z"],
            "weight": [1.0, 1.0],
        }
    )

    result = compile_auxiliary_requirements([sources])

    assert result["context"].tolist() == ["FRA", "GBR"]


def test_compile_auxiliary_requirements_deduplicates_periods():
    """Remove duplicate auxiliary requirements."""
    sources = pd.DataFrame(
        {
            "context": ["GBR", "GBR"],
            "start": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "end": ["2025-01-02T00:00:00Z", "2025-01-02T00:00:00Z"],
            "weight": [1.0, 1.0],
        }
    )

    result = compile_auxiliary_requirements([sources])

    assert len(result) == 1


def test_compile_auxiliary_requirements_validates_source_periods():
    """Reject invalid source-period definitions before planning."""
    sources = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": ["2025-01-01T00:00:00Z"],
            "end": ["2025-01-02T00:00:00Z"],
        }
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        compile_auxiliary_requirements([sources])


def test_get_basic_cleaning_context_returns_zero_without_rules():
    """Require no surrounding context when no basic rules are configured."""
    left, right = get_basic_cleaning_context([])

    assert left == pd.Timedelta(0)
    assert right == pd.Timedelta(0)


def test_get_basic_cleaning_context_for_linear_interpolation():
    """Include gap-classification context for linear interpolation."""
    rules = [{"method": "linear_interpolation", "max_gap": "3h"}]

    left, right = get_basic_cleaning_context(rules)

    assert left == pd.Timedelta(hours=3)
    assert right == pd.Timedelta(hours=3)


def test_get_basic_cleaning_context_for_previous_period_copy():
    """Include backward source context required by a copy rule."""
    rules = [{"method": "copy_periods", "max_gap": "2h", "source_offset": "-24h"}]

    left, right = get_basic_cleaning_context(rules)

    assert left == pd.Timedelta(hours=24)
    assert right == pd.Timedelta(hours=2)


def test_get_basic_cleaning_context_for_following_period_copy():
    """Include forward source context required by a copy rule."""
    rules = [{"method": "copy_periods", "max_gap": "2h", "source_offset": "24h"}]

    left, right = get_basic_cleaning_context(rules)

    assert left == pd.Timedelta(hours=2)
    assert right == pd.Timedelta(hours=24)


def test_get_basic_cleaning_context_for_average_periods():
    """Include context for all offsets used by an averaging rule."""
    rules = [
        {
            "method": "average_periods",
            "max_gap": "3h",
            "source_offsets": ["-24h", "24h"],
        }
    ]

    left, right = get_basic_cleaning_context(rules)

    assert left == pd.Timedelta(hours=24)
    assert right == pd.Timedelta(hours=24)


def test_get_basic_cleaning_context_accumulates_across_rules():
    """Accumulate context through ordered basic cleaning rules."""
    rules = [
        {"method": "copy_periods", "max_gap": "2h", "source_offset": "-24h"},
        {"method": "copy_periods", "max_gap": "2h", "source_offset": "-168h"},
    ]

    left, right = get_basic_cleaning_context(rules)

    assert left == pd.Timedelta(hours=192)
    assert right == pd.Timedelta(hours=2)


def test_get_basic_cleaning_context_rejects_unknown_method():
    """Reject unsupported basic cleaning methods."""
    rules = [{"method": "unknown", "max_gap": "2h"}]

    with pytest.raises(ValueError, match="Unsupported basic gap-filling method"):
        get_basic_cleaning_context(rules)


def test_expand_auxiliary_requirements_adds_context():
    """Expand exact requirements by basic-cleaning context."""
    requirements = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": [pd.Timestamp("2025-01-10T00:00:00Z")],
            "end": [pd.Timestamp("2025-01-11T00:00:00Z")],
        }
    )

    rules = [{"method": "copy_periods", "max_gap": "2h", "source_offset": "-24h"}]

    result = expand_auxiliary_requirements(requirements, rules=rules)

    assert result.loc[0, "start"] == pd.Timestamp("2025-01-09T00:00:00Z")

    assert result.loc[0, "end"] == pd.Timestamp("2025-01-11T02:00:00Z")


def test_expand_auxiliary_requirements_preserves_exact_period_when_disabled():
    """Do not add cleaning context when basic cleaning is disabled."""
    requirements = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": [pd.Timestamp("2025-01-10T00:00:00Z")],
            "end": [pd.Timestamp("2025-01-11T00:00:00Z")],
        }
    )

    rules = [{"method": "copy_periods", "max_gap": "2h", "source_offset": "-24h"}]

    result = expand_auxiliary_requirements(requirements, rules=rules, enabled=False)

    pd.testing.assert_frame_equal(result, requirements, check_dtype=False)


def test_expand_auxiliary_requirements_merges_after_expansion():
    """Merge requirements that overlap after context is added."""
    requirements = pd.DataFrame(
        {
            "context": ["GBR", "GBR"],
            "start": [
                pd.Timestamp("2025-01-01T00:00:00Z"),
                pd.Timestamp("2025-01-03T00:00:00Z"),
            ],
            "end": [
                pd.Timestamp("2025-01-02T00:00:00Z"),
                pd.Timestamp("2025-01-04T00:00:00Z"),
            ],
        }
    )

    rules = [
        {
            "method": "average_periods",
            "max_gap": "1h",
            "source_offsets": ["-24h", "24h"],
        }
    ]

    result = expand_auxiliary_requirements(requirements, rules=rules)

    assert len(result) == 1

    assert result.loc[0, "start"] == pd.Timestamp("2024-12-31T00:00:00Z")

    assert result.loc[0, "end"] == pd.Timestamp("2025-01-05T00:00:00Z")


def test_validate_source_capabilities_accepts_explicit_context():
    """Accept a source with explicit context coverage."""
    capabilities = pd.DataFrame({"source": ["neso"], "context": ["GBR"]})

    result = validate_source_capabilities(capabilities)

    assert result.loc[0, "source"] == "neso"
    assert result.loc[0, "context"] == "GBR"


def test_validate_source_capabilities_accepts_all_context_wildcard():
    """Accept a missing context as wildcard coverage."""
    capabilities = pd.DataFrame({"source": ["entsoe"], "context": [None]})

    result = validate_source_capabilities(capabilities)

    assert result.loc[0, "source"] == "entsoe"
    assert pd.isna(result.loc[0, "context"])


def test_validate_source_capabilities_rejects_duplicates():
    """Reject duplicate source-context capability definitions."""
    capabilities = pd.DataFrame({"source": ["neso", "neso"], "context": ["GBR", "GBR"]})

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_source_capabilities(capabilities)


def test_validate_source_capabilities_rejects_mixed_wildcard_and_explicit():
    """Reject mixed wildcard and explicit coverage for one source."""
    capabilities = pd.DataFrame(
        {"source": ["entsoe", "entsoe"], "context": [None, "GBR"]}
    )

    with pytest.raises(pandera.errors.SchemaErrors):
        validate_source_capabilities(capabilities)


def test_build_auxiliary_source_requests_maps_explicit_capability():
    """Map a requirement to an explicitly capable source."""
    requirements = pd.DataFrame(
        {
            "context": ["GBR"],
            "start": [pd.Timestamp("2025-01-01T00:00:00Z")],
            "end": [pd.Timestamp("2025-01-02T00:00:00Z")],
        }
    )

    capabilities = pd.DataFrame({"source": ["neso"], "context": ["GBR"]})

    result = build_auxiliary_source_requests(
        requirements, source_capabilities=capabilities
    )

    assert list(result[["source", "context"]].itertuples(index=False, name=None)) == [
        ("neso", "GBR")
    ]


def test_build_auxiliary_source_requests_applies_wildcard_to_all_contexts():
    """Map wildcard source coverage to every required context."""
    requirements = pd.DataFrame(
        {
            "context": ["GBR", "FRA"],
            "start": [
                pd.Timestamp("2025-01-01T00:00:00Z"),
                pd.Timestamp("2025-01-01T00:00:00Z"),
            ],
            "end": [
                pd.Timestamp("2025-01-02T00:00:00Z"),
                pd.Timestamp("2025-01-02T00:00:00Z"),
            ],
        }
    )

    capabilities = pd.DataFrame({"source": ["entsoe"], "context": [None]})

    result = build_auxiliary_source_requests(
        requirements, source_capabilities=capabilities
    )

    assert list(result[["source", "context"]].itertuples(index=False, name=None)) == [
        ("entsoe", "FRA"),
        ("entsoe", "GBR"),
    ]


def test_build_auxiliary_source_requests_combines_wildcard_and_specific_sources():
    """Combine wildcard and context-specific source capabilities."""
    requirements = pd.DataFrame(
        {
            "context": ["GBR", "FRA"],
            "start": [
                pd.Timestamp("2025-01-01T00:00:00Z"),
                pd.Timestamp("2025-01-01T00:00:00Z"),
            ],
            "end": [
                pd.Timestamp("2025-01-02T00:00:00Z"),
                pd.Timestamp("2025-01-02T00:00:00Z"),
            ],
        }
    )

    capabilities = pd.DataFrame(
        {"source": ["entsoe", "neso"], "context": [None, "GBR"]}
    )

    result = build_auxiliary_source_requests(
        requirements, source_capabilities=capabilities
    )

    assert list(result[["source", "context"]].itertuples(index=False, name=None)) == [
        ("entsoe", "FRA"),
        ("entsoe", "GBR"),
        ("neso", "GBR"),
    ]


def test_build_auxiliary_source_requests_rejects_uncovered_context():
    """Reject requirements unsupported by every configured source."""
    requirements = pd.DataFrame(
        {
            "context": ["FRA"],
            "start": [pd.Timestamp("2025-01-01T00:00:00Z")],
            "end": [pd.Timestamp("2025-01-02T00:00:00Z")],
        }
    )

    capabilities = pd.DataFrame({"source": ["neso"], "context": ["GBR"]})

    with pytest.raises(ValueError, match="No configured auxiliary source supports"):
        build_auxiliary_source_requests(requirements, source_capabilities=capabilities)


def test_build_auxiliary_source_requests_returns_empty_without_requirements():
    """Return no requests when no auxiliary data are required."""
    requirements = pd.DataFrame(
        {
            "context": pd.Series(dtype="string"),
            "start": pd.Series(dtype="datetime64[ns, UTC]"),
            "end": pd.Series(dtype="datetime64[ns, UTC]"),
        }
    )

    capabilities = pd.DataFrame({"source": ["entsoe"], "context": [None]})

    result = build_auxiliary_source_requests(
        requirements, source_capabilities=capabilities
    )

    assert result.empty

    assert result.columns.tolist() == ["source", "context", "start", "end"]


def test_select_active_advanced_rules_keeps_intersecting_rules():
    """Select rules whose context and period intersect the target scope."""
    rules = pd.DataFrame(
        {
            "rule_name": ["inside", "wrong_context", "wrong_period"],
            "method": ["external_profile", "external_profile", "external_profile"],
            "source": ["test_source", "test_source", "test_source"],
            "context": ["GBR", "FRA", "GBR"],
            "start": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
                "2025-01-01T00:00:00Z",
            ],
            "end": [
                "2026-01-03T00:00:00Z",
                "2026-01-03T00:00:00Z",
                "2025-01-03T00:00:00Z",
            ],
            "scope": ["fill_gaps", "fill_gaps", "fill_gaps"],
        }
    )

    result = select_active_advanced_rules(
        rules,
        target_contexts=["GBR"],
        target_start="2026-01-01T00:00:00Z",
        target_end="2026-02-01T00:00:00Z",
    )

    assert result["rule_name"].tolist() == ["inside"]


def test_select_active_advanced_rules_accepts_partial_period_overlap():
    """Select a rule that overlaps only part of the target period."""
    rules = pd.DataFrame(
        {
            "rule_name": ["partial"],
            "method": ["external_profile"],
            "source": ["test_source"],
            "context": ["GBR"],
            "start": ["2025-12-31T00:00:00Z"],
            "end": ["2026-01-02T00:00:00Z"],
            "scope": ["fill_gaps"],
        }
    )

    result = select_active_advanced_rules(
        rules,
        target_contexts=["GBR"],
        target_start="2026-01-01T00:00:00Z",
        target_end="2026-02-01T00:00:00Z",
    )

    assert result["rule_name"].tolist() == ["partial"]


def test_select_active_advanced_rules_excludes_rule_ending_at_target_start():
    """Exclude a rule ending exactly when the target period begins."""
    rules = pd.DataFrame(
        {
            "rule_name": ["before"],
            "method": ["external_profile"],
            "source": ["test_source"],
            "context": ["GBR"],
            "start": ["2025-12-01T00:00:00Z"],
            "end": ["2026-01-01T00:00:00Z"],
            "scope": ["fill_gaps"],
        }
    )

    result = select_active_advanced_rules(
        rules,
        target_contexts=["GBR"],
        target_start="2026-01-01T00:00:00Z",
        target_end="2026-02-01T00:00:00Z",
    )

    assert result.empty


def test_select_active_advanced_rules_preserves_rule_order():
    """Preserve configured execution order when selecting active rules."""
    rules = pd.DataFrame(
        {
            "rule_name": ["second_chronologically", "first_chronologically"],
            "method": ["external_profile", "external_profile"],
            "source": ["test_source", "test_source"],
            "context": ["GBR", "GBR"],
            "start": ["2026-01-10T00:00:00Z", "2026-01-01T00:00:00Z"],
            "end": ["2026-01-11T00:00:00Z", "2026-01-02T00:00:00Z"],
            "scope": ["overwrite", "overwrite"],
        }
    )

    result = select_active_advanced_rules(
        rules,
        target_contexts=["GBR"],
        target_start="2026-01-01T00:00:00Z",
        target_end="2026-02-01T00:00:00Z",
    )

    assert result["rule_name"].tolist() == [
        "second_chronologically",
        "first_chronologically",
    ]
