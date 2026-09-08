"""Tests for supported public imports."""

import tclean
from tclean import TimeGrid
from tclean.data_quality import QualityEvaluation, evaluate, validate_quality_tests
from tclean.gap_filling import (
    apply_advanced_rules,
    build_auxiliary_acquisition_requirements,
    build_auxiliary_source_requests,
    build_cleaning_method_ranks,
    build_gap_report,
    construct_from_sources,
    derive_cleaning_method_rank,
    fill_gaps,
    read_external_profile,
    select_active_advanced_rules,
    validate_advanced_fill_rules,
    validate_advanced_source,
    validate_auxiliary_requirements,
    validate_auxiliary_source_requests,
    validate_basic_rules,
    validate_source_capabilities,
    validate_source_periods,
)


def test_shared_top_level_public_api_is_importable() -> None:
    """Expose only shared T-Clean primitives at the package root."""
    assert TimeGrid is not None
    assert not hasattr(tclean, "clean")
    assert not hasattr(tclean, "TCleanConfig")
    assert not hasattr(tclean, "fill_gaps")


def test_gap_filling_public_api_is_importable() -> None:
    """Expose supported gap-filling machinery from one public namespace."""
    public_objects = [
        apply_advanced_rules,
        build_auxiliary_acquisition_requirements,
        build_auxiliary_source_requests,
        build_cleaning_method_ranks,
        build_gap_report,
        construct_from_sources,
        derive_cleaning_method_rank,
        fill_gaps,
        read_external_profile,
        select_active_advanced_rules,
        validate_advanced_fill_rules,
        validate_advanced_source,
        validate_auxiliary_requirements,
        validate_auxiliary_source_requests,
        validate_basic_rules,
        validate_source_capabilities,
        validate_source_periods,
    ]

    assert all(item is not None for item in public_objects)


def test_data_quality_public_api_is_importable() -> None:
    """Retain the supported data-quality public namespace."""
    assert QualityEvaluation is not None
    assert evaluate is not None
    assert validate_quality_tests is not None
