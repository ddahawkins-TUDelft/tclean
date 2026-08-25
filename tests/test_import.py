"""Tests for import functionality."""
from tclean import TCleanConfig, TimeGrid, clean
from tclean.advanced import (
    build_auxiliary_acquisition_requirements,
    build_auxiliary_source_requests,
    build_gap_report,
    construct_from_sources,
    read_external_profile,
    select_active_advanced_rules,
)


def test_top_level_public_api_is_importable() -> None:
    """Expose the supported top-level T-Clean API."""
    assert TCleanConfig is not None
    assert TimeGrid is not None
    assert clean is not None


def test_advanced_public_api_is_importable() -> None:
    """Expose the supported advanced T-Clean API."""
    assert build_auxiliary_acquisition_requirements is not None
    assert build_auxiliary_source_requests is not None
    assert build_gap_report is not None
    assert construct_from_sources is not None
    assert read_external_profile is not None
    assert select_active_advanced_rules is not None
