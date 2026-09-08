"""Public gap-filling API."""

from tclean.gap_filling.advanced.apply import apply_advanced_rules
from tclean.gap_filling.advanced.gap_report import build_gap_report
from tclean.gap_filling.advanced.methods.construct_from_sources import (
    construct_from_sources,
)
from tclean.gap_filling.advanced.methods.external_profile import read_external_profile
from tclean.gap_filling.advanced.planning import (
    build_auxiliary_acquisition_requirements,
    build_auxiliary_source_requests,
    select_active_advanced_rules,
)
from tclean.gap_filling.basic.rule_validation import validate_basic_rules
from tclean.gap_filling.pipeline import fill_gaps
from tclean.gap_filling.provenance import (
    build_cleaning_method_ranks,
    derive_cleaning_method_rank,
)
from tclean.gap_filling.validation import (
    validate_advanced_fill_rules,
    validate_advanced_source,
    validate_auxiliary_requirements,
    validate_auxiliary_source_requests,
    validate_source_capabilities,
    validate_source_periods,
)

__all__ = [
    "apply_advanced_rules",
    "build_auxiliary_acquisition_requirements",
    "build_auxiliary_source_requests",
    "build_cleaning_method_ranks",
    "build_gap_report",
    "construct_from_sources",
    "derive_cleaning_method_rank",
    "fill_gaps",
    "read_external_profile",
    "select_active_advanced_rules",
    "validate_advanced_fill_rules",
    "validate_advanced_source",
    "validate_auxiliary_requirements",
    "validate_auxiliary_source_requests",
    "validate_basic_rules",
    "validate_source_capabilities",
    "validate_source_periods",
]
