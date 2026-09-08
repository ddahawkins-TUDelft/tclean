"""Validation interfaces for canonical data-quality result tables."""

import pandas as pd

from tclean.data_quality._schemas import QUALITY_FAILURES_SCHEMA, QUALITY_ISSUES_SCHEMA
from tclean.time_grid import TimeGrid
from tclean.validation import _validate_periods_against_grid


def validate_quality_failures(
    failures: pd.DataFrame, *, grid: TimeGrid
) -> pd.DataFrame:
    """Validate canonical data-quality failure events."""
    validated = QUALITY_FAILURES_SCHEMA.validate(failures, lazy=True)
    _validate_periods_against_grid(validated, grid=grid)
    return validated


def validate_quality_issues(issues: pd.DataFrame, *, grid: TimeGrid) -> pd.DataFrame:
    """Validate canonical data-quality evaluation issues."""
    validated = QUALITY_ISSUES_SCHEMA.validate(issues, lazy=True)
    _validate_periods_against_grid(validated, grid=grid)
    return validated
