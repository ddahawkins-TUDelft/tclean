"""Validation interfaces for gap-filling data contracts."""

import pandas as pd

from tclean.gap_filling._schemas import (
    ADVANCED_FILL_RULES_SCHEMA,
    ADVANCED_SOURCE_SCHEMA,
    AUXILIARY_REQUIREMENTS_SCHEMA,
    AUXILIARY_SOURCE_REQUESTS_SCHEMA,
    PROVENANCE_SCHEMA,
    SOURCE_CAPABILITIES_SCHEMA,
    SOURCE_PERIODS_SCHEMA,
)
from tclean.time_grid import TimeGrid
from tclean.validation import _require_datetime_index, _validate_periods_against_grid


def validate_source_periods(
    source_periods: pd.DataFrame, *, grid: TimeGrid
) -> pd.DataFrame:
    """Validate and normalize source-period definitions."""
    validated = SOURCE_PERIODS_SCHEMA.validate(source_periods, lazy=True)
    _validate_periods_against_grid(validated, grid=grid)
    return validated


def validate_provenance(
    provenance: pd.DataFrame, *, data: pd.DataFrame
) -> pd.DataFrame:
    """Validate and normalize gap-filling provenance data."""
    validated = PROVENANCE_SCHEMA.validate(provenance, lazy=True)

    if not validated.index.equals(data.index):
        raise ValueError("Cleaning-method index must exactly match data index.")

    if not validated.columns.equals(data.columns):
        raise ValueError("Cleaning-method columns must exactly match data columns.")

    return validated


def validate_cleaning_method(
    cleaning_method: pd.DataFrame, *, data: pd.DataFrame
) -> pd.DataFrame:
    """Validate cleaning-method provenance aligned with values."""
    return validate_provenance(cleaning_method, data=data)


def validate_data_source(
    data_source: pd.DataFrame, *, data: pd.DataFrame
) -> pd.DataFrame:
    """Validate data-source provenance aligned with values."""
    return validate_provenance(data_source, data=data)


def validate_advanced_fill_rules(
    rules: pd.DataFrame, *, grid: TimeGrid
) -> pd.DataFrame:
    """Validate and normalize advanced-fill rule definitions."""
    validated = ADVANCED_FILL_RULES_SCHEMA.validate(rules, lazy=True)
    _validate_periods_against_grid(validated, grid=grid)
    return validated


def validate_auxiliary_requirements(
    requirements: pd.DataFrame, *, grid: TimeGrid
) -> pd.DataFrame:
    """Validate and normalize auxiliary data requirements."""
    validated = AUXILIARY_REQUIREMENTS_SCHEMA.validate(requirements, lazy=True)
    _validate_periods_against_grid(validated, grid=grid)
    return validated


def validate_source_capabilities(capabilities: pd.DataFrame) -> pd.DataFrame:
    """Validate source-context capability definitions."""
    return SOURCE_CAPABILITIES_SCHEMA.validate(capabilities, lazy=True)


def validate_auxiliary_source_requests(
    requests: pd.DataFrame, *, grid: TimeGrid
) -> pd.DataFrame:
    """Validate and normalize auxiliary source requests."""
    validated = AUXILIARY_SOURCE_REQUESTS_SCHEMA.validate(requests, lazy=True)
    _validate_periods_against_grid(validated, grid=grid)
    return validated


def validate_advanced_source(source: pd.Series, *, grid: TimeGrid) -> pd.Series:
    """Validate and normalize an advanced time-series source."""
    validated = ADVANCED_SOURCE_SCHEMA.validate(source, lazy=True)
    index = _require_datetime_index(validated.index, field="Advanced source index")
    grid.validate_sparse_index(index)
    return validated.astype(float)
