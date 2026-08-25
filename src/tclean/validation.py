"""Validation interfaces for canonical T-Clean data."""

import pandas as pd

from tclean._schemas import (
    ADVANCED_FILL_RULES_SCHEMA,
    ADVANCED_SOURCE_SCHEMA,
    AUXILIARY_REQUIREMENTS_SCHEMA,
    AUXILIARY_SOURCE_REQUESTS_SCHEMA,
    PROVENANCE_SCHEMA,
    SOURCE_CAPABILITIES_SCHEMA,
    SOURCE_PERIODS_SCHEMA,
    TIME_SERIES_SCHEMA,
    TIMESTAMP_INDEX_SCHEMA,
)
from tclean._temporal import normalize_temporal_range
from tclean.time_grid import TimeGrid


def validate_time_series(data: pd.DataFrame, *, grid: TimeGrid) -> pd.DataFrame:
    """Validate canonical time-series data.

    Args:
        data: Time-series data with timestamps on the index and one or
            more contexts in the columns.
        grid: Temporal grid against which timestamps are validated.

    Returns:
        Validated canonical time-series data.

    Raises:
        pandera.errors.SchemaErrors: If the data violate the canonical
            T-Clean time-series contract.
        ValueError: If timestamps do not form a complete configured grid.
    """
    validated = TIME_SERIES_SCHEMA.validate(data, lazy=True)

    index = _require_datetime_index(validated.index, field="Time-series index")

    grid.validate_complete_index(index)

    return validated


def validate_temporal_range(
    start: object, end: object
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate and normalize a temporal range.

    Args:
        start: Start of the temporal range.
        end: End of the temporal range.

    Returns:
        Validated UTC start and end timestamps.
    """
    return normalize_temporal_range(start=start, end=end)


def validate_timestamp_index(
    index: pd.DatetimeIndex, *, grid: TimeGrid
) -> pd.DatetimeIndex:
    """Validate a canonical regular timestamp index.

    Args:
        index: Timestamp index to validate.
        grid: Temporal grid against which timestamps are validated.

    Returns:
        Validated timestamp index.

    Raises:
        TypeError: If index is not a pandas DatetimeIndex.
        ValueError: If timestamps do not form a complete configured grid.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("'index' must be a pandas DatetimeIndex.")

    frame = pd.DataFrame(index=index)
    validated = TIMESTAMP_INDEX_SCHEMA.validate(frame, lazy=True)

    validated_index = _require_datetime_index(validated.index, field="Timestamp index")

    grid.validate_complete_index(validated_index)

    return validated_index


def validate_source_periods(
    source_periods: pd.DataFrame, *, grid: TimeGrid
) -> pd.DataFrame:
    """Validate and normalize source-period definitions.

    Args:
        source_periods: Source definitions containing context, start,
            end, and weight columns.
        grid: Temporal grid against which period boundaries are validated.

    Returns:
        Validated source-period definitions with UTC timestamps and
        floating-point positive weights.

    Raises:
        pandera.errors.SchemaErrors: If the source-period definitions
            violate the T-Clean source-period contract.
        ValueError: If period boundaries do not align with the configured grid.
    """
    validated = SOURCE_PERIODS_SCHEMA.validate(source_periods, lazy=True)

    _validate_periods_against_grid(validated, grid=grid)

    return validated


def validate_provenance(
    provenance: pd.DataFrame, *, data: pd.DataFrame
) -> pd.DataFrame:
    """Validate and normalize cleaning-method provenance data.

    Args:
        provenance: Provenance labels corresponding to values.
        data: Canonical data whose shape and axes must be matched.

    Returns:
        Validated cleaning-method data aligned exactly with ``data``.

    Raises:
        pandera.errors.SchemaErrors: If provenance values violate the
            provenance schema.
        ValueError: If provenance index or columns do not exactly match data.
    """
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
    """Validate and normalize advanced-fill rule definitions.

    Args:
        rules: Advanced-fill rules containing rule name, method, source,
            context, start, end, and scope.
        grid: Temporal grid against which period boundaries are validated.

    Returns:
        Validated rule definitions with UTC timestamps.

    Raises:
        pandera.errors.SchemaErrors: If the rules violate the
            T-Clean advanced-fill contract.
        ValueError: If rule periods do not align with the configured grid.
    """
    validated = ADVANCED_FILL_RULES_SCHEMA.validate(rules, lazy=True)

    _validate_periods_against_grid(validated, grid=grid)

    return validated


def validate_auxiliary_requirements(
    requirements: pd.DataFrame, *, grid: TimeGrid
) -> pd.DataFrame:
    """Validate and normalize auxiliary data requirements.

    Args:
        requirements: Auxiliary context-period requirements.
        grid: Temporal grid against which period boundaries are validated.

    Returns:
        Validated requirements with canonical UTC timestamps.

    Raises:
        pandera.errors.SchemaErrors: If requirements violate the
            T-Clean auxiliary-requirements contract.
        ValueError: If requirement periods do not align with the configured grid.
    """
    validated = AUXILIARY_REQUIREMENTS_SCHEMA.validate(requirements, lazy=True)

    _validate_periods_against_grid(validated, grid=grid)

    return validated


def validate_source_capabilities(capabilities: pd.DataFrame) -> pd.DataFrame:
    """Validate source-context capability definitions.

    Args:
        capabilities: Explicit source-context capability pairs. A missing
            context means that the source supports all contexts.

    Returns:
        Validated source capabilities.

    Raises:
        pandera.errors.SchemaErrors: If the capabilities violate the
            T-Clean source-capability contract.
    """
    return SOURCE_CAPABILITIES_SCHEMA.validate(capabilities, lazy=True)


def validate_auxiliary_source_requests(
    requests: pd.DataFrame, *, grid: TimeGrid
) -> pd.DataFrame:
    """Validate and normalize auxiliary source requests.

    Args:
        requests: Source-context-period acquisition requests.
        grid: Temporal grid against which period boundaries are validated.

    Returns:
        Validated source requests with canonical UTC timestamps.

    Raises:
        pandera.errors.SchemaErrors: If the requests violate the
            T-Clean auxiliary-source-request contract.
        ValueError: If request periods do not align with the configured grid.
    """
    validated = AUXILIARY_SOURCE_REQUESTS_SCHEMA.validate(requests, lazy=True)

    _validate_periods_against_grid(validated, grid=grid)

    return validated


def validate_advanced_source(source: pd.Series, *, grid: TimeGrid) -> pd.Series:
    """Validate and normalize an advanced time-series source.

    Args:
        source: Time-series values indexed by UTC timestamps.
        grid: Temporal grid against which source timestamps are validated.

    Returns:
        Validated numeric source with canonical UTC timestamps.

    Raises:
        pandera.errors.SchemaErrors: If the source violates the
            T-Clean advanced-source contract.
        ValueError: If source timestamps do not lie on the configured grid.
    """
    validated = ADVANCED_SOURCE_SCHEMA.validate(source, lazy=True)

    index = _require_datetime_index(validated.index, field="Advanced source index")

    grid.validate_sparse_index(index)

    return validated.astype(float)


def _validate_periods_against_grid(periods: pd.DataFrame, *, grid: TimeGrid) -> None:
    """Validate period boundaries against a configured time grid."""
    for start, end in periods[["start", "end"]].itertuples(index=False, name=None):
        grid.validate_period(start=pd.Timestamp(start), end=pd.Timestamp(end))


def _require_datetime_index(index: pd.Index, *, field: str) -> pd.DatetimeIndex:
    """Narrow an already validated index to a DatetimeIndex.

    Args:
        index: Index produced by validated tabular data.
        field: Human-readable field name for error messages.

    Returns:
        Timestamp index.

    Raises:
        TypeError: If the validated index is not a DatetimeIndex.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{field} must be a pandas DatetimeIndex.")

    return index
