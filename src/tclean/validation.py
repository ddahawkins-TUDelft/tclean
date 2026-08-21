"""Validation interfaces for canonical T-Clean data."""

import pandas as pd

from tclean._schemas import (
    ADVANCED_FILL_RULES_SCHEMA,
    ADVANCED_SOURCE_SCHEMA,
    AUXILIARY_REQUIREMENTS_SCHEMA,
    AUXILIARY_SOURCE_REQUESTS_SCHEMA,
    HOURLY_TIMESTAMP_INDEX_SCHEMA,
    PROVENANCE_SCHEMA,
    SOURCE_CAPABILITIES_SCHEMA,
    SOURCE_PERIODS_SCHEMA,
    TEMPORAL_RANGE_SCHEMA,
    TIME_SERIES_SCHEMA,
)


def validate_time_series(data: pd.DataFrame) -> pd.DataFrame:
    """Validate canonical time-series data.

    Args:
        data: Time-series data with timestamps on the index and one or
            more contexts in the columns.

    Returns:
        Validated canonical time-series data.

    Raises:
        pandera.errors.SchemaErrors: If the data violate the canonical
            T-Clean time-series contract.
    """
    return TIME_SERIES_SCHEMA.validate(data, lazy=True)


def validate_temporal_range(
    *, start: object, end: object
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate and normalize a temporal range.

    Args:
        start: Start of the temporal range.
        end: End of the temporal range.

    Returns:
        Validated UTC start and end timestamps.

    Raises:
        pandera.errors.SchemaErrors: If the temporal range violates
            the T-Clean temporal contract.
    """
    temporal_range = pd.DataFrame({"start": [start], "end": [end]})

    validated = TEMPORAL_RANGE_SCHEMA.validate(temporal_range, lazy=True)

    return (
        pd.Timestamp(validated.loc[0, "start"]),
        pd.Timestamp(validated.loc[0, "end"]),
    )


def infer_regular_timestep(index: pd.Index) -> pd.Timedelta:
    """Return the timestep of an already validated datetime index.

    Args:
        index: Validated regular datetime index.

    Returns:
        The temporal difference between consecutive timestamps.
    """
    return index[1] - index[0]


def validate_hourly_timestamp_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Validate and normalize a canonical hourly timestamp index.

    Args:
        index: Timestamp index to validate.

    Returns:
        Validated UTC hourly timestamp index named ``timestamp``.

    Raises:
        pandera.errors.SchemaErrors: If the index violates the canonical
            hourly timestamp contract.
    """
    frame = pd.DataFrame(index=index)

    validated = HOURLY_TIMESTAMP_INDEX_SCHEMA.validate(frame, lazy=True)

    return validated.index


def validate_source_periods(source_periods: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize source-period definitions.

    Args:
        source_periods: Source definitions containing context, start,
            end, and weight columns.

    Returns:
        Validated source-period definitions with UTC timestamps and
        floating-point positive weights.

    Raises:
        pandera.errors.SchemaErrors: If the source-period definitions
            violate the T-Clean source-period contract.
    """
    return SOURCE_PERIODS_SCHEMA.validate(source_periods, lazy=True)


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


def validate_advanced_fill_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize advanced-fill rule definitions.

    Args:
        rules: Advanced-fill rules containing rule name, method,
            context, start, end, and scope.

    Returns:
        Validated rule definitions with UTC timestamps.

    Raises:
        pandera.errors.SchemaErrors: If the rules violate the
            T-Clean advanced-fill contract.
    """
    return ADVANCED_FILL_RULES_SCHEMA.validate(rules, lazy=True)


def validate_auxiliary_requirements(requirements: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize auxiliary data requirements.

    Args:
        requirements: Auxiliary context-period requirements.

    Returns:
        Validated requirements with canonical UTC timestamps.

    Raises:
        pandera.errors.SchemaErrors: If requirements violate the
            T-Clean auxiliary-requirements contract.
    """
    return AUXILIARY_REQUIREMENTS_SCHEMA.validate(requirements, lazy=True)


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


def validate_auxiliary_source_requests(requests: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize auxiliary source requests.

    Args:
        requests: Source-context-period acquisition requests.

    Returns:
        Validated source requests with canonical UTC timestamps.

    Raises:
        pandera.errors.SchemaErrors: If the requests violate the
            T-Clean auxiliary-source-request contract.
    """
    return AUXILIARY_SOURCE_REQUESTS_SCHEMA.validate(requests, lazy=True)


def validate_advanced_source(source: pd.Series) -> pd.Series:
    """Validate and normalise advanced time-series source.

    Args:
        source: Time-series values indexed by UTC timestamps.

    Returns:
        Validated numeric source with canonical UTC timestamps.

    Raises:
        pandera.errors.SchemaErrors: If the source violates the
            T-Clean advanced-source contract.
    """
    validated = ADVANCED_SOURCE_SCHEMA.validate(source, lazy=True)

    if not validated.index.is_monotonic_increasing:
        raise ValueError("Advanced source timestamps must be sorted.")

    # pandera is not coercing despite coerce=True, so forcing coersion here.
    return validated.astype(float)
