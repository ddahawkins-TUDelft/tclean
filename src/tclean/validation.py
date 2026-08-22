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
    TEMPORAL_RANGE_SCHEMA,
    TIME_SERIES_SCHEMA,
    TIMESTAMP_INDEX_SCHEMA,
)


def validate_time_series(data: pd.DataFrame, frequency: pd.Timedelta) -> pd.DataFrame:
    """Validate canonical time-series data.

    Args:
        data: Time-series data with timestamps on the index and one or
            more contexts in the columns.
        frequency: pd.Timedelta of time series frequency.

    Returns:
        Validated canonical time-series data.

    Raises:
        pandera.errors.SchemaErrors: If the data violate the canonical
            T-Clean time-series contract.
    """
    validated = TIME_SERIES_SCHEMA.validate(data, lazy=True)

    if not validated.index.is_monotonic_increasing:
        raise ValueError("Timestamps must be sorted.")

    _validate_frequency_grid(
        validated.index, frequency=frequency, require_complete=True
    )

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


def validate_timestamp_index(
    index: pd.DatetimeIndex, *, frequency: pd.Timedelta
) -> pd.DatetimeIndex:
    """Validate a canonical regular timestamp index.

    Args:
        index: Timestamp index to validate.
        frequency: Fixed interval between consecutive timestamps.

    Returns:
        Validated timestamp index.

    Raises:
        TypeError: If index is not a pandas DatetimeIndex.
        ValueError: If timestamps are unsorted or do not follow the
            configured frequency.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("'index' must be a pandas DatetimeIndex.")

    frame = pd.DataFrame(index=index)

    validated = TIMESTAMP_INDEX_SCHEMA.validate(frame, lazy=True)

    validated_index = validated.index

    if not validated_index.is_monotonic_increasing:
        raise ValueError("Timestamps must be sorted.")

    _validate_frequency_grid(
        validated_index, frequency=frequency, require_complete=True
    )

    return validated_index


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


def validate_advanced_source(source: pd.Series, frequency: pd.Timedelta) -> pd.Series:
    """Validate and normalise advanced time-series source.

    Args:
        source: Time-series values indexed by UTC timestamps.
        frequency: pd.Timesamp of time series frequency.

    Returns:
        Validated numeric source with canonical UTC timestamps.

    Raises:
        pandera.errors.SchemaErrors: If the source violates the
            T-Clean advanced-source contract.
    """
    validated = ADVANCED_SOURCE_SCHEMA.validate(source, lazy=True)
    del frequency
    if not validated.index.is_monotonic_increasing:
        raise ValueError("Advanced source timestamps must be sorted.")

    # pandera is not coercing despite coerce=True, so forcing coersion here.
    return validated.astype(float)


def _validate_frequency_grid(
    index: pd.DatetimeIndex, *, frequency: pd.Timedelta, require_complete: bool
) -> None:
    """Validate timestamp spacing against a fixed frequency."""
    if len(index) < 2:
        return

    differences = index.to_series().diff().dropna()

    if require_complete:
        valid = differences.eq(frequency).all()
    else:
        valid = (differences % frequency).eq(pd.Timedelta(0)).all()

    if not valid:
        requirement = (
            "exactly one configured interval apart"
            if require_complete
            else "integer multiples of the configured interval"
        )

        raise ValueError(
            f"Timestamps must be {requirement}. Configured frequency: {frequency}."
        )
