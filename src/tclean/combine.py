"""Combine prepared sources in configured priority order."""

from collections.abc import Mapping

import pandas as pd

from tclean.time_grid import TimeGrid
from tclean.validation import (
    validate_cleaning_method,
    validate_data_source,
    validate_time_series,
)


def _validate_source_alignment(sources: Mapping[str, pd.DataFrame]) -> None:
    """Require all prepared sources to use the same target grid."""
    source_items = list(sources.items())

    reference_name, reference = source_items[0]

    for source_name, source in source_items[1:]:
        if not source.index.equals(reference.index):
            raise ValueError(
                f"Source {source_name!r} does not use the "
                f"same timestamp index as {reference_name!r}."
            )

        if not source.columns.equals(reference.columns):
            raise ValueError(
                f"Source {source_name!r} does not use the "
                f"same context columns as {reference_name!r}."
            )


def combine_sources(
    sources: Mapping[str, pd.DataFrame], *, grid: TimeGrid
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Combine prepared time-series sources in mapping order.

    Earlier sources have higher priority. Later sources contribute only
    where all earlier sources are missing.

    Args:
        sources: Prepared canonical time-series frames keyed by source name.
            Mapping insertion order defines source priority from highest to
            lowest.
        grid: Temporal grid against which source timestamps are validated.

    Returns:
        Combined data, data-source provenance, and cleaning-method
        provenance.

    Raises:
        ValueError: If no sources are supplied or source grids are not
            aligned.
        pandera.errors.SchemaErrors: If any source violates the canonical
            time-series contract.
    """
    if not sources:
        raise ValueError("At least one time-series source must be supplied.")

    validated_sources = {
        name: validate_time_series(data, grid=grid) for name, data in sources.items()
    }

    _validate_source_alignment(validated_sources)

    source_names = list(validated_sources)

    first_source = source_names[0]
    combined = validated_sources[first_source].copy()

    data_source = pd.DataFrame(
        pd.NA, index=combined.index, columns=combined.columns, dtype="string"
    )

    cleaning_method = pd.DataFrame(
        pd.NA, index=combined.index, columns=combined.columns, dtype="string"
    )

    first_source_values = combined.notna()

    data_source = data_source.mask(first_source_values, first_source)

    cleaning_method = cleaning_method.mask(
        first_source_values, f"observed_{first_source}"
    )

    for source_name in source_names[1:]:
        candidate = validated_sources[source_name]

        newly_supplied = combined.isna() & candidate.notna()

        combined = combined.combine_first(candidate)

        data_source = data_source.mask(newly_supplied, source_name)

        cleaning_method = cleaning_method.mask(
            newly_supplied, f"observed_{source_name}"
        )

    combined = validate_time_series(combined, grid=grid)

    data_source = validate_data_source(data_source, data=combined)

    cleaning_method = validate_cleaning_method(cleaning_method, data=combined)

    return (combined, data_source, cleaning_method)


def combine_auxiliary_sources(
    datas: Mapping[str, pd.DataFrame], *, grid: TimeGrid
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Combine available auxiliary sources in mapping order.

    Earlier sources have higher priority. Context coverage may differ
    between auxiliary sources.

    Args:
        datas: Available prepared auxiliary time-series data keyed by source
            name. Mapping insertion order defines source priority.
        grid: Temporal grid against which source timestamps are validated.

    Returns:
        Combined auxiliary data, source provenance, and cleaning-method
        provenance.
    """
    if not datas:
        empty = pd.DataFrame()

        return (empty, empty.copy(), empty.copy())

    validated_datas = {
        name: validate_time_series(data, grid=grid) for name, data in datas.items()
    }

    columns = sorted(
        {column for data in validated_datas.values() for column in data.columns}
    )

    aligned = {
        source: data.reindex(columns=columns)
        for source, data in validated_datas.items()
    }

    return combine_sources(aligned, grid=grid)
