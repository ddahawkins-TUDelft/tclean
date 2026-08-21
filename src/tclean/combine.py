"""Combine prepared demand sources in configured priority order."""

from collections.abc import Mapping, Sequence

import pandas as pd

from tclean.validation import (
    validate_cleaning_method,
    validate_data_source,
    validate_load,
)


def _validate_source_priority(
    sources: Mapping[str, pd.DataFrame], priority: Sequence[str]
) -> None:
    """Require an explicit unique priority for every supplied source."""
    if not sources:
        raise ValueError("At least one demand source must be supplied.")

    if not priority:
        raise ValueError("Source priority must contain at least one source.")

    if len(priority) != len(set(priority)):
        raise ValueError("Source priority must not contain duplicate sources.")

    source_names = set(sources)
    priority_names = set(priority)

    if source_names != priority_names:
        missing = sorted(source_names - priority_names)
        unknown = sorted(priority_names - source_names)

        raise ValueError(
            "Source priority must contain every supplied source "
            "exactly once. "
            f"Missing from priority: {missing!r}; "
            f"not supplied: {unknown!r}."
        )


def _validate_source_alignment(sources: Mapping[str, pd.DataFrame]) -> None:
    """Require all prepared sources to use the same target grid."""
    source_items = list(sources.items())

    reference_name, reference = source_items[0]

    for source_name, source in source_items[1:]:
        if not source.index.equals(reference.index):
            raise ValueError(
                f"Demand source {source_name!r} does not use the "
                f"same timestamp index as {reference_name!r}."
            )

        if not source.columns.equals(reference.columns):
            raise ValueError(
                f"Demand source {source_name!r} does not use the "
                f"same country columns as {reference_name!r}."
            )


def combine_sources(
    sources: Mapping[str, pd.DataFrame], *, priority: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Combine prepared demand sources in explicit priority order.

    Higher-priority sources supply values first. Lower-priority sources
    contribute only where all higher-priority sources are missing.

    Args:
        sources: Prepared canonical demand frames keyed by source name.
        priority: Source names from highest to lowest priority. Every
            supplied source must appear exactly once.

    Returns:
        Combined demand, data-source provenance, and cleaning-method
        provenance.

    Raises:
        ValueError: If source priority or source alignment is invalid.
        pandera.errors.SchemaErrors: If any source violates the canonical
            demand contract.
    """
    _validate_source_priority(sources, priority)

    validated_sources = {name: validate_load(load) for name, load in sources.items()}

    selected = {source: validated_sources[source] for source in priority}

    _validate_source_alignment(selected)

    first_source = priority[0]
    combined = selected[first_source].copy()

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

    for source_name in priority[1:]:
        candidate = selected[source_name]

        newly_supplied = combined.isna() & candidate.notna()

        combined = combined.combine_first(candidate)

        data_source = data_source.mask(newly_supplied, source_name)

        cleaning_method = cleaning_method.mask(
            newly_supplied, f"observed_{source_name}"
        )

    combined = validate_load(combined)

    data_source = validate_data_source(data_source, load=combined)

    cleaning_method = validate_cleaning_method(cleaning_method, load=combined)

    return (combined, data_source, cleaning_method)


def combine_auxiliary_sources(
    loads: Mapping[str, pd.DataFrame], *, priority: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Combine available auxiliary sources using configured priority.

    Configured sources that did not supply auxiliary data are skipped.
    Every supplied auxiliary source must nevertheless occur in the
    configured priority.

    Args:
        loads: Available prepared auxiliary demand by source.
        priority: Complete configured source priority.

    Returns:
        Combined auxiliary demand, source provenance, and cleaning-method
        provenance.

    Raises:
        ValueError: If a supplied auxiliary source is absent from the
            configured priority.
    """
    if not loads:
        empty = pd.DataFrame()
        return (empty, empty.copy(), empty.copy())

    if len(priority) != len(set(priority)):
        raise ValueError("Source priority must not contain duplicate sources.")

    unconfigured = sorted(set(loads) - set(priority))

    if unconfigured:
        raise ValueError(
            "Auxiliary data were supplied by sources absent from "
            f"the configured priority: {unconfigured!r}."
        )

    validated_loads = {name: validate_load(load) for name, load in loads.items()}

    available_priority = [source for source in priority if source in validated_loads]

    columns = sorted(
        {column for load in validated_loads.values() for column in load.columns}
    )

    aligned = {
        source: load.reindex(columns=columns)
        for source, load in validated_loads.items()
    }

    return combine_sources(aligned, priority=available_priority)
