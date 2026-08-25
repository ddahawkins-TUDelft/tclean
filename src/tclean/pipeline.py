"""High-level time-series cleaning pipeline."""

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from tclean.advanced.apply import apply_advanced_rules
from tclean.basic.apply import fill_basic_gaps
from tclean.combine import combine_sources
from tclean.config import TCleanConfig


def clean(
    sources: Mapping[str, pd.DataFrame],
    *,
    config: TCleanConfig,
    basic_rules: Sequence[Mapping[str, Any]] | None = None,
    advanced_rules: pd.DataFrame | None = None,
    advanced_sources: Mapping[str, pd.Series] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Combine and clean time-series sources.

    Source mapping order defines source priority from highest to lowest.
    Basic cleaning is applied before advanced cleaning.

    Args:
        sources: Canonical time-series frames keyed by source name.
            Mapping insertion order defines source priority.
        config: tclean configuration object.
        basic_rules: Ordered basic cleaning rules. If omitted, basic
            cleaning is skipped.
        advanced_rules: Validated advanced-fill rule definitions. If omitted,
            advanced cleaning is skipped.
        advanced_sources: Advanced time-series sources keyed by source name.

    Returns:
        Cleaned time series, data-source provenance, and cleaning-method
        provenance.

    Raises:
        ValueError: If advanced sources are supplied without advanced rules,
            or advanced rules are supplied without an advanced-source mapping.
    """
    grid = config.grid
    cleaned, data_source, cleaning_method = combine_sources(sources, grid=grid)

    if basic_rules is not None:
        cleaned, cleaning_method = fill_basic_gaps(
            cleaned, cleaning_method=cleaning_method, rules=basic_rules, grid=grid
        )

    if advanced_rules is None:
        if advanced_sources is not None:
            raise ValueError("Advanced sources were supplied without advanced rules.")
        cleaning_method = cleaning_method.fillna("missing")
        return (cleaned, data_source, cleaning_method)

    if advanced_sources is None:
        advanced_sources = {}

    cleaned, data_source, cleaning_method = apply_advanced_rules(
        cleaned,
        data_source,
        cleaning_method,
        rules=advanced_rules,
        advanced_sources=advanced_sources,
        grid=grid,
    )
    cleaning_method = cleaning_method.fillna("missing")
    grid.validate_target_coverage(cleaned.index)

    cleaned = grid.crop(cleaned)
    data_source = grid.crop(data_source)
    cleaning_method = grid.crop(cleaning_method)

    return (cleaned, data_source, cleaning_method)
