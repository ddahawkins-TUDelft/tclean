"""High-level time-series gap-filling pipeline."""

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from tclean.gap_filling.advanced.apply import apply_advanced_rules
from tclean.gap_filling.basic.apply import fill_basic_gaps
from tclean.gap_filling.combine import combine_sources
from tclean.time_grid import TimeGrid


def fill_gaps(
    sources: Mapping[str, pd.DataFrame],
    *,
    grid: TimeGrid,
    basic_rules: Sequence[Mapping[str, Any]] | None = None,
    advanced_rules: pd.DataFrame | None = None,
    advanced_sources: Mapping[str, pd.Series] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Combine sources and fill configured time-series gaps.

    Source mapping order defines source priority from highest to lowest.
    Basic gap filling is applied before advanced gap filling.

    Args:
        sources: Canonical time-series frames keyed by source name.
            Mapping insertion order defines source priority.
        grid: Fixed-frequency temporal grid and requested output window.
        basic_rules: Ordered basic gap-filling rules. If omitted, basic
            gap filling is skipped.
        advanced_rules: Validated advanced-fill rule definitions. If omitted,
            advanced gap filling is skipped.
        advanced_sources: Advanced time-series sources keyed by source name.

    Returns:
        Filled time series, data-source provenance, and cleaning-method
        provenance.

    Raises:
        ValueError: If advanced sources are supplied without advanced rules,
            or advanced rules are supplied without an advanced-source mapping.
    """
    filled, data_source, cleaning_method = combine_sources(sources, grid=grid)

    if basic_rules is not None:
        filled, cleaning_method = fill_basic_gaps(
            filled, cleaning_method=cleaning_method, rules=basic_rules, grid=grid
        )

    if advanced_rules is None:
        if advanced_sources is not None:
            raise ValueError("Advanced sources were supplied without advanced rules.")
    else:
        if advanced_sources is None:
            advanced_sources = {}

        filled, data_source, cleaning_method = apply_advanced_rules(
            filled,
            data_source,
            cleaning_method,
            rules=advanced_rules,
            advanced_sources=advanced_sources,
            grid=grid,
        )

    cleaning_method = cleaning_method.fillna("missing")

    grid.validate_target_coverage(filled.index)

    filled = grid.crop(filled)
    data_source = grid.crop(data_source)
    cleaning_method = grid.crop(cleaning_method)

    return filled, data_source, cleaning_method
