"""Cleaning-method provenance and ranking helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


def build_cleaning_method_ranks(
    source_names: Sequence[str],
    *,
    basic_rule_names: Sequence[str] = (),
    advanced_rule_names: Sequence[str] = (),
) -> dict[str, int]:
    """Build cleaning-method ranks from execution order.

    Args:
        source_names: Primary source names in priority order.
        basic_rule_names: Basic cleaning rule names in execution order.
        advanced_rule_names: Advanced cleaning rule names in execution order.

    Returns:
        Mapping from cleaning-method provenance labels to integer ranks.

    Raises:
        ValueError: If any generated provenance label would be duplicated.
    """
    labels = [
        *(f"observed_{source}" for source in source_names),
        *basic_rule_names,
        *advanced_rule_names,
        "missing",
    ]

    duplicates = sorted({label for label in labels if labels.count(label) > 1})

    if duplicates:
        raise ValueError(
            "Cleaning-method provenance labels must be unique. "
            f"Duplicates: {duplicates!r}."
        )

    return {label: rank for rank, label in enumerate(labels)}


def derive_cleaning_method_rank(
    *, cleaning_method: pd.DataFrame, ranks: Mapping[str, int]
) -> pd.DataFrame:
    """Translate cleaning-method names to integer provenance ranks."""
    present_methods = set(cleaning_method.stack().astype(str).unique())
    unknown_methods = sorted(present_methods - set(ranks))

    if unknown_methods:
        raise ValueError(f"No cleaning-method rank is defined for: {unknown_methods}")

    cleaning_method_rank = cleaning_method.replace(ranks)

    return cleaning_method_rank.astype("int16")
