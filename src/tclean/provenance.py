"""Cleaning-method provenance and ranking helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


def build_cleaning_method_ranks(
    *,
    source_priority: Sequence[str],
    rules: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Build cleaning-method ranks from source and rule priority order."""
    ranks: dict[str, int] = {}

    for rank, source_name in enumerate(source_priority):
        ranks[f"observed_{source_name}"] = rank

    first_gap_filling_rank = len(source_priority)

    for rule_position, rule in enumerate(rules):
        ranks[str(rule["name"])] = first_gap_filling_rank + rule_position

    ranks["missing"] = len(source_priority) + len(rules)

    return ranks


def derive_cleaning_method_rank(
    *,
    cleaning_method: pd.DataFrame,
    ranks: Mapping[str, int],
) -> pd.DataFrame:
    """Translate cleaning-method names to integer provenance ranks."""
    present_methods = set(cleaning_method.stack().astype(str).unique())
    unknown_methods = sorted(present_methods - set(ranks))

    if unknown_methods:
        raise ValueError(
            f"No cleaning-method rank is defined for: {unknown_methods}"
        )

    cleaning_method_rank = cleaning_method.replace(ranks)

    return cleaning_method_rank.astype("int16")


def combine_cleaning_rules(
    *,
    basic_rules: Sequence[Mapping[str, Any]],
    advanced_rules: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Combine basic and advanced cleaning rules in provenance order."""
    return [
        *(dict(rule) for rule in basic_rules),
        *(dict(rule) for rule in advanced_rules),
    ]