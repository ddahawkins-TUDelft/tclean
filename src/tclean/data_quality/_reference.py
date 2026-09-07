"""Helpers for constructing quality-test reference data."""

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


def reference_exclusion_mask(
    data: pd.DataFrame,
    *,
    source_name: str,
    contexts: Sequence[str],
    preceding_failures: Sequence[Mapping[str, Any]],
    include_failed_periods_from: Sequence[str] = (),
) -> pd.DataFrame:
    """Return observations excluded from a test's reference population.

    Failures from preceding tests are excluded only when they belong to the
    same source and context. Failures produced by explicitly included tests
    remain eligible for this test's reference population.

    Args:
        data: Canonical time-series data for one source.
        source_name: Name of the source being evaluated.
        contexts: Contexts being evaluated for the current test.
        preceding_failures: Failure records from preceding tests only.
        include_failed_periods_from: Preceding tests whose failed periods
            should remain eligible for this reference population.

    Returns:
        Boolean frame aligned with the selected source data. ``True`` marks
        observations that must be excluded from reference construction.
    """
    selected_contexts = list(contexts)

    excluded = pd.DataFrame(
        False, index=data.index, columns=selected_contexts, dtype=bool
    )

    included_tests = set(include_failed_periods_from)

    for failure in preceding_failures:
        if failure["source"] != source_name:
            continue

        context = failure["context"]

        if context not in excluded.columns:
            continue

        if failure["test_name"] in included_tests:
            continue

        period = (data.index >= failure["start"]) & (data.index < failure["end"])

        excluded.loc[period, context] = True

    return excluded


def build_reference_data(
    data: pd.DataFrame,
    *,
    source_name: str,
    contexts: Sequence[str],
    preceding_failures: Sequence[Mapping[str, Any]],
    include_failed_periods_from: Sequence[str] = (),
) -> pd.DataFrame:
    """Build failure-filtered reference data for one source and test."""
    selected = data.loc[:, list(contexts)].copy()

    excluded = reference_exclusion_mask(
        data,
        source_name=source_name,
        contexts=contexts,
        preceding_failures=preceding_failures,
        include_failed_periods_from=include_failed_periods_from,
    )

    return selected.mask(excluded)
