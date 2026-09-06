"""Evaluate configured data-quality tests."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from tclean.data_quality._periods import failure_mask_to_periods
from tclean.data_quality.methods import METHODS
from tclean.data_quality.methods import flatline as flatline_method
from tclean.data_quality.methods import low_variability as low_variability_method
from tclean.data_quality.methods import range as range_method
from tclean.data_quality.methods import value_run as value_run_method
from tclean.data_quality.rule_validation import validate_quality_tests
from tclean.time_grid import TimeGrid
from tclean.validation import (
    validate_quality_failures,
    validate_quality_issues,
    validate_time_series,
)

_FAILURE_COLUMNS = [
    "context",
    "source",
    "start",
    "end",
    "test_name",
    "method",
    "details",
]

_ISSUE_COLUMNS = [
    "context",
    "source",
    "start",
    "end",
    "test_name",
    "method",
    "severity",
    "code",
    "details",
]


@dataclass(frozen=True)
class QualityEvaluation:
    """Results of configured data-quality evaluation."""

    failures: pd.DataFrame
    issues: pd.DataFrame


def _empty_quality_failures() -> pd.DataFrame:
    """Return an empty canonical quality-failure table."""
    return pd.DataFrame(
        {
            "context": pd.Series(dtype="string"),
            "source": pd.Series(dtype="string"),
            "start": pd.Series(dtype="datetime64[ns, UTC]"),
            "end": pd.Series(dtype="datetime64[ns, UTC]"),
            "test_name": pd.Series(dtype="string"),
            "method": pd.Series(dtype="string"),
            "details": pd.Series(dtype="object"),
        }
    )


def _empty_quality_issues() -> pd.DataFrame:
    """Return an empty canonical quality-issue table."""
    return pd.DataFrame(
        {
            "context": pd.Series(dtype="string"),
            "source": pd.Series(dtype="string"),
            "start": pd.Series(dtype="datetime64[ns, UTC]"),
            "end": pd.Series(dtype="datetime64[ns, UTC]"),
            "test_name": pd.Series(dtype="string"),
            "method": pd.Series(dtype="string"),
            "severity": pd.Series(dtype="string"),
            "code": pd.Series(dtype="string"),
            "details": pd.Series(dtype="object"),
        }
    )


def _validate_sources(
    sources: Mapping[str, pd.DataFrame],
    *,
    grid: TimeGrid,
) -> dict[str, pd.DataFrame]:
    """Validate named source frames independently."""
    if not isinstance(sources, Mapping):
        raise TypeError("Data-quality sources must be supplied as a mapping.")

    if not sources:
        raise ValueError(
            "At least one time-series source must be supplied "
            "for data-quality evaluation."
        )

    validated: dict[str, pd.DataFrame] = {}

    for source_name, data in sources.items():
        if not isinstance(source_name, str) or not source_name.strip():
            raise ValueError(
                "Data-quality source names must be non-empty strings."
            )

        validated[source_name] = validate_time_series(
            data,
            grid=grid,
        )

    return validated


def _selected_source_names(
    sources: Mapping[str, pd.DataFrame],
    *,
    test: Mapping[str, Any],
) -> list[str]:
    """Resolve selected sources in supplied source-mapping order."""
    if "sources" not in test:
        return list(sources)

    requested = set(test["sources"])
    unknown = requested - set(sources)

    if unknown:
        raise ValueError(
            f"Data-quality test {test['name']!r} references "
            f"unknown sources: {sorted(unknown)!r}."
        )

    return [
        source_name
        for source_name in sources
        if source_name in requested
    ]


def _validate_requested_contexts_exist(
    sources: Mapping[str, pd.DataFrame],
    *,
    source_names: Sequence[str],
    test: Mapping[str, Any],
) -> None:
    """Require requested contexts to exist in at least one selected source."""
    if "contexts" not in test:
        return

    available = {
        context
        for source_name in source_names
        for context in sources[source_name].columns
    }

    missing = set(test["contexts"]) - available

    if missing:
        raise ValueError(
            f"Data-quality test {test['name']!r} references contexts "
            "that are unavailable in all selected sources: "
            f"{sorted(missing)!r}."
        )


def _selected_contexts(
    data: pd.DataFrame,
    *,
    test: Mapping[str, Any],
) -> list[str]:
    """Resolve selected contexts in source-column order."""
    if "contexts" not in test:
        return list(data.columns)

    requested = set(test["contexts"])

    return [
        context
        for context in data.columns
        if context in requested
    ]


def _evaluate_test_for_source(
    data: pd.DataFrame,
    *,
    source_name: str,
    contexts: Sequence[str],
    test: Mapping[str, Any],
    grid: TimeGrid,
) -> list[dict[str, Any]]:
    """Evaluate one quality test for one source."""
    if not contexts:
        return []

    method = METHODS[test["method"]]

    selected = data.loc[:, list(contexts)]

    mask = method.evaluate(
        selected,
        test=test,
        grid=grid,
    )

    failures: list[dict[str, Any]] = []

    for context in contexts:
        periods = failure_mask_to_periods(
            mask[context],
            grid=grid,
        )

        for start, end in periods:
            failures.append(
                {
                    "context": context,
                    "source": source_name,
                    "start": start,
                    "end": end,
                    "test_name": test["name"],
                    "method": test["method"],
                    "details": method.build_details(
                        data[context],
                        start=start,
                        end=end,
                        test=test,
                        grid=grid,
                    ),
                }
            )

    return failures


def _build_failure_frame(
    rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build a canonical quality-failure frame from event records."""
    if not rows:
        return _empty_quality_failures()

    failures = pd.DataFrame(rows, columns=_FAILURE_COLUMNS)

    for column in [
        "context",
        "source",
        "test_name",
        "method",
    ]:
        failures[column] = failures[column].astype("string")

    failures["start"] = pd.to_datetime(
        failures["start"],
        utc=True,
    )
    failures["end"] = pd.to_datetime(
        failures["end"],
        utc=True,
    )

    return failures


def evaluate(
    sources: Mapping[str, pd.DataFrame],
    *,
    tests: Sequence[Mapping[str, Any]],
    grid: TimeGrid,
) -> QualityEvaluation:
    """Evaluate configured data-quality tests across named sources.

    Tests execute in configured order. Source-specific tests are evaluated
    independently for every selected source and applicable context.

    This function does not modify source data.

    Args:
        sources: Canonical time-series frames keyed by source name.
        tests: Ordered data-quality test configurations.
        grid: Temporal grid against which sources and failure periods are
            evaluated.

    Returns:
        Quality failures and evaluation issues.

    Raises:
        TypeError: If sources or tests violate their structural contracts.
        ValueError: If source/context selectors cannot be resolved or test
            configuration is invalid.
        pandera.errors.SchemaErrors: If source or result frames violate
            canonical T-Clean schemas.
    """
    validated_sources = _validate_sources(
        sources,
        grid=grid,
    )

    validated_tests = validate_quality_tests(
        tests,
        grid=grid,
    )

    failure_rows: list[dict[str, Any]] = []

    for test in validated_tests:
        source_names = _selected_source_names(
            validated_sources,
            test=test,
        )

        _validate_requested_contexts_exist(
            validated_sources,
            source_names=source_names,
            test=test,
        )

        for source_name in source_names:
            data = validated_sources[source_name]

            contexts = _selected_contexts(
                data,
                test=test,
            )

            failure_rows.extend(
                _evaluate_test_for_source(
                    data,
                    source_name=source_name,
                    contexts=contexts,
                    test=test,
                    grid=grid,
                )
            )

    failures = validate_quality_failures(
        _build_failure_frame(failure_rows),
        grid=grid,
    )

    issues = validate_quality_issues(
        _empty_quality_issues(),
        grid=grid,
    )

    return QualityEvaluation(
        failures=failures,
        issues=issues,
    )
