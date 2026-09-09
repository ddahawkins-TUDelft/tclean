"""Evaluate configured data-quality tests."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from time import perf_counter
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype

from tclean.data_quality._method import MethodContext, MethodIssue, MethodResult
from tclean.data_quality._periods import failure_mask_to_periods
from tclean.data_quality._validation import (
    validate_quality_failures,
    validate_quality_issues,
)
from tclean.data_quality.methods import METHODS
from tclean.data_quality.rule_validation import validate_quality_tests
from tclean.time_grid import TimeGrid
from tclean.validation import validate_time_series

logger = logging.getLogger(__name__)

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
    sources: Mapping[str, pd.DataFrame], *, grid: TimeGrid
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
            raise ValueError("Data-quality source names must be non-empty strings.")

        validated[source_name] = validate_time_series(data, grid=grid)

    return validated


def _validate_threads(threads: int) -> int:
    """Validate and normalize the available thread count."""
    if isinstance(threads, bool) or not isinstance(threads, Integral):
        raise TypeError("threads must be an integer.")

    threads = int(threads)

    if threads < 1:
        raise ValueError("threads must be at least 1.")

    return threads


def _selected_source_names(
    sources: Mapping[str, pd.DataFrame], *, test: Mapping[str, Any]
) -> list[str]:
    """Resolve selected focal sources in supplied source-mapping order."""
    if "sources" not in test:
        return list(sources)

    requested = set(test["sources"])
    unknown = requested - set(sources)

    if unknown:
        raise ValueError(
            f"Data-quality test {test['name']!r} references "
            f"unknown sources: {sorted(unknown)!r}."
        )

    return [source_name for source_name in sources if source_name in requested]


def _validate_requested_contexts_exist(
    sources: Mapping[str, pd.DataFrame],
    *,
    source_names: Sequence[str],
    test: Mapping[str, Any],
) -> None:
    """Require requested contexts to exist in at least one focal source."""
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


def _selected_contexts(data: pd.DataFrame, *, test: Mapping[str, Any]) -> list[str]:
    """Resolve selected contexts in focal-source column order."""
    if "contexts" not in test:
        return list(data.columns)

    requested = set(test["contexts"])

    return [context for context in data.columns if context in requested]


def _validate_method_result(result: MethodResult, *, context: MethodContext) -> None:
    """Require a method result to satisfy the framework mask contract."""
    if not isinstance(result, MethodResult):
        raise TypeError(
            f"Data-quality method {context.test['method']!r} must return MethodResult."
        )

    mask = result.mask
    target = context.target_data

    if not isinstance(mask, pd.DataFrame):
        raise TypeError(
            f"Data-quality method {context.test['method']!r} returned a non-DataFrame "
            "failure mask."
        )

    if not mask.index.equals(target.index) or not mask.columns.equals(target.columns):
        raise ValueError(
            f"Data-quality method {context.test['method']!r} returned a failure mask "
            "that is not aligned with its focal data."
        )

    if any(not is_bool_dtype(dtype) for dtype in mask.dtypes):
        raise TypeError(
            f"Data-quality method {context.test['method']!r} returned a failure mask "
            "with non-Boolean columns."
        )

    if mask.isna().to_numpy().any():
        raise ValueError(
            f"Data-quality method {context.test['method']!r} returned a failure mask "
            "containing missing values."
        )

    if any(not isinstance(issue, MethodIssue) for issue in result.issues):
        raise TypeError(
            f"Data-quality method {context.test['method']!r} returned an invalid "
            "method issue."
        )

    unknown_issue_contexts = {
        issue.context for issue in result.issues if issue.context not in target.columns
    }

    if unknown_issue_contexts:
        raise ValueError(
            f"Data-quality method {context.test['method']!r} returned issues for "
            f"unknown focal contexts: {sorted(unknown_issue_contexts)!r}."
        )


def _evaluate_test_for_source(
    sources: Mapping[str, pd.DataFrame],
    *,
    source_name: str,
    contexts: Sequence[str],
    test: Mapping[str, Any],
    grid: TimeGrid,
    preceding_failures: Sequence[Mapping[str, Any]],
    threads: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate one quality test for one focal source."""
    if not contexts:
        return [], []

    method = METHODS[test["method"]]

    context = MethodContext(
        source_name=source_name,
        sources=sources,
        contexts=tuple(contexts),
        test=test,
        grid=grid,
        threads=threads,
        preceding_failures=tuple(preceding_failures),
    )

    result = method.evaluate(context)
    _validate_method_result(result, context=context)

    failures: list[dict[str, Any]] = []

    for context_name in contexts:
        periods = failure_mask_to_periods(result.mask[context_name], grid=grid)

        for start, end in periods:
            details = method.build_details(
                context, result, context_name=context_name, start=start, end=end
            )

            failures.append(
                {
                    "context": context_name,
                    "source": source_name,
                    "start": start,
                    "end": end,
                    "test_name": test["name"],
                    "method": method.name,
                    "details": details,
                }
            )

    issues = [
        {
            "context": issue.context,
            "source": source_name,
            "start": issue.start,
            "end": issue.end,
            "test_name": test["name"],
            "method": method.name,
            "severity": issue.severity,
            "code": issue.code,
            "details": dict(issue.details),
        }
        for issue in result.issues
    ]

    return failures, issues


def _build_failure_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a canonical quality-failure frame from event records."""
    if not rows:
        return _empty_quality_failures()

    failures = pd.DataFrame(rows, columns=_FAILURE_COLUMNS)

    for column in ["context", "source", "test_name", "method"]:
        failures[column] = failures[column].astype("string")

    failures["start"] = pd.to_datetime(failures["start"], utc=True)
    failures["end"] = pd.to_datetime(failures["end"], utc=True)

    return failures


def _build_issue_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a canonical quality-issue frame from event records."""
    if not rows:
        return _empty_quality_issues()

    issues = pd.DataFrame(rows, columns=_ISSUE_COLUMNS)

    for column in ["context", "source", "test_name", "method", "severity", "code"]:
        issues[column] = issues[column].astype("string")

    issues["start"] = pd.to_datetime(issues["start"], utc=True)
    issues["end"] = pd.to_datetime(issues["end"], utc=True)

    return issues


def evaluate(
    sources: Mapping[str, pd.DataFrame],
    *,
    tests: Sequence[Mapping[str, Any]],
    grid: TimeGrid,
    threads: int = 1,
) -> QualityEvaluation:
    """Evaluate configured data-quality tests across named sources.

    Tests execute in configured order. ``sources`` selectors choose focal
    sources for a test; every supplied source remains available to the method
    through its execution context for cross-source evidence when needed.

    This function does not modify source data.

    Args:
        sources: Canonical time-series frames keyed by source name.
        tests: Ordered data-quality test configurations.
        grid: Temporal grid against which sources and failure periods are
            evaluated.
        threads: Maximum number of threads available to data-quality methods.
            Methods may use fewer threads when parallel execution would not
            be beneficial. Defaults to 1.

    Returns:
        Quality failures and evaluation issues.

    Raises:
        TypeError: If sources or tests violate their structural contracts.
        ValueError: If source/context selectors cannot be resolved or test
            configuration is invalid.
        pandera.errors.SchemaErrors: If source or result frames violate
            canonical T-Clean schemas.
    """
    threads = _validate_threads(threads)
    validated_sources = _validate_sources(sources, grid=grid)
    validated_tests = validate_quality_tests(tests, grid=grid)

    logger.info(
        "Tclean data-quality evaluation: tests=%d | threads=%d",
        len(validated_tests),
        threads,
    )

    failure_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []

    for test_number, test in enumerate(validated_tests, start=1):
        test_started = perf_counter()

        logger.debug(
            "Starting data-quality test %d/%d: %s [%s]",
            test_number,
            len(validated_tests),
            test["name"],
            test["method"],
        )
        preceding_failures = tuple(failure_rows)
        current_test_failure_rows: list[dict[str, Any]] = []
        current_test_issue_rows: list[dict[str, Any]] = []

        source_names = _selected_source_names(validated_sources, test=test)

        _validate_requested_contexts_exist(
            validated_sources, source_names=source_names, test=test
        )

        for source_name in source_names:
            data = validated_sources[source_name]
            contexts = _selected_contexts(data, test=test)

            source_failures, source_issues = _evaluate_test_for_source(
                validated_sources,
                source_name=source_name,
                contexts=contexts,
                test=test,
                grid=grid,
                preceding_failures=preceding_failures,
                threads=threads,
            )

            current_test_failure_rows.extend(source_failures)
            current_test_issue_rows.extend(source_issues)

        failure_rows.extend(current_test_failure_rows)
        issue_rows.extend(current_test_issue_rows)

        logger.debug(
            "Completed data-quality test %d/%d: %s [%s] in %.2fs",
            test_number,
            len(validated_tests),
            test["name"],
            test["method"],
            perf_counter() - test_started,
        )

    failures = validate_quality_failures(_build_failure_frame(failure_rows), grid=grid)
    issues = validate_quality_issues(_build_issue_frame(issue_rows), grid=grid)

    return QualityEvaluation(failures=failures, issues=issues)
