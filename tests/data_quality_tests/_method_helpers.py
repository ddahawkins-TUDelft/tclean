"""Helpers for exercising registered data-quality methods in isolation."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from tclean.data_quality._method import MethodContext, MethodIssue, MethodResult
from tclean.time_grid import TimeGrid


def method_context(
    data: pd.DataFrame | pd.Series,
    *,
    test: Mapping[str, Any],
    grid: TimeGrid,
    source_name: str = "primary",
    sources: Mapping[str, pd.DataFrame] | None = None,
    contexts: Sequence[str] | None = None,
    preceding_failures: Sequence[Mapping[str, Any]] = (),
) -> MethodContext:
    """Build a method context for focused method tests."""
    if isinstance(data, pd.Series):
        context_name = data.name if isinstance(data.name, str) else "A"
        focal_data = data.to_frame(context_name)
    else:
        focal_data = data

    supplied_sources = {source_name: focal_data} if sources is None else sources
    selected_contexts = tuple(focal_data.columns if contexts is None else contexts)

    return MethodContext(
        source_name=source_name,
        sources=supplied_sources,
        contexts=selected_contexts,
        test=test,
        grid=grid,
        preceding_failures=tuple(preceding_failures),
    )


def method_evaluation(
    evaluate: Callable[[MethodContext], MethodResult],
    data: pd.DataFrame | pd.Series,
    *,
    test: Mapping[str, Any],
    grid: TimeGrid,
) -> MethodResult:
    """Evaluate one method against a minimal focal-source context."""
    return evaluate(method_context(data, test=test, grid=grid))


def method_mask(
    evaluate: Callable[[MethodContext], MethodResult],
    data: pd.DataFrame | pd.Series,
    *,
    test: Mapping[str, Any],
    grid: TimeGrid,
) -> pd.DataFrame:
    """Return only the failure mask from an isolated method evaluation."""
    return method_evaluation(evaluate, data, test=test, grid=grid).mask


def method_details(
    evaluate: Callable[[MethodContext], MethodResult],
    build_details: Callable[..., dict[str, Any]],
    data: pd.DataFrame | pd.Series,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    test: Mapping[str, Any],
    grid: TimeGrid,
    context_name: str | None = None,
) -> dict[str, Any]:
    """Build details for one isolated method failure period."""
    context = method_context(data, test=test, grid=grid)
    result = evaluate(context)

    if context_name is None:
        context_name = context.contexts[0]

    return build_details(
        context, result, context_name=context_name, start=start, end=end
    )


def method_evaluation_pair(
    evaluate: Callable[[MethodContext], MethodResult],
    data: pd.DataFrame | pd.Series,
    *,
    test: Mapping[str, Any],
    grid: TimeGrid,
) -> tuple[pd.DataFrame, tuple[MethodIssue, ...]]:
    """Return mask and issues for concise focused method tests."""
    result = method_evaluation(evaluate, data, test=test, grid=grid)
    return result.mask, result.issues
