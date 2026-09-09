"""Core interfaces for data-quality evaluation methods."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

from tclean.data_quality._reference import build_reference_data
from tclean.time_grid import TimeGrid


@dataclass(frozen=True)
class MethodIssue:
    """One undecorated issue produced while evaluating a quality method."""

    context: str
    start: pd.Timestamp
    end: pd.Timestamp
    severity: str
    code: str
    details: Mapping[str, Any]


@dataclass(frozen=True)
class MethodResult:
    """Result returned by every data-quality method evaluation."""

    mask: pd.DataFrame
    issues: tuple[MethodIssue, ...] = ()


@dataclass(frozen=True)
class MethodContext:
    """Data and execution state available to one method/source evaluation.

    A method always has one focal source, but may inspect any supplied source.
    ``reference_data`` applies the framework's preceding-failure eligibility
    rules to whichever source is requested. This lets methods construct
    same-source historical references, cross-source comparisons, or future
    combinations without requiring evaluator-level execution modes.
    """

    source_name: str
    sources: Mapping[str, pd.DataFrame]
    contexts: tuple[str, ...]
    test: Mapping[str, Any]
    grid: TimeGrid
    threads: int = 1
    preceding_failures: tuple[Mapping[str, Any], ...] = ()
    _reference_cache: dict[tuple[str, tuple[str, ...]], pd.DataFrame] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def source_names(self) -> tuple[str, ...]:
        """Return all supplied source names in their configured order."""
        return tuple(self.sources)

    @property
    def target_data(self) -> pd.DataFrame:
        """Return selected contexts for the focal source."""
        return self.source_data(self.source_name)

    def available_contexts(
        self, source_name: str, *, contexts: Sequence[str] | None = None
    ) -> tuple[str, ...]:
        """Return requested contexts available in one supplied source."""
        data = self.sources[source_name]
        requested = self.contexts if contexts is None else tuple(contexts)

        return tuple(context for context in requested if context in data.columns)

    def source_data(
        self, source_name: str, *, contexts: Sequence[str] | None = None
    ) -> pd.DataFrame:
        """Return requested available contexts from one supplied source."""
        selected_contexts = self.available_contexts(source_name, contexts=contexts)

        return self.sources[source_name].loc[:, list(selected_contexts)]

    def reference_data(
        self, source_name: str, *, contexts: Sequence[str] | None = None
    ) -> pd.DataFrame:
        """Return failure-filtered evidence data for one supplied source.

        Failures from preceding tests are masked according to the current
        test's ``include_failed_periods_from`` configuration. Results are
        cached for the lifetime of this method context.
        """
        selected_contexts = self.available_contexts(source_name, contexts=contexts)
        cache_key = (source_name, selected_contexts)

        if cache_key not in self._reference_cache:
            self._reference_cache[cache_key] = build_reference_data(
                self.sources[source_name],
                source_name=source_name,
                contexts=selected_contexts,
                preceding_failures=self.preceding_failures,
                include_failed_periods_from=self.test.get(
                    "include_failed_periods_from", []
                ),
            )

        return self._reference_cache[cache_key]


class MethodValidator(Protocol):
    """Callable contract for quality-method configuration validation."""

    def __call__(
        self, test: Mapping[str, Any], *, grid: TimeGrid
    ) -> dict[str, Any]: ...


class MethodEvaluator(Protocol):
    """Callable contract for quality-method evaluation."""

    def __call__(self, context: MethodContext) -> MethodResult: ...


class MethodDetailsBuilder(Protocol):
    """Callable contract for failed-period diagnostic construction."""

    def __call__(
        self,
        context: MethodContext,
        result: MethodResult,
        *,
        context_name: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class MethodSpec:
    """Complete framework contract for one registered quality method."""

    name: str
    validate: MethodValidator
    evaluate: MethodEvaluator
    build_details: MethodDetailsBuilder
