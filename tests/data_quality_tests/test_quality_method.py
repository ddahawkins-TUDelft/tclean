"""Tests for the data-quality method execution contract."""

import pandas as pd
import pytest

from tclean import TimeGrid
from tclean.data_quality._method import MethodContext, MethodResult, MethodSpec
from tclean.data_quality.methods import METHODS, build_method_registry


def _grid() -> TimeGrid:
    """Return a four-hour test grid."""
    return TimeGrid(
        start="2026-01-01T00:00:00Z", end="2026-01-01T04:00:00Z", frequency="1h"
    )


def _source(**contexts: list[float]) -> pd.DataFrame:
    """Build source data on the test grid."""
    return pd.DataFrame(contexts, index=_grid().target_index)


def _context(*, preceding_failures=()) -> MethodContext:
    """Build a multi-source execution context."""
    sources = {
        "primary": _source(A=[1, 2, 3, 4], B=[10, 20, 30, 40]),
        "secondary": _source(A=[5, 6, 7, 8]),
    }

    return MethodContext(
        source_name="primary",
        sources=sources,
        contexts=("A", "B"),
        test={"name": "example", "method": "example"},
        grid=_grid(),
        preceding_failures=tuple(preceding_failures),
    )


def test_method_context_exposes_all_sources_but_one_focal_target():
    """Keep every supplied source available while selecting one focal source."""
    context = _context()

    assert context.source_names == ("primary", "secondary")
    assert context.target_data.columns.tolist() == ["A", "B"]
    assert context.target_data.iloc[0].tolist() == [1, 10]


def test_method_context_uses_only_available_requested_contexts_for_other_source():
    """Intersect selected contexts with contexts available in another source."""
    context = _context()

    secondary = context.source_data("secondary")

    assert secondary.columns.tolist() == ["A"]
    assert secondary["A"].tolist() == [5, 6, 7, 8]


def test_method_context_reference_data_filters_preceding_failures_by_source():
    """Apply preceding-failure eligibility rules to whichever source is requested."""
    failure = {
        "context": "A",
        "source": "secondary",
        "start": pd.Timestamp("2026-01-01T01:00:00Z"),
        "end": pd.Timestamp("2026-01-01T02:00:00Z"),
        "test_name": "earlier",
        "method": "range",
        "details": {},
    }

    context = _context(preceding_failures=[failure])

    secondary_reference = context.reference_data("secondary")
    primary_reference = context.reference_data("primary")

    assert pd.isna(secondary_reference.loc[failure["start"], "A"])
    assert primary_reference.loc[failure["start"], "A"] == 2


def test_method_context_reference_data_is_cached():
    """Reuse an already constructed failure-filtered reference frame."""
    context = _context()

    first = context.reference_data("secondary")
    second = context.reference_data("secondary")

    assert second is first


def _validate(test, *, grid):
    del grid
    return dict(test)


def _evaluate(context):
    data = context.target_data
    return MethodResult(
        mask=pd.DataFrame(False, index=data.index, columns=data.columns, dtype=bool)
    )


def _build_details(context, result, *, context_name, start, end):
    del context, result, context_name, start, end
    return {}


def _method(name: str) -> MethodSpec:
    return MethodSpec(
        name=name, validate=_validate, evaluate=_evaluate, build_details=_build_details
    )


def test_registered_methods_are_explicit_method_specs():
    """Register concrete method specifications rather than Python modules."""
    assert METHODS
    assert all(isinstance(method, MethodSpec) for method in METHODS.values())
    assert tuple(METHODS) == tuple(method.name for method in METHODS.values())


def test_method_registry_rejects_duplicate_names():
    """Reject ambiguous method registrations immediately."""
    with pytest.raises(ValueError, match="Duplicate data-quality method name"):
        build_method_registry([_method("same"), _method("same")])


def test_method_registry_is_immutable():
    """Prevent runtime mutation of the registered method contract."""
    registry = build_method_registry([_method("example")])

    with pytest.raises(TypeError):
        registry["other"] = _method("other")  # type: ignore[index]
