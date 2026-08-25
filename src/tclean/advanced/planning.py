"""Plan auxiliary data required for advanced time series cleaning."""

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from tclean.basic.methods.average_periods import METHOD_NAME as AVERAGE_PERIODS
from tclean.basic.methods.copy_periods import METHOD_NAME as COPY_PERIODS
from tclean.basic.methods.linear_interpolation import (
    METHOD_NAME as LINEAR_INTERPOLATION,
)
from tclean.basic.rule_validation import validate_basic_rules
from tclean.time_grid import TimeGrid
from tclean.validation import (
    validate_advanced_fill_rules,
    validate_auxiliary_requirements,
    validate_auxiliary_source_requests,
    validate_source_capabilities,
    validate_source_periods,
)

REQUIREMENT_COLUMNS = ["context", "start", "end"]

SOURCE_REQUEST_COLUMNS = ["source", "context", "start", "end"]


def compile_auxiliary_requirements(
    source_periods: Sequence[pd.DataFrame], *, grid: TimeGrid
) -> pd.DataFrame:
    """Compile and merge auxiliary context-period requirements."""
    if not source_periods:
        return _empty_requirements()

    validated_periods = [
        validate_source_periods(periods, grid=grid) for periods in source_periods
    ]

    requirements = pd.concat(
        [periods[["context", "start", "end"]] for periods in validated_periods],
        ignore_index=True,
    )

    requirements = (
        requirements.drop_duplicates()
        .sort_values(["context", "start", "end"])
        .reset_index(drop=True)
    )

    merged = _merge_requirements(requirements)

    return validate_auxiliary_requirements(merged, grid=grid)


def _empty_requirements() -> pd.DataFrame:
    """Return an empty canonical auxiliary-requirements table."""
    return pd.DataFrame(
        {
            "context": pd.Series(dtype="string"),
            "start": pd.Series(dtype="datetime64[ns, UTC]"),
            "end": pd.Series(dtype="datetime64[ns, UTC]"),
        }
    )


def _merge_requirements(requirements: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping or adjacent context-period requirements."""
    if requirements.empty:
        return requirements.copy()

    merged_rows: list[dict[str, object]] = []

    for context, context_requirements in requirements.groupby("context", sort=True):
        ordered = context_requirements.sort_values(["start", "end"])

        current_start = ordered.iloc[0]["start"]
        current_end = ordered.iloc[0]["end"]

        for row in ordered.iloc[1:].itertuples(index=False):
            if row.start <= current_end:
                current_end = max(current_end, row.end)
                continue

            merged_rows.append(
                {"context": context, "start": current_start, "end": current_end}
            )

            current_start = row.start
            current_end = row.end

        merged_rows.append(
            {"context": context, "start": current_start, "end": current_end}
        )

    return pd.DataFrame(merged_rows, columns=REQUIREMENT_COLUMNS)


def get_basic_cleaning_context(
    rules: Sequence[Mapping[str, Any]], *, grid: TimeGrid
) -> tuple[pd.Timedelta, pd.Timedelta]:
    """Calculate context required by ordered basic cleaning rules.

    Context accumulates through the rule sequence because later rules may
    depend on values that earlier rules can themselves only construct using
    additional surrounding data.

    Args:
        rules: Ordered basic cleaning-rule definitions.
        grid: Temporal grid against which the rules are validated.

    Returns:
        Required context before and after the exact auxiliary period.

    Raises:
        ValueError: If a rule uses an unsupported basic cleaning method.
    """
    rules = validate_basic_rules(rules, grid=grid)

    left_context = pd.Timedelta(0)
    right_context = pd.Timedelta(0)

    for rule in rules:
        method = rule["method"]
        max_gap = rule["max_gap"]

        # Context is required on both sides to classify complete missing
        # runs correctly at the boundary of the requested period.
        rule_left = -max_gap
        rule_right = max_gap

        if method == LINEAR_INTERPOLATION:
            offsets = (-grid.frequency, grid.frequency)

        elif method == COPY_PERIODS:
            offsets = (rule["source_offset"],)

        elif method == AVERAGE_PERIODS:
            offsets = tuple(rule["source_offsets"])

        else:
            raise ValueError(f"Unsupported basic gap-filling method: {method!r}")

        previous_left = left_context
        previous_right = right_context

        for offset in offsets:
            rule_left = min(rule_left, offset + previous_left)

            rule_right = max(rule_right, offset + previous_right)

        left_context = min(previous_left, rule_left)

        right_context = max(previous_right, rule_right)

    return -left_context, right_context


def expand_auxiliary_requirements(
    requirements: pd.DataFrame,
    *,
    rules: Sequence[Mapping[str, Any]],
    grid: TimeGrid,
    enabled: bool = True,
) -> pd.DataFrame:
    """Expand auxiliary requirements for basic-cleaning context.

    Args:
        requirements: Exact auxiliary context-period requirements.
        rules: Ordered basic cleaning rules.
        grid: Temporal grid against which requirements and rules are
            validated.
        enabled: Whether basic cleaning will be applied to auxiliary data.

    Returns:
        Requirements expanded by the context needed for basic cleaning.
    """
    requirements = validate_auxiliary_requirements(requirements, grid=grid)

    if requirements.empty or not enabled or not rules:
        return requirements.copy()

    left_context, right_context = get_basic_cleaning_context(rules, grid=grid)

    expanded = requirements.copy()

    expanded["start"] = expanded["start"] - left_context

    expanded["end"] = expanded["end"] + right_context

    expanded = _merge_requirements(expanded)

    return validate_auxiliary_requirements(expanded, grid=grid)


def build_auxiliary_acquisition_requirements(
    source_periods: Sequence[pd.DataFrame],
    *,
    basic_rules: Sequence[Mapping[str, Any]],
    grid: TimeGrid,
    basic_cleaning_enabled: bool,
) -> pd.DataFrame:
    """Build auxiliary periods required for acquisition.

    Args:
        source_periods: Source-period tables required by advanced cleaning.
        basic_rules: Ordered rules used to clean acquired auxiliary data.
        grid: Temporal grid against which requirements and rules are
            validated.
        basic_cleaning_enabled: Whether basic cleaning is applied to
            auxiliary data.

    Returns:
        Merged acquisition requirements including required cleaning context.
    """
    exact_requirements = compile_auxiliary_requirements(source_periods, grid=grid)

    return expand_auxiliary_requirements(
        exact_requirements, rules=basic_rules, grid=grid, enabled=basic_cleaning_enabled
    )


def _empty_source_requests() -> pd.DataFrame:
    """Return an empty canonical auxiliary-source request table."""
    return pd.DataFrame(
        {
            "source": pd.Series(dtype="string"),
            "context": pd.Series(dtype="string"),
            "start": pd.Series(dtype="datetime64[ns, UTC]"),
            "end": pd.Series(dtype="datetime64[ns, UTC]"),
        }
    )


def build_auxiliary_source_requests(
    requirements: pd.DataFrame, *, source_capabilities: pd.DataFrame, grid: TimeGrid
) -> pd.DataFrame:
    """Map auxiliary requirements onto capable data sources.

    A capability row with a missing context applies to all required
    contexts. Explicit context capabilities apply only to that context.

    Args:
        requirements: Required auxiliary context-periods.
        source_capabilities: Explicit source capability definitions.
        grid: Temporal grid against which requests are validated.

    Returns:
        Source-context-period acquisition requests.

    Raises:
        ValueError: If a required context is unsupported by every configured
            source.
        pandera.errors.SchemaErrors: If an input violates its T-Clean
            data contract.
    """
    requirements = validate_auxiliary_requirements(requirements, grid=grid)

    if requirements.empty:
        return _empty_source_requests()

    source_capabilities = validate_source_capabilities(source_capabilities)

    request_frames: list[pd.DataFrame] = []

    for capability in source_capabilities.itertuples(index=False):
        if pd.isna(capability.context):
            applicable = requirements.copy()

        else:
            applicable = requirements.loc[
                requirements["context"] == capability.context
            ].copy()

        if applicable.empty:
            continue

        applicable.insert(0, "source", capability.source)

        request_frames.append(applicable)

    if not request_frames:
        missing_contexts = sorted(requirements["context"].unique().tolist())

        raise ValueError(
            "No configured auxiliary source supports required "
            f"contexts: {missing_contexts!r}."
        )

    requests = pd.concat(request_frames, ignore_index=True)

    covered_contexts = set(requests["context"])

    required_contexts = set(requirements["context"])

    missing_contexts = sorted(required_contexts - covered_contexts)

    if missing_contexts:
        raise ValueError(
            "No configured auxiliary source supports required "
            f"contexts: {missing_contexts!r}."
        )

    requests = (
        requests.drop_duplicates()
        .sort_values(["source", "context", "start", "end"])
        .reset_index(drop=True)
    )

    return validate_auxiliary_source_requests(requests, grid=grid)


def select_active_advanced_rules(
    rules: pd.DataFrame, *, target_contexts: Sequence[str], grid: TimeGrid
) -> pd.DataFrame:
    """Select advanced-fill rules intersecting the target model scope.

    Args:
        rules: Canonical advanced-fill rules in execution order.
        target_contexts: Contexts included in the target model scope.
        grid: Temporal grid defining the target model period.

    Returns:
        Active advanced-fill rules in their original execution order.

    Raises:
        pandera.errors.SchemaErrors: If the advanced rules violate their
            T-Clean contract.
    """
    rules = validate_advanced_fill_rules(rules, grid=grid)

    context_intersects = rules["context"].isin(target_contexts)

    period_intersects = (rules["start"] < grid.end) & (rules["end"] > grid.start)

    return rules.loc[context_intersects & period_intersects].reset_index(drop=True)
