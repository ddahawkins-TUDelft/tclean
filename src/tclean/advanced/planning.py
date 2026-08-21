"""Plan auxiliary data required for advanced demand cleaning."""

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from tclean.basic.methods.average_periods import METHOD_NAME as AVERAGE_PERIODS
from tclean.basic.methods.copy_periods import METHOD_NAME as COPY_PERIODS
from tclean.basic.methods.linear_interpolation import (
    METHOD_NAME as LINEAR_INTERPOLATION,
)
from tclean.validation import (
    validate_advanced_fill_rules,
    validate_auxiliary_requirements,
    validate_auxiliary_source_requests,
    validate_source_capabilities,
    validate_source_periods,
    validate_temporal_range,
)

REQUIREMENT_COLUMNS = ["country", "start", "end"]

SOURCE_REQUEST_COLUMNS = ["source", "country", "start", "end"]


def compile_auxiliary_requirements(
    source_periods: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Compile and merge auxiliary country-period requirements."""
    if not source_periods:
        return _empty_requirements()

    validated_periods = [validate_source_periods(periods) for periods in source_periods]

    requirements = pd.concat(
        [periods[["country", "start", "end"]] for periods in validated_periods],
        ignore_index=True,
    )

    requirements = (
        requirements.drop_duplicates()
        .sort_values(["country", "start", "end"])
        .reset_index(drop=True)
    )

    merged = _merge_requirements(requirements)

    return validate_auxiliary_requirements(merged)


def _empty_requirements() -> pd.DataFrame:
    """Return an empty canonical auxiliary-requirements table."""
    return pd.DataFrame(
        {
            "country": pd.Series(dtype="string"),
            "start": pd.Series(dtype="datetime64[ns, UTC]"),
            "end": pd.Series(dtype="datetime64[ns, UTC]"),
        }
    )


def _merge_requirements(requirements: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping or adjacent country-period requirements."""
    if requirements.empty:
        return requirements.copy()

    merged_rows: list[dict[str, object]] = []

    for country, country_requirements in requirements.groupby("country", sort=True):
        ordered = country_requirements.sort_values(["start", "end"])

        current_start = ordered.iloc[0]["start"]
        current_end = ordered.iloc[0]["end"]

        for row in ordered.iloc[1:].itertuples(index=False):
            if row.start <= current_end:
                current_end = max(current_end, row.end)
                continue

            merged_rows.append(
                {"country": country, "start": current_start, "end": current_end}
            )

            current_start = row.start
            current_end = row.end

        merged_rows.append(
            {"country": country, "start": current_start, "end": current_end}
        )

    return pd.DataFrame(merged_rows, columns=REQUIREMENT_COLUMNS)


def get_basic_cleaning_context(
    rules: Sequence[Mapping[str, Any]],
) -> tuple[pd.Timedelta, pd.Timedelta]:
    """Calculate context required by ordered basic cleaning rules.

    Context accumulates through the rule sequence because later rules may
    depend on values that earlier rules can themselves only construct using
    additional surrounding data.

    Args:
        rules: Ordered basic cleaning-rule definitions.

    Returns:
        Required context before and after the exact auxiliary period.

    Raises:
        ValueError: If a rule uses an unsupported basic cleaning method.
    """
    left_context = pd.Timedelta(0)
    right_context = pd.Timedelta(0)

    for rule in rules:
        method = rule["method"]
        max_gap = pd.Timedelta(rule["max_gap"])

        # Context is required on both sides to classify complete missing
        # runs correctly at the boundary of the requested period.
        rule_left = -max_gap
        rule_right = max_gap

        if method == LINEAR_INTERPOLATION:
            offsets = (-pd.Timedelta(hours=1), pd.Timedelta(hours=1))

        elif method == COPY_PERIODS:
            offsets = (pd.Timedelta(rule["source_offset"]),)

        elif method == AVERAGE_PERIODS:
            offsets = tuple(pd.Timedelta(offset) for offset in rule["source_offsets"])

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
    enabled: bool = True,
) -> pd.DataFrame:
    """Expand auxiliary requirements for basic-cleaning context.

    Args:
        requirements: Exact auxiliary country-period requirements.
        rules: Ordered basic cleaning rules.
        enabled: Whether basic cleaning will be applied to auxiliary data.

    Returns:
        Requirements expanded by the context needed for basic cleaning.
    """
    requirements = validate_auxiliary_requirements(requirements)

    if requirements.empty or not enabled or not rules:
        return requirements.copy()

    left_context, right_context = get_basic_cleaning_context(rules)

    expanded = requirements.copy()

    expanded["start"] = expanded["start"] - left_context
    expanded["end"] = expanded["end"] + right_context

    expanded = _merge_requirements(expanded)

    return validate_auxiliary_requirements(expanded)


def build_auxiliary_acquisition_requirements(
    source_periods: Sequence[pd.DataFrame],
    *,
    basic_rules: Sequence[Mapping[str, Any]],
    basic_cleaning_enabled: bool,
) -> pd.DataFrame:
    """Build auxiliary periods required for acquisition.

    Args:
        source_periods: Source-period tables required by advanced cleaning.
        basic_rules: Ordered rules used to clean acquired auxiliary data.
        basic_cleaning_enabled: Whether basic cleaning is applied to
            auxiliary data.

    Returns:
        Merged acquisition requirements including required cleaning context.
    """
    exact_requirements = compile_auxiliary_requirements(source_periods)

    return expand_auxiliary_requirements(
        exact_requirements, rules=basic_rules, enabled=basic_cleaning_enabled
    )


def _empty_source_requests() -> pd.DataFrame:
    """Return an empty canonical auxiliary-source request table."""
    return pd.DataFrame(
        {
            "source": pd.Series(dtype="string"),
            "country": pd.Series(dtype="string"),
            "start": pd.Series(dtype="datetime64[ns, UTC]"),
            "end": pd.Series(dtype="datetime64[ns, UTC]"),
        }
    )


def build_auxiliary_source_requests(
    requirements: pd.DataFrame, *, source_capabilities: pd.DataFrame
) -> pd.DataFrame:
    """Map auxiliary requirements onto capable data sources.

    A capability row with a missing country applies to all required
    countries. Explicit country capabilities apply only to that country.

    Args:
        requirements: Required auxiliary country-periods.
        source_capabilities: Explicit source capability definitions.

    Returns:
        Source-country-period acquisition requests.

    Raises:
        ValueError: If a required country is unsupported by every configured
            source.
        pandera.errors.SchemaErrors: If an input violates its T-Clean
            data contract.
    """
    requirements = validate_auxiliary_requirements(requirements)

    if requirements.empty:
        return _empty_source_requests()

    source_capabilities = validate_source_capabilities(source_capabilities)

    request_frames: list[pd.DataFrame] = []

    for capability in source_capabilities.itertuples(index=False):
        if pd.isna(capability.country):
            applicable = requirements.copy()
        else:
            applicable = requirements.loc[
                requirements["country"] == capability.country
            ].copy()

        if applicable.empty:
            continue

        applicable.insert(0, "source", capability.source)

        request_frames.append(applicable)

    if not request_frames:
        missing_countries = sorted(requirements["country"].unique().tolist())

        raise ValueError(
            "No configured auxiliary source supports required "
            f"countries: {missing_countries!r}."
        )

    requests = pd.concat(request_frames, ignore_index=True)

    covered_countries = set(requests["country"])
    required_countries = set(requirements["country"])

    missing_countries = sorted(required_countries - covered_countries)

    if missing_countries:
        raise ValueError(
            "No configured auxiliary source supports required "
            f"countries: {missing_countries!r}."
        )

    requests = (
        requests.drop_duplicates()
        .sort_values(["source", "country", "start", "end"])
        .reset_index(drop=True)
    )

    return validate_auxiliary_source_requests(requests)


def select_active_advanced_rules(
    rules: pd.DataFrame,
    *,
    target_countries: Sequence[str],
    target_start: object,
    target_end: object,
) -> pd.DataFrame:
    """Select advanced-fill rules intersecting the target model scope.

    Args:
        rules: Canonical advanced-fill rules in execution order.
        target_countries: Countries included in the target model scope.
        target_start: Inclusive start of the target model period.
        target_end: Exclusive end of the target model period.

    Returns:
        Active advanced-fill rules in their original execution order.

    Raises:
        pandera.errors.SchemaErrors: If the advanced rules or temporal
            range violate their T-Clean contracts.
    """
    rules = validate_advanced_fill_rules(rules)

    target_start, target_end = validate_temporal_range(
        start=target_start, end=target_end
    )

    country_intersects = rules["country"].isin(target_countries)

    period_intersects = (rules["start"] < target_end) & (rules["end"] > target_start)

    return rules.loc[country_intersects & period_intersects].reset_index(drop=True)
