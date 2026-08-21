"""Apply validated advanced auxiliary-fill rules."""

from collections.abc import Mapping

import pandas as pd

from tclean.advanced.methods.construct_from_sources import (
    METHOD_NAME as CONSTRUCT_FROM_SOURCES,
)
from tclean.advanced.methods.external_profile import METHOD_NAME as EXTERNAL_PROFILE
from tclean.validation import (
    validate_advanced_fill_rules,
    validate_cleaning_method,
    validate_load,
)

LEAVE_MISSING = "leave_missing"

_EXACT_ALIGNMENT = "exact"
_OVERLAP_ALIGNMENT = "overlap"

_SOURCE_ALIGNMENT = {
    CONSTRUCT_FROM_SOURCES: _EXACT_ALIGNMENT,
    EXTERNAL_PROFILE: _OVERLAP_ALIGNMENT,
}


def _apply_advanced_source(
    load: pd.DataFrame,
    cleaning_method: pd.DataFrame,
    source: pd.Series,
    *,
    country: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    scope: str,
    rule_name: str,
    alignment: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply an advanced source to the target time series."""
    if country not in load.columns:
        raise ValueError(f"Target country {country!r} is not present in load data.")

    target_index = load.index[(load.index >= start) & (load.index < end)]

    if alignment == _EXACT_ALIGNMENT:
        if not source.index.equals(target_index):
            raise ValueError(
                "Constructed source index must exactly match the target period."
            )

        candidate = source

    elif alignment == _OVERLAP_ALIGNMENT:
        candidate = source.loc[(source.index >= start) & (source.index < end)]

        candidate = candidate.loc[candidate.index.intersection(load.index)]

    else:
        raise ValueError(f"Unsupported source alignment {alignment!r}.")

    if scope == "fill_gaps":
        replace_index = candidate.index[load.loc[candidate.index, country].isna()]

    elif scope == "overwrite":
        replace_index = candidate.index

    else:
        raise ValueError(f"Unsupported advanced fill scope: {scope!r}")

    filled = load.copy()
    methods = cleaning_method.copy()

    filled.loc[replace_index, country] = candidate.loc[replace_index]

    methods.loc[replace_index, country] = rule_name

    return filled, methods


def _validate_advanced_sources(
    rules: pd.DataFrame, advanced_sources: Mapping[str, pd.Series]
) -> None:
    """Require exact agreement between rules and supplied sources."""
    required_sources = set(rules.loc[rules["source"].notna(), "source"])

    supplied_sources = set(advanced_sources)

    missing = sorted(required_sources - supplied_sources)

    unused = sorted(supplied_sources - required_sources)

    if missing or unused:
        raise ValueError(
            "Advanced sources must exactly match the sources referenced "
            "by advanced rules. "
            f"Missing: {missing!r}; unused: {unused!r}."
        )


def apply_auxiliary_fill_rule(
    load: pd.DataFrame,
    cleaning_method: pd.DataFrame,
    *,
    rule: pd.Series,
    source: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply one validated advanced-fill rule.

    Args:
        load: Canonical hourly time-series data.
        cleaning_method: Provenance labels aligned with ``load``.
        rule: One validated advanced-fill rule.
        source: Advanced time-series source referenced by the rule,
            if required.

    Returns:
        Updated time-series and provenance data.

    Raises:
        ValueError: If a required source is absent or the rule cannot
            be applied.
    """
    method = rule["method"]
    rule_name = rule["rule_name"]

    if method == LEAVE_MISSING:
        return (load.copy(), cleaning_method.copy())

    if method not in _SOURCE_ALIGNMENT:
        raise ValueError(f"Unsupported advanced-fill method {method!r}.")

    if source is None:
        raise ValueError(
            f"Advanced-fill rule {rule_name!r} requires an advanced source."
        )

    return _apply_advanced_source(
        load,
        cleaning_method,
        source,
        country=rule["country"],
        start=rule["start"],
        end=rule["end"],
        scope=rule["scope"],
        rule_name=rule_name,
        alignment=_SOURCE_ALIGNMENT[method],
    )


def apply_auxiliary_fill_rules(
    load: pd.DataFrame,
    cleaning_method: pd.DataFrame,
    *,
    rules: pd.DataFrame,
    advanced_sources: Mapping[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply validated advanced-fill rules in table order.

    Args:
        load: Canonical hourly time-series data.
        cleaning_method: Provenance labels aligned with ``load``.
        rules: Advanced-fill rules in the order they should be applied.
        advanced_sources: Advanced time-series sources keyed by source name.

    Returns:
        Time-series and provenance data after sequential rule application.

    Raises:
        ValueError: If supplied advanced sources do not exactly match
            the sources referenced by the rules.
    """
    filled = validate_load(load)

    methods = validate_cleaning_method(cleaning_method, load=filled)

    rules = validate_advanced_fill_rules(rules)

    _validate_advanced_sources(rules, advanced_sources)

    filled = filled.copy()
    methods = methods.copy()

    for _, rule in rules.iterrows():
        source = None if pd.isna(rule["source"]) else advanced_sources[rule["source"]]

        filled, methods = apply_auxiliary_fill_rule(
            filled, methods, rule=rule, source=source
        )

    return (filled, methods)
