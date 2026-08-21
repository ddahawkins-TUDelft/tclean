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

_PROFILE_ALIGNMENT = {
    CONSTRUCT_FROM_SOURCES: _EXACT_ALIGNMENT,
    EXTERNAL_PROFILE: _OVERLAP_ALIGNMENT,
}


def _apply_profile(
    load: pd.DataFrame,
    cleaning_method: pd.DataFrame,
    profile: pd.Series,
    *,
    country: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    scope: str,
    rule_name: str,
    alignment: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a validated auxiliary profile to target demand."""
    if country not in load.columns:
        raise ValueError(f"Target country {country!r} is not present in load data.")

    target_index = load.index[(load.index >= start) & (load.index < end)]

    if alignment == _EXACT_ALIGNMENT:
        if not profile.index.equals(target_index):
            raise ValueError(
                "Constructed profile index must exactly match the target period."
            )

        candidate = profile

    elif alignment == _OVERLAP_ALIGNMENT:
        candidate = profile.loc[(profile.index >= start) & (profile.index < end)]

        candidate = candidate.loc[candidate.index.intersection(load.index)]

    else:
        raise ValueError(f"Unsupported profile alignment {alignment!r}.")

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


def apply_auxiliary_fill_rule(
    load: pd.DataFrame,
    cleaning_method: pd.DataFrame,
    *,
    rule: pd.Series,
    profile: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply one validated advanced-fill rule.

    Args:
        load: Canonical hourly demand data.
        cleaning_method: Provenance labels aligned with ``load``.
        rule: One validated advanced-fill rule.
        profile: Auxiliary profile associated with the rule, if required.

    Returns:
        Updated demand and provenance data.

    Raises:
        ValueError: If a required profile is absent or the rule cannot
            be applied.
    """
    method = rule["method"]
    rule_name = rule["rule_name"]

    if method == LEAVE_MISSING:
        return load.copy(), cleaning_method.copy()

    if method not in _PROFILE_ALIGNMENT:
        raise ValueError(f"Unsupported advanced-fill method {method!r}.")

    if profile is None:
        raise ValueError(
            f"Advanced-fill rule {rule_name!r} requires an auxiliary profile."
        )

    return _apply_profile(
        load,
        cleaning_method,
        profile,
        country=rule["country"],
        start=rule["start"],
        end=rule["end"],
        scope=rule["scope"],
        rule_name=rule_name,
        alignment=_PROFILE_ALIGNMENT[method],
    )


def apply_auxiliary_fill_rules(
    load: pd.DataFrame,
    cleaning_method: pd.DataFrame,
    *,
    rules: pd.DataFrame,
    profiles: Mapping[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply validated advanced-fill rules in table order.

    Args:
        load: Canonical hourly demand data.
        cleaning_method: Provenance labels aligned with ``load``.
        rules: Advanced-fill rules in the order they should be applied.
        profiles: Auxiliary profiles keyed by rule name.

    Returns:
        Demand and provenance data after sequential rule application.
    """
    filled = validate_load(load)

    methods = validate_cleaning_method(cleaning_method, load=filled)

    rules = validate_advanced_fill_rules(rules)

    filled = filled.copy()
    methods = methods.copy()

    for _, rule in rules.iterrows():
        rule_name = rule["rule_name"]

        profile = profiles.get(rule_name)

        filled, methods = apply_auxiliary_fill_rule(
            filled, methods, rule=rule, profile=profile
        )

    return filled, methods
