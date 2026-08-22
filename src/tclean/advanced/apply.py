"""Apply validated advanced fill rules."""

from collections.abc import Mapping

import pandas as pd

from tclean.advanced.methods.construct_from_sources import (
    METHOD_NAME as CONSTRUCT_FROM_SOURCES,
)
from tclean.advanced.methods.external_profile import METHOD_NAME as EXTERNAL_PROFILE
from tclean.validation import (
    validate_advanced_fill_rules,
    validate_advanced_source,
    validate_cleaning_method,
    validate_data_source,
    validate_time_series,
)

LEAVE_MISSING = "leave_missing"

_EXACT_ALIGNMENT = "exact"
_OVERLAP_ALIGNMENT = "overlap"

_SOURCE_ALIGNMENT = {
    CONSTRUCT_FROM_SOURCES: _EXACT_ALIGNMENT,
    EXTERNAL_PROFILE: _OVERLAP_ALIGNMENT,
}


def _apply_advanced_source(
    data: pd.DataFrame,
    data_source: pd.DataFrame,
    cleaning_method: pd.DataFrame,
    source: pd.Series,
    *,
    source_name: str,
    context: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    scope: str,
    rule_name: str,
    alignment: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply an advanced source to the target time series."""
    if context not in data.columns:
        raise ValueError(f"Target context {context!r} is not present in data data.")

    target_index = data.index[(data.index >= start) & (data.index < end)]

    if alignment == _EXACT_ALIGNMENT:
        if not source.index.equals(target_index):
            raise ValueError(
                "Constructed source index must exactly match the target period."
            )

        candidate = source

    elif alignment == _OVERLAP_ALIGNMENT:
        candidate = source.loc[(source.index >= start) & (source.index < end)]

        candidate = candidate.loc[candidate.index.intersection(data.index)]

    else:
        raise ValueError(f"Unsupported source alignment {alignment!r}.")

    if scope == "fill_gaps":
        replace_index = candidate.index[data.loc[candidate.index, context].isna()]

    elif scope == "overwrite":
        replace_index = candidate.index

    else:
        raise ValueError(f"Unsupported advanced fill scope: {scope!r}")

    filled = data.copy()
    sources = data_source.copy()
    methods = cleaning_method.copy()

    filled.loc[replace_index, context] = candidate.loc[replace_index]

    sources.loc[replace_index, context] = source_name

    methods.loc[replace_index, context] = rule_name

    return (filled, sources, methods)


def _validate_advanced_sources(
    rules: pd.DataFrame,
    advanced_sources: Mapping[str, pd.Series],
    *,
    frequency: pd.Timedelta,
) -> dict[str, pd.Series]:
    """Validate advanced-source references and source data.

    Args:
        rules: Validated advanced cleaning rules.
        advanced_sources: Advanced time-series sources keyed by source name.
        frequency: pd.Timedelta of time series frequency.

    Returns:
        Validated advanced sources keyed by source name.

    Raises:
        TypeError: If an advanced source is not a pandas Series.
        ValueError: If supplied sources do not exactly match those referenced
            by the advanced rules.
        pandera.errors.SchemaErrors: If a source violates the advanced-source
            data contract.
    """
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

    validated_sources: dict[str, pd.Series] = {}

    for name, source in advanced_sources.items():
        if not isinstance(source, pd.Series):
            raise TypeError(f"Advanced source {name!r} must be a pandas Series.")

        validated_sources[name] = validate_advanced_source(source, frequency=frequency)

    return validated_sources


def apply_advanced_rule(
    data: pd.DataFrame,
    data_source: pd.DataFrame,
    cleaning_method: pd.DataFrame,
    *,
    rule: pd.Series,
    source: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply one validated advanced-fill rule.

    Args:
        data: Canonical hourly time-series data.
        data_source: Provenance of data values.
        cleaning_method: Provenance labels aligned with ``data``.
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
        return (data.copy(), data_source.copy(), cleaning_method.copy())

    if method not in _SOURCE_ALIGNMENT:
        raise ValueError(f"Unsupported advanced-fill method {method!r}.")

    if source is None:
        raise ValueError(
            f"Advanced-fill rule {rule_name!r} requires an advanced source."
        )

    source_name = rule["source"]

    return _apply_advanced_source(
        data,
        data_source,
        cleaning_method,
        source,
        source_name=source_name,
        context=rule["context"],
        start=rule["start"],
        end=rule["end"],
        scope=rule["scope"],
        rule_name=rule_name,
        alignment=_SOURCE_ALIGNMENT[method],
    )


def apply_advanced_rules(
    data: pd.DataFrame,
    data_source: pd.DataFrame,
    cleaning_method: pd.DataFrame,
    *,
    rules: pd.DataFrame,
    advanced_sources: Mapping[str, pd.Series],
    frequency: pd.Timedelta,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply validated advanced-fill rules in table order.

    Args:
        data: Canonical hourly time-series data.
        data_source: Data value provenance.
        cleaning_method: Provenance labels aligned with ``data``.
        rules: Advanced-fill rules in the order they should be applied.
        advanced_sources: Advanced time-series sources keyed by source name.

    Returns:
        Time-series and provenance data after sequential rule application.

    Raises:
        ValueError: If supplied advanced sources do not exactly match
            the sources referenced by the rules.
    """
    filled = validate_time_series(data, frequency=frequency)

    sources = validate_data_source(data_source, data=filled)

    methods = validate_cleaning_method(cleaning_method, data=filled)

    rules = validate_advanced_fill_rules(rules)

    advanced_sources = _validate_advanced_sources(
        rules, advanced_sources, frequency=frequency
    )

    filled = filled.copy()
    sources = sources.copy()
    methods = methods.copy()

    for _, rule in rules.iterrows():
        source = None if pd.isna(rule["source"]) else advanced_sources[rule["source"]]

        filled, sources, methods = apply_advanced_rule(
            filled, sources, methods, rule=rule, source=source
        )

    return (filled, sources, methods)
