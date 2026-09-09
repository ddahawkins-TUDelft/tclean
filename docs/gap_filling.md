# T-Clean gap filling

This document is the detailed reference for the `tclean.gap_filling` namespace.

Gap filling is the **transformative** side of T-Clean: it combines observed sources, reconstructs configured gaps, optionally applies targeted advanced replacements, and returns provenance describing how every final value was produced.

For shared concepts such as `TimeGrid`, contexts, sources, the canonical time-series contract, and package design principles, see the [main README](../README.md).

---

## Contents

- [Public API](#public-api)
- [Pipeline overview](#pipeline-overview)
- [Input sources and source priority](#input-sources-and-source-priority)
- [Outputs and provenance](#outputs-and-provenance)
- [Basic gap filling](#basic-gap-filling)
  - [`linear_interpolation`](#linear_interpolation)
  - [`copy_periods`](#copy_periods)
  - [`average_periods`](#average_periods)
- [Original gap duration](#original-gap-duration)
- [Gap reporting](#gap-reporting)
- [Advanced gap filling](#advanced-gap-filling)
- [Advanced rule methods](#advanced-rule-methods)
- [Advanced scopes](#advanced-scopes)
- [Selecting active advanced rules](#selecting-active-advanced-rules)
- [Auxiliary planning](#auxiliary-planning)
- [Source-period definitions](#source-period-definitions)
- [Acquisition requirements](#acquisition-requirements)
- [Source capabilities and requests](#source-capabilities-and-requests)
- [Constructing profiles from sources](#constructing-profiles-from-sources)
- [Scaling constructed profiles](#scaling-constructed-profiles)
- [External profiles](#external-profiles)
- [Provenance ranks](#provenance-ranks)
- [Validation and failure behaviour](#validation-and-failure-behaviour)
- [Adding a new basic method](#adding-a-new-basic-method)
- [Adding a new advanced method](#adding-a-new-advanced-method)

---

# Public API

The primary interface is:

```python
from tclean.gap_filling import fill_gaps
```

The namespace also exposes the main supporting interfaces:

```python
from tclean.gap_filling import (
    apply_advanced_rules,
    build_auxiliary_acquisition_requirements,
    build_auxiliary_source_requests,
    build_cleaning_method_ranks,
    build_gap_report,
    construct_from_sources,
    derive_cleaning_method_rank,
    fill_gaps,
    read_external_profile,
    select_active_advanced_rules,
    validate_advanced_fill_rules,
    validate_advanced_source,
    validate_auxiliary_requirements,
    validate_auxiliary_source_requests,
    validate_basic_rules,
    validate_source_capabilities,
    validate_source_periods,
)
```

Most applications should begin with `fill_gaps(...)` and use the planning or construction helpers only when orchestrating advanced reconstruction.

---

# Pipeline overview

The high-level pipeline is:

```python
filled, data_source, cleaning_method = fill_gaps(
    sources,
    grid=grid,
    basic_rules=basic_rules,
    advanced_rules=advanced_rules,
    advanced_sources=advanced_sources,
)
```

Conceptually:

```text
prepared primary sources
        |
        v
validate source contracts
        |
        v
combine sources by mapping priority
        |
        +--> data_source provenance
        +--> observed_* cleaning provenance
        |
        v
apply ordered basic gap-filling rules
        |
        v
apply ordered advanced rules, if configured
        |
        v
mark unresolved cells as "missing"
        |
        v
validate complete target coverage
        |
        v
crop to the requested TimeGrid window
        |
        +--> filled
        +--> data_source
        +--> cleaning_method
```

Basic gap filling is applied before advanced gap filling.

The consuming application remains responsible for any external acquisition needed between advanced planning and advanced application.

---

# Input sources and source priority

Primary inputs are supplied as a mapping of source names to canonical DataFrames:

```python
sources = {
    "primary": primary_data,
    "secondary": secondary_data,
    "fallback": fallback_data,
}
```

Mapping insertion order defines priority from highest to lowest.

For every timestamp/context cell, T-Clean:

1. uses `primary` when it has an observed value;
2. otherwise uses `secondary`;
3. otherwise uses `fallback`;
4. otherwise leaves the value missing.

Later sources never overwrite values already supplied by an earlier source.

All primary source frames must use the same timestamp index and context columns. Each source is validated against the supplied `TimeGrid` before combination.

The combination operation occurs **before** basic gap filling. This ensures that available observed data from lower-priority sources are preferred over synthetic reconstruction.

---

# Outputs and provenance

`fill_gaps(...)` returns three aligned DataFrames:

```python
filled, data_source, cleaning_method = fill_gaps(...)
```

## `filled`

The resulting time series after source combination and configured reconstruction.

## `data_source`

Records which observed or advanced source supplied the final value where applicable.

For observed source combination, values resemble:

```text
timestamp                 A
2026-01-01 00:00       primary
2026-01-01 01:00     secondary
2026-01-01 02:00          <NA>
```

## `cleaning_method`

Records the provenance label responsible for the final value.

Observed values are labelled:

```text
observed_<source_name>
```

For example:

```text
observed_primary
observed_secondary
```

Values filled by a basic or advanced rule use the configured **rule name**.

Cells that remain unresolved after the pipeline are labelled:

```text
missing
```

Rule names should therefore be unique, stable, and descriptive.

---

# Basic gap filling

Basic rules are deterministic gap-filling operations applied sequentially after source combination.

Rules are supplied as an ordered sequence of mappings:

```python
basic_rules = [
    {
        "name": "interpolate_short_gaps",
        "method": "linear_interpolation",
        "max_gap": "3h",
    },
    {
        "name": "average_adjacent_weeks",
        "method": "average_periods",
        "max_gap": "48h",
        "source_offsets": ["-7d", "7d"],
    },
    {
        "name": "copy_previous_week",
        "method": "copy_periods",
        "max_gap": "168h",
        "source_offset": "-7d",
        "require_complete_source": True,
    },
]
```

Order matters. Once an earlier rule fills a cell, that cell is no longer missing and is not available for a later basic rule to fill.

Each basic rule requires:

- `name`: a unique non-empty provenance label;
- `method`: the registered method name;
- method-specific fields.

Unknown or extra configuration fields are rejected.

You can validate rules independently before running the pipeline:

```python
from tclean.gap_filling import validate_basic_rules

validated = validate_basic_rules(basic_rules, grid=grid)
```

---

## `linear_interpolation`

Fills eligible bounded gaps using time-based linear interpolation.

```python
{
    "name": "interpolate_short_gaps",
    "method": "linear_interpolation",
    "max_gap": "3h",
}
```

### Configuration

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | yes | unique rule/provenance name |
| `method` | yes | must be `linear_interpolation` |
| `max_gap` | yes | maximum original contiguous gap duration eligible for filling |

`max_gap` must be positive and an integer multiple of the configured grid frequency.

Interpolation requires the missing run to be bounded by observed values. Boundary gaps are therefore not filled merely because their duration is below `max_gap`.

---

## `copy_periods`

Fills an eligible gap using corresponding values at one configured temporal offset.

```python
{
    "name": "copy_previous_week",
    "method": "copy_periods",
    "max_gap": "168h",
    "source_offset": "-7d",
    "require_complete_source": True,
}
```

### Configuration

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | yes | unique rule/provenance name |
| `method` | yes | must be `copy_periods` |
| `max_gap` | yes | maximum original contiguous gap duration |
| `source_offset` | yes | non-zero fixed temporal offset from target to source |
| `require_complete_source` | yes | whether every source value required for a gap must be present |

A negative `source_offset` refers to an earlier period; a positive offset refers to a later period.

When `require_complete_source=True`, a target gap remains eligible only when the complete corresponding source period exists.

---

## `average_periods`

Fills an eligible gap using the mean of corresponding values at multiple temporal offsets.

```python
{
    "name": "average_adjacent_weeks",
    "method": "average_periods",
    "max_gap": "48h",
    "source_offsets": ["-7d", "7d"],
}
```

### Configuration

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | yes | unique rule/provenance name |
| `method` | yes | must be `average_periods` |
| `max_gap` | yes | maximum original contiguous gap duration |
| `source_offsets` | yes | non-empty sequence of non-zero temporal offsets |

For each eligible target timestamp, all configured source values must be available before their mean is used.

---

# Original gap duration

Basic-rule eligibility is based on the **original contiguous missing-run structure presented to that rule**.

T-Clean explicitly calculates the duration of each missing run before a rule executes.

This matters for rules such as:

> interpolate only gaps up to three hours.

Without explicit run-duration tracking, filling part of a longer gap could incorrectly cause the remainder to appear short enough for the same rule.

Later rules see the data produced by earlier rules, but each rule still evaluates the missing-run structure that exists at the start of its own execution.

---

# Gap reporting

`build_gap_report(...)` describes unresolved contiguous gaps without filling them.

```python
from tclean.gap_filling import build_gap_report

report = build_gap_report(filled, grid=grid, enabled=True)
```

The report contains one row per unresolved gap:

| Column | Meaning |
| --- | --- |
| `context` | affected context |
| `gap_start` | inclusive gap start |
| `gap_end` | exclusive gap end |
| `gap_duration` | total missing duration |
| `touches_start_boundary` | whether the gap begins at the supplied data boundary |
| `touches_end_boundary` | whether the gap ends at the supplied data boundary |

When reporting is disabled, or no gaps remain, the function returns an empty DataFrame with the expected columns.

---

# Advanced gap filling

Advanced rules support explicit period-specific interventions after basic gap filling.

They are supplied as a canonical DataFrame with columns in this order:

```text
rule_name
method
source
context
start
end
scope
```

Example:

```python
advanced_rules = pd.DataFrame(
    {
        "rule_name": ["replace_a"],
        "method": ["external_profile"],
        "source": ["replacement_profile"],
        "context": ["A"],
        "start": ["2026-01-01T00:00:00Z"],
        "end": ["2026-01-02T00:00:00Z"],
        "scope": ["fill_gaps"],
    }
)
```

Advanced rules execute in DataFrame row order.

The rule table is deliberately generic data rather than application-specific configuration. It does not contain provider credentials, file paths, or workflow-specific fields.

Rules can be validated independently:

```python
from tclean.gap_filling import validate_advanced_fill_rules

validated = validate_advanced_fill_rules(advanced_rules, grid=grid)
```

Rule names must be unique, periods must be valid grid-aligned half-open intervals, and source use must match the method.

---

# Advanced rule methods

## `external_profile`

Applies an externally supplied profile to the configured context and period.

The rule must define a `source` name, and that name must exist in the `advanced_sources` mapping supplied to `fill_gaps(...)` or `apply_advanced_rules(...)`.

External-profile values are aligned by timestamp.

## `construct_from_sources`

Applies a previously constructed advanced source to the target period.

The rule must define a `source` name. The construction itself is performed separately with `construct_from_sources(...)`; T-Clean does not acquire supporting data inside the rule application step.

Constructed-source values are aligned positionally to the configured target interval after their source has been validated for the expected temporal structure.

## `leave_missing`

Explicitly leaves the targeted interval unchanged.

A `leave_missing` rule must not define a `source`.

This method is useful when an application wants an explicit configured decision that a period should remain unresolved rather than accidentally appearing unhandled.

---

# Advanced scopes

Every advanced rule has one of two scopes.

## `fill_gaps`

Replace only target cells that are currently missing.

Existing observed or previously filled values are preserved.

## `overwrite`

Replace every target cell in the configured context/period with the supplied advanced source.

Use this only when the configured source is explicitly intended to be authoritative for that period.

Scope is separate from rule activation. An advanced rule can belong to the target model scope even if a `fill_gaps` rule later finds that no matching gaps remain.

---

# Selecting active advanced rules

`select_active_advanced_rules(...)` filters a validated rule table to rules that intersect the current model scope.

```python
from tclean.gap_filling import select_active_advanced_rules

active = select_active_advanced_rules(
    advanced_rules,
    target_contexts=["A", "B"],
    grid=grid,
)
```

A rule is active when:

- its `context` is included in `target_contexts`; and
- its half-open period intersects the target `TimeGrid`.

Activity is deliberately **not gap-aware**.

This keeps planning deterministic:

- **activation** answers whether a configured rule belongs to the target domain;
- **scope** determines which target cells the rule is allowed to replace when applied.

---

# Auxiliary planning

Advanced reconstruction can require supporting data that is not part of the primary target dataset.

T-Clean provides planning helpers so the consuming application can determine what data it needs **before** performing acquisition.

The intended boundary is:

```text
advanced source definitions
          |
          v
T-Clean derives exact source periods
          |
          v
T-Clean expands periods for auxiliary cleaning context
          |
          v
T-Clean maps requirements to capable sources
          |
          v
application acquires and prepares data
          |
          v
T-Clean constructs advanced profiles
```

T-Clean determines temporal requirements; the consuming application determines how data are downloaded, cached, or scheduled.

---

# Source-period definitions

Weighted historical source periods use the canonical columns:

```text
context
start
end
weight
```

Example:

```python
source_periods = pd.DataFrame(
    {
        "context": ["A", "A"],
        "start": ["2024-01-01", "2025-01-01"],
        "end": ["2024-02-01", "2025-02-01"],
        "weight": [1.0, 2.0],
    }
)
```

Requirements include:

- at least one row;
- non-empty contexts;
- positive finite weights;
- `end > start`;
- grid-aligned period boundaries.

Validate independently with:

```python
from tclean.gap_filling import validate_source_periods

validated = validate_source_periods(source_periods, grid=grid)
```

---

# Acquisition requirements

`build_auxiliary_acquisition_requirements(...)` converts one or more source-period tables into merged context-period requirements and optionally expands them for the temporal support needed by basic cleaning.

```python
from tclean.gap_filling import build_auxiliary_acquisition_requirements

requirements = build_auxiliary_acquisition_requirements(
    [source_periods],
    basic_rules=basic_rules,
    grid=grid,
    basic_cleaning_enabled=True,
)
```

The output contains:

```text
context
start
end
```

Overlapping or adjacent requirements for the same context are merged.

## Why expansion is necessary

Suppose an advanced construction needs:

```text
2024-01-01 -> 2024-02-01
```

but acquired auxiliary data will itself be cleaned by a basic rule that copies from seven days earlier.

T-Clean can determine that acquisition must begin before the exact construction interval so that auxiliary cleaning has enough temporal context.

Context accumulates through ordered basic rules because a later basic rule may depend on values that an earlier rule can itself construct only using additional surrounding data.

---

# Source capabilities and requests

A consuming application tells T-Clean which acquisition sources can provide which contexts using a canonical capability table:

```text
source
context
```

A missing `context` acts as a wildcard: that source can serve all required contexts.

A source must use either:

- wildcard coverage; or
- explicit context rows;

not both.

Example:

```python
source_capabilities = pd.DataFrame(
    {
        "source": ["provider_a", "provider_b"],
        "context": [pd.NA, "special_context"],
    }
)
```

Map requirements to source requests with:

```python
from tclean.gap_filling import build_auxiliary_source_requests

requests = build_auxiliary_source_requests(
    requirements,
    source_capabilities=source_capabilities,
    grid=grid,
)
```

The output contains:

```text
source
context
start
end
```

If no configured source can provide a required context, planning fails explicitly.

The returned request table describes what the application should acquire. T-Clean does not perform the acquisition itself.

---

# Constructing profiles from sources

`construct_from_sources(...)` builds one complete target profile from weighted historical periods that have already been acquired and prepared.

```python
from tclean.gap_filling import construct_from_sources

profile = construct_from_sources(
    auxiliary_data,
    target_index=target_index,
    sources=source_periods,
    grid=grid,
)
```

For each configured source period T-Clean:

1. validates the requested context and period;
2. extracts the period;
3. aligns leap-day calendar shape when necessary;
4. requires the source period to produce the same number of values as the target;
5. requires complete source values;
6. remaps the values onto `target_index`;
7. multiplies by the configured weight.

The resulting profile is the weighted mean of the supplied source periods.

## Leap-day handling

Equivalent calendar periods can differ in length between leap and non-leap years.

T-Clean handles supported February 29 transformations explicitly rather than silently truncating or shifting data.

The configured frequency must divide a whole day cleanly when leap-day alignment is required. Invalid or ambiguous transformations fail explicitly.

## Construction is separate from acquisition

`construct_from_sources(...)` does not download anything.

It assumes the consuming application has already acquired, combined, and prepared the necessary supporting data.

---

# Scaling constructed profiles

A constructed profile can optionally be scaled after its temporal shape has been built:

```python
profile = construct_from_sources(
    auxiliary_data,
    target_index=target_index,
    sources=source_periods,
    scaling_method="normalise_mean",
    grid=grid,
)
```

Supported scaling methods are:

- `match_total`;
- `normalise_mean`;
- `normalise_max`.

## No scaling

When `scaling_method=None`, the weighted constructed profile is returned as-is.

Supplying `scaling_sources` without a `scaling_method` is invalid.

## `match_total`

Scales the constructed profile so that its total matches the weighted mean total of configured reference periods.

```python
profile = construct_from_sources(
    auxiliary_data,
    target_index=target_index,
    sources=source_periods,
    scaling_method="match_total",
    scaling_sources=scaling_periods,
    grid=grid,
)
```

`match_total` requires `scaling_sources`, using the same canonical `context`, `start`, `end`, and `weight` schema as construction sources.

Conceptually:

```text
reference total = weighted mean total of scaling periods
scale factor    = reference total / constructed profile total
scaled profile  = constructed profile * scale factor
```

A zero-total constructed profile cannot be matched to a reference total.

## `normalise_mean`

Scales the profile so that:

```text
profile.mean() == 1
```

Conceptually:

```python
profile = profile / profile.mean()
```

This method does not use `scaling_sources` and fails when the profile mean is zero.

## `normalise_max`

Scales the profile so that:

```text
profile.max() == 1
```

Conceptually:

```python
profile = profile / profile.max()
```

This method does not use `scaling_sources` and requires a positive profile maximum.

---

# External profiles

An external profile is an application/user-supplied time series read separately from the primary source mapping.

```python
from tclean.gap_filling import read_external_profile

profile = read_external_profile(path, grid=grid)
```

The input file must contain exactly:

```text
timestamp
value
```

The reader validates:

- required and unexpected columns;
- UTC timestamps;
- duplicate timestamps;
- timestamp ordering;
- numeric values;
- missing values;
- grid phase and frequency compatibility.

The returned object is a canonical floating-point pandas Series.

T-Clean deliberately does not decide where external profile files live. The consuming application resolves the path and passes it to the reader.

---

# Provenance ranks

Human-readable `cleaning_method` labels can be mapped to deterministic integer ranks for plotting or compact downstream representation.

```python
from tclean.gap_filling import (
    build_cleaning_method_ranks,
    derive_cleaning_method_rank,
)

ranks = build_cleaning_method_ranks(
    ["primary", "secondary"],
    basic_rule_names=["interpolate_short_gaps"],
    advanced_rule_names=["replace_bad_period"],
)

rank_frame = derive_cleaning_method_rank(
    cleaning_method=cleaning_method,
    ranks=ranks,
)
```

Rank order is:

1. observed source labels in supplied source order;
2. basic rule names in execution order;
3. advanced rule names in execution order;
4. `missing`.

Generated provenance labels must be unique.

---

# Validation and failure behaviour

Gap filling follows T-Clean's fail-fast philosophy.

Typical configuration or data-contract errors include:

- unsupported basic or advanced methods;
- unknown configuration fields;
- duplicated rule names;
- non-grid-aligned durations or periods;
- missing required advanced sources;
- unused or unexpected advanced sources;
- incomplete construction periods;
- unavailable construction contexts;
- invalid source capabilities;
- auxiliary requirements no configured source can satisfy.

T-Clean does not silently resample, truncate, repair, or reinterpret malformed input.

---

# Adding a new basic method

A basic method should be added only when it is:

- deterministic;
- domain-neutral;
- expressible from the supplied time series and grid;
- appropriate to execute in sequence with existing basic rules.

When adding one:

1. implement it under `tclean/gap_filling/basic/methods/`;
2. define an explicit configuration contract;
3. extend basic-rule validation;
4. extend basic dispatch;
5. update auxiliary-context planning if the method requires data outside the target period;
6. ensure filled cells receive the configured rule name as provenance;
7. add focused and interaction tests;
8. document the semantics, configuration, support-period requirements, and failure conditions here.

A method requiring surrounding data must also update planning semantics. Otherwise primary gap filling may work while auxiliary reconstruction fails because the application was not told to acquire enough support data.

---

# Adding a new advanced method

Advanced methods are appropriate for explicitly configured period-specific intervention rather than generic local gap filling.

When adding one:

1. decide whether the method requires a named advanced source;
2. extend the canonical advanced-rule validation contract if necessary;
3. implement transformation logic without application-specific assumptions;
4. respect the existing `fill_gaps` and `overwrite` scopes;
5. extend advanced dispatch;
6. define auxiliary planning requirements if supporting acquisition is needed;
7. test source matching, target alignment, scope behaviour, provenance, and inactive-rule behaviour;
8. document the new method here.

Acquisition itself should remain outside the method.
