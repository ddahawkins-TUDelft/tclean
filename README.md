# T-Clean

**T-Clean** is a small, domain-neutral Python library for combining, validating, cleaning, reconstructing, and tracking provenance in regular time series.

It was originally extracted from the Modelblocks electricity-demand module so that the generic time-series cleaning logic could be tested, versioned, documented, and reused independently of Snakemake, electricity-demand providers, or any other application-specific workflow.

T-Clean deliberately does **not** know what a time series represents. It does not know about electricity, countries, MW, ENTSO-E, OPSD, NESO, Modelblocks, or Snakemake. Instead, it works with:

- a regular time grid;
- one or more named time-series sources;
- opaque **contexts** represented by DataFrame columns;
- ordered basic cleaning rules;
- optional advanced rules and advanced source profiles;
- explicit provenance describing where every value came from and how it was produced.

This separation is intentional. Applications such as Modelblocks are responsible for obtaining and preparing source data; T-Clean is responsible for the generic time-series operations that follow.

> **Version note**
>
> This README is written for the `0.1.x` API. Sections describing `normalise_mean` and `normalise_max` assume their inclusion in T-Clean `0.1.1`.

---

## Contents

- [Why T-Clean exists](#why-t-clean-exists)
- [Installation](#installation)
- [Core concepts](#core-concepts)
- [Public API](#public-api)
- [Data contract](#data-contract)
- [TimeGrid](#timegrid)
- [TCleanConfig](#tcleanconfig)
- [The cleaning pipeline](#the-cleaning-pipeline)
- [Basic cleaning](#basic-cleaning)
- [Advanced cleaning](#advanced-cleaning)
- [Constructing profiles from auxiliary sources](#constructing-profiles-from-auxiliary-sources)
- [Scaling constructed profiles](#scaling-constructed-profiles)
- [External profiles](#external-profiles)
- [Advanced planning](#advanced-planning)
- [Provenance](#provenance)
- [Validation and failure behaviour](#validation-and-failure-behaviour)
- [How Modelblocks uses T-Clean](#how-modelblocks-uses-t-clean)
- [Adding a new basic cleaning method](#adding-a-new-basic-cleaning-method)
- [Adding a new advanced method](#adding-a-new-advanced-method)
- [Adding a new construction/scaling method](#adding-a-new-constructionscaling-method)
- [Design principles](#design-principles)
- [Development](#development)
- [Release workflow](#release-workflow)

---

# Why T-Clean exists

Time-series cleaning often starts as application code:

1. combine several observed data sources;
2. resolve short gaps;
3. copy or average neighbouring periods;
4. diagnose unresolved gaps;
5. acquire supporting data;
6. reconstruct missing periods;
7. track what happened to each value.

Those operations are generic. They should not need to be reimplemented separately for electricity demand, weather, traffic, industrial activity, or another regularly sampled time series.

T-Clean therefore separates **time-series semantics** from **application orchestration**.

A consuming application is expected to handle concerns such as:

- APIs and downloads;
- provider-specific file formats;
- authentication;
- domain-specific units;
- application configuration;
- workflow scheduling;
- persistence of intermediate files.

T-Clean handles concerns such as:

- timestamp-grid validation;
- source priority;
- deterministic basic gap filling;
- advanced period-specific replacement;
- construction of profiles from auxiliary periods;
- external profile validation;
- acquisition-requirement planning;
- provenance.

This boundary is particularly important in workflow systems. A large upstream download should not need to rerun because a generic gap-filling rule changed, and a generic time-series library should not need to import a workflow engine just to clean a DataFrame.

---

# Installation

From PyPI:

```bash
pip install tclean
```

Once available from conda-forge:

```bash
conda install -c conda-forge tclean
```

For development:

```bash
git clone https://github.com/ddahawkins-TUDelft/tclean.git
cd tclean
pixi install
```

---

# Core concepts

T-Clean is easiest to understand in terms of five concepts.

## 1. Grid

Every operation is anchored to a `TimeGrid`.

The grid defines:

- the inclusive start;
- the exclusive end;
- the fixed frequency;
- the timestamp phase.

For example:

```python
from tclean import TimeGrid

grid = TimeGrid(
    start="2026-01-01T00:00:00Z",
    end="2026-01-03T00:00:00Z",
    frequency="1h",
)
```

represents the timestamps:

```text
2026-01-01 00:00 UTC
2026-01-01 01:00 UTC
...
2026-01-02 23:00 UTC
```

The end timestamp itself is not part of the target index.

## 2. Context

T-Clean calls each independently cleaned DataFrame column a **context**.

A context could be:

- a country;
- a weather station;
- an industrial site;
- a traffic counter;
- a model region;
- any other logical stream.

T-Clean treats context labels as opaque identifiers. It does not interpret their meaning.

## 3. Source

A source is a named provider of time-series values.

For example:

```python
sources = {
    "primary": primary_data,
    "secondary": secondary_data,
}
```

Mapping order defines source priority: earlier sources have priority over later sources.

## 4. Rule

A rule has:

- a unique name;
- a method;
- method-specific parameters.

Rule names are not cosmetic. They become provenance labels and should therefore describe what happened.

## 5. Provenance

T-Clean tracks both:

- the **observed source** from which an original value came;
- the **cleaning method/rule** responsible for a reconstructed value.

This makes the final series auditable at the cell level.

---

# Public API

The intentionally small top-level API is:

```python
from tclean import TCleanConfig, TimeGrid, clean
```

The advanced planning and construction APIs can be imported explicitly when an application needs to orchestrate advanced cleaning:

```python
from tclean.advanced.gap_report import build_gap_report
from tclean.advanced.methods.construct_from_sources import construct_from_sources
from tclean.advanced.methods.external_profile import read_external_profile
from tclean.advanced.planning import (
    build_auxiliary_acquisition_requirements,
    build_auxiliary_source_requests,
    select_active_advanced_rules,
)
```

Basic-rule validation is also available independently:

```python
from tclean.basic.rule_validation import validate_basic_rules
```

Provenance helpers are available from:

```python
from tclean.provenance import (
    build_cleaning_method_ranks,
    derive_cleaning_method_rank,
)
```

The distinction is intentional:

- `clean(...)` is the normal high-level entry point;
- planning helpers support workflow orchestration;
- construction helpers transform already acquired auxiliary data;
- validation helpers can be used before expensive workflow steps.

---

# Data contract

## Primary time-series data

Primary sources are supplied as pandas DataFrames:

```python
import pandas as pd

index = pd.date_range(
    "2026-01-01",
    periods=4,
    freq="1h",
    tz="UTC",
    name="timestamp",
)

primary = pd.DataFrame(
    {
        "A": [10.0, None, 12.0, 13.0],
        "B": [20.0, 21.0, None, 23.0],
    },
    index=index,
)
```

The canonical shape is:

```text
timestamp                 A       B
2026-01-01 00:00        10.0    20.0
2026-01-01 01:00         NaN    21.0
2026-01-01 02:00        12.0     NaN
2026-01-01 03:00        13.0    23.0
```

Requirements include:

- a `DatetimeIndex`;
- UTC-aware timestamps;
- sorted timestamps;
- no duplicate timestamps;
- no duplicate context columns;
- timestamps aligned with the configured grid;
- numeric values or missing values.

T-Clean fails explicitly when these contracts are violated.

## Sparse versus complete data

The target series passed to the main cleaning pipeline is expected to live on the configured target grid.

Auxiliary data can extend beyond the target period. This is necessary because a rule such as “copy one week earlier” requires data outside the target period.

Support data may therefore have a broader temporal extent, but it must use the **same frequency and phase** as the target grid.

---

# TimeGrid

`TimeGrid` is the temporal contract used throughout T-Clean.

```python
from tclean import TimeGrid

grid = TimeGrid(
    start="2026-01-01T00:30:00Z",
    end="2026-01-02T00:30:00Z",
    frequency="1h",
)
```

This is a valid hourly grid offset by 30 minutes.

The grid is **not** implicitly aligned to wall-clock boundaries. Its phase is defined by `start`.

A timestamp is aligned when:

```text
(timestamp - grid.start) % grid.frequency == 0
```

This means that for the grid above:

```text
00:30, 01:30, 02:30, ...
```

are aligned, while:

```text
00:00, 01:00, 02:00, ...
```

are not.

## Half-open periods

T-Clean uses half-open intervals:

```text
[start, end)
```

This convention is used consistently for:

- the target grid;
- advanced rules;
- auxiliary periods;
- source construction;
- scaling periods.

It avoids double-counting boundary timestamps when adjacent periods meet.

## Fixed-frequency requirement

The frequency must be a fixed duration and the total target duration must contain an integer number of timesteps.

Invalid grids fail immediately.

---

# TCleanConfig

`TCleanConfig` groups temporal configuration used by the pipeline.

Typical use:

```python
from tclean import TCleanConfig, TimeGrid

grid = TimeGrid(
    start="2026-01-01",
    end="2026-02-01",
    frequency="1h",
)

config = TCleanConfig(grid=grid)
```

Keeping the grid inside an explicit configuration object avoids individual cleaning functions independently inferring frequency or timestamp alignment from potentially incomplete data.

---

# The cleaning pipeline

The high-level pipeline is:

```python
cleaned, data_source, cleaning_method = clean(
    sources,
    basic_rules=basic_rules,
    advanced_rules=advanced_rules,
    advanced_sources=advanced_sources,
    config=config,
)
```

Conceptually, `clean(...)` performs the following operations:

```text
prepared primary sources
        |
        v
validate source contracts
        |
        v
combine sources by priority
        |
        +--> data_source provenance
        |
        v
apply ordered basic rules
        |
        v
optionally apply advanced rules
        |
        v
crop/validate final target grid
        |
        +--> cleaning_method provenance
        |
        v
cleaned time series
```

The application remains responsible for any advanced acquisition that must occur between diagnosis/planning and advanced application.

## Source combination

Given:

```python
sources = {
    "source_a": a,
    "source_b": b,
    "source_c": c,
}
```

T-Clean combines them in mapping order.

For every timestamp/context cell:

1. use `source_a` if it has a value;
2. otherwise use `source_b`;
3. otherwise use `source_c`;
4. otherwise leave the value missing.

Later sources do not overwrite values from earlier sources.

The resulting `data_source` DataFrame records which source supplied each observed value.

## Basic cleaning happens after combination

Basic rules are applied to the **combined** time series, not independently to each primary provider.

This ensures that a value available from a lower-priority observed source is used before synthetic gap filling is attempted.

## Advanced cleaning happens after basic cleaning

Advanced rules are designed for explicit, period-specific intervention after the normal deterministic cleaning stage.

This ordering is deliberate:

```text
observed data
    ↓
source combination
    ↓
basic deterministic reconstruction
    ↓
advanced targeted reconstruction/overwrite
```

---

# Basic cleaning

Basic rules are deterministic and applied sequentially in configuration order.

Example:

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

Order matters. Once a value is filled by an earlier rule, it is no longer available for a later rule to fill.

The rule's `name` is written into cleaning provenance.

## `linear_interpolation`

Fills sufficiently short gaps using linear interpolation.

Example:

```python
{
    "name": "interpolate_short_gaps",
    "method": "linear_interpolation",
    "max_gap": "3h",
}
```

`max_gap` limits which contiguous missing runs are eligible.

## `average_periods`

Fills a missing period using the average of corresponding values at configured temporal offsets.

Example:

```python
{
    "name": "average_adjacent_weeks",
    "method": "average_periods",
    "max_gap": "48h",
    "source_offsets": ["-7d", "7d"],
}
```

For each eligible timestamp, this example takes values from the same timestamp one week earlier and one week later and averages them.

## `copy_periods`

Copies the corresponding source period at a configured offset.

Example:

```python
{
    "name": "copy_previous_week",
    "method": "copy_periods",
    "max_gap": "168h",
    "source_offset": "-7d",
    "require_complete_source": True,
}
```

A negative offset refers to an earlier source period; a positive offset refers to a later source period.

When `require_complete_source=True`, a gap is filled only when the source period needed for that gap is complete.

## Original gap duration

Eligibility is based on the original contiguous gap structure presented to the rule. T-Clean explicitly tracks gap duration rather than treating each missing cell independently.

This is important for rules such as:

> interpolate only gaps shorter than three hours.

Without explicit run-duration handling, partially filled gaps could otherwise change eligibility during execution.

---

# Advanced cleaning

Advanced cleaning handles explicitly configured interventions that require more than ordinary local gap rules.

The canonical advanced-rule table contains:

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

Advanced rules are data, rather than application-specific configuration objects. This is part of the boundary between T-Clean and consuming applications.

## Active versus inactive rules

T-Clean exposes `select_active_advanced_rules(...)`.

A rule is active when:

- its target context exists in the current target data; and
- its target period overlaps the configured target grid.

Activity is deliberately **not gap-aware**.

A `fill_gaps` rule does not become “inactive” simply because the current post-basic data happens not to contain a matching gap.

This distinction is important:

- **activation** answers whether a rule belongs to the current target domain;
- **scope** answers what the rule is allowed to replace.

Keeping those concepts separate makes planning deterministic and avoids changing acquisition logic based on incidental current missingness.

## Advanced scopes

Two scopes are supported.

### `fill_gaps`

Replace only values that remain missing in the target period.

Existing values are preserved.

### `overwrite`

Replace every target value in the rule period, including previously observed or basic-cleaned values.

Use this when a configured profile is explicitly considered authoritative for the target interval.

---

# Constructing profiles from auxiliary sources

`construct_from_sources(...)` builds a target-length profile from one or more auxiliary periods.

Typical use:

```python
from tclean.advanced.methods.construct_from_sources import construct_from_sources

profile = construct_from_sources(
    auxiliary,
    target_index=target_index,
    sources=[
        {
            "context": "A",
            "start": "2024-01-01",
            "end": "2024-02-01",
            "weight": 1.0,
        }
    ],
    grid=grid,
)
```

> Applications may use their own domain-specific field names while building these canonical inputs. T-Clean's internal concept is an opaque context, not a country.

## Construction algorithm

Conceptually:

```text
source period 1 ─┐
source period 2 ─┼─> temporal alignment ─> weighted combination ─> target profile
source period 3 ─┘
```

For each configured source period T-Clean:

1. extracts the requested context and half-open period;
2. validates that the source values are complete;
3. aligns calendar shape where necessary;
4. checks that the resulting number of timesteps matches the target;
5. applies the configured weight;
6. combines the source periods into one weighted profile.

Source weights must be meaningful and the total weighting must be valid.

## Leap-day handling

Construction can require aligning equivalent calendar periods from leap and non-leap years.

T-Clean contains explicit leap-day handling rather than silently truncating or shifting timestamps.

Where a leap-day transformation is required, the configured frequency must support the transformation cleanly. Invalid or ambiguous temporal transformations fail explicitly.

## Why source construction is separate from acquisition

`construct_from_sources(...)` does not download anything.

It assumes the consuming application has already acquired and prepared the auxiliary time series.

That separation is critical:

```text
T-Clean planning
    ↓
application acquires data
    ↓
application prepares data
    ↓
T-Clean constructs profile
```

T-Clean therefore remains reusable without knowing anything about APIs, credentials, filesystems, providers, or workflow engines.

---

# Scaling constructed profiles

A constructed profile can optionally be scaled after its shape has been built.

The scaling operation changes magnitude while preserving the constructed temporal shape.

## No scaling

Without a `scaling` configuration, the weighted constructed profile is returned as-is.

## `match_energy`

`match_energy` scales the constructed profile so its total equals the weighted mean total of one or more reference periods.

Conceptually:

```text
scale factor = target reference energy / constructed profile energy
scaled profile = constructed profile × scale factor
```

The reference periods are read from already prepared auxiliary data.

Example concept:

```python
scaling = {
    "method": "match_energy",
    "periods": [
        {
            "context": "A",
            "start": "2024-01-01",
            "end": "2024-02-01",
            "weight": 1.0,
        }
    ],
}
```

The scaling period is distinct from the construction period:

- **construction periods** define profile shape;
- **scaling periods** define target magnitude.

A zero-energy constructed profile cannot be energy-matched and raises an error.

## `normalise_mean` — 0.1.1

`normalise_mean` scales the constructed profile so:

```text
profile.mean() == 1
```

Conceptually:

```python
profile = profile / profile.mean()
```

Example:

```python
scaling = {
    "method": "normalise_mean",
}
```

Unlike `match_energy`, this method requires no reference periods.

The operation should fail if the profile mean is zero because the scaling factor would be undefined.

This method is useful when the constructed output is intended as a dimensionless shape that will be scaled elsewhere by the consuming application.

## `normalise_max` — 0.1.1

`normalise_max` scales the constructed profile so:

```text
profile.max() == 1
```

Conceptually:

```python
profile = profile / profile.max()
```

Example:

```python
scaling = {
    "method": "normalise_max",
}
```

It also requires no reference periods.

The operation should fail when the profile maximum is zero.

Like `normalise_mean`, this produces a reusable dimensionless shape rather than matching the magnitude of another observed period.

---

# External profiles

An external profile is a user/application-supplied time series read independently of the primary sources.

T-Clean provides:

```python
from tclean.advanced.methods.external_profile import read_external_profile
```

External profile ingestion is deliberately strict.

The reader validates:

- required columns;
- unexpected columns;
- timestamps;
- numeric values;
- missing values;
- duplicate timestamps;
- ordering;
- timestamp phase/frequency compatibility.

A valid external profile is returned as a canonical typed pandas Series.

## Filesystem boundary

T-Clean does **not** decide where external profile files live.

The application resolves the path:

```python
profile = read_external_profile(
    resolved_path,
    grid=grid,
)
```

For example, Modelblocks may resolve a file through a Snakemake pathvar. T-Clean sees only the resolved path.

This keeps filesystem layout outside the generic library.

---

# Advanced planning

Advanced reconstruction often needs data that is not part of the primary target dataset.

T-Clean provides planning primitives so a workflow can determine what it must acquire **before** doing the acquisition.

This is one of the most important interfaces between T-Clean and workflow systems.

## Stage 1: source-period requirements

A `construct_from_sources` definition identifies exact context/periods needed to build a profile.

Canonical source-period rows contain:

```text
context
start
end
weight
```

## Stage 2: acquisition requirements

Use:

```python
build_auxiliary_acquisition_requirements(
    source_periods,
    basic_rules=basic_rules,
    grid=grid,
    basic_cleaning_enabled=True,
)
```

This does more than simply return the exact construction periods.

If auxiliary data will itself be basic-cleaned, T-Clean expands the acquisition window to include any surrounding temporal context required by those rules.

For example:

```text
construction needs:
    2024-01-01 → 2024-02-01

basic auxiliary rule:
    copy from -7d

acquisition may need:
    2023-12-25 → 2024-02-01
```

The workflow can therefore acquire enough data in one planned operation rather than discovering later that the reconstruction rule requires unavailable context.

## Stage 3: map requirements to capable sources

Use:

```python
build_auxiliary_source_requests(
    requirements,
    source_capabilities=source_capabilities,
    grid=grid,
)
```

The workflow supplies source capabilities because T-Clean does not know what providers exist.

T-Clean maps each required context/period onto configured sources capable of supplying it.

The output is a canonical request table containing fields such as:

```text
source
context
start
end
```

If no configured source can satisfy a required period, planning fails explicitly.

## Why planning belongs in T-Clean

The workflow engine owns execution, but the amount of temporal context required is determined by cleaning semantics.

For example, only the cleaning library knows that:

```text
copy_previous_week
```

requires a preceding seven-day support period.

Therefore:

- **T-Clean decides what data is required**;
- **the consuming workflow decides how to obtain it**.

This is the central orchestration boundary.

---

# Provenance

T-Clean treats provenance as a first-class output, not as logging metadata.

## Data-source provenance

`data_source` records which observed source supplied a value.

For example:

```text
timestamp                 A
2026-01-01 00:00       primary
2026-01-01 01:00     secondary
2026-01-01 02:00          <NA>
```

## Cleaning-method provenance

`cleaning_method` records the rule responsible for the final value.

Observed values are labelled according to their source-derived provenance; reconstructed values are labelled with the cleaning rule name.

This allows a downstream application to answer questions such as:

- how much of the final series is observed?
- which provider supplied it?
- how much was interpolated?
- how much was copied from another period?
- which explicit advanced rule replaced a given interval?

## Cleaning-method ranks

For plotting or compact storage, provenance labels can be mapped to ordered integer ranks:

```python
ranks = build_cleaning_method_ranks(
    source_names,
    basic_rule_names=basic_rule_names,
    advanced_rule_names=advanced_rule_names,
)

rank_frame = derive_cleaning_method_rank(
    cleaning_method=cleaning_method,
    ranks=ranks,
)
```

The ordering is deterministic and follows the configured source/rule order.

Applications can use the integer representation for visualisation without losing the human-readable provenance table.

---

# Validation and failure behaviour

T-Clean follows a fail-fast philosophy.

It does not silently:

- infer malformed rules;
- repair duplicate timestamps;
- resample incompatible frequencies;
- reinterpret misaligned timestamps;
- ignore unknown methods;
- accept missing required source periods;
- discard unsupported configuration fields;
- use incomplete construction periods where completeness is required.

Errors are intended to identify invalid assumptions as early as possible.

## Pandera

Structured DataFrame and Series contracts are validated with Pandera where appropriate.

This provides validation for both structure and semantic constraints.

## Rule validation

Basic-rule configuration can be validated independently:

```python
validated = validate_basic_rules(
    rules,
    grid=grid,
)
```

This is useful before an expensive workflow begins.

## Temporal validation

The grid is also reusable independently of the cleaning pipeline.

Applications can therefore validate their temporal configuration before downloading or processing data.

---

# How Modelblocks uses T-Clean

Modelblocks provides a useful example of the intended library boundary.

The electricity-demand module still owns:

- electricity-specific configuration;
- ENTSO-E/NESO/OPSD acquisition;
- provider-specific preparation;
- country-code handling;
- filesystem locations;
- Snakemake rules;
- checkpoints and dynamic DAG expansion;
- plotting and final module outputs.

T-Clean owns:

- generic temporal validation;
- source combination;
- basic rule application;
- gap diagnosis;
- advanced-rule selection;
- auxiliary requirement calculation;
- source-request planning;
- profile construction;
- external-profile validation;
- advanced rule application;
- provenance.

## Main handoff

Modelblocks prepares each provider into the same canonical form:

```text
DatetimeIndex(timestamp, UTC)
×
context columns
×
numeric values
```

It then constructs a T-Clean grid/config and passes the prepared source mapping to `clean(...)`.

Conceptually:

```python
grid = TimeGrid(
    start=temporal_start,
    end=temporal_end,
    frequency=frequency,
)

config = TCleanConfig(grid=grid)

cleaned, data_source, cleaning_method = clean(
    prepared_sources,
    basic_rules=basic_rules,
    config=config,
)
```

At that point T-Clean no longer knows that the contexts are countries or that the values are electricity demand.

## Advanced Modelblocks flow

The full advanced flow is approximately:

```text
Modelblocks provider acquisition/preparation
                |
                v
          tclean.clean(...)
        basic cleaning stage
                |
                v
        build_gap_report(...)
                |
                v
      select active advanced rules
                |
                v
T-Clean calculates auxiliary requirements
                |
                v
Modelblocks maps plan into Snakemake jobs
                |
                v
Modelblocks downloads/prepares auxiliary sources
                |
                v
T-Clean combines/cleans auxiliary data
                |
                v
construct_from_sources(...) / read_external_profile(...)
                |
                v
T-Clean applies advanced rules
                |
                v
Modelblocks writes final data + provenance
```

This division is intentional.

A reviewer should therefore not expect T-Clean itself to:

- call ENTSO-E;
- know that `ALB` means Albania;
- build Snakemake jobs;
- know where Modelblocks resources live.

Conversely, generic reconstruction algorithms should not live in the Modelblocks module simply because that was where they were first needed.

---

# Adding a new basic cleaning method

A new basic method should be added only when it is:

- deterministic;
- generic;
- expressible from the supplied time series/grid;
- appropriate to execute in ordered sequence with the existing rules.

Suppose we want to add:

```text
rolling_median
```

## 1. Implement the method

Add a module under:

```text
src/tclean/basic/methods/
```

For example:

```text
rolling_median.py
```

The function should:

- accept canonical validated data;
- make eligibility explicit;
- avoid mutating caller-owned inputs unexpectedly;
- return values in the same index/column structure;
- contain no domain-specific assumptions.

## 2. Define the configuration contract

Decide exactly what fields are required.

For example:

```python
{
    "name": "median_short_gaps",
    "method": "rolling_median",
    "max_gap": "6h",
    "window": "24h",
}
```

Avoid implicit defaults for settings whose absence could hide a configuration error.

## 3. Extend basic-rule validation

Update:

```text
tclean.basic.rule_validation
```

Validation should reject:

- missing fields;
- unexpected fields;
- invalid durations;
- incompatible grid values;
- invalid types.

The normalized rule returned by validation should be directly executable.

## 4. Add dispatch in the basic application layer

Update the basic application dispatcher so:

```text
method == "rolling_median"
```

calls the new implementation.

The dispatcher should remain explicit. Unknown methods must continue to raise an error.

## 5. Consider auxiliary planning context

This is easily missed.

If the new method requires data outside the period being cleaned, update the logic that calculates surrounding auxiliary context.

For example, a centred 24-hour rolling window may require support data before and after the exact requested period.

If this step is omitted, primary cleaning might work while advanced auxiliary cleaning fails because the acquisition planner did not request enough data.

## 6. Add provenance tests

Confirm that newly filled cells receive the configured **rule name**, not merely the method name.

## 7. Add tests

At minimum test:

- valid filling;
- ineligible gaps;
- boundary behaviour;
- invalid configuration;
- missing support data;
- grid alignment;
- provenance;
- interaction with earlier/later rules.

## 8. Document it

Add the new method to this README with:

- semantics;
- configuration fields;
- a minimal example;
- support-period requirements;
- failure conditions.

---

# Adding a new advanced method

Advanced methods are appropriate when a rule requires an explicitly supplied profile or has period-specific semantics that should not be applied as a generic local gap rule.

Existing advanced methods include:

```text
construct_from_sources
external_profile
leave_missing
```

## 1. Define whether the method requires a source

An advanced rule has a `source` reference when it needs a supplied profile.

For example:

```text
external_profile → requires source
construct_from_sources → requires source
leave_missing → does not require source
```

This distinction affects validation and planning.

## 2. Extend the canonical rule validator

The canonical advanced-rule table should remain stable:

```text
rule_name
method
source
context
start
end
scope
```

Do not add application-specific fields such as provider names to the rule table.

## 3. Implement transformation logic

The method should operate on:

- canonical target data;
- target provenance;
- a target context;
- a half-open target period;
- the supplied advanced source/profile if required;
- the rule scope.

## 4. Respect `fill_gaps` versus `overwrite`

Advanced method code should not invent a third interpretation of scope.

`fill_gaps`:

```text
replace only missing target cells
```

`overwrite`:

```text
replace every cell in the targeted interval
```

## 5. Extend advanced dispatch

Add an explicit method branch to the advanced rule application layer.

Unknown methods must still fail.

## 6. Extend planning if necessary

If the new method requires auxiliary acquisition, define how its source definition becomes canonical source-period requirements.

Do not perform acquisition inside the method.

## 7. Test source/provenance behaviour

Tests should cover:

- missing required source;
- unused source;
- wrong target context;
- wrong target index;
- both scopes;
- provenance;
- inactive rule behaviour.

---

# Adding a new construction/scaling method

Scaling is intentionally narrower than a full advanced method. It operates on an already constructed profile.

Current/planned methods are:

```text
match_energy
normalise_mean
normalise_max
```

## `match_energy`

Needs external reference periods and therefore affects auxiliary planning.

## `normalise_mean`

Needs only the constructed profile:

```python
denominator = profile.mean()

if denominator == 0:
    raise ValueError(...)

return profile / denominator
```

## `normalise_max`

Needs only the constructed profile:

```python
denominator = profile.max()

if denominator == 0:
    raise ValueError(...)

return profile / denominator
```

## Adding another scaling method

When adding a method, answer these questions explicitly:

1. Does it require only the constructed profile?
2. Does it require additional reference periods?
3. Does it change auxiliary acquisition requirements?
4. What values make the scaling undefined?
5. Is the operation meaningful for negative-valued series?
6. Does it preserve the index exactly?

Then:

- implement the scaling function;
- extend `_apply_scaling` dispatch;
- update scaling configuration validation;
- update auxiliary source-period extraction if extra data is required;
- add unit tests;
- document the formula and failure conditions.

---

# Design principles

## Domain neutrality

No T-Clean API should require knowledge of:

```text
electricity
countries
MW
weather
traffic
specific providers
Snakemake
Modelblocks
```

A context is simply a context.

## Explicit temporal semantics

Time-series bugs are often timestamp bugs.

T-Clean therefore treats:

- start;
- end;
- frequency;
- timezone;
- phase;
- half-open periods

as explicit contracts.

## No silent repair

Malformed input should not be quietly transformed into something that merely looks plausible.

## Ordered deterministic behaviour

Source priority and rule priority are explicit and stable.

## Provenance is part of the result

A cleaned time series without a record of how it was cleaned is incomplete for many analytical workflows.

## Planning is separate from acquisition

T-Clean may determine:

> I require context A from 10 January through 20 January.

It should not decide:

> call provider X with credential Y and save it under directory Z.

## Transformation is separate from orchestration

The package can therefore be tested entirely without Snakemake.

---

# Development

The repository uses Pixi for development.

Typical checks:

```bash
pixi run format
pixi run lint
pixi run test
```

The test suite covers:

- grid semantics;
- source validation;
- source combination;
- basic rule validation;
- basic methods;
- advanced application;
- advanced planning;
- source construction;
- external profiles;
- gap reports;
- provenance;
- pipeline behaviour.

A new method should normally include both focused unit tests and at least one pipeline-level interaction test.

---

# Release workflow

T-Clean follows semantic versioning.

For a normal patch release such as `0.1.1`:

1. implement and test changes;
2. update documentation;
3. bump the package version;
4. commit the release state;
5. create Git tag `v0.1.1`;
6. create a matching GitHub Release;
7. build and validate the Python distribution;
8. publish `0.1.1` to PyPI;
9. publish/update the development conda package if required;
10. allow the conda-forge feedstock to update once available.

GitHub release tags, PyPI versions, and conda versions should refer to the same source state.

A GitHub Release is useful even though PyPI remains the canonical Python package distribution channel because it provides:

- an immutable project milestone;
- human-readable release notes;
- an obvious diff between versions;
- a clear reference for reviewers.

---

# License

T-Clean is released under the MIT License.
