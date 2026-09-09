# T-Clean

**T-Clean** is a small, domain-neutral Python library for validating, evaluating, and reconstructing regular time series.

It was originally extracted from the Modelblocks electricity-demand module so that generic time-series logic could be tested, versioned, documented, and reused independently of Snakemake, electricity-demand providers, or any other application-specific workflow.

T-Clean deliberately does **not** know what a time series represents. It does not know about electricity, countries, MW, ENTSO-E, weather, traffic, or any other domain. Instead, it works with:

- a regular `TimeGrid`;
- one or more named time-series sources;
- opaque **contexts** represented by DataFrame columns;
- explicit configuration describing what should be evaluated or reconstructed;
- structured outputs that make data-quality findings and data provenance auditable.

T-Clean currently provides two main capabilities:

| Capability | Purpose | Main interface |
| --- | --- | --- |
| **Gap filling** | Combine sources, fill configured gaps, perform advanced reconstruction, and track provenance. | `tclean.gap_filling.fill_gaps` |
| **Data quality** | Evaluate time series for suspicious values and patterns without modifying the supplied data. | `tclean.data_quality.evaluate` |

Detailed guides are available for each capability:

- [Gap filling](docs/gap_filling.md)
- [Data quality](docs/data_quality.md)

---

## Contents

- [Why T-Clean exists](#why-t-clean-exists)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Core concepts](#core-concepts)
- [Shared data contract](#shared-data-contract)
- [Public interfaces](#public-interfaces)
- [Gap filling](#gap-filling)
- [Data quality](#data-quality)
- [Validation and failure behaviour](#validation-and-failure-behaviour)
- [Application boundary](#application-boundary)
- [Design principles](#design-principles)
- [Development](#development)
- [Release workflow](#release-workflow)

---

# Why T-Clean exists

Time-series workflows often mix several different concerns:

1. acquire data from external providers;
2. prepare provider-specific files and units;
3. validate temporal structure;
4. detect suspicious observations or patterns;
5. combine overlapping sources;
6. fill or reconstruct gaps;
7. track how each final value was produced;
8. write workflow-specific reports and outputs.

Only some of those concerns are domain-specific.

Operations such as validating a regular timestamp grid, identifying a flatline, copying a corresponding period, comparing repeated profiles, or tracking reconstruction provenance can be useful in many domains. They should not need to be reimplemented separately for electricity demand, weather, traffic, industrial activity, or another regularly sampled time series.

T-Clean therefore separates **time-series semantics** from **application orchestration**.

A consuming application is expected to handle concerns such as:

- APIs and downloads;
- provider-specific file formats;
- authentication;
- domain-specific units;
- application configuration;
- workflow scheduling;
- persistence of intermediate and final files.

T-Clean handles concerns such as:

- timestamp-grid validation;
- generic data-quality evaluation;
- source priority and combination;
- deterministic gap filling;
- advanced period-specific reconstruction;
- construction of profiles from auxiliary periods;
- acquisition-requirement planning;
- provenance.

This boundary is particularly useful in workflow systems. A large upstream download should not need to rerun because a generic time-series method changed, and a generic time-series library should not need to import a workflow engine to evaluate or reconstruct a DataFrame.

---

# Installation

From PyPI:

```bash
pip install tclean
```

From conda-forge:

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

# Quick start

Both major T-Clean interfaces share the same `TimeGrid` and canonical time-series structure.

```python
import pandas as pd

from tclean import TimeGrid

index = pd.date_range("2026-01-01", periods=48, freq="1h", tz="UTC", name="timestamp")

grid = TimeGrid(
    start="2026-01-01T00:00:00Z", end="2026-01-03T00:00:00Z", frequency="1h"
)

primary = pd.DataFrame({"A": range(48), "B": range(100, 148)}, index=index, dtype=float)
```

## Evaluate data quality

Data-quality evaluation reports suspicious periods but does not modify the source data.

```python
from tclean.data_quality import evaluate

quality = evaluate(
    {"primary": primary},
    tests=[
        {
            "name": "nonnegative",
            "method": "range",
            "minimum": {"value_mode": "fixed", "value": 0},
        },
        {"name": "flatline_6h", "method": "flatline", "minimum_duration": "6h"},
    ],
    grid=grid,
)

print(quality.failures)
print(quality.issues)
```

See [Data quality](docs/data_quality.md) for all methods and configuration options.

## Fill gaps

Gap filling combines supplied sources in priority order and then applies configured reconstruction rules.

```python
from tclean.gap_filling import fill_gaps

secondary = primary.copy()
primary.loc[index[10:12], "A"] = float("nan")

filled, data_source, cleaning_method = fill_gaps(
    {"primary": primary, "secondary": secondary},
    basic_rules=[
        {
            "name": "interpolate_short_gaps",
            "method": "linear_interpolation",
            "max_gap": "3h",
        }
    ],
    grid=grid,
)
```

The returned objects are:

- `filled`: the resulting time series;
- `data_source`: which observed source supplied each value;
- `cleaning_method`: which observed source or reconstruction rule produced each final value.

See [Gap filling](docs/gap_filling.md) for the full pipeline, advanced reconstruction, planning, and provenance interfaces.

---

# Core concepts

T-Clean is easiest to understand in terms of a small set of shared concepts.

## 1. Grid

Every operation is anchored to a `TimeGrid`.

```python
from tclean import TimeGrid

grid = TimeGrid(
    start="2026-01-01T00:30:00Z", end="2026-01-02T00:30:00Z", frequency="1h"
)
```

The grid defines:

- an **inclusive** start;
- an **exclusive** end;
- a fixed frequency;
- the timestamp phase.

For the example above, valid timestamps are:

```text
00:30, 01:30, 02:30, ...
```

not:

```text
00:00, 01:00, 02:00, ...
```

T-Clean does not implicitly realign timestamps to wall-clock boundaries.

### Half-open periods

T-Clean consistently represents periods as:

```text
[start, end)
```

The start is included and the end is excluded. This avoids double-counting when adjacent periods meet and is used for target grids, data-quality events, gaps, advanced rules, auxiliary periods, and constructed profiles.

### Fixed-frequency requirement

The configured frequency must be a fixed duration, and the target interval from `start` to `end` must contain a whole number of grid steps.

Durations supplied to methods must also align with the configured grid where required.

## 2. Context

T-Clean calls each logical DataFrame column a **context**.

A context could be:

- a country;
- a weather station;
- an industrial site;
- a traffic counter;
- a model region;
- any other logical stream.

T-Clean treats context labels as opaque identifiers and does not interpret their meaning.

## 3. Source

A source is a named provider of time-series values.

```python
sources = {"primary": primary_data, "secondary": secondary_data}
```

The meaning of source order depends on the operation:

- in **gap filling**, mapping order defines source priority;
- in **data quality**, every source can be evaluated independently, while methods such as `source_disagreement` may also inspect other supplied sources as evidence.

## 4. Ordered configuration

Both major T-Clean capabilities use ordered configuration.

For gap filling, rule order determines which reconstruction gets the first opportunity to fill a value.

For data quality, test order determines which earlier failures are excluded from the reference data used by later reference-aware tests.

Names are therefore meaningful identifiers rather than cosmetic labels.

## 5. Provenance and evidence

Gap filling returns provenance describing how final values were produced.

Data quality returns structured failure and issue tables describing what was detected and, in the `details` field, the evidence associated with each event.

T-Clean therefore treats auditability as part of the result rather than as logging metadata.

---

# Shared data contract

Primary time-series data are supplied as pandas DataFrames with timestamps on the index and contexts in the columns.

```text
timestamp                 A       B
2026-01-01 00:00        10.0    20.0
2026-01-01 01:00         NaN    21.0
2026-01-01 02:00        12.0     NaN
2026-01-01 03:00        13.0    23.0
```

Canonical time-series data require:

- a pandas `DatetimeIndex` named `timestamp`;
- UTC-aware timestamps;
- ascending timestamp order;
- no duplicate timestamps;
- one or more uniquely named context columns;
- numeric values or missing values;
- a complete consecutive index on the configured `TimeGrid`.

The data can extend beyond the target output window where an operation requires supporting temporal context, but timestamps must remain on the same configured frequency and phase.

Malformed inputs fail explicitly rather than being silently repaired, resampled, sorted, or realigned.

---

# Public interfaces

The shared package-level API is intentionally small:

```python
from tclean import TimeGrid
```

Gap-filling functionality is grouped under:

```python
from tclean.gap_filling import fill_gaps
```

The `tclean.gap_filling` namespace also exposes validation, planning, advanced reconstruction, profile construction, and provenance helpers. See the [gap-filling reference](docs/gap_filling.md).

Data-quality functionality is grouped under:

```python
from tclean.data_quality import QualityEvaluation, evaluate, validate_quality_tests
```

See the [data-quality reference](docs/data_quality.md).

The namespace split is intentional: gap filling and data-quality evaluation share temporal concepts and validation rules, but they solve different problems and have different result contracts.

---

# Gap filling

Gap filling is a **transformative** operation: it can change missing values into reconstructed values and, through explicitly configured advanced overwrite rules, can replace existing values.

At a high level:

```text
prepared sources
      |
      v
validate and combine by source priority
      |
      v
apply ordered basic gap-filling rules
      |
      v
optionally apply ordered advanced rules
      |
      v
crop to target grid and validate coverage
      |
      +--> filled data
      +--> data-source provenance
      +--> cleaning-method provenance
```

Basic methods currently include:

- `linear_interpolation`;
- `copy_periods`;
- `average_periods`.

Advanced rule methods currently include:

- `external_profile`;
- `construct_from_sources`;
- `leave_missing`.

T-Clean also provides helpers for:

- unresolved-gap reporting;
- auxiliary acquisition planning;
- mapping auxiliary requirements to capable sources;
- constructing weighted profiles from historical periods;
- profile scaling;
- external-profile validation;
- provenance ranking.

See [Gap filling](docs/gap_filling.md) for the complete reference.

---

# Data quality

Data-quality evaluation is **diagnostic**: it does not modify the supplied source data.

Configured tests run in order and return:

```python
QualityEvaluation(failures=..., issues=...)
```

A **failure** means a test was evaluable and identified suspicious data according to its configured criterion.

An **issue** means evaluation encountered a limitation that should be reported separately from a failure, for example because a derived threshold or contextual reference population could not be evaluated.

Available methods are:

| Method | Detects |
| --- | --- |
| `range` | values outside configured lower and/or upper bounds |
| `value_run` | sustained runs near a configured value |
| `flatline` | sustained runs of effectively unchanged observations |
| `low_variability` | complete rolling windows with unusually small range |
| `repeated_pattern` | repeated non-overlapping temporal blocks |
| `rate_of_change` | excessive adjacent fixed or relative changes |
| `level_shift` | localized boundaries supported by persistent level shifts |
| `contextual_level` | individual observations unusual relative to temporal references |
| `contextual_profile` | temporal shapes unusual relative to reference profiles |
| `source_disagreement` | focal values that disagree with peer sources |

Several methods support **derived value specifications**, allowing thresholds to be calculated from eligible focal data rather than hard-coded. Contextual methods additionally support calendar-aware historical reference lattices.

See [Data quality](docs/data_quality.md) for complete configuration and output details.

---

# Validation and failure behaviour

T-Clean follows a fail-fast philosophy.

It does not silently:

- infer malformed configuration;
- accept unknown configuration fields;
- repair duplicate timestamps;
- sort malformed input;
- resample incompatible frequencies;
- reinterpret misaligned timestamps;
- ignore unknown methods;
- accept invalid temporal ranges;
- use incomplete supporting periods where completeness is required.

Structured DataFrame and Series contracts are validated with Pandera where appropriate.

This distinction is important for data quality:

- **invalid configuration or malformed input** raises an exception;
- **valid configuration that cannot be evaluated for part of the data** can produce a structured `issues` row instead.

---

# Application boundary

T-Clean is designed to be usable inside larger applications and workflow systems without depending on them.

For example, an application such as Modelblocks may own:

- provider-specific acquisition;
- credentials and authentication;
- domain-specific preparation;
- filesystem layout;
- Snakemake rules;
- workflow resources;
- plotting and final outputs.

T-Clean owns generic operations such as:

- temporal validation;
- gap-filling semantics;
- data-quality methods;
- reference construction;
- auxiliary requirement calculation;
- profile construction;
- provenance.

The intended boundary is:

```text
application prepares canonical time series
                |
                v
       T-Clean evaluates and/or fills
                |
                v
application consumes structured results
```

A generic T-Clean method should therefore not call a domain API, resolve credentials, build workflow jobs, or infer what a context label represents.

---

# Design principles

## Domain neutrality

No T-Clean API should require knowledge of the domain represented by a time series.

A context is simply a context, and a source is simply a named source.

## Explicit temporal semantics

Time-series bugs are often timestamp bugs.

T-Clean therefore treats start, end, frequency, timezone, phase, and half-open period semantics as explicit contracts.

## No silent repair

Malformed input should not be quietly transformed into something that merely looks plausible.

## Ordered deterministic behaviour

Source priority, gap-filling rule priority, and data-quality test order are explicit and stable.

## Evaluation is separate from transformation

Data-quality evaluation reports evidence without modifying data.

Gap filling reconstructs data only through explicitly configured rules.

Keeping those operations distinct lets consuming applications decide how findings should influence later processing.

## Provenance and diagnostics are part of the result

A reconstructed time series without provenance, or a quality flag without evidence, is incomplete for many analytical workflows.

## Planning is separate from acquisition

T-Clean may determine that a context-period is required. The consuming application decides how to obtain it.

## Transformation is separate from orchestration

T-Clean can be tested and used entirely without Snakemake or another workflow engine.

---

# Development

The repository uses Pixi for development.

Typical checks are:

```bash
pixi run ruff format .
pixi run ruff check .
pixi run pytest
```

The test suite covers shared temporal contracts, gap filling, data-quality evaluation, validation, planning, provenance, and pipeline interactions.

New functionality should normally include focused tests for its own semantics and at least one interaction-level test where ordering or another subsystem matters.

---

# Release workflow

T-Clean follows semantic versioning.

A normal release should:

1. implement and test changes;
2. update documentation;
3. bump the package version;
4. commit the release state;
5. create the matching Git tag and GitHub Release;
6. build and validate the Python distribution;
7. publish the matching version to PyPI;
8. allow/update downstream conda packaging as required.

GitHub tags, PyPI versions, and conda package versions should refer to the same source state.

---

# License

T-Clean is released under the MIT License.
