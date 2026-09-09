# T-Clean data quality

This document is the detailed reference for the `tclean.data_quality` namespace.

Data-quality evaluation is the **diagnostic** side of T-Clean: it evaluates configured tests across one or more time-series sources and returns structured failures and evaluation issues without modifying the supplied data.

For shared concepts such as `TimeGrid`, contexts, sources, the canonical time-series contract, and package design principles, see the [main README](../README.md).

---

## Contents

- [Public API](#public-api)
- [Evaluation model](#evaluation-model)
- [Configuring tests](#configuring-tests)
- [Source and context selectors](#source-and-context-selectors)
- [Ordered tests and reference eligibility](#ordered-tests-and-reference-eligibility)
- [Value specifications](#value-specifications)
- [Outputs](#outputs)
- [Failures](#failures)
- [Issues](#issues)
- [Methods overview](#methods-overview)
- [`range`](#range)
- [`value_run`](#value_run)
- [`flatline`](#flatline)
- [`low_variability`](#low_variability)
- [`repeated_pattern`](#repeated_pattern)
- [`rate_of_change`](#rate_of_change)
- [`level_shift`](#level_shift)
- [`contextual_level`](#contextual_level)
- [`contextual_profile`](#contextual_profile)
- [`source_disagreement`](#source_disagreement)
- [Contextual reference lattices](#contextual-reference-lattices)
- [Threads and execution](#threads-and-execution)
- [Validation and failure behaviour](#validation-and-failure-behaviour)
- [Adding a new data-quality method](#adding-a-new-data-quality-method)

---

# Public API

The public interface is:

```python
from tclean.data_quality import QualityEvaluation, evaluate, validate_quality_tests
```

The high-level entry point is:

```python
evaluation = evaluate(
    sources,
    tests=tests,
    grid=grid,
    threads=1,
)
```

It returns:

```python
QualityEvaluation(
    failures=evaluation.failures,
    issues=evaluation.issues,
)
```

`evaluate(...)` does **not** modify the supplied source DataFrames.

Configuration can be validated independently before an expensive workflow begins:

```python
validated_tests = validate_quality_tests(tests, grid=grid)
```

---

# Evaluation model

A data-quality evaluation consists of:

- one or more named source DataFrames;
- an ordered sequence of test configurations;
- one shared `TimeGrid`;
- an optional maximum number of Python worker threads available to methods that support explicit parallel execution.

Conceptually:

```text
named canonical sources
          |
          v
validate sources and ordered tests
          |
          v
for each test in configuration order
          |
          +--> select focal sources
          |
          +--> select focal contexts
          |
          +--> expose earlier failures as reference exclusions
          |
          +--> evaluate method
          |
          +--> convert failed timestamps to half-open failure periods
          |
          +--> build structured diagnostic details
          |
          +--> collect evaluation issues
          |
          v
QualityEvaluation(failures, issues)
```

Configured tests remain sequential because test order can affect the reference data available to later tests.

For a single test, each focal source is evaluated against the same snapshot of failures produced by **preceding tests only**. Failures produced by one source during the current test do not alter the reference population used for another source during that same test.

---

# Configuring tests

Every test is a mapping with at least:

```python
{
    "name": "unique_test_name",
    "method": "registered_method",
}
```

`name` must be a non-empty unique string across the ordered test list.

`method` must be one of the registered data-quality methods.

Every method defines an explicit configuration contract. Missing required fields and unknown extra fields are rejected rather than ignored.

Example:

```python
tests = [
    {
        "name": "nonnegative",
        "method": "range",
        "minimum": {"value_mode": "fixed", "value": 0},
    },
    {
        "name": "flatline_12h",
        "method": "flatline",
        "minimum_duration": "12h",
        "tolerance": {"value_mode": "fixed", "value": 0.01},
    },
]
```

Test order is semantically meaningful and is preserved.

---

# Source and context selectors

Most tests can optionally restrict their focal domain with:

```python
"sources": ["primary", "secondary"],
"contexts": ["A", "B"],
```

Both selectors must be non-empty ordered sequences of unique non-empty strings.

## `sources`

When omitted, every supplied source is a focal source for the test.

When provided, only the named sources are evaluated as focal sources. Unknown source names are rejected.

A source selector does **not** hide other supplied sources from the method framework. This matters for methods such as `source_disagreement`, which evaluates a focal source while using non-focal supplied sources as peer evidence.

## `contexts`

When omitted, every context available in each focal source is evaluated.

When provided, T-Clean evaluates requested contexts that are present in each focal source. A requested context must exist in at least one selected focal source; otherwise configuration fails.

Context output order follows the focal source's column order rather than the order in the selector list.

---

# Ordered tests and reference eligibility

Some methods need a reference population derived from focal data or other supplied sources.

By default, observations belonging to failures from **preceding tests** are excluded from later reference populations when the failure belongs to the same source and context.

This helps prevent an earlier-detected anomaly from contaminating a later derived threshold or contextual reference.

Example:

```python
tests = [
    {
        "name": "negative_values",
        "method": "range",
        "minimum": {"value_mode": "fixed", "value": 0},
    },
    {
        "name": "large_change",
        "method": "rate_of_change",
        "difference_mode": "fixed",
        "threshold": {
            "value_mode": "median_absolute_increment",
            "multiplier": 10,
        },
    },
]
```

The derived `large_change` threshold is calculated from eligible focal data after periods failed by `negative_values` have been masked from its reference data.

## `include_failed_periods_from`

A later test can explicitly keep failed periods from selected **preceding** tests eligible for its reference population:

```python
{
    "name": "large_change",
    "method": "rate_of_change",
    "difference_mode": "fixed",
    "threshold": {
        "value_mode": "median_absolute_increment",
        "multiplier": 10,
    },
    "include_failed_periods_from": ["negative_values"],
}
```

References may name preceding tests only. Forward references and unknown/later names are rejected.

For methods using ordinary derived value specifications, `include_failed_periods_from` is accepted only when at least one configured scalar is actually derived from reference data.

Contextual methods and `source_disagreement` also support the field because their evidence populations are inherently reference-aware.

---

# Value specifications

Several data-quality methods allow a configurable scalar to be either fixed or derived from eligible focal data.

A value specification is always a mapping containing `value_mode`.

Supported modes are:

```text
fixed
mean
median
quantile
quantile_range
standard_deviation
mean_absolute_increment
median_absolute_increment
```

Derived values are resolved independently for each focal source/context from that test's eligible reference data.

## `fixed`

Use a literal value:

```python
{"value_mode": "fixed", "value": 100.0}
```

No reference data are required.

## `mean`

Use the arithmetic mean of eligible observations:

```python
{"value_mode": "mean"}
```

Optionally multiply it:

```python
{"value_mode": "mean", "multiplier": 2.0}
```

## `median`

Use the median of eligible observations:

```python
{"value_mode": "median", "multiplier": 1.5}
```

## `quantile`

Use one quantile between 0 and 1:

```python
{
    "value_mode": "quantile",
    "quantile": 0.99,
    "multiplier": 1.0,
}
```

## `quantile_range`

Use the difference between two configured quantiles:

```python
{
    "value_mode": "quantile_range",
    "lower_quantile": 0.25,
    "upper_quantile": 0.75,
    "multiplier": 3.0,
}
```

This example resolves to three times the interquartile range.

The lower quantile must be strictly less than the upper quantile.

## `standard_deviation`

Use the population standard deviation (`ddof=0`) of eligible observations:

```python
{
    "value_mode": "standard_deviation",
    "multiplier": 6.0,
}
```

## `mean_absolute_increment`

Calculate absolute adjacent increments from eligible data and use their mean:

```python
{
    "value_mode": "mean_absolute_increment",
    "multiplier": 10.0,
}
```

## `median_absolute_increment`

Calculate absolute adjacent increments from eligible data and use their median:

```python
{
    "value_mode": "median_absolute_increment",
    "multiplier": 10.0,
}
```

## Derived values that cannot be resolved

A derived value can be unavailable, for example because:

- no eligible observations remain;
- an increment-based mode has no eligible increments;
- the calculated property is non-finite;
- the final multiplied value is non-finite;
- a consuming method requires a positive/non-negative value and the resolved result violates that constraint.

This is reported as a structured `issues` event with:

```text
severity = "not_evaluable"
code     = "derived_value_not_evaluable"
```

rather than converting a configuration that was valid in principle into an arbitrary fallback value.

## Dimensionless thresholds

Some relative comparisons are dimensionless. Their thresholds must therefore use `value_mode: fixed` rather than being derived from a quantity with the original data units.

This applies to the `threshold` field of relative `rate_of_change` and relative `source_disagreement` tests.

---

# Outputs

`evaluate(...)` returns a frozen `QualityEvaluation` containing two canonical DataFrames:

```python
evaluation.failures
evaluation.issues
```

Both are returned even when empty, with stable expected columns and dtypes.

---

# Failures

A failure means the configured test was evaluable and the data met that method's failure criterion.

The canonical failure columns are:

| Column | Meaning |
| --- | --- |
| `context` | focal context |
| `source` | focal source |
| `start` | inclusive failure-period start |
| `end` | exclusive failure-period end |
| `test_name` | configured test name |
| `method` | registered method name |
| `details` | method-specific evidence mapping |

Methods first produce a Boolean failure mask aligned with the focal data. Consecutive failed grid timestamps are converted into contiguous half-open periods `[start, end)`.

The `details` mapping is method-specific and is intended to make the failure auditable rather than merely returning a Boolean flag.

---

# Issues

An issue means evaluation encountered a reportable limitation that is distinct from a quality failure.

The canonical issue columns are:

| Column | Meaning |
| --- | --- |
| `context` | focal context |
| `source` | focal source |
| `start` | inclusive affected-period start |
| `end` | exclusive affected-period end |
| `test_name` | configured test name |
| `method` | registered method name |
| `severity` | issue severity |
| `code` | stable machine-readable issue code |
| `details` | structured diagnostics |

Current severities include:

- `not_evaluable`: the configured criterion cannot be evaluated over the affected observations;
- `warning`: evaluation was possible in part, but one configured criterion or evidence component was unavailable.

Current issue codes can include:

- `derived_value_not_evaluable`;
- `insufficient_peer_data`;
- `incomplete_target_profile`;
- `insufficient_reference_profiles`;
- `insufficient_reference_data`;
- `criterion_not_evaluable`.

Invalid input or invalid configuration still raises an exception. `issues` are for limitations encountered while executing otherwise valid evaluation configuration.

---

# Methods overview

| Method | Core question |
| --- | --- |
| `range` | Is an observed value outside configured bounds? |
| `value_run` | Has the series remained near a configured value for long enough? |
| `flatline` | Have consecutive observations changed by no more than a tolerance for long enough? |
| `low_variability` | Does a complete rolling window have a sufficiently small total range? |
| `repeated_pattern` | Does a complete temporal block reproduce another block closely enough? |
| `rate_of_change` | Is the change from the previous observation too large? |
| `level_shift` | Is there evidence for a persistent change in level around a localized boundary? |
| `contextual_level` | Is one observation unusual relative to analogous historical timestamps? |
| `contextual_profile` | Is one temporal shape unusual relative to analogous historical profiles? |
| `source_disagreement` | Does a focal source disagree too strongly with peer sources? |

---

# `range`

Flags observed values below a configured minimum and/or above a configured maximum.

At least one bound must be supplied.

## Configuration

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | yes | unique test name |
| `method` | yes | `range` |
| `minimum` | at least one bound | value specification for inclusive lower acceptable boundary |
| `maximum` | at least one bound | value specification for inclusive upper acceptable boundary |
| `sources` | no | focal source selector |
| `contexts` | no | focal context selector |
| `include_failed_periods_from` | conditional | prior-failure exceptions when a bound is derived |

Fixed `minimum` and fixed `maximum` must not be inverted.

If both are derived, or one is derived, the resolved values are checked independently for each source/context. A source/context becomes not evaluable if its resolved minimum exceeds its resolved maximum.

## Example

```python
{
    "name": "plausible_range",
    "method": "range",
    "minimum": {"value_mode": "fixed", "value": 0},
    "maximum": {
        "value_mode": "quantile",
        "quantile": 0.999,
        "multiplier": 1.2,
    },
}
```

## Failure rule

An observed value fails when:

```text
value < resolved minimum
```

or:

```text
value > resolved maximum
```

Equality with a bound is accepted.

## Failure details

Details include observed extrema over the failed period, resolved bounds and their resolution metadata, and maximum excursions below/above the configured bounds.

---

# `value_run`

Flags sufficiently long contiguous runs whose observations remain within an absolute tolerance of a configured value.

## Configuration

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | yes | unique test name |
| `method` | yes | `value_run` |
| `value` | yes | target value specification |
| `minimum_duration` | yes | minimum failing run duration |
| `tolerance` | no | non-negative absolute tolerance; defaults to fixed `0` |
| `sources` | no | focal source selector |
| `contexts` | no | focal context selector |
| `include_failed_periods_from` | conditional | prior-failure exceptions when `value` or `tolerance` is derived |

`minimum_duration` must be positive and grid-aligned.

A fixed tolerance must be non-negative.

## Example

```python
{
    "name": "long_zero_run",
    "method": "value_run",
    "value": {"value_mode": "fixed", "value": 0},
    "tolerance": {"value_mode": "fixed", "value": 0.01},
    "minimum_duration": "12h",
}
```

## Failure rule

An observed timestamp belongs to the matching set when:

```text
abs(value - target_value) <= tolerance
```

A contiguous matching run fails when its number of grid observations is at least `minimum_duration / grid.frequency`.

## Failure details

Details include the resolved target value, resolved tolerance, run duration, required duration, observed min/max, and maximum absolute deviation from the target value.

---

# `flatline`

Flags sufficiently long runs of observations that are effectively unchanged from one step to the next.

## Configuration

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | yes | unique test name |
| `method` | yes | `flatline` |
| `minimum_duration` | yes | minimum failing run duration |
| `tolerance` | no | maximum allowed absolute step change; defaults to fixed `0` |
| `sources` | no | focal source selector |
| `contexts` | no | focal context selector |
| `include_failed_periods_from` | conditional | prior-failure exceptions when tolerance is derived |

`minimum_duration` must span at least two grid steps.

A fixed tolerance must be non-negative.

## Example

```python
{
    "name": "flatline_24h",
    "method": "flatline",
    "minimum_duration": "24h",
    "tolerance": {"value_mode": "fixed", "value": 0.001},
}
```

## Failure rule

Two adjacent observed values belong to the same stable run when:

```text
abs(current - previous) <= tolerance
```

A stable observed run fails once its total number of observations reaches the configured minimum duration.

The criterion is based on **adjacent step changes**, not on the total range across the whole run. With a non-zero tolerance, a long run can therefore drift gradually while each individual step remains within tolerance.

## Failure details

Details include duration, minimum duration, resolved tolerance, observed range, and maximum observed step change within the failed period.

---

# `low_variability`

Flags observations that belong to at least one complete rolling window whose total observed range is less than or equal to a configured maximum.

## Configuration

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | yes | unique test name |
| `method` | yes | `low_variability` |
| `window_duration` | yes | complete rolling-window duration |
| `maximum_range` | yes | maximum acceptable `max - min` value specification |
| `sources` | no | focal source selector |
| `contexts` | no | focal context selector |
| `include_failed_periods_from` | conditional | prior-failure exceptions when `maximum_range` is derived |

`window_duration` must span at least two grid steps.

A fixed `maximum_range` must be non-negative.

## Example

```python
{
    "name": "low_variability_24h",
    "method": "low_variability",
    "window_duration": "24h",
    "maximum_range": {
        "value_mode": "quantile_range",
        "lower_quantile": 0.25,
        "upper_quantile": 0.75,
        "multiplier": 0.1,
    },
}
```

## Failure rule

Only complete windows with no missing observations are evaluated.

A window qualifies when:

```text
window.max() - window.min() <= maximum_range
```

Every observation belonging to at least one qualifying window is flagged. Overlapping qualifying windows can therefore merge into a longer contiguous failure period.

## Failure details

Details include the window duration, resolved maximum range, number of contributing windows, minimum/maximum qualifying window range, and observed range across the resulting failed period.

---

# `repeated_pattern`

Flags complete non-overlapping temporal blocks whose values reproduce enough other complete blocks within an absolute tolerance.

## Configuration

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | yes | unique test name |
| `method` | yes | `repeated_pattern` |
| `pattern_duration` | yes | duration of each block |
| `minimum_matches` | yes | minimum total number of mutually matching blocks, including the focal block |
| `tolerance` | no | maximum pointwise absolute difference; defaults to fixed `0` |
| `sources` | no | focal source selector |
| `contexts` | no | focal context selector |
| `include_failed_periods_from` | conditional | prior-failure exceptions when tolerance is derived |

`pattern_duration` must span at least two grid steps.

`minimum_matches` must be an integer of at least 2.

A fixed tolerance must be non-negative.

## Example

```python
{
    "name": "repeated_day_shape",
    "method": "repeated_pattern",
    "pattern_duration": "24h",
    "minimum_matches": 3,
    "tolerance": {"value_mode": "fixed", "value": 0.01},
}
```

## Block construction

Blocks are:

- non-overlapping;
- aligned from the beginning of the supplied focal series;
- exactly `pattern_duration` long;
- ignored when any value in the block is missing.

## Failure rule

With `tolerance = 0`, blocks must match exactly.

With a positive tolerance, two blocks match when their **maximum pointwise absolute difference** is less than or equal to the tolerance.

A block fails when:

```text
number of direct matching blocks + itself >= minimum_matches
```

All timestamps within each qualifying block are flagged.

## Failure details

Details include the pattern duration, minimum matches, resolved tolerance, number of failed blocks, and per-block information about matching periods and their maximum absolute differences.

---

# `rate_of_change`

Flags adjacent observed transitions whose change exceeds a configured fixed or relative threshold.

## Configuration

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | yes | unique test name |
| `method` | yes | `rate_of_change` |
| `difference_mode` | yes | `fixed` or `relative` |
| `threshold` | yes | failing change threshold |
| `reference_magnitude_threshold` | relative only | minimum absolute previous value eligible for relative comparison; defaults to fixed `0` |
| `sources` | no | focal source selector |
| `contexts` | no | focal context selector |
| `include_failed_periods_from` | conditional | prior-failure exceptions for derived scalar fields |

## Fixed mode

```python
{
    "name": "large_absolute_change",
    "method": "rate_of_change",
    "difference_mode": "fixed",
    "threshold": {
        "value_mode": "median_absolute_increment",
        "multiplier": 10,
    },
}
```

The transition ending at `current` fails when:

```text
abs(current - previous) > threshold
```

The threshold may be fixed or derived.

`reference_magnitude_threshold` is not allowed in fixed mode.

## Relative mode

```python
{
    "name": "large_relative_change",
    "method": "rate_of_change",
    "difference_mode": "relative",
    "threshold": {"value_mode": "fixed", "value": 0.5},
    "reference_magnitude_threshold": {
        "value_mode": "fixed",
        "value": 1.0,
    },
}
```

A transition is eligible when both values are observed and:

```text
abs(previous) > reference_magnitude_threshold
```

It fails when:

```text
abs(current - previous) / abs(previous) > threshold
```

The relative `threshold` must be fixed because it is dimensionless.

`reference_magnitude_threshold` may be fixed or derived and must resolve to a non-negative value.

## Failure details

Details include the difference mode, resolved thresholds, transition count, and for each failing transition the current/previous timestamps and values, absolute change, and relative change where applicable.

---

# `level_shift`

Detects localized boundaries supported by a persistent change in level across two adjacent windows.

## Configuration

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | yes | unique test name |
| `method` | yes | `level_shift` |
| `window_duration` | yes | evidence-window duration on each side of a candidate boundary |
| `threshold` | yes | minimum absolute median paired shift |
| `sources` | no | focal source selector |
| `contexts` | no | focal context selector |
| `include_failed_periods_from` | conditional | prior-failure exceptions when threshold is derived |

`window_duration` must span at least two grid steps.

A fixed threshold must be positive.

## Example

```python
{
    "name": "persistent_level_shift",
    "method": "level_shift",
    "window_duration": "24h",
    "threshold": {
        "value_mode": "standard_deviation",
        "multiplier": 2,
    },
}
```

## Failure rule

For a window containing `N` grid observations, T-Clean forms paired differences between each value and the value `N` steps earlier. It then uses the median paired difference as a signed shift score around candidate boundaries.

A boundary qualifies when the evidence window is complete and:

```text
abs(shift_score) > threshold
```

Contiguous runs of qualifying boundaries are treated as one shift event. T-Clean localizes that event to the boundary with the strongest absolute shift score; ties across a maximum plateau are resolved to the middle qualifying position.

The resulting failure is therefore localized to a single grid timestamp per detected shift event rather than flagging the whole supporting evidence window.

## Failure details

Details include the configured window duration, resolved threshold, localized change point, estimated signed shift, number of qualifying boundaries supporting the event, and the temporal extent of pre/post evidence.

---

# `contextual_level`

Evaluates individual observations against values at analogous historical timestamps defined by a configurable temporal reference lattice.

This is useful when the expected level varies systematically with temporal context and a global threshold would be inappropriate.

## Configuration

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | yes | unique test name |
| `method` | yes | `contextual_level` |
| `reference_orders` | yes | ordered temporal reference-lattice definitions |
| `robust_deviation_threshold` | at least one criterion | positive robust-deviation threshold |
| `maximum_predictive_probability` | at least one criterion | probability in `(0, 1)` |
| `sources` | no | focal source selector |
| `contexts` | no | focal context selector |
| `include_failed_periods_from` | no | preceding failures explicitly retained in reference data |

At least one of `robust_deviation_threshold` or `maximum_predictive_probability` must be configured. Both can be used together.

## Example

```python
{
    "name": "unusual_level",
    "method": "contextual_level",
    "reference_orders": [
        {"period": "7D", "radius": 4},
        {"period": "1Y", "radius": 2},
    ],
    "robust_deviation_threshold": 6,
}
```

For the reference-lattice semantics, see [Contextual reference lattices](#contextual-reference-lattices).

## Robust deviation criterion

The method extracts observed reference values and calculates:

```text
reference median
MAD = median(abs(reference - median))
robust scale = 1.4826 * MAD
```

The target's robust deviation is:

```text
abs(target - reference median) / robust scale
```

when the robust scale is non-zero.

It fails when the robust deviation is strictly greater than `robust_deviation_threshold`.

If the reference population has zero robust variation, an identical target passes; a different target fails.

At least one observed reference value is required for this criterion.

## Predictive probability criterion

The method calculates the reference mean and sample standard deviation and evaluates the target using a two-sided Student-t predictive probability with:

```text
df = reference observations - 1
prediction SE = reference SD * sqrt(1 + 1 / n)
```

It fails when:

```text
predictive_probability <= maximum_predictive_probability
```

At least two observed reference values are required.

## Combining criteria

When both criteria are configured, a target observation fails if **either** available criterion fails.

The failure details record which criteria failed.

If no contextual reference observations are available, the affected target can produce:

```text
severity = "not_evaluable"
code     = "insufficient_reference_data"
```

If some reference data exist but one configured criterion cannot be evaluated while another can, T-Clean can report:

```text
severity = "warning"
code     = "criterion_not_evaluable"
```

## Failure details

Per-observation details can include:

- timestamp and value;
- reference observation count;
- failed criteria;
- reference median/MAD/robust scale;
- signed and standardized robust deviation;
- reference mean/SD;
- predictive standard error;
- Student-t statistic and degrees of freedom;
- predictive probability.

Contiguous failing observations are grouped into one failure period with their observation-level evidence retained in `details`.

---

# `contextual_profile`

Evaluates complete temporal **shapes** against analogous historical profiles defined by a contextual reference lattice.

Unlike `contextual_level`, which asks whether one observation has an unusual magnitude, `contextual_profile` centers and scales each profile so that comparisons focus on shape rather than absolute level.

## Configuration

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | yes | unique test name |
| `method` | yes | `contextual_profile` |
| `profile_duration` | yes | length of each target/reference profile |
| `reference_orders` | yes | ordered temporal reference-lattice definitions |
| `profile_offset` | no | offset from `grid.start` used to anchor target profiles; defaults to `0` |
| `robust_deviation_threshold` | at least one criterion | positive robust shape-deviation threshold |
| `maximum_predictive_probability` | at least one criterion | probability in `(0, 1)` |
| `sources` | no | focal source selector |
| `contexts` | no | focal context selector |
| `include_failed_periods_from` | no | preceding failures explicitly retained in reference data |

`profile_duration` must span at least two grid steps.

`profile_offset` must be non-negative, grid-aligned, and strictly less than `profile_duration`.

At least one criterion must be configured.

## Example

```python
{
    "name": "unusual_daily_shape",
    "method": "contextual_profile",
    "profile_duration": "24h",
    "profile_offset": "0h",
    "reference_orders": [
        {"period": "7D", "radius": 4},
        {"period": "1Y", "radius": 10},
    ],
    "robust_deviation_threshold": 6,
}
```

## Target profiles

Target profiles are non-overlapping windows anchored from:

```text
grid.start + profile_offset
```

with spacing equal to `profile_duration`.

Only profiles completely contained in the available focal data and target grid are considered.

A target profile containing missing observations cannot be evaluated and is reported with:

```text
severity = "not_evaluable"
code     = "incomplete_target_profile"
```

## Shape normalization

Each complete raw profile is converted to shape-only values:

1. subtract the profile mean;
2. calculate the root mean square of the centered values;
3. divide centered values by that scale.

Conceptually:

```text
centered = values - mean(values)
scale    = sqrt(mean(centered ** 2))
shape    = centered / scale
```

A profile with zero variation becomes an all-zero normalized shape.

## Reference profiles

Reference-lattice timestamps identify candidate **profile starts**.

A candidate reference profile is used only when:

- its complete period exists;
- it has no missing observations after preceding-failure exclusions;
- it does not overlap the target profile.

Profile distance is aligned root-mean-square error between normalized shapes:

```text
RMSE(target_shape, reference_shape)
```

## Robust deviation criterion

At least three complete reference profiles are required.

For the reference profiles, T-Clean calculates pairwise shape distances. Each reference profile receives a peer score equal to its median distance to all other reference profiles.

The reference peer scores define a robust location/scale using their median and MAD-based scale.

The target score is its median distance to the reference profiles.

Only positive target deviation above the reference median contributes to robust nonconformity:

```text
max(target_distance - reference_distance_median, 0) / robust_scale
```

The criterion fails when this robust deviation is strictly greater than `robust_deviation_threshold`.

## Predictive probability criterion

At least one complete reference profile is required.

T-Clean calculates peer-distance scores for the reference profiles plus the target profile, then assigns the target a rank-based predictive probability:

```text
number of profiles at least as nonconforming as target
------------------------------------------------------
             total comparison profiles
```

It fails when:

```text
predictive_probability <= maximum_predictive_probability
```

## Combining criteria

When both criteria are configured, a profile fails when either available criterion fails.

Insufficient reference profiles can produce:

```text
severity = "not_evaluable"
code     = "insufficient_reference_profiles"
```

If one criterion is unavailable but another can be evaluated, T-Clean can report:

```text
severity = "warning"
code     = "criterion_not_evaluable"
```

## Failure periods

When a target profile fails, all grid observations in that complete target profile are flagged. Adjacent failed profiles can therefore merge into a longer contiguous failure period.

## Failure details

Per-profile details can include:

- profile start/end;
- number of complete reference profiles;
- failed criteria;
- target profile distance;
- robust reference distance median/MAD/scale;
- robust deviation;
- predictive rank counts and probability.

---

# `source_disagreement`

Compares one focal source with all eligible non-focal supplied sources for the same context and timestamp.

This method is source-aware: `sources` selects which sources should be evaluated as focal sources, while other supplied sources remain available as peers.

## Configuration

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | yes | unique test name |
| `method` | yes | `source_disagreement` |
| `difference_mode` | yes | `fixed` or `relative` |
| `threshold` | yes | disagreement threshold |
| `peer_aggregation_mode` | no | `median` or `mean`; defaults to `median` |
| `sources` | no | focal source selector |
| `contexts` | no | focal context selector |
| `include_failed_periods_from` | no | preceding failures explicitly retained in focal/peer reference data |

A fixed threshold must be positive.

A relative threshold must use `value_mode: fixed` because it is dimensionless.

## Example

```python
{
    "name": "cross_source_disagreement",
    "method": "source_disagreement",
    "sources": ["primary"],
    "difference_mode": "relative",
    "threshold": {"value_mode": "fixed", "value": 0.1},
    "peer_aggregation_mode": "median",
}
```

This evaluates `primary` as the focal source while using any other supplied source containing the same context as peer evidence.

## Peer reference

For each focal context and timestamp, T-Clean collects eligible values from all non-focal sources containing that context.

Available peer values are aggregated row-wise by:

- `median` (default); or
- `mean`.

Peer sources can have different context coverage.

When a focal observation exists but no eligible peer observation exists, T-Clean reports:

```text
severity = "not_evaluable"
code     = "insufficient_peer_data"
```

## Fixed difference

The focal observation fails when:

```text
abs(focal - peer_reference) > threshold
```

The threshold may be fixed or derived from eligible focal reference data.

## Relative difference

When the peer reference is non-zero:

```text
relative_difference = abs(focal - peer_reference) / abs(peer_reference)
```

and the observation fails when this exceeds the fixed threshold.

When both focal and peer reference equal zero, the relative difference is treated as zero.

When the peer reference is zero but the focal value is non-zero, the disagreement is unbounded and fails.

## Failure details

Per-observation details include:

- focal value;
- participating peer sources and values;
- peer observation count;
- aggregated peer reference value;
- signed and absolute difference;
- relative difference when configured;
- resolved threshold information.

---

# Contextual reference lattices

`contextual_level` and `contextual_profile` use `reference_orders` to define analogous historical timestamps.

Example:

```python
"reference_orders": [
    {"period": "7D", "radius": 4},
    {"period": "1Y", "radius": 2},
]
```

Each order contains exactly:

| Field | Meaning |
| --- | --- |
| `period` | one fixed-duration or calendar-aware temporal step |
| `radius` | positive integer number of steps explored on either side |

The sequence must be non-empty and duplicate order definitions are rejected.

## Fixed periods

Values such as:

```text
7D
24h
30min
```

are interpreted as fixed durations and must align with the configured grid.

## Calendar-aware periods

Strings of the form:

```text
1mo
2mo
1Y
10Y
```

are interpreted as calendar months or calendar years rather than fixed numbers of days.

This distinction is deliberate. `1Y` means the analogous calendar timestamp in another year; it is **not** converted to `365D`.

If a calendar shift would produce an invalid timestamp, for example shifting February 29 into a non-leap year, that branch is discarded rather than coerced to another date.

## Lattice construction

Orders are configured from lower to higher temporal scale but expanded from higher to lower order.

For each target timestamp, T-Clean constructs combinations of coefficients from `-radius` through `+radius` at every order, removes the target timestamp itself, removes invalid calendar branches, removes timestamps not present in the available reference index, deduplicates, and sorts the result.

With:

```python
[
    {"period": "7D", "radius": 1},
    {"period": "1Y", "radius": 1},
]
```

references can include the same weekly neighbourhood around the equivalent timestamp in the previous year, current year, and next year, subject to data availability and removal of the focal target itself.

For `contextual_profile`, these timestamps are candidate **profile starts**, after which incomplete and overlapping reference profiles are removed.

## Preceding failures

The available reference index remains temporal, but values failed by preceding tests are masked from the reference data unless explicitly retained with `include_failed_periods_from`.

Consequently, a lattice timestamp can exist while its value/profile is unavailable as usable evidence.

---

# Threads and execution

`evaluate(...)` accepts:

```python
threads=1
```

`threads` must be an integer of at least 1.

It represents the maximum number of Python worker threads available to data-quality methods that support explicit parallel execution. Methods may use fewer threads when parallel execution is not beneficial.

The default is deliberately conservative:

```python
threads=1
```

When the supplied thread count is 1, contextual method execution uses the serial path and does not create a `ThreadPoolExecutor`.

Native numerical libraries used by NumPy or SciPy can manage their own threading independently of this parameter. Workflow engines and numerical runtimes may therefore impose additional resource limits outside T-Clean's explicit Python worker setting.

Test order itself is never parallelized because later tests can depend on failures produced by earlier tests.

---

# Validation and failure behaviour

Data quality follows T-Clean's fail-fast philosophy.

Configuration errors raise rather than becoming issue rows. Examples include:

- tests not supplied as an ordered sequence;
- duplicate test names;
- unsupported methods;
- unknown configuration fields;
- invalid source/context selectors;
- invalid durations;
- invalid fixed threshold values;
- invalid quantiles;
- `include_failed_periods_from` referencing a later test;
- a relative dimensionless threshold configured with a derived value mode;
- malformed source time-series data.

By contrast, a valid test can produce an `issues` row when its data-dependent evidence is unavailable during evaluation.

This distinction keeps configuration errors explicit while allowing incomplete real-world evidence to be represented as structured output.

---

# Adding a new data-quality method

Every registered method is represented by a `MethodSpec` containing three contracts:

```text
validate
    normalize and validate method configuration

evaluate
    return a MethodResult with an aligned Boolean failure mask

build_details
    construct structured diagnostics for one contiguous failed period
```

A method evaluation receives one `MethodContext` containing:

- the focal source name;
- all supplied source frames;
- selected focal contexts;
- normalized test configuration;
- the `TimeGrid`;
- the available Python thread count;
- a snapshot of failures from preceding tests.

The context exposes helpers for:

- selected focal data;
- other source data;
- failure-filtered reference data;
- cross-source evidence.

When adding a method:

1. define an explicit configuration validator;
2. avoid domain-specific assumptions;
3. return a Boolean DataFrame mask exactly aligned with `context.target_data`;
4. return structured `MethodIssue` objects for valid-but-not-evaluable situations;
5. keep output deterministic in source/context/timestamp order;
6. use reference data through the framework so preceding-failure semantics remain consistent;
7. build meaningful failure details rather than only Boolean flags;
8. register the `MethodSpec` in the method registry;
9. add validation, method, issue, detail, ordering, and integration tests;
10. document the method and every configuration field here.

A new method should not introduce application-specific acquisition, filesystem, workflow, or domain logic into T-Clean.
