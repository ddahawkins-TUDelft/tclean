"""Data-quality evaluation methods."""

from . import (
    fixed_rate_of_change,
    flatline,
    level_shift,
    low_variability,
    range,
    relative_rate_of_change,
    repeated_pattern,
    value_run,
)

METHODS = {
    range.METHOD_NAME: range,
    value_run.METHOD_NAME: value_run,
    flatline.METHOD_NAME: flatline,
    low_variability.METHOD_NAME: low_variability,
    repeated_pattern.METHOD_NAME: repeated_pattern,
    fixed_rate_of_change.METHOD_NAME: fixed_rate_of_change,
    relative_rate_of_change.METHOD_NAME: relative_rate_of_change,
    level_shift.METHOD_NAME: level_shift,
}

__all__ = ["METHODS"]
