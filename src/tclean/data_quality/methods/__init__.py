"""Data-quality evaluation methods."""

from . import flatline, low_variability, range, value_run

METHODS = {
    range.METHOD_NAME: range,
    value_run.METHOD_NAME: value_run,
    flatline.METHOD_NAME: flatline,
    low_variability.METHOD_NAME: low_variability,
}

__all__ = ["METHODS"]
