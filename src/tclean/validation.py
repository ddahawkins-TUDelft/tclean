"""Validation utilities for electricity-demand time series."""

from __future__ import annotations

import pandas as pd


def validate_load(load: pd.DataFrame) -> None:
    """Validate the structure of hourly electricity-demand data."""
    if not isinstance(load, pd.DataFrame):
        raise TypeError("Load must be a pandas DataFrame.")

    if load.empty:
        raise ValueError("Load dataframe is empty.")

    timestep = infer_regular_timestep(load.index)

    if timestep != pd.Timedelta(hours=1):
        raise ValueError(
            "Gap filling currently expects hourly load data. "
            f"Found timestep {timestep}."
        )

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in load.dtypes):
        raise TypeError("All load columns must be numeric.")


def infer_regular_timestep(index: pd.Index) -> pd.Timedelta:
    """Infer and validate the regular timestep of a datetime index."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("Load data must use a pandas DatetimeIndex.")

    if not index.is_monotonic_increasing:
        raise ValueError("Load timestamps must be sorted in increasing order.")

    if index.has_duplicates:
        raise ValueError("Load timestamps must not contain duplicates.")

    differences = index.to_series().diff().dropna()

    if differences.empty:
        raise ValueError("At least two timestamps are required for gap filling.")

    timestep = differences.iloc[0]

    if not differences.eq(timestep).all():
        raise ValueError(
            "Load data must have a complete, regular time index before gap filling."
        )

    return timestep
