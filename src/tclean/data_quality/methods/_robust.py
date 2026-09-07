"""Shared robust-statistics helpers for data-quality methods."""

from dataclasses import dataclass

import numpy as np

_MAD_NORMAL_SCALE = 1.4826


@dataclass(frozen=True)
class RobustLocationScale:
    """Median, median absolute deviation, and MAD-based robust scale."""

    median: float
    mad: float
    scale: float


def robust_location_scale(values: np.ndarray) -> RobustLocationScale:
    """Return robust location and scale for a non-empty numeric vector."""
    if values.size == 0:
        raise ValueError("Robust location and scale require at least one value.")

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))

    return RobustLocationScale(
        median=median, mad=mad, scale=float(_MAD_NORMAL_SCALE * mad)
    )
