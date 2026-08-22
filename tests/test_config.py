"""Tests for T-Clean configuration."""

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from tclean import TCleanConfig


def test_config_normalizes_hourly_frequency():
    """Normalize an hourly frequency to a timedelta."""
    config = TCleanConfig(frequency="1h")

    assert config.frequency == pd.Timedelta("1h")


def test_config_normalizes_subhourly_frequency():
    """Normalize a subhourly frequency to a timedelta."""
    config = TCleanConfig(frequency="30min")

    assert config.frequency == pd.Timedelta("30min")


def test_config_accepts_timedelta_frequency():
    """Accept an already normalized timedelta."""
    config = TCleanConfig(frequency=pd.Timedelta("2h"))

    assert config.frequency == pd.Timedelta("2h")


def test_config_rejects_zero_frequency():
    """Reject a zero time-series frequency."""
    with pytest.raises(ValueError, match="greater than zero"):
        TCleanConfig(frequency="0h")


def test_config_rejects_negative_frequency():
    """Reject a negative time-series frequency."""
    with pytest.raises(ValueError, match="greater than zero"):
        TCleanConfig(frequency="-1h")


def test_config_rejects_missing_frequency():
    """Reject a missing timedelta frequency."""
    with pytest.raises(ValueError, match="must not be missing"):
        TCleanConfig(frequency=pd.Timedelta("NaT"))


def test_config_rejects_numeric_frequency():
    """Reject numeric frequencies without explicit units."""
    with pytest.raises(TypeError, match="duration string"):
        TCleanConfig(
            frequency=1  # type: ignore[arg-type]
        )


def test_config_rejects_numeric_string_frequency():
    """Reject string frequencies without explicit units."""
    with pytest.raises(ValueError, match="explicit duration unit"):
        TCleanConfig(frequency="1")


def test_config_rejects_calendar_frequency():
    """Reject frequencies that are not fixed durations."""
    with pytest.raises(ValueError, match="valid fixed duration"):
        TCleanConfig(frequency="1M")


def test_config_is_immutable():
    """Prevent configuration from changing after construction."""
    config = TCleanConfig(frequency="1h")

    with pytest.raises(FrozenInstanceError):
        config.frequency = pd.Timedelta("30min")
