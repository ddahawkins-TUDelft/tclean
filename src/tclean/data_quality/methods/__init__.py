"""Registered data-quality evaluation methods."""

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from tclean.data_quality._method import MethodSpec

from .contextual_level import METHOD as CONTEXTUAL_LEVEL
from .contextual_profile import METHOD as CONTEXTUAL_PROFILE
from .fixed_rate_of_change import METHOD as FIXED_RATE_OF_CHANGE
from .flatline import METHOD as FLATLINE
from .level_shift import METHOD as LEVEL_SHIFT
from .low_variability import METHOD as LOW_VARIABILITY
from .range import METHOD as RANGE
from .relative_rate_of_change import METHOD as RELATIVE_RATE_OF_CHANGE
from .repeated_pattern import METHOD as REPEATED_PATTERN
from .value_run import METHOD as VALUE_RUN


def build_method_registry(methods: Iterable[MethodSpec]) -> Mapping[str, MethodSpec]:
    """Build an immutable method registry and reject duplicate names."""
    registry: dict[str, MethodSpec] = {}

    for method in methods:
        if not isinstance(method, MethodSpec):
            raise TypeError("Data-quality registry entries must be MethodSpec objects.")

        if not isinstance(method.name, str) or not method.name.strip():
            raise ValueError("Data-quality method names must be non-empty strings.")

        if method.name in registry:
            raise ValueError(f"Duplicate data-quality method name: {method.name!r}.")

        registry[method.name] = method

    return MappingProxyType(registry)


METHODS = build_method_registry(
    [
        RANGE,
        VALUE_RUN,
        FLATLINE,
        LOW_VARIABILITY,
        REPEATED_PATTERN,
        FIXED_RATE_OF_CHANGE,
        RELATIVE_RATE_OF_CHANGE,
        LEVEL_SHIFT,
        CONTEXTUAL_LEVEL,
        CONTEXTUAL_PROFILE,
    ]
)

__all__ = ["METHODS"]
