"""Validate and normalize data-quality tests."""

from collections.abc import Mapping, Sequence
from typing import Any

from tclean.data_quality.methods import METHODS
from tclean.time_grid import TimeGrid


def _validate_common_fields(test: Mapping[str, Any]) -> None:
    """Validate fields shared by every data-quality test."""
    name = test.get("name")
    method = test.get("method")

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Data-quality test 'name' must be a non-empty string.")

    if not isinstance(method, str) or not method.strip():
        raise ValueError("Data-quality test 'method' must be a non-empty string.")


def validate_quality_test(test: Mapping[str, Any], *, grid: TimeGrid) -> dict[str, Any]:
    """Validate and normalize one data-quality test."""
    if not isinstance(test, Mapping):
        raise TypeError("Each data-quality test must be a mapping.")

    _validate_common_fields(test)

    method_name = test["method"]

    try:
        method = METHODS[method_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported data-quality method: {method_name!r}.") from exc

    return method.validate(test, grid=grid)


def _validate_prior_failure_references(tests: Sequence[Mapping[str, Any]]) -> None:
    """Require failure references to target only preceding tests."""
    preceding_names: set[str] = set()

    for test in tests:
        references = test.get("include_failed_periods_from", [])

        unavailable = [name for name in references if name not in preceding_names]

        if unavailable:
            raise ValueError(
                "'include_failed_periods_from' may reference only "
                "preceding data-quality tests. "
                f"Unavailable names for test {test['name']!r}: "
                f"{unavailable!r}."
            )

        preceding_names.add(test["name"])


def validate_quality_tests(
    tests: Sequence[Mapping[str, Any]], *, grid: TimeGrid
) -> list[dict[str, Any]]:
    """Validate and normalize ordered data-quality tests.

    Test order is semantically meaningful. Reference-aware methods may use
    failures produced by preceding tests, but never failures from later tests.

    Args:
        tests: Ordered data-quality test configurations.
        grid: Temporal grid against which temporal test parameters are
            validated.

    Returns:
        Validated tests in their original execution order.

    Raises:
        TypeError: If tests are not supplied as an ordered sequence.
        ValueError: If a test is invalid, test names are duplicated, or
            failure references do not point exclusively to preceding tests.
    """
    if isinstance(tests, (str, bytes)) or not isinstance(tests, Sequence):
        raise TypeError("Data-quality tests must be an ordered sequence.")

    normalized = [validate_quality_test(test, grid=grid) for test in tests]

    names = [test["name"] for test in normalized]

    duplicates = sorted({name for name in names if names.count(name) > 1})

    if duplicates:
        raise ValueError(
            f"Data-quality test names must be unique. Duplicates: {duplicates!r}."
        )

    _validate_prior_failure_references(normalized)

    return normalized
