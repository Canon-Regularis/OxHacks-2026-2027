"""Parameter and value checks shared by the adapters."""

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from numbers import Real

from gridpulse.analysis.algorithms.errors import (
    CalculationError,
    InvalidInputError,
    MissingDataError,
)


@contextmanager
def orchid_errors(source: str) -> Iterator[None]:
    """Report an Orchid failure on validated input as ``CalculationError``."""
    try:
        yield
    except (ValueError, ArithmeticError) as exc:  # Orchid's ForecastError and LinAlgError too
        raise CalculationError(f"Orchid {source} failed: {exc}") from exc


def require_real(name: str, value: object) -> float:
    """A real number (not a bool) that fits in a float."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InvalidInputError(f"{name} must be a real number, got {value!r}")
    try:
        return float(value)
    except OverflowError:
        raise InvalidInputError(f"{name} is too large for a float") from None


def optional_number(name: str, index: int, value: object) -> float | None:
    """``value`` as a float, or ``None`` when it is missing (``None`` or NaN)."""
    if value is None:
        return None
    number = require_real(f"{name}[{index}]", value)
    if math.isinf(number):
        raise InvalidInputError(f"{name}[{index}] is infinite")
    return None if math.isnan(number) else number


def finite_values(name: str, values: Sequence[object]) -> list[float]:
    """All of ``values`` as floats; ``MissingDataError`` if any is ``None`` or NaN."""
    numbers = [optional_number(name, i, value) for i, value in enumerate(values)]
    missing = [i for i, number in enumerate(numbers) if number is None]
    if missing:
        raise MissingDataError(missing)
    return [number for number in numbers if number is not None]


def require_int(name: str, value: object, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidInputError(f"{name} must be an integer, got {value!r}")
    if value < minimum or (maximum is not None and value > maximum):
        expected = f">= {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        raise InvalidInputError(f"{name} must be {expected}, got {value}")
    return value


def require_fraction(name: str, value: object) -> float:
    """A level or quantile strictly between 0 and 1 (0.95, not 95)."""
    number = require_real(name, value)
    if not 0 < number < 1:
        raise InvalidInputError(f"{name} must be strictly between 0 and 1, got {value!r}")
    return number


def require_weights(weights: Sequence[float]) -> tuple[float, ...]:
    """Positive, finite weights with a finite sum."""
    checked = tuple(require_real(f"weights[{i}]", weight) for i, weight in enumerate(weights))
    try:
        total = math.fsum(checked)  # as Orchid normalises them
    except OverflowError:
        total = math.inf
    valid = checked and all(0 < weight < math.inf for weight in checked)
    if not valid or not math.isfinite(total):
        raise InvalidInputError(
            f"weights must be non-empty, positive and have a finite sum, got {list(weights)}"
        )
    return checked


def require_levels(levels: Sequence[float]) -> tuple[float, ...]:
    """Distinct interval coverage levels in ascending order."""
    checked = sorted(require_fraction("level", level) for level in levels)
    if not checked or len(set(checked)) != len(checked):
        raise InvalidInputError(f"levels must be distinct and non-empty, got {list(levels)}")
    return tuple(checked)
