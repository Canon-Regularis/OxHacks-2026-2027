"""Adapter errors.

``InvalidInputError`` and its subclasses mean the inputs cannot support a calculation
(report as missing or insufficient inputs). ``CalculationError`` means a validated
calculation failed (report as failed).
"""

from collections.abc import Sequence
from typing import Any


class AlgorithmError(Exception):
    """Base class for algorithm adapter errors."""


class InvalidInputError(AlgorithmError, ValueError):
    """Parameters or data cannot be used for the requested calculation."""


class MissingDataError(InvalidInputError):
    """The calculation needs complete data, but some periods are missing."""

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = tuple(indices)
        shown = ", ".join(map(str, self.indices[:10]))
        more = f" and {len(self.indices) - 10} more" if len(self.indices) > 10 else ""
        super().__init__(f"{len(self.indices)} missing value(s) at index {shown}{more}")

    def __reduce__(self) -> tuple[Any, ...]:  # rebuild from the arguments, not the message
        return type(self), (self.indices,), self.__dict__


class InsufficientDataError(InvalidInputError):
    """Too few observations for the requested calculation."""

    def __init__(self, required: int, available: int, what: str = "observations") -> None:
        self.required = required
        self.available = available
        self.what = what
        super().__init__(f"needs at least {required} {what}, got {available}")

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), (self.required, self.available, self.what), self.__dict__


class DegenerateDataError(InvalidInputError):
    """The input has no variation, so the requested statistic is undefined."""


class CalculationError(AlgorithmError):
    """A validated calculation failed or produced non-finite output."""
