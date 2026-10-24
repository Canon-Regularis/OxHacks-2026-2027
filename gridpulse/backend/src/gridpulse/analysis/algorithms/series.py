"""The adapters' input contract: values on a regular half-hourly UTC grid."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from gridpulse.analysis.algorithms._checks import optional_number
from gridpulse.analysis.algorithms.errors import InvalidInputError, MissingDataError

HALF_HOUR = timedelta(minutes=30)
LONDON = ZoneInfo("Europe/London")


@dataclass(frozen=True)
class HalfHourlySeries:
    """Values on a gap-free 30-minute UTC grid; ``None`` marks a missing period.

    ``values[i]`` belongs to the period starting at ``start + i * HALF_HOUR``. Build the
    grid from UTC instants, never from settlement-period numbers: GB settlement days have
    46, 48 or 50 periods. NaN is stored as ``None``; booleans and infinities are rejected.
    """

    start: datetime
    values: Sequence[float | None]

    def __post_init__(self) -> None:
        if self.start.utcoffset() is None:
            raise InvalidInputError("start must be timezone-aware")
        start = self.start.astimezone(UTC)
        if start.minute % 30 or start.second or start.microsecond:
            raise InvalidInputError(f"start must fall on :00 or :30, got {start.isoformat()}")
        values = tuple(optional_number("values", i, value) for i, value in enumerate(self.values))
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "values", values)

    def __len__(self) -> int:
        return len(self.values)

    @property
    def end(self) -> datetime:
        """Start of the period after the last value."""
        return self.timestamp(len(self.values))

    def timestamp(self, index: int) -> datetime:
        """Start of period ``index``; indices past the end give forecast periods."""
        return self.start + index * HALF_HOUR

    def timestamps(self) -> tuple[datetime, ...]:
        return tuple(self.timestamp(i) for i in range(len(self.values)))

    def window(self, start: int, stop: int) -> "HalfHourlySeries":
        """Periods ``start`` to ``stop - 1`` as a series with the matching start time."""
        if not 0 <= start <= stop <= len(self.values):
            raise InvalidInputError(
                f"window [{start}, {stop}) lies outside a series of length {len(self.values)}"
            )
        return HalfHourlySeries(self.timestamp(start), self.values[start:stop])

    def complete_values(self) -> list[float]:
        """All values; ``MissingDataError`` if any period is missing."""
        missing = [i for i, value in enumerate(self.values) if value is None]
        if missing:
            raise MissingDataError(missing)
        return [value for value in self.values if value is not None]


def crosses_clock_change(start: datetime, stop: datetime) -> bool:
    """Whether periods starting in ``[start, stop)`` straddle a GB clock change."""
    if start.utcoffset() is None or stop.utcoffset() is None:
        raise InvalidInputError("start and stop must be timezone-aware")
    # Arithmetic on Europe/London datetimes uses wall-clock time, so work in UTC.
    start, stop = start.astimezone(UTC), stop.astimezone(UTC)
    last = stop - HALF_HOUR
    if last <= start:
        return False
    offset = start.astimezone(LONDON).utcoffset()
    probe = start
    while probe < last:
        probe = min(probe + timedelta(days=1), last)
        if probe.astimezone(LONDON).utcoffset() != offset:
            return True
    return False
