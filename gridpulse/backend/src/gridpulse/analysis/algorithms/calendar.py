"""Deterministic seasonal basis functions on the GB local clock (P0 #6)."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import cache

from gridpulse.analysis.algorithms._checks import require_int
from gridpulse.analysis.algorithms.errors import InvalidInputError
from gridpulse.analysis.algorithms.orchid import forecasting_baseline as fb
from gridpulse.analysis.algorithms.series import LONDON

DAY_SLOTS = 48
WEEK_SLOTS = 7 * DAY_SLOTS


@dataclass(frozen=True)
class FeatureMatrix:
    columns: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]


def fourier_terms(
    timestamps: Sequence[datetime], *, daily_harmonics: int = 0, weekly_harmonics: int = 0
) -> FeatureMatrix:
    """Sine and cosine terms of each period's Europe/London clock time.

    Phases are local half-hour slots (00:00 is slot 0; weeks start on Monday), so the
    repeated hour of a 50-period day reuses its slots and the skipped hour of a 46-period
    day never occurs. Weekly harmonics 7k duplicate daily harmonic k on these slots and
    are omitted when that daily harmonic is included.
    """
    require_int("daily_harmonics", daily_harmonics, minimum=0, maximum=DAY_SLOTS // 2)
    require_int("weekly_harmonics", weekly_harmonics, minimum=0, maximum=WEEK_SLOTS // 2)
    if daily_harmonics == weekly_harmonics == 0:
        raise InvalidInputError("request at least one daily or weekly harmonic")
    daily = _layout(DAY_SLOTS, daily_harmonics)
    weekly = _layout(WEEK_SLOTS, weekly_harmonics)
    kept = [j for j, (_, k) in enumerate(weekly) if k % 7 or k // 7 > daily_harmonics]
    columns = tuple(f"daily_{kind}_{k}" for kind, k in daily) + tuple(
        f"weekly_{weekly[j][0]}_{weekly[j][1]}" for j in kept
    )
    daily_table = _table(DAY_SLOTS, daily_harmonics)
    weekly_table = _table(WEEK_SLOTS, weekly_harmonics)
    rows = []
    for timestamp in timestamps:
        slot = _day_slot(timestamp)
        weekly_row = weekly_table[timestamp.astimezone(LONDON).weekday() * DAY_SLOTS + slot]
        rows.append(daily_table[slot] + tuple(weekly_row[j] for j in kept))
    return FeatureMatrix(columns, tuple(rows))


def _day_slot(timestamp: datetime) -> int:
    if timestamp.utcoffset() is None:
        raise InvalidInputError(f"timestamp must be timezone-aware, got {timestamp!r}")
    local = timestamp.astimezone(LONDON)
    if local.minute % 30 or local.second or local.microsecond:
        raise InvalidInputError(f"timestamp must start a half-hour period, got {timestamp!r}")
    return 2 * local.hour + local.minute // 30


def _layout(slots: int, harmonics: int) -> list[tuple[str, int]]:
    """(kind, harmonic) of each column of ``fb.fourier_features(n, slots, harmonics)``."""
    layout = []
    for k in range(1, harmonics + 1):
        if not (slots % 2 == 0 and k == slots // 2):  # the Nyquist sine is identically zero
            layout.append(("sin", k))
        layout.append(("cos", k))
    return layout


@cache
def _table(slots: int, harmonics: int) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(row) for row in fb.fourier_features(slots, slots, harmonics))
