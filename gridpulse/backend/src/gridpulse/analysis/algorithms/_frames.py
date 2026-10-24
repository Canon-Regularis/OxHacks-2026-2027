"""Conversions between the service's input frame (pandas) and the adapters' HalfHourlySeries."""

import numpy as np
import pandas as pd

from gridpulse.analysis.algorithms.series import HalfHourlySeries


def half_hourly(column: pd.Series) -> HalfHourlySeries:
    """A frame column as a series; NaN becomes a missing period."""
    values = [None if np.isnan(value) else float(value) for value in column.to_numpy(dtype=float)]
    return HalfHourlySeries(column.index[0].to_pydatetime(), values)


def as_floats(series: HalfHourlySeries) -> list[float]:
    """Values with missing periods as NaN, ready to become a frame column."""
    return [float("nan") if value is None else value for value in series.values]


def longest_complete(frame: pd.DataFrame) -> pd.DataFrame:
    """The longest run of consecutive rows with every column present (the earliest on a tie).

    Correlograms and model fits need gap-free data, and a contiguous stretch keeps every lag
    meaning the same elapsed time, which dropping the missing rows would not."""
    complete = frame.notna().all(axis=1).to_numpy()
    best_start, best_length, start = 0, 0, None
    for i, ok in enumerate([*complete, False]):  # the sentinel closes a final run
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start > best_length:
                best_start, best_length = start, i - start
            start = None
    return frame.iloc[best_start : best_start + best_length]
