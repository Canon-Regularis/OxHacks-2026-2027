"""Trailing imbalance price signals for chart overlays (P0 #3, #4, #5).

Every value uses only prices up to its own period. Window lengths are part of the output names,
so they are fixed rather than parameters.
"""

from datetime import timedelta

import pandas as pd

from gridpulse.analysis.algorithms import signals
from gridpulse.analysis.algorithms._frames import as_floats, half_hourly
from gridpulse.analysis.algorithms.series import HALF_HOUR, HalfHourlySeries
from gridpulse.analysis.contract import Algorithm

DAY = 48  # periods in the 24-hour windows
WEEK = 336  # periods in the 7-day percentile window
GBP = "GBP/MWh"
OUTPUTS = {
    f"price_mean_{DAY}": GBP,
    f"price_weighted_mean_{DAY}": GBP,
    f"price_ewma_{DAY}": GBP,
    f"price_variance_{DAY}": "(GBP/MWh)^2",
    f"price_std_{DAY}": GBP,
    f"price_median_{DAY}": GBP,
    f"price_rms_{DAY}": GBP,
    f"price_p10_{WEEK}": GBP,
    f"price_p90_{WEEK}": GBP,
    "price_change_1": GBP,
    f"price_change_{DAY}": GBP,
}


def _signals(price: HalfHourlySeries) -> dict[str, HalfHourlySeries]:
    return {
        f"price_mean_{DAY}": signals.trailing_mean(price, DAY),
        f"price_weighted_mean_{DAY}": signals.trailing_weighted_mean(price, range(1, DAY + 1)),
        f"price_ewma_{DAY}": signals.exponential_moving_average(price, span=DAY),
        f"price_variance_{DAY}": signals.rolling_variance(price, DAY),
        f"price_std_{DAY}": signals.rolling_std(price, DAY),
        f"price_median_{DAY}": signals.rolling_median(price, DAY),
        f"price_rms_{DAY}": signals.rolling_rms(price, DAY),
        f"price_p10_{WEEK}": signals.rolling_quantile(price, WEEK, 0.1),
        f"price_p90_{WEEK}": signals.rolling_quantile(price, WEEK, 0.9),
        "price_change_1": signals.difference(price, lag=1),
        f"price_change_{DAY}": signals.difference(price, lag=DAY),
    }


def run(frame: pd.DataFrame) -> pd.DataFrame:
    price = half_hourly(frame["price_gbp_mwh"])
    columns = {name: as_floats(series) for name, series in _signals(price).items()}
    return pd.DataFrame(columns, index=frame.index)


ALGORITHM = Algorithm(
    id="price_rolling_signals",
    version="0.1.0",
    description="Trailing price signals: 24-hour simple, linearly weighted and exponential means; "
    "24-hour variance, standard deviation, median and RMS; 7-day 10th/90th percentiles; period "
    "and daily changes.",
    run=run,
    inputs=("price_gbp_mwh",),
    outputs=OUTPUTS,
    warmup=timedelta(minutes=30 * WEEK) - HALF_HOUR,  # the 7-day window's first full value
    limitations="Trailing windows only, so each value was knowable at the end of its period. A "
    "window that touches a missing period gives null; the EWMA restarts after a gap. "
    f"price_change_{DAY} counts {DAY} elapsed half-hours, not the same clock time, across a clock "
    "change. Descriptive, not a forecast.",
)
