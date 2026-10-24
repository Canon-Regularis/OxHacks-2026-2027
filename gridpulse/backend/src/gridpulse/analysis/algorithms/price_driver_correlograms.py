"""Correlograms of the imbalance price and its drivers (P0 #7, #8).

ACF and PACF of the price, and its cross-correlation with residual demand and the wind forecast,
over the longest gap-free stretch of the range. Tables only.
"""

from collections.abc import Sequence

import pandas as pd

from gridpulse.analysis.algorithms import diagnostics
from gridpulse.analysis.algorithms._frames import half_hourly, longest_complete
from gridpulse.analysis.contract import Algorithm, AlgorithmResult

LAG_COLUMNS = ["lag_periods", "lag_hours", "value", "band"]
SAMPLE_COLUMNS = ["first_utc", "last_utc", "periods", "excluded_periods"]
TABLE_COLUMNS = {
    "sample": SAMPLE_COLUMNS,
    "acf_price": LAG_COLUMNS,
    "pacf_price": LAG_COLUMNS,
    "ccf_residual_demand": LAG_COLUMNS,
    "ccf_wind": LAG_COLUMNS,
}
INPUTS = ("price_gbp_mwh", "residual_demand_mw", "wind_mw")


def sample_table(frame: pd.DataFrame, used: pd.DataFrame) -> pd.DataFrame:
    """One row describing the gap-free stretch ``used`` out of ``frame``."""
    first, last = (used.index[0], used.index[-1]) if len(used) else (None, None)
    return pd.DataFrame([[first, last, len(used), len(frame) - len(used)]], columns=SAMPLE_COLUMNS)


def _lags(lags: Sequence[int], values: Sequence[float], band: Sequence[float]) -> pd.DataFrame:
    rows = [[k, k / 2, v, b] for k, v, b in zip(lags, values, band, strict=True)]
    return pd.DataFrame(rows, columns=LAG_COLUMNS)


def run(frame: pd.DataFrame, *, max_lag: int, alpha: float) -> AlgorithmResult:
    used = longest_complete(frame[list(INPUTS)])
    tables = {name: pd.DataFrame(columns=columns) for name, columns in TABLE_COLUMNS.items()}
    tables["sample"] = sample_table(frame, used)
    if len(used) < 2 * max_lag + 2:  # the PACF's minimum; shorter stretches give empty tables
        return AlgorithmResult(tables=tables)

    price = half_hourly(used["price_gbp_mwh"])
    acf = diagnostics.autocorrelation(price, max_lag=max_lag, alpha=alpha)
    pacf = diagnostics.partial_autocorrelation(price, max_lag=max_lag, alpha=alpha)
    tables["acf_price"] = _lags(acf.lags, acf.values, acf.band)
    tables["pacf_price"] = _lags(pacf.lags, pacf.values, pacf.band)
    for name, column in (("ccf_residual_demand", "residual_demand_mw"), ("ccf_wind", "wind_mw")):
        ccf = diagnostics.cross_correlation(
            price, half_hourly(used[column]), max_lag=max_lag, alpha=alpha
        )
        tables[name] = _lags(ccf.lags, ccf.values, ccf.band)
    return AlgorithmResult(tables=tables)


ALGORITHM = Algorithm(
    id="price_driver_correlograms",
    version="0.1.0",
    description="Autocorrelation of the imbalance price and its cross-correlation with residual "
    "demand and the wind forecast, by lag.",
    run=run,
    inputs=INPUTS,
    outputs={},
    tables={
        "sample": "The gap-free stretch of the range the correlations use: its first and last "
        "period, its length and how many periods were left out",
        "acf_price": "Price autocorrelation at lags 1..K with a Bartlett band",
        "pacf_price": "Price partial autocorrelation at lags 1..K with a +/- z/sqrt(n) band",
        "ccf_residual_demand": "corr(price[t+k], residual_demand[t]) at lags -K..K; k > 0 means "
        "demand leads",
        "ccf_wind": "corr(price[t+k], wind_forecast[t]) at lags -K..K; k > 0 means wind leads",
    },
    params={"max_lag": 96, "alpha": 0.05},
    param_descriptions={
        "max_lag": "Largest lag in periods (96 = 48 h); cross-correlations cover -max_lag to "
        "+max_lag",
        "alpha": "Significance level of the band (0.05 = 95% band)",
    },
    limitations="Retrospective association over the whole stretch, not causation. The "
    "cross-correlation band assumes white noise, so it overstates significance between "
    "autocorrelated series. Uses the longest gap-free stretch of the range (see the sample "
    "table); a range too short for the lags gives empty tables.",
)
