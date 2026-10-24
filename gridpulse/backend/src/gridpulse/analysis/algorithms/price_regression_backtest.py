"""Day-ahead imbalance price backtest of trend and seasonal regressions (P0 #1, #6, #11).

The fold machinery is in ``_backtest``; this module picks the models.
"""

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from gridpulse.analysis.algorithms import _backtest
from gridpulse.analysis.algorithms._backtest import WINDOW
from gridpulse.analysis.algorithms.forecasting import (
    CalendarRegression,
    Forecaster,
    FourierRegression,
    SeasonalDummyRegression,
    TrendRegression,
)
from gridpulse.analysis.algorithms.series import HALF_HOUR
from gridpulse.analysis.contract import Algorithm, AlgorithmResult

MODELS: dict[str, Callable[..., Forecaster]] = {
    "trend": TrendRegression,
    "seasonal_dummy": SeasonalDummyRegression,
    "fourier": FourierRegression,
    "calendar": CalendarRegression,
}
OUTPUTS = _backtest.outputs(list(MODELS))
TABLE_COLUMNS = {name: _backtest.TABLE_COLUMNS[name] for name in _backtest.TABLES}


def run(
    frame: pd.DataFrame,
    *,
    horizon: int,
    window: int,
    max_folds: int,
    season_length: int,
    models: Mapping[str, Mapping[str, Any]],
) -> AlgorithmResult:
    return _backtest.run_backtest(
        frame,
        _backtest.build(MODELS, models),
        banded=(),
        horizon=horizon,
        window=window,
        max_folds=max_folds,
        season_length=season_length,
    )


ALGORITHM = Algorithm(
    id="price_regression_backtest",
    version="0.1.0",
    description="Day-ahead imbalance price from linear trend, half-hourly seasonal dummies, daily "
    "Fourier terms and GB-clock calendar (daily and weekly) regressions, scored by a "
    "rolling-origin backtest.",
    run=run,
    inputs=("price_gbp_mwh",),
    outputs=OUTPUTS,
    tables=_backtest.TABLES,
    params=_backtest.SETTINGS
    | {
        "models": {
            "trend": {"degree": 1},
            "seasonal_dummy": {"season_length": 48, "degree": 1},
            "fourier": {"season_length": 48, "harmonics": 4, "degree": 1},
            "calendar": {"daily_harmonics": 4, "weekly_harmonics": 2, "degree": 1},
        },
    },
    param_descriptions=_backtest.SETTING_DESCRIPTIONS,
    warmup=WINDOW * HALF_HOUR,
    limitations="Only calendar follows the Europe/London clock, so it alone is exact on 46- and "
    "50-period days; the others count elapsed half-hours. Point forecasts only are stored. "
    + _backtest.LIMITATIONS,
)
