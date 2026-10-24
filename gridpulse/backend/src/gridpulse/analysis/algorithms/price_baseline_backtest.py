"""Day-ahead imbalance price backtest of the naive and moving-average baselines (P0 #1-#3).

These are the benchmarks the fitted models must beat. The fold machinery is in ``_backtest``.
"""

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from gridpulse.analysis.algorithms import _backtest
from gridpulse.analysis.algorithms._backtest import WINDOW
from gridpulse.analysis.algorithms.forecasting import (
    Drift,
    Forecaster,
    Mean,
    MovingAverage,
    Naive,
    Naive2,
    SeasonalNaive,
    WeightedMovingAverage,
)
from gridpulse.analysis.algorithms.series import HALF_HOUR
from gridpulse.analysis.contract import Algorithm, AlgorithmResult

MODELS: dict[str, Callable[..., Forecaster]] = {
    "naive": Naive,
    "seasonal_naive_day": SeasonalNaive,
    "seasonal_naive_week": SeasonalNaive,
    "naive2": Naive2,
    "drift": Drift,
    "mean": Mean,
    "moving_average": MovingAverage,
    "weighted_moving_average": WeightedMovingAverage,
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
    id="price_baseline_backtest",
    version="0.1.0",
    description="Day-ahead imbalance price from naive, seasonal naive (daily and weekly), Naive2, "
    "drift, mean and moving-average baselines, scored by a rolling-origin backtest.",
    run=run,
    inputs=("price_gbp_mwh",),
    outputs=OUTPUTS,
    tables=_backtest.TABLES,
    params=_backtest.SETTINGS
    | {
        "models": {
            "naive": {},
            "seasonal_naive_day": {"season_length": 48},
            "seasonal_naive_week": {"season_length": 336},
            "naive2": {"season_length": 48},
            "drift": {},
            "mean": {},
            "moving_average": {"window": 48},
            "weighted_moving_average": {"weights": [1, 2, 3, 4]},
        },
    },
    param_descriptions=_backtest.SETTING_DESCRIPTIONS,
    warmup=WINDOW * HALF_HOUR,
    limitations="Benchmarks every other forecaster must beat; point forecasts only are stored. "
    + _backtest.LIMITATIONS,
)
