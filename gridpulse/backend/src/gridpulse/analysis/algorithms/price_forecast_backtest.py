"""Day-ahead imbalance price backtest: seasonal ETS and ARIMA against a seasonal naive benchmark.

The fold machinery is in ``_backtest`` (P0 #1, #9, #10, #12, #13); this module picks the models.
"""

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from gridpulse.analysis.algorithms import _backtest
from gridpulse.analysis.algorithms._backtest import WINDOW
from gridpulse.analysis.algorithms.forecasting import ARIMA, ETS, Forecaster, SeasonalNaive
from gridpulse.analysis.algorithms.series import HALF_HOUR
from gridpulse.analysis.contract import Algorithm, AlgorithmResult

MODELS: dict[str, Callable[..., Forecaster]] = {
    "seasonal_naive": SeasonalNaive,
    "ets": ETS,
    "arima": ARIMA,
}
BANDED = ("ets", "arima")  # models whose prediction intervals are stored as series
OUTPUTS = _backtest.outputs(list(MODELS), BANDED)
TABLE_COLUMNS = _backtest.TABLE_COLUMNS
__all__ = ["ALGORITHM", "MODELS", "OUTPUTS", "TABLE_COLUMNS", "WINDOW", "run"]


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
        banded=BANDED,
        horizon=horizon,
        window=window,
        max_folds=max_folds,
        season_length=season_length,
        fold_params=True,
    )


ALGORITHM = Algorithm(
    id="price_forecast_backtest",
    version="0.1.0",
    description="Day-ahead imbalance price from seasonal ETS and seasonally differenced "
    "ARIMA(2, 0, 1), against a daily seasonal naive benchmark, scored by a rolling-origin backtest.",
    run=run,
    inputs=("price_gbp_mwh",),
    outputs=OUTPUTS,
    tables=_backtest.TABLES | _backtest.FOLD_PARAMS_TABLE,
    params=_backtest.SETTINGS
    | {
        "models": {
            "seasonal_naive": {"season_length": 48},
            "ets": {"trend": None, "damped": False, "seasonal": "additive", "season_length": 48},
            "arima": {"p": 2, "d": 0, "q": 1, "seasonal_differencing": 1, "season_length": 48},
        },
    },
    param_descriptions=_backtest.SETTING_DESCRIPTIONS,
    warmup=WINDOW * HALF_HOUR,
    limitations=_backtest.LIMITATIONS
    + " The seasonal naive's intervals are scored but not stored as series.",
)
