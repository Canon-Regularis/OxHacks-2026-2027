"""Next-day imbalance price forecast with automatic model selection (P0 #2, #9, #10, #13).

AutoETS and AutoARIMA are selected and fitted on the last ``history_periods`` settled prices of
the range and forecast every period of the following settlement day (46, 48 or 50 of them),
with a daily seasonal naive for comparison. The forecast lies after the range, so a run with
``--to`` the latest settled day forecasts the next day.
"""

from collections.abc import Callable, Iterator, Mapping
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from gridpulse.analysis.algorithms import _parallel
from gridpulse.analysis.algorithms._backtest import LEVELS, WINDOW, build, outputs
from gridpulse.analysis.algorithms._checks import require_int
from gridpulse.analysis.algorithms._frames import half_hourly
from gridpulse.analysis.algorithms.errors import AlgorithmError
from gridpulse.analysis.algorithms.forecasting import (
    AutoARIMA,
    AutoETS,
    Forecaster,
    SeasonalNaive,
)
from gridpulse.analysis.algorithms.series import HALF_HOUR, LONDON
from gridpulse.analysis.contract import Algorithm, AlgorithmResult

MODELS: dict[str, Callable[..., Forecaster]] = {
    "seasonal_naive": SeasonalNaive,
    "auto_ets": AutoETS,
    "auto_arima": AutoARIMA,
}
BANDED = ("auto_ets", "auto_arima")
OUTPUTS = outputs(list(MODELS), BANDED)
DAY_COLUMNS = [
    "settlement_date", "first_utc", "last_utc", "periods", "history_first_utc", "history_last_utc",
]  # fmt: skip
SELECTED_COLUMNS = ["model", "source", "parameter", "value"]
WARNING_COLUMNS = ["model", "warning"]
TABLE_COLUMNS = {
    "forecast_day": DAY_COLUMNS,
    "selected_models": SELECTED_COLUMNS,
    "fit_warnings": WARNING_COLUMNS,
}


def forecast_day(end: datetime) -> tuple[str, int]:
    """The settlement day that ``end`` (the first period after the data) falls in, and how many
    periods remain in it from ``end``: the whole day when ``end`` is a London midnight."""
    day = end.astimezone(LONDON).date()
    following = day + timedelta(days=1)
    stop = datetime(following.year, following.month, following.day, tzinfo=LONDON)
    return day.isoformat(), (stop - end) // HALF_HOUR


def run(
    frame: pd.DataFrame,
    *,
    history_periods: int,
    models: Mapping[str, Mapping[str, Any]],
) -> AlgorithmResult:
    forecasters = build(MODELS, models)
    history_periods = require_int("history_periods", history_periods, minimum=1, maximum=WINDOW)
    end = frame.index[-1].to_pydatetime() + HALF_HOUR
    day, periods = forecast_day(end)
    targets = pd.date_range(end, periods=periods, freq="30min")
    price = frame["price_gbp_mwh"].iloc[-history_periods:]
    history = half_hourly(price)

    out = pd.DataFrame(np.nan, index=targets, columns=list(OUTPUTS))
    selected: list[list[object]] = []
    warnings: list[list[object]] = []
    fits = _parallel.forecast_all(list(forecasters.values()), history, periods, levels=LEVELS)
    for name, forecast in zip(forecasters, fits, strict=True):
        if isinstance(forecast, AlgorithmError):
            warnings.append([name, f"not forecast: {forecast}"])
            continue
        out[f"{name}_forecast"] = forecast.mean
        if name in BANDED:
            for band in forecast.intervals:
                out[f"{name}_lower_{round(band.level * 100)}"] = band.lower
                out[f"{name}_upper_{round(band.level * 100)}"] = band.upper
        selected.extend(
            [name, forecast.source, key, value] for key, value in _flat(forecast.params)
        )
        warnings.extend([name, warning] for warning in forecast.warnings)

    first, last = price.index[0], price.index[-1]
    day_row = [day, targets[0], targets[-1], periods, first, last]
    return AlgorithmResult(
        series=out,
        tables={
            "forecast_day": pd.DataFrame([day_row], columns=DAY_COLUMNS),
            "selected_models": pd.DataFrame(selected, columns=SELECTED_COLUMNS),
            "fit_warnings": pd.DataFrame(warnings, columns=WARNING_COLUMNS),
        },
    )


def _flat(params: Mapping[str, object]) -> Iterator[tuple[str, object]]:
    """Forecast.params as (parameter, value) rows: numbers stay numbers, the rest (orders,
    method labels) become text, coefficients get a row each, and unset choices are skipped."""
    for key, value in params.items():
        if key == "coefficients" and isinstance(value, Mapping):
            yield from ((str(term), float(coefficient)) for term, coefficient in value.items())
        elif value is None:
            continue
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            yield key, value
        else:
            yield key, str(value)


ALGORITHM = Algorithm(
    id="price_day_ahead_forecast",
    version="0.1.0",
    description="Next-day imbalance price forecast from AutoETS and AutoARIMA, selected on the last "
    "28 days of settled prices, with a daily seasonal naive benchmark.",
    run=run,
    inputs=("price_gbp_mwh",),
    outputs=OUTPUTS,
    tables={
        "forecast_day": "The settlement day forecast, its first and last period, how many periods "
        "it has (46, 48 or 50) and the history used",
        "selected_models": "Each model's selected specification and fitted parameters, one row per "
        "parameter",
        "fit_warnings": "Conditions flagged while fitting, and models that could not forecast",
    },
    params={
        "history_periods": WINDOW,
        "models": {
            "seasonal_naive": {"season_length": 48},
            "auto_ets": {"season_length": 48},
            "auto_arima": {"season_length": 48, "seasonal_differencing": 1, "max_p": 3, "max_q": 2},
        },
    },
    param_descriptions={
        "history_periods": f"Settled periods before the forecast day used to select and fit each "
        f"model; at most {WINDOW} (28 days)",
        "models": "Settings of each model; null seasonal_differencing would let a test choose, which "
        "on these prices drops the daily cycle",
    },
    warmup=WINDOW * HALF_HOUR,
    horizon=timedelta(hours=25),  # the longest settlement day
    limitations="A forecast of the settlement day after the range, from prices settled by its "
    "end. The model selection is not backtested here; price_forecast_backtest shows how accurate "
    "such forecasts have been. Seasonal lags count elapsed half-hours, so a forecast day with a "
    "clock change is offset by an hour after the change. Intervals are model-based, not "
    "probabilities of a price event.",
)
