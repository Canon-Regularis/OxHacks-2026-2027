"""Day-ahead imbalance price backtest: seasonal ETS and ARIMA against a seasonal naive benchmark.

Each fold refits every model on the ``window`` settled prices before its origin and forecasts
the next ``horizon`` periods (P0 #1, #9, #10, #12, #13). The first origin is the start of the
requested range and origins advance by ``horizon``, so every target period gets one forecast per
model, made only from prices before its origin. Folds go through ``evaluation.backtest`` one at
a time, so a missing price excludes just the folds whose training window or targets contain it
instead of the whole run.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from statistics import fmean
from typing import Any

import numpy as np
import pandas as pd

from gridpulse.analysis.algorithms import evaluation
from gridpulse.analysis.algorithms._checks import require_int
from gridpulse.analysis.algorithms.errors import (
    AlgorithmError,
    InvalidInputError,
    MissingDataError,
)
from gridpulse.analysis.algorithms.forecasting import (
    ARIMA,
    ETS,
    Forecast,
    Forecaster,
    SeasonalNaive,
)
from gridpulse.analysis.algorithms.series import HALF_HOUR, HalfHourlySeries
from gridpulse.analysis.contract import Algorithm, AlgorithmResult

WINDOW = 1344  # 28 days of half-hours: the longest training window and the history before the range
LEVELS = (0.8, 0.95)  # fixed, because the output names carry them
MODELS: dict[str, Callable[..., Forecaster]] = {
    "seasonal_naive": SeasonalNaive,
    "ets": ETS,
    "arima": ARIMA,
}
BANDED = ("ets", "arima")  # models whose prediction intervals are stored as series
GBP = "GBP/MWh"


def _pct(level: float) -> str:
    return f"{round(level * 100)}"


OUTPUTS = {f"{name}_forecast": GBP for name in MODELS} | {
    f"{name}_{edge}_{_pct(level)}": GBP
    for name in BANDED
    for level in LEVELS
    for edge in ("lower", "upper")
}
METRIC_COLUMNS = [
    "model", "interval_method", "n_ok_folds", "n_excluded_folds",
    "mae_gbp_mwh", "rmse_gbp_mwh", "wmape", "smape", "mase", "rmsse",
    *(f"{kind}_{_pct(level)}" for level in LEVELS for kind in ("coverage", "interval_score", "msis")),
    "pinball_q10_gbp_mwh", "pinball_q90_gbp_mwh",
]  # fmt: skip
HORIZON_COLUMNS = [
    "model", "horizon_step", "lead_hours", "mae_gbp_mwh", "rmse_gbp_mwh",
    *(f"coverage_{_pct(level)}" for level in LEVELS),
]  # fmt: skip
FOLD_COLUMNS = [
    "model", "origin_utc", "train_start_utc", "train_size", "status", "message", "mae_gbp_mwh",
]  # fmt: skip
PARAM_COLUMNS = ["model", "origin_utc", "parameter", "value"]


@dataclass(frozen=True)
class _Fold:
    """One origin of one model. ``forecast`` and ``scores`` are set only when status is "ok"."""

    origin: datetime
    train_start: datetime
    train_size: int
    status: str  # evaluation.FoldStatus, or "missing_data"
    message: str | None
    forecast: Forecast | None
    actual: tuple[float, ...]
    scores: Mapping[str, float]


def run(
    frame: pd.DataFrame,
    *,
    horizon: int,
    window: int,
    max_folds: int,
    season_length: int,
    models: Mapping[str, Mapping[str, Any]],
) -> AlgorithmResult:
    horizon = require_int("horizon", horizon, minimum=1)
    window = require_int("window", window, minimum=1, maximum=WINDOW)
    max_folds = require_int("max_folds", max_folds, minimum=1)
    if set(models) != set(MODELS):
        raise InvalidInputError(
            f"models must configure exactly {sorted(MODELS)}, got {sorted(models)}"
        )
    forecasters = {name: MODELS[name](**models[name]) for name in MODELS}

    series = _price_series(frame["price_gbp_mwh"])
    origins = range(WINDOW, len(series) - horizon + 1, horizon)
    if len(origins) > max_folds:
        days = max_folds * horizon * HALF_HOUR.total_seconds() / 86400
        raise InvalidInputError(
            f"{len(origins)} folds exceed max_folds={max_folds}; shorten the range to about "
            f"{days:g} days or raise max_folds"
        )
    folds = {
        name: [
            _fold(forecaster, series, origin, window, horizon, season_length) for origin in origins
        ]
        for name, forecaster in forecasters.items()
    }

    return AlgorithmResult(
        series=_forecast_series(folds, frame.index[WINDOW:]),
        tables={
            "metrics": pd.DataFrame(
                [_metrics(name, model_folds) for name, model_folds in folds.items()],
                columns=METRIC_COLUMNS,
            ),
            "by_horizon": pd.DataFrame(
                [
                    row
                    for name, model_folds in folds.items()
                    for row in _by_horizon(name, model_folds, horizon)
                ],
                columns=HORIZON_COLUMNS,
            ),
            "folds": pd.DataFrame(
                [
                    _fold_row(name, fold)
                    for name, model_folds in folds.items()
                    for fold in model_folds
                ],
                columns=FOLD_COLUMNS,
            ),
            "fold_params": pd.DataFrame(
                [
                    [name, fold.origin, parameter, value]
                    for name, model_folds in folds.items()
                    for fold, forecast in _ok(model_folds)
                    for parameter, value in _fitted(forecast.params)
                ],
                columns=PARAM_COLUMNS,
            ),
        },
    )


def _price_series(price: pd.Series) -> HalfHourlySeries:
    values = [None if np.isnan(value) else float(value) for value in price.to_numpy(dtype=float)]
    return HalfHourlySeries(price.index[0].to_pydatetime(), values)


def _fold(
    forecaster: Forecaster,
    series: HalfHourlySeries,
    origin: int,
    window: int,
    horizon: int,
    season_length: int,
) -> _Fold:
    span = series.window(origin - window, origin + horizon)
    try:
        report = evaluation.backtest(
            forecaster,
            span,
            initial=window,
            horizon=horizon,
            max_folds=1,
            season_length=season_length,
            levels=LEVELS,
        )
    except MissingDataError:
        missing = sum(value is None for value in span.values)
        gap = f"{missing} of its {len(span)} training and target periods are missing"
        return _Fold(
            series.timestamp(origin), span.start, window, "missing_data", gap, None, (), {}
        )
    fold = report.folds[0]
    message = fold.message
    if fold.forecast is not None and fold.forecast.warnings:
        message = "; ".join(fold.forecast.warnings)
    return _Fold(
        fold.origin,
        fold.train_start,
        fold.train_size,
        fold.status,
        message,
        fold.forecast,
        fold.actual,
        fold.scores,
    )


def _ok(folds: Sequence[_Fold]) -> list[tuple[_Fold, Forecast]]:
    return [(fold, fold.forecast) for fold in folds if fold.forecast is not None]


def _band(forecast: Forecast, level: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    band = next(band for band in forecast.intervals if band.level == level)
    return band.lower, band.upper


def _forecast_series(
    folds: Mapping[str, Sequence[_Fold]], targets: pd.DatetimeIndex
) -> pd.DataFrame:
    """One row per target period; NaN where a fold was excluded."""
    out = pd.DataFrame(np.nan, index=targets, columns=list(OUTPUTS))
    for name, model_folds in folds.items():
        for _, forecast in _ok(model_folds):
            index = pd.DatetimeIndex(forecast.timestamps())
            out.loc[index, f"{name}_forecast"] = forecast.mean
            if name in BANDED:
                for level in LEVELS:
                    lower, upper = _band(forecast, level)
                    out.loc[index, f"{name}_lower_{_pct(level)}"] = lower
                    out.loc[index, f"{name}_upper_{_pct(level)}"] = upper
    return out


def _metric(
    metric: Callable[..., float], actual: Sequence[float], *others: Sequence[float]
) -> float:
    """NaN instead of an error when there is nothing to score or the score is undefined."""
    if not actual:
        return float("nan")
    try:
        return metric(actual, *others)
    except AlgorithmError:
        return float("nan")


def _mean(values: Sequence[float]) -> float:
    return fmean(values) if values else float("nan")


def _metrics(name: str, folds: Sequence[_Fold]) -> dict[str, object]:
    """Pooled like evaluation.backtest: MAE, RMSE, WMAPE, sMAPE, coverage, interval and pinball
    scores over every ok forecast; MASE, RMSSE and MSIS averaged over folds."""
    ok = _ok(folds)
    actual = [value for fold, _ in ok for value in fold.actual]
    predicted = [value for _, forecast in ok for value in forecast.mean]
    row: dict[str, object] = {
        "model": name,
        "interval_method": ok[0][1].interval_method if ok else None,
        "n_ok_folds": len(ok),
        "n_excluded_folds": len(folds) - len(ok),
        "mae_gbp_mwh": _metric(evaluation.mae, actual, predicted),
        "rmse_gbp_mwh": _metric(evaluation.rmse, actual, predicted),
        "wmape": _metric(evaluation.wmape, actual, predicted),
        "smape": _metric(evaluation.smape, actual, predicted),
        "mase": _mean([fold.scores["mase"] for fold, _ in ok]),
        "rmsse": _mean([fold.scores["rmsse"] for fold, _ in ok]),
    }
    for level in LEVELS:
        lower = [value for _, forecast in ok for value in _band(forecast, level)[0]]
        upper = [value for _, forecast in ok for value in _band(forecast, level)[1]]
        pct = _pct(level)
        row[f"coverage_{pct}"] = _metric(evaluation.empirical_coverage, actual, lower, upper)
        score = partial(evaluation.interval_score, level=level)
        row[f"interval_score_{pct}"] = _metric(score, actual, lower, upper)
        row[f"msis_{pct}"] = _mean([fold.scores[f"msis_{level:g}"] for fold, _ in ok])
        if level == 0.8:  # its edges are the 10% and 90% quantile forecasts
            low = partial(evaluation.pinball_loss, quantile=0.1)
            high = partial(evaluation.pinball_loss, quantile=0.9)
            row["pinball_q10_gbp_mwh"] = _metric(low, actual, lower)
            row["pinball_q90_gbp_mwh"] = _metric(high, actual, upper)
    return row


def _by_horizon(name: str, folds: Sequence[_Fold], horizon: int) -> Iterator[dict[str, object]]:
    ok = _ok(folds)
    for step in range(horizon):
        actual = [fold.actual[step] for fold, _ in ok]
        predicted = [forecast.mean[step] for _, forecast in ok]
        row: dict[str, object] = {
            "model": name,
            "horizon_step": step + 1,
            "lead_hours": (step + 1) / 2,  # to the end of the target period
            "mae_gbp_mwh": _metric(evaluation.mae, actual, predicted),
            "rmse_gbp_mwh": _metric(evaluation.rmse, actual, predicted),
        }
        for level in LEVELS:
            lower = [_band(forecast, level)[0][step] for _, forecast in ok]
            upper = [_band(forecast, level)[1][step] for _, forecast in ok]
            row[f"coverage_{_pct(level)}"] = _metric(
                evaluation.empirical_coverage, actual, lower, upper
            )
        yield row


def _fold_row(name: str, fold: _Fold) -> list[object]:
    mae = fold.scores.get("mae", float("nan"))
    return [name, fold.origin, fold.train_start, fold.train_size, fold.status, fold.message, mae]


def _fitted(params: Mapping[str, object]) -> Iterator[tuple[str, float]]:
    """Fitted numbers from Forecast.params: ETS smoothing weights, ARIMA coefficients, sigma2
    and AICc. Configuration (orders, method labels, season lengths) is left out."""
    for name, value in params.items():
        if name == "coefficients" and isinstance(value, Mapping):
            yield from ((str(term), float(coefficient)) for term, coefficient in value.items())
        elif isinstance(value, float):
            yield name, value


ALGORITHM = Algorithm(
    id="price_forecast_backtest",
    version="0.1.0",
    description="Day-ahead imbalance price from seasonal ETS and seasonally differenced ARIMA(2, 0, 1), "
    "against a daily seasonal naive benchmark, scored by a rolling-origin backtest.",
    run=run,
    inputs=("price_gbp_mwh",),
    outputs=OUTPUTS,
    tables={
        "metrics": "Scores per model, pooled over the ok folds; MASE, RMSSE and MSIS averaged per fold",
        "by_horizon": "MAE, RMSE and 80%/95% coverage per model and lead step",
        "folds": "Every forecast origin per model with its status; excluded folds carry a message",
        "fold_params": "Fitted parameters per model and origin (ETS smoothing weights, ARIMA "
        "coefficients, sigma2 and AICc)",
    },
    params={
        "horizon": 48,
        "window": WINDOW,
        "max_folds": 60,
        "season_length": 48,
        "models": {
            "seasonal_naive": {"season_length": 48},
            "ets": {"trend": None, "damped": False, "seasonal": "additive", "season_length": 48},
            "arima": {"p": 2, "d": 0, "q": 1, "seasonal_differencing": 1, "season_length": 48},
        },
    },
    param_descriptions={
        "horizon": "Periods forecast from each origin, which is also the step between origins "
        "(48 = one day ahead, once a day)",
        "window": f"Settled periods each fit uses, ending at the origin; at most {WINDOW} (28 days)",
        "max_folds": "Most origins a run may use; a longer range fails instead of running for hours "
        "(60 = about 60 days at horizon 48)",
        "season_length": "Seasonal period in half-hours used to scale MASE, RMSSE and MSIS (48 = daily)",
        "models": "Settings of each model; the names match the output prefixes and the model column "
        "of the tables",
    },
    warmup=WINDOW * HALF_HOUR,
    limitations="Out-of-sample: each forecast uses only prices settled before its origin, and every "
    "model is refitted at every origin. Origins step by elapsed half-hours, so they shift by an "
    "hour of local time across a clock change. Intervals are model-based (the seasonal naive's "
    "are not stored); WMAPE and sMAPE are fractions. A fold with a missing price in its window or "
    "targets is excluded, not imputed. Not a probability of a price spike or a trading signal.",
)
