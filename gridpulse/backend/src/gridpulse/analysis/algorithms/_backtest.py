"""Rolling-origin backtests shared by the backtest algorithms (P0 #1, #12, #13).

Each fold refits a model on the ``window`` settled prices before its origin and forecasts the
next ``horizon`` periods. The first origin is the start of the requested range (``WINDOW``
periods into the frame) and origins advance by ``horizon``, so every target period gets one
forecast per model, made only from prices before its origin. Folds go through
``evaluation.backtest_folds`` one at a time, so a missing price excludes just the folds whose
training window or targets contain it (status "missing_data") instead of the whole run. Every
(model, origin) fit is independent, so they run in ``_parallel``'s process pool. A fold is a
function of its model, its prices and the fold settings alone, so ``memo`` keeps them, as JSON,
under a hash of exactly those: a rerun a day later reuses every fold but the newest.
"""

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime
from functools import partial
from statistics import fmean
from typing import Any

import numpy as np
import pandas as pd

from gridpulse.analysis.algorithms import _parallel, evaluation
from gridpulse.analysis.algorithms._checks import require_int
from gridpulse.analysis.algorithms._frames import half_hourly
from gridpulse.analysis.algorithms.errors import (
    AlgorithmError,
    InvalidInputError,
    MissingDataError,
)
from gridpulse.analysis.algorithms.forecasting import Forecast, Forecaster, PredictionInterval
from gridpulse.analysis.algorithms.series import HALF_HOUR, HalfHourlySeries
from gridpulse.analysis.contract import AlgorithmResult

WINDOW = 1344  # 28 days of half-hours: the longest training window and the history before the range
LEVELS = (0.8, 0.95)  # fixed, because the output names carry them
GBP = "GBP/MWh"

SETTINGS = {"horizon": 48, "window": WINDOW, "max_folds": 60, "season_length": 48}
SETTING_DESCRIPTIONS = {
    "horizon": "Periods forecast from each origin, which is also the step between origins "
    "(48 = one day ahead, once a day)",
    "window": f"Settled periods each fit uses, ending at the origin; at most {WINDOW} (28 days)",
    "max_folds": "Most origins a run may use; a longer range fails instead of running for hours "
    "(60 = about 60 days at horizon 48)",
    "season_length": "Seasonal period in half-hours used to scale MASE, RMSSE and MSIS "
    "(48 = daily)",
    "models": "Settings of each model; the names match the output prefixes and the model column "
    "of the tables",
}
TABLES = {
    "metrics": "Scores per model, pooled over the ok folds; MASE, RMSSE and MSIS averaged per fold",
    "by_horizon": "MAE, RMSE and 80%/95% coverage per model and lead step",
    "folds": "Every forecast origin per model with its status; excluded folds carry a message",
}
FOLD_PARAMS_TABLE = {
    "fold_params": "Fitted parameters per model and origin (ETS smoothing weights, ARIMA "
    "coefficients, sigma2 and AICc)",
}
LIMITATIONS = (
    "Out-of-sample: each forecast uses only prices settled before its origin, and every model is "
    "refitted at every origin. Origins step by elapsed half-hours, so they shift by an hour of "
    "local time across a clock change. Intervals are model-based; WMAPE and sMAPE are fractions. "
    "A fold with a missing price in its window or targets is excluded, not imputed. Not a "
    "probability of a price spike or a trading signal."
)


def _pct(level: float) -> str:
    return f"{round(level * 100)}"


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
TABLE_COLUMNS = {
    "metrics": METRIC_COLUMNS,
    "by_horizon": HORIZON_COLUMNS,
    "folds": FOLD_COLUMNS,
    "fold_params": PARAM_COLUMNS,
}


def outputs(models: Sequence[str], banded: Sequence[str] = ()) -> dict[str, str]:
    """``<model>_forecast`` for every model, plus interval edges for the ``banded`` ones."""
    return {f"{name}_forecast": GBP for name in models} | {
        f"{name}_{edge}_{_pct(level)}": GBP
        for name in banded
        for level in LEVELS
        for edge in ("lower", "upper")
    }


def build(
    catalogue: Mapping[str, Callable[..., Forecaster]], models: Mapping[str, Mapping[str, Any]]
) -> dict[str, Forecaster]:
    """One forecaster per catalogue name, from the ``models`` param."""
    if set(models) != set(catalogue):
        raise InvalidInputError(
            f"models must configure exactly {sorted(catalogue)}, got {sorted(models)}"
        )
    return {name: make(**models[name]) for name, make in catalogue.items()}


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


def run_backtest(
    frame: pd.DataFrame,
    forecasters: Mapping[str, Forecaster],
    *,
    banded: Sequence[str],
    horizon: int,
    window: int,
    max_folds: int,
    season_length: int,
    fold_params: bool = False,
    memo: MutableMapping[str, str] | None = None,
) -> AlgorithmResult:
    """Backtest every forecaster on ``frame["price_gbp_mwh"]``: forecast series for the range
    after the first ``WINDOW`` rows, and the metrics, by_horizon, folds (and fold_params) tables.

    Folds found in ``memo`` are reused and the others are fitted and added to it."""
    horizon = require_int("horizon", horizon, minimum=1)
    window = require_int("window", window, minimum=1, maximum=WINDOW)
    max_folds = require_int("max_folds", max_folds, minimum=1)
    series = half_hourly(frame["price_gbp_mwh"])
    origins = range(WINDOW, len(series) - horizon + 1, horizon)
    if len(origins) > max_folds:
        days = max_folds * horizon * HALF_HOUR.total_seconds() / 86400
        raise InvalidInputError(
            f"{len(origins)} folds exceed max_folds={max_folds}; shorten the range to about "
            f"{days:g} days or raise max_folds"
        )
    spans = [series.window(origin - window, origin + horizon) for origin in origins]
    settings = (window, horizon, season_length)
    tasks = [(f, span) for f in forecasters.values() for span in spans]
    keys = [_fold_key(forecaster, span, *settings) for forecaster, span in tasks]
    memo = {} if memo is None else memo
    known = {key: fold for key in keys if key in memo and (fold := _decoded(memo[key])) is not None}
    missing = [i for i, key in enumerate(keys) if key not in known]
    fit = partial(_fold, window=window, horizon=horizon, season_length=season_length)
    for i, fold in zip(missing, _parallel.run_all(fit, [tasks[i] for i in missing]), strict=True):
        known[keys[i]] = fold
        with suppress(TypeError):  # a param JSON cannot hold: refit it next time instead
            memo[keys[i]] = _encoded(fold)
    ordered = iter(known[key] for key in keys)
    folds = {name: [next(ordered) for _ in spans] for name in forecasters}
    tables = {
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
            [_fold_row(name, fold) for name, model_folds in folds.items() for fold in model_folds],
            columns=FOLD_COLUMNS,
        ),
    }
    if fold_params:
        tables["fold_params"] = pd.DataFrame(
            [
                [name, fold.origin, parameter, value]
                for name, model_folds in folds.items()
                for fold, forecast in _ok(model_folds)
                for parameter, value in _fitted(forecast.params)
            ],
            columns=PARAM_COLUMNS,
        )
    columns = outputs(list(forecasters), banded)
    return AlgorithmResult(
        series=_forecast_series(folds, frame.index[WINDOW:], columns), tables=tables
    )


def _fold(
    task: tuple[Forecaster, HalfHourlySeries], *, window: int, horizon: int, season_length: int
) -> _Fold:
    """One model at one origin; ``task`` is the model and its training window plus targets."""
    forecaster, span = task
    try:
        (fold,) = evaluation.backtest_folds(
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
        return _Fold(span.timestamp(window), span.start, window, "missing_data", gap, None, (), {})
    message = fold.message
    forecast = fold.forecast
    if forecast is not None:
        if forecast.warnings:
            message = "; ".join(forecast.warnings)
        forecast = replace(forecast, residuals=())  # unused here, and the bulk of a fold to pickle
    return _Fold(
        fold.origin,
        fold.train_start,
        fold.train_size,
        fold.status,
        message,
        forecast,
        fold.actual,
        fold.scores,
    )


# Bump when _Fold, Forecast or what a fold computes changes shape, so old entries are not reused.
_FOLD_FORMAT = 1


def _fold_key(
    forecaster: Forecaster, span: HalfHourlySeries, window: int, horizon: int, season_length: int
) -> str:
    """A hash of everything a fold depends on: the model and its settings (a frozen dataclass's
    repr), its prices and their start, the fold settings and the interval levels."""
    inputs = [
        _FOLD_FORMAT, repr(forecaster), span.start.isoformat(), list(span.values),
        window, horizon, season_length, list(LEVELS),
    ]  # fmt: skip
    return hashlib.sha256(json.dumps(inputs).encode()).hexdigest()


def _encoded(fold: _Fold) -> str:
    """The fold as JSON; floats keep their exact value (``repr`` round-trips)."""
    forecast = fold.forecast
    return json.dumps(
        {
            "origin": fold.origin.isoformat(),
            "train_start": fold.train_start.isoformat(),
            "train_size": fold.train_size,
            "status": fold.status,
            "message": fold.message,
            "actual": fold.actual,
            "scores": fold.scores,
            "forecast": None
            if forecast is None
            else {
                "source": forecast.source,
                "params": _tagged(forecast.params),
                "origin": forecast.origin.isoformat(),
                "mean": forecast.mean,
                "intervals": [[i.level, i.lower, i.upper] for i in forecast.intervals],
                "interval_method": forecast.interval_method,
                "warnings": forecast.warnings,
            },
        }
    )


def _decoded(text: str) -> _Fold | None:
    """The fold ``_encoded`` wrote, or None for anything else, which is then refitted."""
    try:
        data = json.loads(text)
        raw = data["forecast"]
        forecast = None
        if raw is not None:
            forecast = Forecast(
                source=raw["source"],
                params=_untagged(raw["params"]),
                origin=datetime.fromisoformat(raw["origin"]),
                mean=tuple(raw["mean"]),
                intervals=tuple(
                    PredictionInterval(level, tuple(lower), tuple(upper))
                    for level, lower, upper in raw["intervals"]
                ),
                interval_method=raw["interval_method"],
                residuals=(),
                warnings=tuple(raw["warnings"]),
            )
        return _Fold(
            datetime.fromisoformat(data["origin"]),
            datetime.fromisoformat(data["train_start"]),
            data["train_size"],
            data["status"],
            data["message"],
            forecast,
            tuple(data["actual"]),
            data["scores"],
        )
    except (ValueError, KeyError, TypeError):
        return None


def _tagged(value: object) -> object:
    """Forecast.params in JSON, with tuples and mappings tagged so they come back as such."""
    if isinstance(value, tuple):
        return {"tuple": [_tagged(item) for item in value]}
    if isinstance(value, Mapping):
        return {"mapping": {str(key): _tagged(item) for key, item in value.items()}}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"cannot store {type(value).__name__} in a fold")


def _untagged(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"tuple"}:
            return tuple(_untagged(item) for item in value["tuple"])
        if set(value) == {"mapping"}:
            return {key: _untagged(item) for key, item in value["mapping"].items()}
        raise ValueError(f"unknown tagged value {sorted(value)}")
    return value


def _ok(folds: Sequence[_Fold]) -> list[tuple[_Fold, Forecast]]:
    return [(fold, fold.forecast) for fold in folds if fold.forecast is not None]


def _band(forecast: Forecast, level: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    band = next(band for band in forecast.intervals if band.level == level)
    return band.lower, band.upper


def _forecast_series(
    folds: Mapping[str, Sequence[_Fold]], targets: pd.DatetimeIndex, columns: Mapping[str, str]
) -> pd.DataFrame:
    """One row per target period; NaN where a fold was excluded."""
    out = pd.DataFrame(np.nan, index=targets, columns=list(columns))
    for name, model_folds in folds.items():
        for _, forecast in _ok(model_folds):
            index = pd.DatetimeIndex(forecast.timestamps())
            out.loc[index, f"{name}_forecast"] = forecast.mean
            if f"{name}_lower_{_pct(LEVELS[0])}" in columns:
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
    and AICc; AR coefficients as ``ar.L<lag>``; structural variances as ``variance.<state>``.
    Configuration (orders, method labels, season lengths) is left out."""
    for name, value in params.items():
        if name == "coefficients" and isinstance(value, Mapping):
            yield from ((str(term), float(coefficient)) for term, coefficient in value.items())
        elif name == "coefficients" and isinstance(value, Sequence):
            yield from ((f"ar.L{lag}", float(c)) for lag, c in enumerate(value, start=1))
        elif name == "variances" and isinstance(value, Mapping):
            yield from ((f"variance.{state}", float(v)) for state, v in value.items())
        elif isinstance(value, float):
            yield name, value
