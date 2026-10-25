"""Backtests, accuracy and interval metrics and Diebold-Mariano tests backed by Orchid.

Families P0 #1, #12, #13 and P1 #25.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from statistics import fmean, pvariance
from typing import Literal

from gridpulse.analysis.algorithms._checks import (
    finite_values,
    orchid_errors,
    require_fraction,
    require_int,
    require_levels,
)
from gridpulse.analysis.algorithms.errors import (
    AlgorithmError,
    CalculationError,
    DegenerateDataError,
    InsufficientDataError,
    InvalidInputError,
    MissingDataError,
)
from gridpulse.analysis.algorithms.forecasting import (
    DEFAULT_LEVELS,
    Forecast,
    Forecaster,
    PredictionInterval,
)
from gridpulse.analysis.algorithms.orchid import autoregressive as ar
from gridpulse.analysis.algorithms.series import HalfHourlySeries

_MIN_DM_ERRORS = 30
# Orchid clamps smaller scales and denominators to this, silently distorting the result.
_ORCHID_EPS = 1e-12

FoldStatus = Literal["ok", "insufficient_data", "degenerate_data", "failed"]
Loss = Literal["squared", "absolute"]


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    return _call(ar.mae, *_arrays(actual=actual, predicted=predicted))


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    return _call(ar.rmse, *_arrays(actual=actual, predicted=predicted))


def wmape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Weighted MAPE as a fraction: sum |actual - predicted| / sum |actual|."""
    a, p = _arrays(actual=actual, predicted=predicted)
    total = sum(abs(value) for value in a)
    if not math.isfinite(total) and a != p:  # Orchid would divide by inf and return 0.0
        raise CalculationError("the sum of the actual values overflows a float")
    if total < _ORCHID_EPS:
        raise DegenerateDataError("wmape is undefined when every actual value is zero")
    return _call(ar.wmape, a, p, as_percent=False)


def smape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Symmetric MAPE as a fraction from 0 to 2; pairs that are both zero are skipped."""
    a, p = _arrays(actual=actual, predicted=predicted)
    sizes = [abs(x) + abs(y) for x, y in zip(a, p, strict=True)]
    if any(math.isinf(size) and x != y for size, x, y in zip(sizes, a, p, strict=True)):
        raise CalculationError("an actual and forecast pair overflows a float")  # Orchid: 0
    if all(size <= _ORCHID_EPS for size in sizes):
        raise DegenerateDataError("smape is undefined when every actual and forecast is zero")
    return _call(ar.smape, a, p, as_percent=False)


def mase(
    actual: Sequence[float],
    predicted: Sequence[float],
    *,
    training: Sequence[float],
    season_length: int,
) -> float:
    """MAE divided by the in-sample MAE of the seasonal naive forecast of ``training``."""
    m = require_int("season_length", season_length, minimum=1)
    a, p = _arrays(actual=actual, predicted=predicted)
    return _call(ar.mase, a, p, y_train=_training(training, m, squared=False), seasonality=m)


def rmsse(
    actual: Sequence[float],
    predicted: Sequence[float],
    *,
    training: Sequence[float],
    season_length: int,
) -> float:
    """RMSE divided by the in-sample RMSE of the seasonal naive forecast of ``training``."""
    m = require_int("season_length", season_length, minimum=1)
    a, p = _arrays(actual=actual, predicted=predicted)
    return _call(ar.rmsse, a, p, y_train=_training(training, m, squared=True), seasonality=m)


def pinball_loss(
    actual: Sequence[float], quantile_forecast: Sequence[float], *, quantile: float
) -> float:
    """Mean pinball loss of forecasts of the ``quantile`` quantile; at 0.5 it is half the MAE."""
    q = require_fraction("quantile", quantile)
    arrays = _arrays(actual=actual, quantile_forecast=quantile_forecast)
    return _call(ar.pinball_loss, *arrays, quantile=q)


def interval_score(
    actual: Sequence[float], lower: Sequence[float], upper: Sequence[float], *, level: float
) -> float:
    """Mean interval width plus 2 / (1 - level) times the distance of each miss."""
    checked = require_fraction("level", level)
    return _call(ar.interval_score, *_intervals(actual, lower, upper), level=checked)


def empirical_coverage(
    actual: Sequence[float], lower: Sequence[float], upper: Sequence[float]
) -> float:
    """Fraction of actual values within their closed intervals."""
    return _call(ar.empirical_coverage, *_intervals(actual, lower, upper))


def msis(
    actual: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    training: Sequence[float],
    season_length: int,
    level: float,
) -> float:
    """``interval_score`` divided by the same seasonal naive MAE as ``mase``."""
    checked = require_fraction("level", level)
    m = require_int("season_length", season_length, minimum=1)
    a, low, high = _intervals(actual, lower, upper)
    y_train = _training(training, m, squared=False)
    return _call(ar.msis, a, low, high, y_train=y_train, seasonality=m, level=checked)


@dataclass(frozen=True)
class DieboldMariano:
    """A Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.

    A positive statistic means ``errors_a`` have the larger mean loss, so the forecast behind
    ``errors_b`` was the more accurate. The two-sided p-value comes from Student's t with
    n - 1 degrees of freedom; below about 1e-16 it is reported as 0.0, because Orchid computes
    1 - CDF. ``horizon`` is the one used, and ``warnings`` say why when it is not the one
    requested.
    """

    statistic: float
    p_value: float
    horizon: int
    loss: Loss
    warnings: tuple[str, ...]


def diebold_mariano(
    errors_a: Sequence[float],
    errors_b: Sequence[float],
    *,
    horizon: int = 1,
    loss: Loss = "squared",
) -> DieboldMariano:
    """Test whether two forecasts of the same targets have equal expected loss.

    The errors, at least 30 of each, must be aligned target by target and in time order;
    ``horizon`` is at most n // 2. Errors of ``horizon``-step forecasts are autocorrelated to
    lag ``horizon - 1``, so the variance of the mean loss differential sums its
    autocovariances to that lag. Where that sum is not positive (at most 1e-12 of the
    variance), horizon 1 is used, as in R's ``dm.test``. For backtest errors at lead ``L`` (1
    for each fold's first forecast period) from origins ``step`` apart, pass
    ``horizon=ceil(L / step)``: errors from origins fewer than ``L`` periods apart share
    shocks. ``DegenerateDataError`` means the loss differential is constant to within 1e-12
    of the largest loss.
    """
    if loss not in ("squared", "absolute"):
        raise InvalidInputError(f"loss must be 'squared' or 'absolute', got {loss!r}")
    if len(errors_a) == len(errors_b) < _MIN_DM_ERRORS:  # before _arrays, which needs only 1
        raise InsufficientDataError(_MIN_DM_ERRORS, len(errors_a), what="forecast errors")
    a, b = _arrays(errors_a=errors_a, errors_b=errors_b)
    require_int("horizon", horizon, minimum=1, maximum=len(a) // 2)
    power = 2 if loss == "squared" else 1
    a, b = _scaled_errors(a, b, power)
    warnings: tuple[str, ...] = ()
    # With the differential at unit variance, Orchid's 1e-12 floor is 1e-12 of the variance.
    if _long_run_variance(_differential(a, b, power), horizon) <= _ORCHID_EPS:
        warnings = (f"the long-run variance at horizon {horizon} is not positive; used horizon 1",)
        horizon = 1
    with orchid_errors("diebold_mariano"):
        result = ar.diebold_mariano(a, b, horizon=horizon, loss=loss)
    if not all(map(math.isfinite, (result.statistic, result.pvalue))):
        raise CalculationError("Orchid diebold_mariano returned non-finite values")
    return DieboldMariano(result.statistic, result.pvalue, horizon, loss, warnings)


@dataclass(frozen=True)
class FoldResult:
    """One forecast origin. ``forecast`` and ``scores`` are set only for "ok" folds, and
    ``message`` otherwise."""

    origin: datetime
    train_start: datetime
    train_size: int
    status: FoldStatus
    message: str | None
    forecast: Forecast | None
    actual: tuple[float, ...]
    scores: Mapping[str, float]


@dataclass(frozen=True)
class BacktestReport:
    """Scores of the "ok" folds; empty mappings when there are none.

    ``n_excluded`` counts the folds with any other status. ``metrics`` pools every ok
    forecast for MAE, RMSE, coverage and interval score, and averages the per-fold MASE,
    RMSSE and MSIS, each scaled by its own training window. ``by_horizon`` scores each lead
    step across the ok folds.
    """

    folds: tuple[FoldResult, ...]
    n_ok: int
    n_excluded: int
    metrics: Mapping[str, float]
    by_horizon: tuple[Mapping[str, float], ...]


def backtest(
    forecaster: Forecaster,
    series: HalfHourlySeries,
    *,
    initial: int,
    horizon: int,
    step: int | None = None,
    window: int | None = None,
    max_folds: int = 50,
    season_length: int = 48,
    levels: Sequence[float] = DEFAULT_LEVELS,
) -> BacktestReport:
    """``backtest_folds`` with the folds scored together; see ``BacktestReport``."""
    folds = backtest_folds(
        forecaster,
        series,
        initial=initial,
        horizon=horizon,
        step=step,
        window=window,
        max_folds=max_folds,
        season_length=season_length,
        levels=levels,
    )
    return _report(list(folds), horizon, require_levels(levels))


def backtest_folds(
    forecaster: Forecaster,
    series: HalfHourlySeries,
    *,
    initial: int,
    horizon: int,
    step: int | None = None,
    window: int | None = None,
    max_folds: int = 50,
    season_length: int = 48,
    levels: Sequence[float] = DEFAULT_LEVELS,
) -> tuple[FoldResult, ...]:
    """Refit ``forecaster`` at successive origins and score each forecast.

    Origins start at index ``initial`` and advance by ``step`` (default ``horizon``). Each
    fold trains on every earlier value, or on the last ``window`` (at most ``initial``) of
    them. A fold keeps a non-"ok" status, and is left out of the scores, when its training
    window cannot scale MASE, RMSSE and MSIS, the forecaster raises ``InvalidInputError`` or
    ``CalculationError``, or the forecast is malformed; other exceptions propagate. ``step``
    and ``season_length`` count half-hours. The last ``(n - initial - horizon) % step``
    values are never tested: to end at the latest data with ``k`` folds, use
    ``initial = n - horizon - (k - 1) * step``. ``backtest`` scores them.
    """
    require_int("initial", initial, minimum=1)
    require_int("horizon", horizon, minimum=1)
    step = horizon if step is None else require_int("step", step, minimum=1)
    if window is not None:
        require_int("window", window, minimum=1, maximum=initial)
    require_int("max_folds", max_folds, minimum=1)
    require_int("season_length", season_length, minimum=1)
    levels = require_levels(levels)
    if len({f"{level:g}" for level in levels}) < len(levels):
        raise InvalidInputError(f"levels {list(levels)} must differ within 6 significant digits")
    shortest = initial if window is None else window
    if shortest <= season_length:
        raise InvalidInputError(
            f"training windows of {shortest} values cannot scale MASE at season_length "
            f"{season_length}"
        )
    if shortest < forecaster.min_history:
        raise InvalidInputError(
            f"training windows of {shortest} values are shorter than the forecaster's "
            f"min_history of {forecaster.min_history}"
        )
    values = series.complete_values()
    n = len(values)
    n_folds = (n - horizon - initial) // step + 1 if n >= initial + horizon else 0
    if n_folds == 0:
        raise InsufficientDataError(initial + horizon, n)
    if n_folds > max_folds:
        raise InvalidInputError(
            f"{n_folds} folds exceed max_folds={max_folds}; use a larger step or initial"
        )
    splitter = ar.RollingOriginSplitter(
        initial=initial,
        horizon=horizon,
        step=step,
        mode="expanding" if window is None else "sliding",
        window=window,
    )
    folds: list[FoldResult] = []
    for train, test in splitter.split(n):
        history = series.window(train.start, train.stop)
        actual = tuple(values[test.start : test.stop])
        training = values[train.start : train.stop]
        try:
            _check_scale(training, season_length, squared=False)  # before paying for a fit
            _check_scale(training, season_length, squared=True)
            forecast = forecaster.forecast(history, horizon, levels=levels)
            scores = _score(forecast, actual, training, season_length, levels)
        except (InvalidInputError, CalculationError) as exc:
            status = _status(exc)
            fold = FoldResult(
                history.end, history.start, len(history), status, str(exc), None, actual, {}
            )
        else:
            fold = FoldResult(
                history.end, history.start, len(history), "ok", None, forecast, actual, scores
            )
        folds.append(fold)
    return tuple(folds)


def _call(metric: Callable[..., float], *args: object, **kwargs: object) -> float:
    with orchid_errors(metric.__name__):
        value = metric(*args, **kwargs)
    if not math.isfinite(value):
        raise CalculationError(f"Orchid {metric.__name__} returned {value}")
    return value


def _arrays(**arrays: Sequence[float]) -> list[list[float]]:
    """The named arrays as floats; they must be non-empty, equally long and complete."""
    lengths = {name: len(values) for name, values in arrays.items()}
    if len(set(lengths.values())) > 1:
        raise InvalidInputError(f"inputs must have equal lengths, got {lengths}")
    if 0 in lengths.values():
        raise InsufficientDataError(1, 0, what="value")
    return [finite_values(name, values) for name, values in arrays.items()]


def _intervals(
    actual: Sequence[float], lower: Sequence[float], upper: Sequence[float]
) -> tuple[list[float], list[float], list[float]]:
    a, low, high = _arrays(actual=actual, lower=lower, upper=upper)
    crossed = [i for i in range(len(a)) if low[i] > high[i]]
    if crossed:
        raise InvalidInputError(
            f"lower exceeds upper at {len(crossed)} point(s), first at index {crossed[0]}"
        )
    return a, low, high


def _training(training: Sequence[float], season_length: int, *, squared: bool) -> list[float]:
    values = finite_values("training", training)
    _check_scale(values, season_length, squared=squared)
    return values


def _check_scale(values: Sequence[float], season_length: int, *, squared: bool) -> None:
    """Reject training whose mean absolute (or, if ``squared``, root mean square) seasonal
    difference Orchid would clamp, or which is lost in rounding at the training level."""
    if len(values) <= season_length:
        raise InsufficientDataError(season_length + 1, len(values), what="training values")
    differences = [values[t] - values[t - season_length] for t in range(season_length, len(values))]
    try:
        scale = fmean(d * d if squared else abs(d) for d in differences)
        level = fmean(abs(value) for value in values)
    except OverflowError as exc:
        raise CalculationError("training values overflow a float") from exc
    if not math.isfinite(scale):
        raise CalculationError("training values overflow a float")
    # Orchid clamps the mean absolute difference at 1e-12 and the mean square at 1e-12.
    spread, floor = (math.sqrt(scale), math.sqrt(_ORCHID_EPS)) if squared else (scale, _ORCHID_EPS)
    if spread <= max(floor, _ORCHID_EPS * level):
        raise DegenerateDataError(f"training has no usable variation at lag {season_length}")


def _status(error: AlgorithmError) -> FoldStatus:
    if isinstance(error, InsufficientDataError):
        return "insufficient_data"
    if isinstance(error, DegenerateDataError):
        return "degenerate_data"
    return "failed"


def _band(forecast: Forecast, level: float) -> PredictionInterval:
    return next(band for band in forecast.intervals if band.level == level)


def _score(
    forecast: Forecast,
    actual: Sequence[float],
    training: Sequence[float],
    season_length: int,
    levels: Sequence[float],
) -> dict[str, float]:
    """One fold's scores, from Orchid on the already validated actuals and training window.

    ``CalculationError`` for a forecast or band of the wrong length, a missing level or NaN
    values; ``InvalidInputError`` for infinite or crossed forecast values.
    """
    missing = sorted(set(levels) - {band.level for band in forecast.intervals})
    if missing:
        raise CalculationError(f"forecast lacks intervals at levels {missing}")
    bands = [_band(forecast, level) for level in levels]
    lengths = {len(forecast.mean), *(len(edge) for b in bands for edge in (b.lower, b.upper))}
    if lengths != {len(actual)}:
        raise CalculationError(
            f"forecast arrays have {sorted(lengths)} values for a horizon of {len(actual)}"
        )
    m = season_length
    try:
        a, p = _arrays(actual=actual, predicted=forecast.mean)
        scores = {
            "mae": _call(ar.mae, a, p),
            "rmse": _call(ar.rmse, a, p),
            "mase": _call(ar.mase, a, p, y_train=training, seasonality=m),
            "rmsse": _call(ar.rmsse, a, p, y_train=training, seasonality=m),
        }
        for level, band in zip(levels, bands, strict=True):
            _, low, high = _intervals(a, band.lower, band.upper)
            scores[f"coverage_{level:g}"] = _call(ar.empirical_coverage, a, low, high)
            scores[f"interval_score_{level:g}"] = _call(
                ar.interval_score, a, low, high, level=level
            )
            scores[f"msis_{level:g}"] = _call(
                ar.msis, a, low, high, y_train=training, seasonality=m, level=level
            )
    except MissingDataError as exc:  # actuals and training are complete, so it is the forecast
        raise CalculationError(f"forecast has NaN values: {exc}") from exc
    return scores


def _report(folds: list[FoldResult], horizon: int, levels: Sequence[float]) -> BacktestReport:
    ok = [(fold, fold.forecast) for fold in folds if fold.forecast is not None]
    n_excluded = len(folds) - len(ok)
    if not ok:
        return BacktestReport(tuple(folds), 0, n_excluded, {}, tuple({} for _ in range(horizon)))
    actual = [value for fold, _ in ok for value in fold.actual]
    predicted = [value for _, forecast in ok for value in forecast.mean]
    metrics = {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mase": fmean(fold.scores["mase"] for fold, _ in ok),
        "rmsse": fmean(fold.scores["rmsse"] for fold, _ in ok),
    }
    for level in levels:
        lower = [value for _, forecast in ok for value in _band(forecast, level).lower]
        upper = [value for _, forecast in ok for value in _band(forecast, level).upper]
        metrics[f"coverage_{level:g}"] = empirical_coverage(actual, lower, upper)
        metrics[f"interval_score_{level:g}"] = interval_score(actual, lower, upper, level=level)
        metrics[f"msis_{level:g}"] = fmean(fold.scores[f"msis_{level:g}"] for fold, _ in ok)
    forecasts = [(fold.actual, forecast) for fold, forecast in ok]
    by_horizon = tuple(_step_scores(forecasts, step, levels) for step in range(horizon))
    return BacktestReport(tuple(folds), len(ok), n_excluded, metrics, by_horizon)


def _step_scores(
    ok: Sequence[tuple[tuple[float, ...], Forecast]], step: int, levels: Sequence[float]
) -> dict[str, float]:
    actual = [values[step] for values, _ in ok]
    predicted = [forecast.mean[step] for _, forecast in ok]
    scores = {"mae": mae(actual, predicted), "rmse": rmse(actual, predicted)}
    for level in levels:
        bands = [_band(forecast, level) for _, forecast in ok]
        lower = [band.lower[step] for band in bands]
        upper = [band.upper[step] for band in bands]
        scores[f"coverage_{level:g}"] = empirical_coverage(actual, lower, upper)
    return scores


def _scaled_errors(
    a: Sequence[float], b: Sequence[float], power: int
) -> tuple[list[float], list[float]]:
    """``a`` and ``b`` divided by the factor that gives their loss differential unit variance.

    Orchid floors the long-run variance at an absolute 1e-12. Dividing by the largest error
    alone would leave the differential's variance below that whenever a few errors dominate
    the losses. The differential scales with the errors to the power ``power``.
    """
    peak = max(abs(value) for value in (*a, *b))
    if peak == 0:
        raise DegenerateDataError("every forecast error is zero")
    a, b = [value / peak for value in a], [value / peak for value in b]  # every loss is <= 1
    differential = _differential(a, b, power)
    # Rounding alone spreads a constant differential over about 1e-16 of the largest loss.
    if max(differential) - min(differential) <= 1e-12:
        raise DegenerateDataError(
            "the loss differential is constant to within 1e-12 of the largest loss"
        )
    unit = math.pow(pvariance(differential), 0.5 / power)
    return [value / unit for value in a], [value / unit for value in b]


def _differential(a: Sequence[float], b: Sequence[float], power: int) -> list[float]:
    """The loss of ``a`` minus the loss of ``b`` at each target, as Orchid computes it."""
    return [abs(x) ** power - abs(y) ** power for x, y in zip(a, b, strict=True)]


def _long_run_variance(differential: Sequence[float], horizon: int) -> float:
    """Orchid's estimate, which can be negative: the variance plus twice the autocovariances
    to lag ``horizon - 1``, each about the overall mean, divided by n and unweighted."""
    n = len(differential)
    mean = math.fsum(differential) / n
    centred = [value - mean for value in differential]
    autocovariances = [
        math.fsum(x * y for x, y in zip(centred[lag:], centred[: n - lag], strict=True)) / n
        for lag in range(horizon)
    ]
    return autocovariances[0] + 2 * math.fsum(autocovariances[1:])
