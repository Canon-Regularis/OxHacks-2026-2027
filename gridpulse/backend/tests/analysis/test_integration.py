"""The adapters compose: real forecasters feed backtests and residual diagnostics."""

import math
from collections.abc import Callable
from statistics import fmean

import pytest

from gridpulse.analysis.algorithms.diagnostics import residual_diagnostics
from gridpulse.analysis.algorithms.evaluation import backtest, mase
from gridpulse.analysis.algorithms.forecasting import ARIMA, SeasonalNaive
from gridpulse.analysis.algorithms.series import HalfHourlySeries

MakeSeries = Callable[..., HalfHourlySeries]


def test_backtest_scores_a_real_forecaster(make_series: MakeSeries) -> None:
    series = make_series(48 * 10)
    report = backtest(SeasonalNaive(48), series, initial=48 * 7, horizon=48)
    assert (report.n_ok, report.n_excluded) == (3, 0)
    values = series.complete_values()
    folds = [(fold, fold.forecast) for fold in report.folds if fold.forecast is not None]
    fold_mase = [  # expanding folds train from the first value
        mase(fold.actual, forecast.mean, training=values[: fold.train_size], season_length=48)
        for fold, forecast in folds
    ]
    assert report.metrics["mase"] == pytest.approx(fmean(fold_mase))
    inside = [
        low <= actual <= high
        for fold, forecast in folds
        for actual, low, high in zip(
            fold.actual, forecast.intervals[-1].lower, forecast.intervals[-1].upper, strict=True
        )
    ]
    assert report.metrics["coverage_0.95"] == pytest.approx(fmean(inside))


def test_forecast_residuals_feed_residual_diagnostics(make_series: MakeSeries) -> None:
    residuals = ARIMA(1, 0, 0).forecast(make_series(48 * 6), 48).residuals
    result = residual_diagnostics(residuals, ljung_box_lags=24, model_df=1)
    assert result.ljung_box.df == tuple(range(24))  # each lag less model_df
    last = result.ljung_box.p_values[-1]
    assert last is not None
    assert last < 0.01  # AR(1) leaves the daily cycle in the residuals
    steps = zip(residuals[1:], residuals[:-1], strict=True)
    durbin_watson = math.fsum((a - b) ** 2 for a, b in steps) / math.fsum(e * e for e in residuals)
    assert result.durbin_watson == pytest.approx(durbin_watson)
