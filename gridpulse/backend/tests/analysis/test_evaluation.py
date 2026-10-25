import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from statistics import fmean

import pytest

from gridpulse.analysis.algorithms.errors import (
    CalculationError,
    DegenerateDataError,
    InsufficientDataError,
    InvalidInputError,
    MissingDataError,
)
from gridpulse.analysis.algorithms.evaluation import (
    BacktestReport,
    Loss,
    backtest,
    diebold_mariano,
    empirical_coverage,
    interval_score,
    mae,
    mase,
    msis,
    pinball_loss,
    rmse,
    rmsse,
    smape,
    wmape,
)
from gridpulse.analysis.algorithms.forecasting import DEFAULT_LEVELS, Forecast, PredictionInterval
from gridpulse.analysis.algorithms.orchid import autoregressive as ar
from gridpulse.analysis.algorithms.series import HalfHourlySeries

START = datetime(2026, 10, 1, tzinfo=UTC)
MakeSeries = Callable[..., HalfHourlySeries]
MakeValues = Callable[..., list[float]]
Metric = Callable[[Sequence[float], Sequence[float]], float]

TRAINING = [0.0, 1.0, 3.0, 2.0, 6.0, 4.0]  # varies at lag 2
# Errors of two forecasts whose squared and absolute loss differentials both vary.
ERRORS_A = [math.sin(i) for i in range(40)]
ERRORS_B = [0.5 * math.cos(i) for i in range(40)]


def _above(values: Sequence[float]) -> list[float]:
    return [value + 1.0 for value in values]


# Each metric as a function of the actual values and one forecast array; interval metrics
# take that array as the lower bound and put the upper bound 1 above it.
METRICS: dict[str, Metric] = {
    "mae": mae,
    "rmse": rmse,
    "wmape": wmape,
    "smape": smape,
    "mase": lambda a, p: mase(a, p, training=TRAINING, season_length=2),
    "rmsse": lambda a, p: rmsse(a, p, training=TRAINING, season_length=2),
    "pinball_loss": lambda a, q: pinball_loss(a, q, quantile=0.9),
    "interval_score": lambda a, lo: interval_score(a, lo, _above(lo), level=0.9),
    "empirical_coverage": lambda a, lo: empirical_coverage(a, lo, _above(lo)),
    "msis": lambda a, lo: msis(a, lo, _above(lo), training=TRAINING, season_length=2, level=0.9),
}
SCALED: dict[str, Callable[[Sequence[float]], float]] = {
    "mase": lambda training: mase([1.0, 2.0], [1.5, 2.5], training=training, season_length=48),
    "rmsse": lambda training: rmsse([1.0, 2.0], [1.5, 2.5], training=training, season_length=48),
    "msis": lambda training: msis(
        [1.0, 2.0], [0.0, 1.0], [2.0, 3.0], training=training, season_length=48, level=0.8
    ),
}


@dataclass
class LastValue:
    """Repeats the last training value inside intervals of half-width ``2 * level``."""

    min_history: int = 1
    histories: list[HalfHourlySeries] = field(default_factory=list)

    def forecast(
        self, history: HalfHourlySeries, horizon: int, *, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> Forecast:
        self.histories.append(history)
        last = history.complete_values()[-1]
        intervals = tuple(
            PredictionInterval(level, (last - 2 * level,) * horizon, (last + 2 * level,) * horizon)
            for level in levels
        )
        return Forecast("last value", {}, history.end, (last,) * horizon, intervals, "fixed", ())


@dataclass
class Faulty(LastValue):
    """Raises, or malforms its forecast, at the history lengths in ``faults``."""

    faults: Mapping[int, Exception | Callable[[Forecast], Forecast]] = field(default_factory=dict)

    def forecast(
        self, history: HalfHourlySeries, horizon: int, *, levels: Sequence[float] = DEFAULT_LEVELS
    ) -> Forecast:
        fault = self.faults.get(len(history))
        if isinstance(fault, Exception):
            raise fault
        forecast = super().forecast(history, horizon, levels=levels)
        return forecast if fault is None else fault(forecast)


def _crossed(forecast: Forecast) -> Forecast:
    band = forecast.intervals[0]
    swapped = PredictionInterval(band.level, band.upper, band.lower)
    return replace(forecast, intervals=(swapped, *forecast.intervals[1:]))


@pytest.mark.parametrize("metric", METRICS.values(), ids=METRICS.keys())
@pytest.mark.parametrize(
    ("actual", "other", "error"),
    [
        ([1.0, 2.0, 3.0], [1.0, 2.0], InvalidInputError),
        ([], [], InsufficientDataError),
        ([1.0, None, 3.0], [1.0, 2.0, 3.0], MissingDataError),
        ([1.0, 2.0, 3.0], [1.0, math.nan, 3.0], MissingDataError),
        ([1.0, 2.0, math.inf], [1.0, 2.0, 3.0], InvalidInputError),
    ],
    ids=["length mismatch", "empty", "None", "NaN", "inf"],
)
def test_metrics_reject_unusable_arrays(
    metric: Metric, actual: list[float | None], other: list[float], error: type[Exception]
) -> None:
    with pytest.raises(error) as excinfo:
        metric(actual, other)  # type: ignore[arg-type]
    assert excinfo.type is error


@pytest.mark.parametrize("metric", SCALED.values(), ids=SCALED.keys())
def test_scaled_metrics_reject_short_or_constant_training(
    metric: Callable[[Sequence[float]], float], synthetic_values: MakeValues
) -> None:
    with pytest.raises(InsufficientDataError):  # Orchid scores this as a perfect 0.0
        metric(synthetic_values(48))
    with pytest.raises(DegenerateDataError):  # Orchid divides by 1e-12 instead
        metric([5.0] * 96)
    assert metric(synthetic_values(49)) > 0


def test_training_values_are_checked() -> None:
    with pytest.raises(MissingDataError):
        mase([1.0], [1.0], training=[0.0, None, 1.0, 2.0], season_length=1)  # type: ignore[list-item]
    with pytest.raises(InvalidInputError) as excinfo:
        mase([1.0], [1.0], training=[0.0, math.inf, 1.0, 2.0], season_length=1)
    assert excinfo.type is InvalidInputError
    # Scale-relative: a 1e-7 wobble on a level of 1e6 is no seasonal variation at all.
    flat = [1e6 + 1e-7 * (i % 2) for i in range(96)]
    with pytest.raises(DegenerateDataError):
        mase([1e6], [1e6], training=flat, season_length=1)
    # RMSSE's squared scale (1e-14) falls below Orchid's clamp where MASE's (1e-7) does not.
    tiny = [1e-7 * (i % 2) for i in range(10)]
    assert mase([0.0], [1e-7], training=tiny, season_length=1) == pytest.approx(1.0)
    with pytest.raises(DegenerateDataError):
        rmsse([0.0], [1e-7], training=tiny, season_length=1)
    # Both scales are compared with the level in the values' units.
    wobble = [1e6 + 5e-4 * (i % 2) for i in range(10)]
    assert rmsse([1e6], [1e6 + 5e-4], training=wobble, season_length=1) == pytest.approx(1.0)


def test_percentage_errors_reject_all_zero_inputs() -> None:
    with pytest.raises(DegenerateDataError):
        wmape([0.0, 0.0], [1.0, -1.0])
    with pytest.raises(DegenerateDataError):  # below Orchid's 1e-12 denominator clamp
        wmape([1e-13, 0.0], [1.0, 0.0])
    with pytest.raises(DegenerateDataError):  # Orchid scores this as a perfect 0.0
        smape([0.0, 0.0], [0.0, 0.0])
    # Overflowing denominators would make Orchid score these as a perfect 0.0 too.
    with pytest.raises(CalculationError):
        wmape([1e308, 1e308], [1e308, 0.0])
    with pytest.raises(CalculationError):
        smape([1.7e308], [1e308])
    assert wmape([1e308, 1e308], [1e308, 1e308]) == smape([1.7e308], [1.7e308]) == 0.0


@pytest.mark.parametrize(
    "call",
    [
        lambda: interval_score([1.0], [0.0], [2.0], level=95),
        lambda: interval_score([1.0], [0.0], [2.0], level=1.0),
        lambda: msis([1.0], [0.0], [2.0], training=TRAINING, season_length=2, level=95),
        lambda: msis([1.0], [0.0], [2.0], training=TRAINING, season_length=2, level=1.0),
        lambda: pinball_loss([1.0], [1.0], quantile=0.0),
        lambda: pinball_loss([1.0], [1.0], quantile=1.0),
        lambda: mase([1.0], [1.0], training=TRAINING, season_length=0),
        lambda: rmsse([1.0], [1.0], training=TRAINING, season_length=0),
        lambda: msis([1.0], [0.0], [2.0], training=TRAINING, season_length=0, level=0.9),
        lambda: interval_score([1.0, 2.0], [0.0, 2.5], [2.0, 2.4], level=0.9),
        lambda: empirical_coverage([1.0, 2.0], [0.0, 2.5], [2.0, 2.4]),
        lambda: msis(
            [1.0, 2.0], [0.0, 2.5], [2.0, 2.4], training=TRAINING, season_length=2, level=0.9
        ),
    ],
    ids=[
        "interval_score level 95",
        "interval_score level 1",
        "msis level 95",
        "msis level 1",
        "quantile 0",
        "quantile 1",
        "mase season length 0",
        "rmsse season length 0",
        "msis season length 0",
        "interval_score crossed",
        "empirical_coverage crossed",
        "msis crossed",
    ],
)
def test_rejects_invalid_parameters_and_crossed_bounds(call: Callable[[], float]) -> None:
    with pytest.raises(InvalidInputError) as excinfo:
        call()
    assert excinfo.type is InvalidInputError


def test_orchid_failures_and_non_finite_results_raise_calculation_error() -> None:
    with pytest.raises(CalculationError) as excinfo:
        rmse([1e200], [0.0])  # Orchid's ** raises OverflowError
    assert isinstance(excinfo.value.__cause__, OverflowError)
    with pytest.raises(CalculationError):
        mae([1.7e308], [-1.7e308])  # the error overflows to inf
    with pytest.raises(CalculationError):
        wmape([1e308, 1e308], [0.0, 0.0])
    with pytest.raises(CalculationError):  # the training scale overflows
        mase([1.0], [1.0], training=[1e308, -1e308, 1e308], season_length=1)


def test_metrics_return_orchid_values_on_negative_and_zero_values(
    synthetic_values: MakeValues,
) -> None:
    actual = [-12.5, 0.0, 3.0, 40.0, -1.0, 0.0]
    predicted = [-10.0, 2.0, 0.0, 35.5, 1.0, 0.0]
    lower = [value - 4.0 for value in predicted]
    upper = [value + 3.0 for value in predicted]
    training = synthetic_values(100, level=0.0)  # frequent negative prices
    assert mae(actual, predicted) == ar.mae(actual, predicted)
    assert rmse(actual, predicted) == ar.rmse(actual, predicted)
    assert wmape(actual, predicted) == ar.wmape(actual, predicted, as_percent=False)
    assert smape(actual, predicted) == ar.smape(actual, predicted, as_percent=False)
    assert mase(actual, predicted, training=training, season_length=48) == ar.mase(
        actual, predicted, y_train=training, seasonality=48
    )
    assert rmsse(actual, predicted, training=training, season_length=48) == ar.rmsse(
        actual, predicted, y_train=training, seasonality=48
    )
    assert pinball_loss(actual, lower, quantile=0.1) == ar.pinball_loss(actual, lower, quantile=0.1)
    assert interval_score(actual, lower, upper, level=0.8) == ar.interval_score(
        actual, lower, upper, level=0.8
    )
    assert empirical_coverage(actual, lower, upper) == ar.empirical_coverage(actual, lower, upper)
    assert msis(actual, lower, upper, training=training, season_length=48, level=0.8) == ar.msis(
        actual, lower, upper, y_train=training, seasonality=48, level=0.8
    )


@pytest.mark.parametrize("scale", [1e-3, 1e3])
def test_rmsse_is_scale_invariant(synthetic_values: MakeValues, scale: float) -> None:
    values = synthetic_values(200, level=0.0)
    training, actual = values[:150], values[150:]
    predicted = synthetic_values(50, level=0.0, seed=1)
    expected = rmsse(actual, predicted, training=training, season_length=48)
    scaled = rmsse(
        [scale * value for value in actual],
        [scale * value for value in predicted],
        training=[scale * value for value in training],
        season_length=48,
    )
    assert scaled == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("level", [0.5, 0.8, 0.95])
def test_interval_score_is_the_scaled_pinball_loss_of_the_bounds(level: float) -> None:
    rng = random.Random(0)
    actual = [rng.gauss(0.0, 10.0) for _ in range(200)]
    bounds = [sorted([rng.gauss(0.0, 10.0), rng.gauss(0.0, 10.0)]) for _ in range(200)]
    lower = [low for low, _ in bounds]
    upper = [high for _, high in bounds]
    tails = pinball_loss(actual, lower, quantile=(1 - level) / 2) + pinball_loss(
        actual, upper, quantile=(1 + level) / 2
    )
    assert interval_score(actual, lower, upper, level=level) == pytest.approx(
        2 / (1 - level) * tails, rel=1e-12
    )


@pytest.mark.parametrize("window", [None, 60])
def test_folds_roll_forward_without_leakage(make_series: MakeSeries, window: int | None) -> None:
    series = make_series(160)
    spy = LastValue()
    report = backtest(spy, series, initial=100, horizon=12, step=10, window=window)
    origins = range(100, 141, 10)  # the last test window ends at 152 <= 160
    starts = [0 if window is None else origin - window for origin in origins]
    assert report.n_ok == len(report.folds) == 5
    assert [fold.origin for fold in report.folds] == [series.timestamp(i) for i in origins]
    assert [fold.train_start for fold in report.folds] == [series.timestamp(i) for i in starts]
    assert [fold.train_size for fold in report.folds] == [
        origin - start for origin, start in zip(origins, starts, strict=True)
    ]
    assert [fold.actual for fold in report.folds] == [series.values[i : i + 12] for i in origins]
    # The forecaster sees exactly its training window, never a value at or after the origin.
    assert [history.end for history in spy.histories] == [fold.origin for fold in report.folds]
    assert [history.values for history in spy.histories] == [
        series.values[start:origin] for start, origin in zip(starts, origins, strict=True)
    ]


def test_failed_folds_are_kept_in_order_and_left_out_of_the_scores(
    make_series: MakeSeries,
) -> None:
    series = make_series(196)  # folds at 100, 112, ..., 184
    faults: dict[int, Exception | Callable[[Forecast], Forecast]] = {
        112: CalculationError("solver diverged"),
        124: InsufficientDataError(200, 124),
        136: DegenerateDataError("constant history"),
        148: lambda forecast: replace(forecast, mean=forecast.mean[:-1]),
        160: lambda forecast: replace(forecast, intervals=forecast.intervals[:1]),
    }
    report = backtest(Faulty(faults=faults), series, initial=100, horizon=12)
    assert [fold.origin for fold in report.folds] == [
        series.timestamp(i) for i in range(100, 185, 12)
    ]
    assert [fold.status for fold in report.folds] == [
        "ok",
        "failed",
        "insufficient_data",
        "degenerate_data",
        "failed",
        "failed",
        "ok",
        "ok",
    ]
    assert [fold.message for fold in report.folds[:4]] == [
        None,
        "solver diverged",
        "needs at least 200 observations, got 124",
        "constant history",
    ]
    assert all(fold.message for fold in report.folds[4:6])
    assert (report.n_ok, report.n_excluded) == (3, 5)
    ok = [fold for fold in report.folds if fold.status == "ok"]
    assert all(fold.scores and fold.forecast for fold in ok)
    assert not any(fold.scores or fold.forecast for fold in report.folds if fold.status != "ok")
    # Three folds with unequal MAEs, so a median would give a different answer.
    assert report.metrics["mae"] == pytest.approx(fmean(fold.scores["mae"] for fold in ok))
    with pytest.raises(ZeroDivisionError):  # other exceptions are bugs, so they propagate
        backtest(Faulty(faults={112: ZeroDivisionError()}), series, initial=100, horizon=12)


def _short_band(forecast: Forecast) -> Forecast:
    band = forecast.intervals[0]
    return replace(
        forecast, intervals=(replace(band, lower=band.lower[:-1]), *forecast.intervals[1:])
    )


def test_invalid_forecasts_fail_their_fold(make_series: MakeSeries) -> None:
    faults: dict[int, Exception | Callable[[Forecast], Forecast]] = {
        112: MissingDataError([0]),
        124: _crossed,
        136: lambda forecast: replace(forecast, mean=(math.nan, *forecast.mean[1:])),
        148: _short_band,
    }
    report = backtest(Faulty(faults=faults), make_series(172), initial=100, horizon=12)
    statuses = [fold.status for fold in report.folds]
    assert statuses == ["ok", "failed", "failed", "failed", "failed", "ok"]
    messages = [fold.message or "" for fold in report.folds]
    assert messages[3].startswith("forecast has NaN values")
    assert messages[4].startswith("forecast arrays have")


def test_folds_that_cannot_be_scored_skip_the_fit(make_series: MakeSeries) -> None:
    # Exactly daily-periodic values have no seasonal-naive error to scale MASE by.
    periodic = HalfHourlySeries(START, [math.sin(2 * math.pi * t / 48) for t in range(160)])
    spy = LastValue()
    report = backtest(spy, periodic, initial=100, horizon=12)
    assert {fold.status for fold in report.folds} == {"degenerate_data"}
    assert spy.histories == []
    assert (report.n_ok, report.n_excluded, report.metrics) == (0, 5, {})
    assert report.by_horizon == ({},) * 12


def test_folds_whose_rmsse_scale_orchid_would_clamp_skip_the_fit() -> None:
    # The mean absolute seasonal difference (1e-7) can scale MASE; its square cannot scale RMSSE.
    wobble = HalfHourlySeries(START, [1e-7 * (i % 2) for i in range(40)])
    spy = LastValue()
    report = backtest(spy, wobble, initial=20, horizon=4, season_length=1, levels=(0.8,))
    assert {fold.status for fold in report.folds} == {"degenerate_data"}
    assert spy.histories == []


def test_scores_match_a_hand_computation() -> None:
    # Two folds forecast 5 and 7 for actuals (4, 7) and (5.75, 3), with half-widths 1 and 1.5.
    # Their lag-2 training scales are 1.5 and 1.75 (absolute) and 2.5 and 3.25 (squared).
    series = HalfHourlySeries(START, [1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 5.75, 3.0])

    def run() -> BacktestReport:
        return backtest(
            LastValue(), series, initial=4, horizon=2, season_length=2, levels=(0.5, 0.75)
        )

    report = run()
    assert report.n_ok == 2
    assert report.folds[0].scores == pytest.approx(
        {
            "mae": 1.5,
            "rmse": math.sqrt(2.5),
            "mase": 1.0,
            "rmsse": 1.0,
            "coverage_0.5": 0.5,
            "interval_score_0.5": 4.0,
            "msis_0.5": 4.0 / 1.5,
            "coverage_0.75": 0.5,
            "interval_score_0.75": 5.0,
            "msis_0.75": 5.0 / 1.5,
        }
    )
    # MAE, RMSE, coverage and interval scores pool all four points; MASE, RMSSE and MSIS
    # average the per-fold values.
    assert report.metrics == pytest.approx(
        {
            "mae": (1.0 + 2.0 + 1.25 + 4.0) / 4,
            "rmse": math.sqrt((1.0 + 4.0 + 1.5625 + 16.0) / 4),
            "mase": (1.0 + 2.625 / 1.75) / 2,
            "rmsse": (1.0 + math.sqrt(8.78125 / 3.25)) / 2,
            "coverage_0.5": 0.25,
            "interval_score_0.5": (2.0 + 6.0 + 3.0 + 14.0) / 4,
            "msis_0.5": (4.0 / 1.5 + 8.5 / 1.75) / 2,
            "coverage_0.75": 0.5,
            "interval_score_0.75": (3.0 + 7.0 + 3.0 + 23.0) / 4,
            "msis_0.75": (5.0 / 1.5 + 13.0 / 1.75) / 2,
        }
    )
    assert list(report.by_horizon) == [
        pytest.approx(
            {"mae": 1.125, "rmse": math.sqrt(1.28125), "coverage_0.5": 0.5, "coverage_0.75": 1.0}
        ),
        pytest.approx(
            {"mae": 3.0, "rmse": math.sqrt(10.0), "coverage_0.5": 0.0, "coverage_0.75": 0.0}
        ),
    ]
    assert run() == report


@pytest.mark.parametrize(
    ("settings", "error"),
    [
        ({"step": 1, "max_folds": 40}, InvalidInputError),  # 49 folds
        ({"initial": 150}, InsufficientDataError),  # no room for a 12-period test window
        ({"initial": 0}, InvalidInputError),
        ({"horizon": 0}, InvalidInputError),
        ({"step": 0}, InvalidInputError),  # Orchid would loop forever
        ({"window": 0}, InvalidInputError),  # Orchid would silently expand the window
        ({"window": 101}, InvalidInputError),  # early folds would expand, later ones slide
        ({"window": 48}, InvalidInputError),  # too short to scale MASE at season_length 48
        ({"max_folds": 50.0}, InvalidInputError),
        ({"season_length": 0}, InvalidInputError),
        ({"levels": (95,)}, InvalidInputError),
        ({"levels": ()}, InvalidInputError),
        ({"levels": (0.95, 0.9500001)}, InvalidInputError),  # their metric keys would collide
    ],
)
def test_backtest_rejects_unusable_settings(
    make_series: MakeSeries, settings: dict[str, object], error: type[Exception]
) -> None:
    options = {"initial": 100, "horizon": 12, **settings}
    with pytest.raises(error) as excinfo:
        backtest(LastValue(), make_series(160), **options)  # type: ignore[arg-type]
    assert excinfo.type is error


def test_backtest_checks_the_forecasters_min_history(make_series: MakeSeries) -> None:
    with pytest.raises(InvalidInputError, match="min_history"):
        backtest(LastValue(min_history=101), make_series(160), initial=100, horizon=12)


def test_max_folds_is_inclusive(make_series: MakeSeries) -> None:
    series = make_series(160)
    report = backtest(LastValue(), series, initial=100, horizon=12, step=1, max_folds=49)
    assert len(report.folds) == 49
    with pytest.raises(InvalidInputError, match="larger step or initial"):
        backtest(LastValue(), series, initial=100, horizon=12, step=1, max_folds=48)


def test_backtest_requires_complete_data(make_series: MakeSeries) -> None:
    values = list(make_series(160).values)
    values[3] = None
    with pytest.raises(MissingDataError):
        backtest(LastValue(), HalfHourlySeries(START, values), initial=100, horizon=12)


@pytest.mark.parametrize("loss", ["squared", "absolute"])
@pytest.mark.parametrize("horizon", [1, 4])
@pytest.mark.parametrize("scale", [1.0, 1e-7])  # Orchid's absolute 1e-12 floor bites at 1e-7
def test_diebold_mariano_matches_orchid_at_unit_scale(
    synthetic_values: MakeValues, scale: float, horizon: int, loss: Loss
) -> None:
    errors_a = synthetic_values(200, level=0.0, amplitude=0.0, noise=1.2, seed=1)
    errors_b = synthetic_values(200, level=0.0, amplitude=0.0, noise=1.0, seed=2)
    expected = ar.diebold_mariano(errors_a, errors_b, horizon=horizon, loss=loss)
    result = diebold_mariano(
        [scale * error for error in errors_a],
        [scale * error for error in errors_b],
        horizon=horizon,
        loss=loss,
    )
    assert (result.horizon, result.loss, result.warnings) == (horizon, loss, ())
    assert result.statistic == pytest.approx(expected.statistic, rel=1e-12)
    assert result.p_value == pytest.approx(expected.pvalue, rel=1e-12)


@pytest.mark.parametrize("loss", ["squared", "absolute"])
def test_diebold_mariano_uses_horizon_1_when_the_long_run_variance_is_negative(
    loss: Loss,
) -> None:
    # The loss differential alternates, so its lag-1 autocovariance outweighs its variance.
    errors_a, errors_b = [3.0, 0.0] * 20, [1.0] * 40
    assert ar.diebold_mariano(errors_a, errors_b, horizon=2, loss=loss).pvalue == 0.0  # floored
    expected = ar.diebold_mariano(errors_a, errors_b, horizon=1, loss=loss)
    result = diebold_mariano(errors_a, errors_b, horizon=2, loss=loss)
    [warning] = result.warnings
    assert "horizon 2" in warning
    assert result.horizon == 1
    assert result.statistic == pytest.approx(expected.statistic, rel=1e-12)
    assert result.p_value == pytest.approx(expected.pvalue, rel=1e-12)


def test_a_large_error_both_forecasts_share_does_not_mask_their_difference() -> None:
    rng = random.Random(1)
    errors_a = [rng.gauss(0.0, 1.0) for _ in range(100)]
    errors_b = [1.1 * error for error in errors_a]
    errors_a[50] = errors_b[50] = 1e3
    expected = ar.diebold_mariano(errors_a, errors_b)  # Orchid's floor is harmless at this scale
    # Divided by the largest error alone, the other squared losses fall below Orchid's floor.
    divided = ar.diebold_mariano(
        [error / 1e3 for error in errors_a], [error / 1e3 for error in errors_b]
    )
    assert divided.statistic != pytest.approx(expected.statistic)
    result = diebold_mariano(errors_a, errors_b)
    assert result.statistic == pytest.approx(expected.statistic, rel=1e-12)


@pytest.mark.parametrize("loss", ["squared", "absolute"])
def test_a_positive_diebold_mariano_statistic_means_errors_b_are_more_accurate(
    loss: Loss,
) -> None:
    rng = random.Random(3)
    errors_a = [rng.gauss(0.0, 2.0) for _ in range(100)]
    errors_b = [rng.gauss(0.0, 1.0) for _ in range(100)]
    result = diebold_mariano(errors_a, errors_b, loss=loss)
    assert result.statistic > 0
    assert ar.diebold_mariano(errors_a, errors_b, loss=loss).better == "model2"
    swapped = diebold_mariano(errors_b, errors_a, loss=loss)
    assert (swapped.statistic, swapped.p_value) == (-result.statistic, result.p_value)


@pytest.mark.parametrize(
    ("errors_a", "errors_b", "options", "error"),
    [
        (ERRORS_A, ERRORS_B[:-1], {}, InvalidInputError),  # Orchid would truncate
        (ERRORS_A[:29], ERRORS_B[:28], {}, InvalidInputError),
        ([*ERRORS_A[:-1], None], ERRORS_B, {}, MissingDataError),
        (ERRORS_A, [*ERRORS_B[:-1], math.nan], {}, MissingDataError),
        ([*ERRORS_A[:-1], math.inf], ERRORS_B, {}, InvalidInputError),
        (ERRORS_A, ERRORS_B, {"horizon": 0}, InvalidInputError),
        (ERRORS_A, ERRORS_B, {"horizon": 21}, InvalidInputError),
        (ERRORS_A, ERRORS_B, {"horizon": 2.0}, InvalidInputError),
        (ERRORS_A, ERRORS_B, {"loss": "mse"}, InvalidInputError),  # Orchid: absolute loss
        ([1.0, -1.0] * 20, [0.5] * 40, {}, DegenerateDataError),  # squared losses 1 and 0.25
        (ERRORS_A, ERRORS_A, {}, DegenerateDataError),
        ([0.0] * 40, [0.0] * 40, {}, DegenerateDataError),
        (  # absolute losses that differ by 0.1, but for rounding
            [2.0 + error for error in ERRORS_A],
            [2.1 + error for error in ERRORS_A],
            {"loss": "absolute"},
            DegenerateDataError,
        ),
    ],
    ids=[
        "length mismatch",
        "length mismatch below 30",
        "None",
        "NaN",
        "inf",
        "horizon 0",
        "horizon above n // 2",
        "float horizon",
        "unknown loss",
        "constant differential",
        "identical errors",
        "zero errors",
        "constant differential but for rounding",
    ],
)
def test_diebold_mariano_rejects_unusable_inputs(
    errors_a: list[float | None],
    errors_b: list[float],
    options: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error) as excinfo:
        diebold_mariano(errors_a, errors_b, **options)  # type: ignore[arg-type]
    assert excinfo.type is error


def test_diebold_mariano_needs_30_errors_and_allows_a_horizon_of_half_of_them() -> None:
    diebold_mariano(ERRORS_A[:30], ERRORS_B[:30])
    assert diebold_mariano(ERRORS_A, ERRORS_B, horizon=20).horizon == 20
    for n in (0, 29):
        with pytest.raises(InsufficientDataError) as excinfo:
            diebold_mariano(ERRORS_A[:n], ERRORS_B[:n])
        assert (excinfo.value.required, excinfo.value.available) == (30, n)


def test_diebold_mariano_reports_orchid_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise ZeroDivisionError("boom")

    monkeypatch.setattr(ar, "diebold_mariano", broken)
    with pytest.raises(CalculationError) as excinfo:
        diebold_mariano(ERRORS_A, ERRORS_B)
    assert isinstance(excinfo.value.__cause__, ZeroDivisionError)
    nan = ar.DMResult(math.nan, 1.0, 1, "squared", "tie")
    monkeypatch.setattr(ar, "diebold_mariano", lambda *args, **kwargs: nan)
    with pytest.raises(CalculationError):
        diebold_mariano(ERRORS_A, ERRORS_B)
