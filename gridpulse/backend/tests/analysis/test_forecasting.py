import math
import random
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from statistics import NormalDist

import pytest

from gridpulse.analysis.algorithms.errors import (
    CalculationError,
    DegenerateDataError,
    InsufficientDataError,
    InvalidInputError,
    MissingDataError,
)
from gridpulse.analysis.algorithms.forecasting import (
    ARIMA,
    ETS,
    AutoARIMA,
    AutoETS,
    CalendarRegression,
    Drift,
    Forecaster,
    FourierRegression,
    Mean,
    MovingAverage,
    Naive,
    Naive2,
    SeasonalDummyRegression,
    SeasonalNaive,
    TrendRegression,
    WeightedMovingAverage,
)
from gridpulse.analysis.algorithms.orchid import autoregressive as ar
from gridpulse.analysis.algorithms.orchid import forecasting_baseline as fb
from gridpulse.analysis.algorithms.series import HALF_HOUR, LONDON, HalfHourlySeries

START = datetime(2026, 9, 1, tzinfo=UTC)
CLOCK_CHANGE_DAY = datetime(2026, 10, 25, tzinfo=UTC)  # clocks go back at 01:00 UTC
CLOCK_CHANGE = "seasonal model spans a GB clock change: seasonal lags count UTC half-hours"
NOT_ADJUSTED = "seasonal adjustment not applied: no significant seasonality detected"
MakeSeries = Callable[..., HalfHourlySeries]

FB, AR = "forecasting_baseline", "autoregressive"
# Every forecaster with the shortest history it accepts, its Orchid source and interval method.
CASES: list[tuple[Forecaster, int, str, str]] = [
    (Naive(), 2, f"{FB}.Naive", "random-walk"),
    (SeasonalNaive(48), 49, f"{FB}.SeasonalNaive", "seasonal-random-walk"),
    (Drift(), 3, f"{FB}.Drift", "sqrt-horizon"),
    (Mean(), 2, f"{FB}.MeanForecast", "analytic"),
    (Naive2(48), 97, f"{FB}.Naive2", "sqrt-horizon"),
    (MovingAverage(4), 5, f"{FB}.MovingAverageForecaster", "sqrt-horizon"),
    (WeightedMovingAverage((1.0, 2.0, 3.0)), 4, f"{FB}.WeightedMovingAverage", "sqrt-horizon"),
    (ETS(trend="additive", damped=True), 4, f"{FB}.ETSForecaster", "analytic"),
    (
        ETS(trend="additive", damped=True, seasonal="additive", season_length=48),
        96,
        f"{FB}.ETSForecaster",
        "analytic",
    ),
    (AutoETS(season_length=48), 96, f"{FB}.auto_ets", "analytic"),
    (ARIMA(2, 0, 1), 100, f"{AR}.ARIMA", "analytic"),
    (
        ARIMA(1, 1, 1, seasonal_differencing=1, season_length=48),
        149,
        f"{AR}.ARIMA",
        "analytic",
    ),
    (AutoARIMA(season_length=48, max_p=1, max_q=1), 288, f"{AR}.auto_arima", "analytic"),
    (TrendRegression(degree=2), 4, f"{FB}.TrendForecaster", "sqrt-horizon"),
    (SeasonalDummyRegression(48), 50, f"{FB}.SeasonalDummyForecaster", "sqrt-horizon"),
    (FourierRegression(48, 3), 48, f"{FB}.FourierForecaster", "sqrt-horizon"),
    (
        CalendarRegression(2, weekly_harmonics=1),
        336,
        f"{FB}.LinearRegressionForecaster",
        "sqrt-horizon",
    ),
]
SHORTEST = [(forecaster, minimum) for forecaster, minimum, _, _ in CASES]
FORECASTERS = [forecaster for forecaster, _ in SHORTEST]
REGRESSIONS: list[Forecaster] = [
    TrendRegression(),
    SeasonalDummyRegression(48),
    FourierRegression(48, 3),
    CalendarRegression(2),
]
SHORT = HalfHourlySeries(START, [1.0, 2.0, 0.0, 3.0])
CONSTANT = HalfHourlySeries(START, [42.0] * 200)
LINE = HalfHourlySeries(START, [float(t) for t in range(200)])
DAILY_REPEAT = HalfHourlySeries(START, [float(t % 48) for t in range(300)])


@pytest.mark.parametrize(("forecaster", "minimum", "source", "method"), CASES, ids=repr)
def test_forecast_contract_on_negative_prices(
    make_series: MakeSeries, forecaster: Forecaster, minimum: int, source: str, method: str
) -> None:
    n = max(240, minimum)
    history = make_series(n, level=0.0)  # crosses zero, often negative
    result = forecaster.forecast(history, 50, levels=(0.95, 0.8))

    assert result == forecaster.forecast(history, 50, levels=(0.95, 0.8))
    assert (result.source, result.interval_method) == (source, method)
    assert result.origin == history.end
    assert result.timestamps() == tuple(history.timestamp(n + i) for i in range(50))
    assert [interval.level for interval in result.intervals] == [0.8, 0.95]
    inner, outer = result.intervals
    bands = list(zip(outer.lower, inner.lower, result.mean, inner.upper, outer.upper, strict=True))
    assert len(bands) == 50
    assert all(list(band) == sorted(band) for band in bands)
    assert all(math.isfinite(value) for band in bands for value in band)
    assert all(math.isfinite(value) for value in result.residuals)


@pytest.mark.parametrize("forecaster", FORECASTERS, ids=repr)
def test_missing_value_is_reported(
    synthetic_values: Callable[..., list[float]], forecaster: Forecaster
) -> None:
    values: list[float | None] = [*synthetic_values(240)]
    values[100] = None
    with pytest.raises(MissingDataError):
        forecaster.forecast(HalfHourlySeries(START, values), 4)


@pytest.mark.parametrize(("forecaster", "minimum"), SHORTEST, ids=repr)
def test_min_history_is_exact(
    make_series: MakeSeries, forecaster: Forecaster, minimum: int
) -> None:
    assert forecaster.min_history == minimum
    forecaster.forecast(make_series(minimum), 4)
    with pytest.raises(InsufficientDataError) as excinfo:
        forecaster.forecast(make_series(minimum - 1), 4)
    assert excinfo.value.required == minimum


@pytest.mark.parametrize(
    ("forecaster", "orchid", "warm_up"),
    [
        (Naive(), fb.Naive, 1),
        (SeasonalNaive(48), lambda: fb.SeasonalNaive(48), 48),
        (Drift(), fb.Drift, 1),
        (Naive2(48), lambda: fb.Naive2(48, model="additive"), 1),
        (MovingAverage(4), lambda: fb.MovingAverageForecaster(4), 4),
        (
            WeightedMovingAverage((1.0, 2.0, 3.0)),
            lambda: fb.WeightedMovingAverage((1.0, 2.0, 3.0)),
            3,
        ),
    ],
    # Named by forecaster: a lambda's repr holds its address, which differs between pytest-xdist
    # workers and makes them disagree about which tests exist.
    ids=["Naive", "SeasonalNaive", "Drift", "Naive2", "MovingAverage", "WeightedMovingAverage"],
)
def test_interval_sigma_excludes_warm_up_residuals(
    make_series: MakeSeries,
    forecaster: Forecaster,
    orchid: Callable[[], fb.BaseForecaster],
    warm_up: int,
) -> None:
    history = make_series(150)
    expected = tuple(orchid().fit(history.complete_values()).resid()[warm_up:])
    result = forecaster.forecast(history, 3, levels=(0.95,))
    assert result.residuals == expected
    sigma = math.sqrt(math.fsum(e * e for e in expected) / len(expected))
    half_width = result.intervals[0].upper[0] - result.mean[0]
    assert half_width == pytest.approx(NormalDist().inv_cdf(0.975) * sigma, rel=1e-9)


@pytest.mark.parametrize(
    ("call", "error"),
    [
        pytest.param(lambda: SeasonalNaive(0), InvalidInputError, id="season_length=0"),
        pytest.param(lambda: MovingAverage(0), InvalidInputError, id="window=0"),
        pytest.param(lambda: WeightedMovingAverage([]), InvalidInputError, id="no weights"),
        pytest.param(
            lambda: WeightedMovingAverage([1.0, 0.0]), InvalidInputError, id="zero weight"
        ),
        pytest.param(
            lambda: WeightedMovingAverage([1.0, math.nan]), InvalidInputError, id="NaN weight"
        ),
        pytest.param(
            lambda: WeightedMovingAverage([1e308, 1e308]), InvalidInputError, id="infinite sum"
        ),
        pytest.param(  # a plain sum stays finite on Python 3.11; Orchid's fsum overflows
            lambda: WeightedMovingAverage([sys.float_info.max, 0.6 * 2**970, 0.6 * 2**970]),
            InvalidInputError,
            id="sum overflowing only when exact",
        ),
        pytest.param(lambda: ETS(damped=True), InvalidInputError, id="damped without trend"),
        pytest.param(
            lambda: ETS(trend="multiplicative"),  # type: ignore[arg-type]
            InvalidInputError,
            id="multiplicative trend",
        ),
        pytest.param(
            lambda: ETS(seasonal="additive"), InvalidInputError, id="season without length"
        ),
        pytest.param(lambda: AutoETS(season_length=1), InvalidInputError, id="season_length=1"),
        pytest.param(lambda: ARIMA(4, 0, 0), InvalidInputError, id="p=4"),
        pytest.param(lambda: ARIMA(0, 2, 0), InvalidInputError, id="d=2"),
        pytest.param(
            lambda: ARIMA(1, 0, 1, seasonal_differencing=1),
            InvalidInputError,
            id="seasonal difference without length",
        ),
        pytest.param(
            lambda: ARIMA(1, 0, 1, season_length=48),
            InvalidInputError,
            id="season length without difference",
        ),
        pytest.param(lambda: AutoARIMA(max_q=3), InvalidInputError, id="max_q=3"),
        pytest.param(
            lambda: AutoARIMA(season_length=2), InvalidInputError, id="seasonal test at length 2"
        ),
        pytest.param(lambda: TrendRegression(degree=3), InvalidInputError, id="degree=3"),
        pytest.param(lambda: SeasonalDummyRegression(336), InvalidInputError, id="336 dummies"),
        pytest.param(lambda: FourierRegression(48, 25), InvalidInputError, id="harmonics > m/2"),
        pytest.param(lambda: FourierRegression(336, 32), InvalidInputError, id="66 columns"),
        pytest.param(lambda: CalendarRegression(0), InvalidInputError, id="no harmonics"),
        pytest.param(
            lambda: CalendarRegression(24, weekly_harmonics=30),
            InvalidInputError,
            id="over 64 columns",
        ),
        pytest.param(lambda: Naive().forecast(SHORT, 0), InvalidInputError, id="horizon=0"),
        pytest.param(
            lambda: Naive().forecast(SHORT, 1, levels=(95,)), InvalidInputError, id="level=95"
        ),
        pytest.param(
            lambda: ETS(seasonal="multiplicative", season_length=2).forecast(SHORT, 1),
            InvalidInputError,
            id="multiplicative season on a zero",
        ),
        pytest.param(lambda: ARIMA(1, 0, 0).forecast(CONSTANT, 1), DegenerateDataError, id="ARIMA"),
        pytest.param(
            lambda: AutoARIMA().forecast(CONSTANT, 1), DegenerateDataError, id="AutoARIMA"
        ),
        pytest.param(
            lambda: ARIMA(1, 1, 0).forecast(LINE, 1),
            DegenerateDataError,
            id="ARIMA with a constant difference",
        ),
        pytest.param(
            lambda: AutoARIMA().forecast(LINE, 1),
            DegenerateDataError,
            id="AutoARIMA with a constant difference",
        ),
        pytest.param(
            lambda: ARIMA(1, 0, 0, seasonal_differencing=1, season_length=48).forecast(
                DAILY_REPEAT, 1
            ),
            DegenerateDataError,
            id="ARIMA with a constant seasonal difference",
        ),
        pytest.param(
            lambda: AutoARIMA(season_length=48).forecast(DAILY_REPEAT, 1),
            DegenerateDataError,
            id="AutoARIMA with a constant seasonal difference",
        ),
    ],
)
def test_errors_are_classified(call: Callable[[], object], error: type[InvalidInputError]) -> None:
    with pytest.raises(error) as excinfo:
        call()
    assert type(excinfo.value) is error


def test_seasonal_naive_repeats_last_season(make_series: MakeSeries) -> None:
    m, n = 48, 150
    values = make_series(n).complete_values()
    result = SeasonalNaive(m).forecast(make_series(n), m + 5)
    assert result.mean == tuple(values[n - m + i % m] for i in range(m + 5))


def test_additive_naive2_matches_closed_form(make_series: MakeSeries) -> None:
    m, n = 48, 250
    history = make_series(n, noise=0.5)
    values = history.complete_values()
    s = fb.seasonal_indices(values, m, model="additive")
    result = Naive2(m).forecast(history, 60)

    assert result.params["seasonal_adjustment"] is True
    assert result.mean == tuple(values[n - 1] - s[(n - 1) % m] + s[(n + i) % m] for i in range(60))


@pytest.mark.parametrize(
    ("forecaster", "n", "horizon", "expected"),
    [
        (SeasonalNaive(48), 192, 48, (CLOCK_CHANGE,)),
        (SeasonalNaive(48), 192, 1, ()),
        (CalendarRegression(2), 192, 48, ()),
        (Naive2(48), 192, 48, (CLOCK_CHANGE,)),
    ],
    ids=repr,
)
def test_clock_change_warning_follows_seasonal_lags(
    make_series: MakeSeries,
    forecaster: Forecaster,
    n: int,
    horizon: int,
    expected: tuple[str, ...],
) -> None:
    history = make_series(n, start=CLOCK_CHANGE_DAY - n * HALF_HOUR)
    assert forecaster.forecast(history, horizon).warnings == expected


def test_naive2_without_seasonality_is_naive_with_a_warning(make_series: MakeSeries) -> None:
    noise = make_series(192, amplitude=0.0, phi=0.0, start=CLOCK_CHANGE_DAY - 192 * HALF_HOUR)
    result = Naive2(48).forecast(noise, 48)
    assert result.params["seasonal_adjustment"] is False
    assert result.warnings == (NOT_ADJUSTED,)  # no clock-change warning: nothing is seasonal
    assert result.mean == Naive().forecast(noise, 48).mean


@pytest.mark.parametrize("forecaster", [Naive(), *REGRESSIONS], ids=repr)
def test_zero_width_intervals_are_flagged(forecaster: Forecaster) -> None:
    # Least squares leaves rounding-level widths, which must still count as zero.
    result = forecaster.forecast(HalfHourlySeries(START, [42.0] * 336), 3)
    assert result.warnings == ("zero-width intervals: residual variance is zero",)
    for interval in result.intervals:
        assert interval.lower == pytest.approx(result.mean, rel=1e-12)
        assert interval.upper == pytest.approx(result.mean, rel=1e-12)


def test_zero_width_tolerance_follows_the_data_not_the_forecast() -> None:
    line = HalfHourlySeries(START, [1000.0 - 5 * t for t in range(200)])  # the next value is 0
    result = TrendRegression().forecast(line, 1)
    assert result.warnings == ("zero-width intervals: residual variance is zero",)


@pytest.mark.parametrize(
    ("forecaster", "components", "method"),
    [
        (ETS(), {"alpha"}, "analytic"),
        (ETS(trend="additive", damped=True), {"alpha", "beta", "phi"}, "analytic"),
        (ETS(seasonal="multiplicative", season_length=48), {"alpha", "gamma"}, "sqrt-horizon"),
    ],
    # Not repr: a set's repr order changes with the process's string hash seed, which makes
    # pytest-xdist workers disagree about which tests exist.
    ids=["level", "damped-trend", "multiplicative-seasonal"],
)
def test_ets_reports_only_fitted_components(
    make_series: MakeSeries, forecaster: Forecaster, components: set[str], method: str
) -> None:
    result = forecaster.forecast(make_series(240), 4)
    assert set(result.params) == {"season_length", "method", *components}
    assert result.interval_method == method


@pytest.mark.parametrize(
    ("forecaster", "minimum"),
    [
        (AutoARIMA(season_length=120, max_p=1, max_q=1), 720),  # six seasons to test D
        (AutoARIMA(season_length=120, seasonal_differencing=1, max_p=1, max_q=1), 221),
    ],
    ids=repr,
)
def test_auto_arima_min_history_leaves_room_for_the_seasonal_test(
    forecaster: Forecaster, minimum: int
) -> None:
    assert forecaster.min_history == minimum


def _number(params: Mapping[str, object], key: str) -> float:
    value = params[key]
    assert isinstance(value, float)
    return value


def _modelled_length(params: Mapping[str, object], n: int) -> int:
    """The number of observations left after the fit's regular and seasonal differences."""
    order, seasonal = params["order"], params["seasonal_order"]
    assert isinstance(order, tuple)
    assert isinstance(seasonal, tuple)
    return n - int(order[1]) - int(seasonal[1]) * int(seasonal[3])


@pytest.mark.parametrize(
    ("forecaster", "trend"), [(ARIMA(1, 0, 1), "c"), (ARIMA(1, 1, 0), "n")], ids=repr
)
def test_arima_reports_the_orchid_fit_in_data_units(
    make_series: MakeSeries, forecaster: ARIMA, trend: str
) -> None:
    values = make_series(240).complete_values()
    raw = ar.ARIMA(
        values, order=(forecaster.p, forecaster.d, forecaster.q), trend=trend, boxcox_lambda=None
    ).fit(method="css", enforce_stationarity=True, enforce_invertibility=True, seed=0)
    result = forecaster.forecast(HalfHourlySeries(START, values), 4)
    assert result.residuals == pytest.approx(raw.resid, rel=1e-4, abs=1e-6)
    assert result.params["coefficients"] == pytest.approx(
        dict(zip(raw.param_names, raw.params, strict=True)), rel=1e-4
    )
    assert _number(result.params, "sigma2") == pytest.approx(raw.sigma2, rel=1e-4)
    assert _number(result.params, "aicc") == pytest.approx(raw.aicc, rel=1e-4)


@pytest.mark.parametrize(
    "forecaster", [ARIMA(1, 0, 1), ARIMA(1, 1, 0), AutoARIMA(max_p=1, max_q=1)], ids=repr
)
def test_arima_results_scale_with_the_data(make_series: MakeSeries, forecaster: Forecaster) -> None:
    # Fitted directly, data this small would put sigma2 at Orchid's absolute 1e-12 floor.
    values = make_series(240).complete_values()
    fits = {
        scale: forecaster.forecast(HalfHourlySeries(START, [v * scale for v in values]), 4)
        for scale in (1.0, 1e-7, 1e3)
    }
    nobs = _modelled_length(fits[1.0].params, len(values))
    for scale in (1e-7, 1e3):
        assert fits[scale].mean == pytest.approx([v * scale for v in fits[1.0].mean], rel=1e-6)
        sigma2 = _number(fits[1.0].params, "sigma2") * scale**2
        assert _number(fits[scale].params, "sigma2") == pytest.approx(sigma2, rel=1e-6)
        shift = _number(fits[scale].params, "aicc") - _number(fits[1.0].params, "aicc")
        assert shift == pytest.approx(2 * nobs * math.log(scale), rel=1e-6)


def test_auto_arima_runs_its_difference_tests_on_scaled_data(make_series: MakeSeries) -> None:
    # Orchid's tests have absolute floors that data in tiny units would hit.
    noise = make_series(288, level=0.0, amplitude=0.0, phi=0.0, noise=1e-7)
    seasonal = AutoARIMA(season_length=48, max_p=1, max_q=1).forecast(noise, 1)
    assert seasonal.params["seasonal_order"] == (0, 0, 0, 0)
    rng = random.Random(0)
    walk = [0.0]
    for _ in range(239):
        walk.append(walk[-1] + rng.gauss(0.0, 1e-8))
    regular = AutoARIMA(max_p=1, max_q=1).forecast(HalfHourlySeries(START, walk), 1)
    order = regular.params["order"]
    assert isinstance(order, tuple)
    assert order[1] == 1


@pytest.mark.parametrize("forecaster", [ARIMA(1, 1, 0), AutoARIMA(max_p=1, max_q=1)], ids=repr)
def test_arima_scales_by_the_differenced_history(forecaster: Forecaster) -> None:
    # The trend sets the levels' range; scaling by it would floor sigma2 at Orchid's 1e-12.
    rng = random.Random(0)
    values = [1000.0 * t + rng.gauss(0.0, 1e-3) for t in range(240)]
    params = forecaster.forecast(HalfHourlySeries(START, values), 2).params
    order, trend = params["order"], params["trend"]
    assert isinstance(order, tuple)
    assert isinstance(trend, str)
    assert order[1] == 1
    raw = ar.ARIMA(values, order=order, trend=trend, boxcox_lambda=None).fit(
        method="css", enforce_stationarity=True, enforce_invertibility=True, seed=0
    )
    assert _number(params, "sigma2") == pytest.approx(raw.sigma2, rel=1e-3)


def test_arima_rejects_fits_at_orchids_sigma2_floor() -> None:
    rng = random.Random(0)
    values = [1e4 * math.sin(2 * math.pi * t / 48) + rng.gauss(0.0, 1e-3) for t in range(240)]
    with pytest.raises(CalculationError):
        ARIMA(2, 0, 0).forecast(HalfHourlySeries(START, values), 2)


def test_arima_reports_overflow_in_the_datas_units_as_calculation_error(
    make_series: MakeSeries,
) -> None:
    huge = HalfHourlySeries(START, [v * 1e160 for v in make_series(240).complete_values()])
    with pytest.raises(CalculationError):  # the range's square overflows
        ARIMA(1, 0, 0).forecast(huge, 1)
    rng = random.Random(0)
    steep = [2e159 * t + 1e153 * rng.gauss(0.0, 1.0) for t in range(240)]
    with pytest.raises(CalculationError):  # sigma2 overflows once rescaled
        ARIMA(0, 1, 1).forecast(HalfHourlySeries(START, steep), 1)


def test_calendar_regression_needs_enough_distinct_clock_slots() -> None:
    # From this start, the clocks skip local 01:00 and 01:30 before either recurs, leaving
    # 46 distinct slots for 24 daily harmonics (47 columns) and the intercept.
    forecaster = CalendarRegression(24)
    start = datetime(2026, 3, 28, 2, tzinfo=UTC)
    history = HalfHourlySeries(start, [math.sin(t) for t in range(forecaster.min_history)])
    with pytest.raises(InsufficientDataError) as excinfo:
        forecaster.forecast(history, 1)
    assert (excinfo.value.required, excinfo.value.available) == (48, 46)


def test_auto_arima_reports_the_selected_fit(make_series: MakeSeries) -> None:
    params = AutoARIMA(max_p=1, max_q=1).forecast(make_series(240), 4).params
    coefficients = params["coefficients"]
    assert isinstance(coefficients, dict)
    assert ("const" in coefficients) == (params["trend"] == "c")
    assert {"sigma2", "aicc", "n_models"} <= set(params)


@pytest.mark.parametrize(
    ("forecaster", "trend", "names"),
    [
        (ARIMA(1, 0, 1), "c", ["ar.L1", "ma.L1", "const"]),
        (ARIMA(1, 1, 1), "n", ["ar.L1", "ma.L1"]),
        (ARIMA(1, 0, 1, seasonal_differencing=1, season_length=48), "n", ["ar.L1", "ma.L1"]),
    ],
    ids=repr,
)
def test_arima_has_a_constant_only_when_undifferenced(
    make_series: MakeSeries, forecaster: Forecaster, trend: str, names: list[str]
) -> None:
    params = forecaster.forecast(make_series(240), 4).params
    coefficients = params["coefficients"]
    assert isinstance(coefficients, dict)
    assert (params["trend"], list(coefficients)) == (trend, names)


def test_calendar_regression_keeps_local_clock_phase_across_clock_change() -> None:
    def price(period: datetime) -> float:
        local = period.astimezone(LONDON)
        slot = 2 * local.hour + local.minute // 30
        return 50 + 10 * math.sin(2 * math.pi * slot / 48) + 4 * math.cos(4 * math.pi * slot / 48)

    # Four BST days of history; the forecast runs through the first two GMT days.
    start = CLOCK_CHANGE_DAY - 192 * HALF_HOUR
    history = HalfHourlySeries(start, [price(start + i * HALF_HOUR) for i in range(192)])
    result = CalendarRegression(2).forecast(history, 96)
    assert result.mean == pytest.approx([price(period) for period in result.timestamps()], abs=1e-6)
