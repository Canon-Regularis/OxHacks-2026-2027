import cmath
import math
import random
import statistics
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from typing import Literal

import pytest

from gridpulse.analysis.algorithms.diagnostics import (
    Correlogram,
    CrossCorrelogram,
    DifferenceOrders,
    UnitRootTest,
    adf_test,
    ar_spectrum,
    autocorrelation,
    cross_correlation,
    difference_orders,
    kpss_test,
    partial_autocorrelation,
    residual_diagnostics,
)
from gridpulse.analysis.algorithms.errors import (
    CalculationError,
    DegenerateDataError,
    InsufficientDataError,
    InvalidInputError,
    MissingDataError,
)
from gridpulse.analysis.algorithms.forecasting import AutoARIMA
from gridpulse.analysis.algorithms.orchid import autoregressive as ar
from gridpulse.analysis.algorithms.orchid import smoothing as sm
from gridpulse.analysis.algorithms.orchid import spectral as sp
from gridpulse.analysis.algorithms.series import HALF_HOUR, HalfHourlySeries

START = datetime(2026, 10, 1, tzinfo=UTC)
Z_95 = statistics.NormalDist().inv_cdf(0.975)
MakeSeries = Callable[..., HalfHourlySeries]
MakeValues = Callable[..., list[float]]
Regression = Literal["c", "ct"]

VARYING = HalfHourlySeries(START, [math.sin(i) for i in range(20)])
CORRELOGRAMS: dict[str, Callable[..., Correlogram | CrossCorrelogram]] = {
    "acf": autocorrelation,
    "pacf": partial_autocorrelation,
    "ccf_response": lambda series, **options: cross_correlation(series, VARYING, **options),
    "ccf_driver": lambda series, **options: cross_correlation(VARYING, series, **options),
}
UNIT_ROOT_TESTS: dict[str, Callable[..., UnitRootTest]] = {"adf": adf_test, "kpss": kpss_test}
# The stationarity diagnostics and the AR spectrum, with options valid for 60 values.
STATIONARITY_AND_SPECTRA: dict[str, Callable[[HalfHourlySeries], object]] = {
    "adf": lambda series: adf_test(series, lags=1),
    "kpss": lambda series: kpss_test(series, lags=1),
    "difference_orders": difference_orders,
    "ar_spectrum": lambda series: ar_spectrum(series, order=1),
}


def _demeaned(series: HalfHourlySeries) -> list[float]:
    values = series.complete_values()
    mean = statistics.fmean(values)
    return [value - mean for value in values]


def _levels(result: ar.UnitRootTestResult) -> dict[float, float]:
    """Orchid's critical values, whose float levels it annotates as strings."""
    return {float(level): value for level, value in result.critical_values.items()}


def test_ar1_acf_decays_geometrically_and_pacf_cuts_off(make_series: MakeSeries) -> None:
    n = 2000
    series = make_series(n, level=0.0, amplitude=0.0, phi=0.6, noise=1.0, seed=1)
    acf = autocorrelation(series, max_lag=4)
    pacf = partial_autocorrelation(series, max_lag=4)
    assert acf.lags == pacf.lags == (1, 2, 3, 4)
    assert acf.values == pytest.approx([0.6**k for k in acf.lags], abs=0.05)
    # Bartlett: se_k^2 = (1 + 2 * sum of r_j^2 for j < k) / n, so the lag-1 band is z / sqrt(n).
    earlier = [math.fsum(r * r for r in acf.values[: k - 1]) for k in acf.lags]
    assert acf.band == pytest.approx([Z_95 * math.sqrt((1 + 2 * s) / n) for s in earlier])
    assert pacf.values[0] == pytest.approx(0.6, abs=0.05)
    assert pacf.band == pytest.approx([Z_95 / math.sqrt(n)] * 4)
    assert all(abs(v) < band for v, band in zip(pacf.values[1:], pacf.band[1:], strict=True))
    assert (acf.method, pacf.method, acf.alpha) == ("bartlett", "white-noise", 0.05)
    z_90 = statistics.NormalDist().inv_cdf(0.95)
    assert autocorrelation(series, max_lag=1, alpha=0.1).band[0] == pytest.approx(
        z_90 / math.sqrt(n)
    )


def test_cross_correlation_peaks_where_the_driver_leads(synthetic_values: MakeValues) -> None:
    values = synthetic_values(1001, seed=2)
    driver = HalfHourlySeries(START, values[1:])
    response = HalfHourlySeries(START, values[:-1])  # response[t] = driver[t - 1]
    ccf = cross_correlation(response, driver, max_lag=3)
    peak, lag = max(zip(ccf.values, ccf.lags, strict=True))
    assert ccf.lags == (-3, -2, -1, 0, 1, 2, 3)
    assert lag == 1
    assert peak == pytest.approx(1.0, abs=0.01)
    assert ccf.band == pytest.approx([Z_95 / math.sqrt(1000)] * 7)
    assert ccf.alpha == 0.05


def test_cross_correlation_matches_spectral_oracle_and_mirrors_when_swapped(
    make_series: MakeSeries,
) -> None:
    response = make_series(300, seed=3)
    driver = make_series(300, level=-20.0, phi=0.9, seed=4)
    ccf = cross_correlation(response, driver, max_lag=10)
    x, y = _demeaned(response), _demeaned(driver)
    raw = sp.correlate(x, y)  # raw[k + n - 1] sums x[t + k] * y[t] over t
    scale = math.sqrt(math.fsum(v * v for v in x) * math.fsum(v * v for v in y))
    expected = [raw[k + len(x) - 1] / scale for k in ccf.lags]
    assert ccf.values == pytest.approx(expected, abs=1e-9)
    swapped = cross_correlation(driver, response, max_lag=10)
    assert swapped.values == pytest.approx(ccf.values[::-1], abs=1e-12)


def test_negating_negative_prices_keeps_acf_and_flips_ccf(make_series: MakeSeries) -> None:
    prices = make_series(500, level=0.0, seed=5)  # level 0 gives frequent negative prices
    negated = HalfHourlySeries(prices.start, [-value for value in prices.complete_values()])
    driver = make_series(500, seed=6)
    assert autocorrelation(negated, max_lag=48).values == pytest.approx(
        autocorrelation(prices, max_lag=48).values, abs=1e-12
    )
    flipped = [-value for value in cross_correlation(prices, driver, max_lag=5).values]
    assert cross_correlation(negated, driver, max_lag=5).values == pytest.approx(flipped, abs=1e-12)


def test_statistics_do_not_depend_on_units(synthetic_values: MakeValues) -> None:
    values = synthetic_values(400, level=0.0, amplitude=0.0, phi=0.6, noise=1.0, seed=11)
    tiny = [value * 1e-7 for value in values]  # Orchid's absolute 1e-12 floors would bite here
    series, tiny_series = HalfHourlySeries(START, values), HalfHourlySeries(START, tiny)
    for compute in (autocorrelation, partial_autocorrelation):
        assert compute(tiny_series, max_lag=5).values == pytest.approx(
            compute(series, max_lag=5).values, abs=1e-12
        )
    full = residual_diagnostics(values, ljung_box_lags=10)
    scaled = residual_diagnostics(tiny, ljung_box_lags=10)
    assert scaled.ljung_box.statistics == pytest.approx(full.ljung_box.statistics, rel=1e-9)
    assert scaled.durbin_watson == pytest.approx(full.durbin_watson, rel=1e-12)
    assert scaled.jarque_bera.statistic == pytest.approx(full.jarque_bera.statistic, rel=1e-9)
    assert scaled.arch_lm.statistic == pytest.approx(full.arch_lm.statistic, rel=1e-9)


@pytest.mark.parametrize("name", list(CORRELOGRAMS))
def test_correlograms_reject_missing_or_constant_values(name: str) -> None:
    gappy = HalfHourlySeries(START, [*VARYING.values[:5], None, *VARYING.values[6:]])
    with pytest.raises(MissingDataError):
        CORRELOGRAMS[name](gappy, max_lag=2)
    with pytest.raises(DegenerateDataError):
        CORRELOGRAMS[name](HalfHourlySeries(START, [-25.0] * 20), max_lag=2)


def test_correlograms_need_enough_values() -> None:
    short = HalfHourlySeries(START, [1.0, 2.0, 0.5])
    with pytest.raises(InsufficientDataError):
        partial_autocorrelation(short, max_lag=1)
    single = HalfHourlySeries(START, [1.0])
    with pytest.raises(InsufficientDataError):
        autocorrelation(single, max_lag=1)
    with pytest.raises(InsufficientDataError):
        cross_correlation(single, single, max_lag=1)


@pytest.mark.parametrize(("name", "largest"), [("acf", 19), ("pacf", 9), ("ccf_response", 19)])
def test_correlograms_validate_max_lag_and_alpha(name: str, largest: int) -> None:
    assert CORRELOGRAMS[name](VARYING, max_lag=largest).lags[-1] == largest
    invalid: list[dict[str, float]] = [
        {"max_lag": 0},
        {"max_lag": largest + 1},
        {"max_lag": 1, "alpha": 0.0},
        {"max_lag": 1, "alpha": 1.0},
        {"max_lag": 1, "alpha": 1e-17},  # no finite critical value
    ]
    for options in invalid:
        with pytest.raises(InvalidInputError):
            CORRELOGRAMS[name](VARYING, **options)


@pytest.mark.parametrize(
    "driver",
    [VARYING.window(0, 19), HalfHourlySeries(START + HALF_HOUR, VARYING.values)],
    ids=["shorter", "later_start"],
)
def test_cross_correlation_requires_the_same_periods(driver: HalfHourlySeries) -> None:
    with pytest.raises(InvalidInputError):
        cross_correlation(VARYING, driver, max_lag=2)


def test_orchid_failures_become_calculation_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise ZeroDivisionError("boom")

    monkeypatch.setattr(ar, "acf", broken)
    with pytest.raises(CalculationError) as excinfo:
        autocorrelation(VARYING, max_lag=2)
    assert isinstance(excinfo.value.__cause__, ZeroDivisionError)
    monkeypatch.setattr(ar, "ccf", lambda *args, **kwargs: ar.AcfResult([0], [0.0], se=[0.1]))
    with pytest.raises(CalculationError):
        cross_correlation(VARYING, VARYING, max_lag=2)


def test_ranges_that_overflow_raise_calculation_error() -> None:
    with pytest.raises(CalculationError):
        autocorrelation(HalfHourlySeries(START, [1e308, -1e308] * 10), max_lag=2)
    with pytest.raises(CalculationError):  # the mostly zero ARCH window's norm overflows
        residual_diagnostics([1e308 if i % 25 == 0 else 0.0 for i in range(500)], ljung_box_lags=10)


def test_white_noise_residuals_pass(synthetic_values: MakeValues) -> None:
    noise = synthetic_values(500, level=0.0, amplitude=0.0, phi=0.0, noise=1.0, seed=9)
    result = residual_diagnostics(noise, ljung_box_lags=10)
    assert result.ljung_box.lags == result.ljung_box.df == tuple(range(1, 11))
    assert all(p is not None and p > 0.01 for p in result.ljung_box.p_values)
    assert result.durbin_watson == pytest.approx(2.0, abs=0.2)
    assert result.jarque_bera.p_value > 0.01
    assert result.arch_lm.p_value > 0.01
    assert result.arch_lm.lags == 12


def test_autocorrelated_residuals_fail_ljung_box(synthetic_values: MakeValues) -> None:
    residuals = synthetic_values(500, level=0.0, amplitude=0.0, phi=0.6, noise=1.0, seed=8)
    result = residual_diagnostics(residuals, ljung_box_lags=10)
    p_value = result.ljung_box.p_values[-1]
    assert p_value is not None
    assert p_value < 0.01
    assert result.durbin_watson < 1.0


def test_jarque_bera_reports_population_skew_and_excess_kurtosis() -> None:
    rng = random.Random(0)
    sample = [rng.expovariate(1.0) for _ in range(500)]
    mean = statistics.fmean(sample)
    m2, m3, m4 = (statistics.fmean([(x - mean) ** k for x in sample]) for k in (2, 3, 4))
    jarque_bera = residual_diagnostics(sample, ljung_box_lags=5).jarque_bera
    assert jarque_bera.skew == pytest.approx(m3 / m2**1.5, rel=1e-9)
    assert jarque_bera.excess_kurtosis == pytest.approx(m4 / m2**2 - 3, rel=1e-9)


def test_ljung_box_lags_within_model_df_are_not_testable(synthetic_values: MakeValues) -> None:
    noise = synthetic_values(200, level=0.0, amplitude=0.0, phi=0.0, seed=9)
    ljung_box = residual_diagnostics(noise, ljung_box_lags=5, model_df=3).ljung_box
    assert ljung_box.df == (-2, -1, 0, 1, 2)
    assert ljung_box.p_values[:3] == (None, None, None)
    assert None not in ljung_box.p_values[3:]


def test_residual_diagnostics_need_30_residuals() -> None:
    residual_diagnostics([math.sin(i) for i in range(30)], ljung_box_lags=5)
    with pytest.raises(InsufficientDataError) as excinfo:
        residual_diagnostics([math.sin(i) for i in range(29)], ljung_box_lags=5)
    assert (excinfo.value.required, excinfo.value.available) == (30, 29)


@pytest.mark.parametrize(
    ("bad", "error"), [(None, MissingDataError), (math.inf, InvalidInputError)]
)
def test_residual_diagnostics_reject_missing_or_infinite_values(
    bad: float | None, error: type[Exception]
) -> None:
    residuals = [*(math.sin(i) for i in range(39)), bad]
    with pytest.raises(error) as excinfo:
        residual_diagnostics(residuals, ljung_box_lags=5)  # type: ignore[arg-type]
    assert excinfo.type is error


@pytest.mark.parametrize(
    "residuals",
    [
        [0.5] * 40,
        [1.0, -1.0] * 20,
        [*(math.sin(i) for i in range(12)), *([0.0] * 48)],  # flat after the 12 ARCH lags
    ],
    ids=["constant", "constant_squares", "constant_regression_tail"],
)
def test_residual_diagnostics_reject_degenerate_residuals(residuals: list[float]) -> None:
    with pytest.raises(DegenerateDataError):
        residual_diagnostics(residuals, ljung_box_lags=5)


@pytest.mark.parametrize("index", [3, 499], ids=["early", "last"])
def test_arch_lm_is_unaffected_by_an_outlier_at_the_edge(index: int) -> None:
    # Dividing by a range this outlier sets would push the other squares into Orchid's floors.
    rng = random.Random(1)
    residuals = [rng.gauss(0.0, 1.0) for _ in range(500)]
    residuals[index] = 3e3
    expected = ar.heteroskedasticity_test(residuals, lags=12)  # exact at unit scale
    arch_lm = residual_diagnostics(residuals, ljung_box_lags=10).arch_lm
    assert arch_lm.statistic == pytest.approx(expected.statistic, rel=1e-6)


def test_arch_lm_scales_a_mostly_zero_window_by_its_own_magnitude() -> None:
    # Large residuals before the regression window must not set the scale of its squares.
    rng = random.Random(2)
    residuals = [1e4 * rng.gauss(0.0, 1.0) for _ in range(12)]
    residuals += [rng.gauss(0.0, 1.0) if i % 3 == 0 else 0.0 for i in range(488)]
    expected = ar.heteroskedasticity_test(residuals, lags=12)  # exact at unit scale
    arch_lm = residual_diagnostics(residuals, ljung_box_lags=10).arch_lm
    assert arch_lm.statistic == pytest.approx(expected.statistic, rel=1e-6)


def test_arch_lm_rejects_squares_too_nearly_constant_for_orchids_ridge() -> None:
    rng = random.Random(3)
    residuals = [(-1) ** i * (1 + 2.5e-5 * rng.gauss(0.0, 1.0)) for i in range(500)]
    with pytest.raises(DegenerateDataError):
        residual_diagnostics(residuals, ljung_box_lags=10)


def test_residual_diagnostics_validate_lags() -> None:
    residuals = [math.sin(i) for i in range(40)]
    result = residual_diagnostics(residuals, ljung_box_lags=39, arch_lags=19)
    assert (result.ljung_box.lags[-1], result.arch_lm.lags) == (39, 19)
    invalid: list[dict[str, int]] = [
        {"ljung_box_lags": 0},
        {"ljung_box_lags": 40},
        {"ljung_box_lags": 5, "model_df": -1},
        {"ljung_box_lags": 5, "model_df": 5},
        {"ljung_box_lags": 5, "arch_lags": 0},
        {"ljung_box_lags": 5, "arch_lags": 20},  # n // 2 would fit the ARCH regression exactly
    ]
    for options in invalid:
        with pytest.raises(InvalidInputError):
            residual_diagnostics(residuals, **options)


@pytest.mark.parametrize(
    ("regression", "kpss_null"), [("c", "level stationary"), ("ct", "trend stationary")]
)
def test_unit_root_tests_equal_orchid_on_centred_range_scaled_data(
    make_series: MakeSeries, regression: Regression, kpss_null: str
) -> None:
    series = make_series(300, level=0.0, seed=12)  # crosses zero, often negative
    values = series.complete_values()
    spread = max(values) - min(values)
    centre = min(values) + spread / 2
    scaled = [(value - centre) / spread for value in values]
    adf = adf_test(series, lags=6, regression=regression)
    expected = ar.adf_test(scaled, maxlag=6, autolag=None, regression=regression)  # type: ignore[arg-type]
    assert (adf.test, adf.null_hypothesis, adf.regression) == ("adf", "unit root", regression)
    assert (adf.statistic, adf.lags, adf.nobs) == (expected.statistic, 6, 293)
    assert adf.critical_values == _levels(expected)
    assert list(adf.critical_values) == [0.01, 0.05, 0.1]
    kpss = kpss_test(series, lags=48, regression=regression)
    expected = ar.kpss_test(scaled, regression=regression, lags=48)
    assert (kpss.test, kpss.null_hypothesis, kpss.regression) == ("kpss", kpss_null, regression)
    assert (kpss.statistic, kpss.lags, kpss.nobs) == (expected.statistic, 48, 300)
    assert kpss.critical_values == _levels(expected)
    assert list(kpss.critical_values) == [0.01, 0.025, 0.05, 0.1]


@pytest.mark.parametrize("regression", ["c", "ct"])
@pytest.mark.parametrize("name", list(UNIT_ROOT_TESTS))
def test_unit_root_statistics_do_not_depend_on_units(
    synthetic_values: MakeValues, name: str, regression: Regression
) -> None:
    test = UNIT_ROOT_TESTS[name]
    values = synthetic_values(300, seed=5)
    expected = test(HalfHourlySeries(START, values), lags=6, regression=regression).statistic
    # Orchid's absolute floors shift its raw ADF and KPSS statistics in units this small, and its
    # ridges shift ADF by about 1e-4 relative at this offset (3e4 ranges) unless it is centred.
    for transformed in ([value * 1e-9 for value in values], [value + 1e6 for value in values]):
        result = test(HalfHourlySeries(START, transformed), lags=6, regression=regression)
        assert result.statistic == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("regression", ["c", "ct"])
def test_a_random_walk_has_a_unit_root_and_white_noise_does_not(
    make_series: MakeSeries, regression: Regression
) -> None:
    walk = make_series(300, level=0.0, amplitude=0.0, phi=1.0, noise=1.0, seed=3)
    noise = make_series(300, level=0.0, amplitude=0.0, phi=0.0, noise=1.0, seed=3)
    assert adf_test(walk, lags=4, regression=regression).rejected_at == ()
    assert kpss_test(walk, lags=12, regression=regression).rejected_at == (0.01, 0.025, 0.05, 0.1)
    assert adf_test(noise, lags=4, regression=regression).rejected_at == (0.01, 0.05, 0.1)
    assert kpss_test(noise, lags=12, regression=regression).rejected_at == ()


@pytest.mark.parametrize(
    ("name", "statistic", "rejected_at"),
    [
        ("adf", -3.5, (0.01, 0.05, 0.1)),
        ("adf", -2.861, (0.1,)),  # the 5% critical value itself
        ("adf", 5.0, ()),
        ("kpss", 0.5, (0.05, 0.1)),
        ("kpss", 0.739, (0.025, 0.05, 0.1)),  # the 1% critical value itself
        ("kpss", 0.1, ()),
    ],
)
def test_rejected_at_lists_the_levels_whose_critical_values_the_statistic_passes(
    monkeypatch: pytest.MonkeyPatch,
    make_series: MakeSeries,
    name: str,
    statistic: float,
    rejected_at: tuple[float, ...],
) -> None:
    orchid: Callable[..., ar.UnitRootTestResult] = getattr(ar, f"{name}_test")

    def fixed(*args: object, **kwargs: object) -> ar.UnitRootTestResult:
        return replace(orchid(*args, **kwargs), statistic=statistic)

    monkeypatch.setattr(ar, f"{name}_test", fixed)
    assert UNIT_ROOT_TESTS[name](make_series(40), lags=1).rejected_at == rejected_at


@pytest.mark.parametrize(
    ("regression", "n", "largest", "nobs"),
    [
        ("c", 31, 13, 17),
        ("ct", 31, 12, 18),  # the trend column costs a lag
        ("ct", 120, 48, 71),  # the cost cap
    ],
)
def test_adf_accepts_lags_up_to_the_largest_orchid_runs_unchanged(
    make_series: MakeSeries, regression: Regression, n: int, largest: int, nobs: int
) -> None:
    # Orchid would cap larger lags, or give a statistic of 0 for "insufficient data".
    series = make_series(n)
    result = adf_test(series, lags=largest, regression=regression)
    assert (result.lags, result.nobs) == (largest, nobs)
    for lags in (-1, largest + 1):
        with pytest.raises(InvalidInputError) as excinfo:
            adf_test(series, lags=lags, regression=regression)
        assert excinfo.type is InvalidInputError


@pytest.mark.parametrize("regression", ["c", "ct"])
def test_kpss_takes_at_most_a_fifth_as_many_lags_as_values(
    make_series: MakeSeries, regression: Regression
) -> None:
    series = make_series(240)
    assert kpss_test(series, lags=48, regression=regression).lags == 48
    for lags in (-1, 49):
        with pytest.raises(InvalidInputError) as excinfo:
            kpss_test(series, lags=lags, regression=regression)
        assert excinfo.type is InvalidInputError


@pytest.mark.parametrize("name", list(UNIT_ROOT_TESTS))
@pytest.mark.parametrize(
    "options",
    [
        {"lags": 2.0},
        {"lags": True},
        {"lags": 2, "regression": "n"},  # Orchid's ADF drops the constant; its KPSS uses "c"
        {"lags": 2, "regression": "ctt"},  # Orchid's ADF raises KeyError; its KPSS uses "c"
        {"lags": 2, "regression": None},
    ],
    ids=["float_lags", "bool_lags", "no_constant", "quadratic_trend", "no_regression"],
)
def test_unit_root_tests_reject_invalid_options(
    make_series: MakeSeries, name: str, options: dict[str, object]
) -> None:
    with pytest.raises(InvalidInputError) as excinfo:
        UNIT_ROOT_TESTS[name](make_series(60), **options)
    assert excinfo.type is InvalidInputError


@pytest.mark.parametrize("name", list(UNIT_ROOT_TESTS))
def test_unit_root_tests_need_30_values(make_series: MakeSeries, name: str) -> None:
    UNIT_ROOT_TESTS[name](make_series(30), lags=1)
    with pytest.raises(InsufficientDataError) as excinfo:
        UNIT_ROOT_TESTS[name](make_series(29), lags=1)
    assert (excinfo.value.required, excinfo.value.available) == (30, 29)


@pytest.mark.parametrize("name", list(STATIONARITY_AND_SPECTRA))
def test_stationarity_and_spectra_reject_missing_or_constant_values(
    make_series: MakeSeries, name: str
) -> None:
    gappy = list(make_series(60).values)
    gappy[7] = None
    with pytest.raises(MissingDataError):
        STATIONARITY_AND_SPECTRA[name](HalfHourlySeries(START, gappy))
    with pytest.raises(DegenerateDataError):
        STATIONARITY_AND_SPECTRA[name](HalfHourlySeries(START, [-25.0] * 60))


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("adf", [1e308, -1e308] * 30),  # the range overflows
        ("kpss", [1e308, -1e308] * 30),
        ("difference_orders", [1e308, -1e308] * 30),
        ("ar_spectrum", [1e160 * math.sin(i) for i in range(60)]),  # Burg's variance overflows
    ],
)
def test_stationarity_and_spectra_report_overflow_as_calculation_error(
    name: str, values: list[float]
) -> None:
    with pytest.raises(CalculationError):
        STATIONARITY_AND_SPECTRA[name](HalfHourlySeries(START, values))


def test_unit_root_tests_report_orchid_failures(
    monkeypatch: pytest.MonkeyPatch, make_series: MakeSeries
) -> None:
    series = make_series(40)
    insufficient = ar.UnitRootTestResult(
        "ADF", 0.0, None, 0, 0, {}, "c", "unit root (non-stationary)", False, ["insufficient data"]
    )
    monkeypatch.setattr(ar, "adf_test", lambda *args, **kwargs: insufficient)
    with pytest.raises(CalculationError):
        adf_test(series, lags=1)
    kpss: Callable[..., ar.UnitRootTestResult] = ar.kpss_test

    def non_finite(*args: object, **kwargs: object) -> ar.UnitRootTestResult:
        return replace(kpss(*args, **kwargs), statistic=math.nan)

    monkeypatch.setattr(ar, "kpss_test", non_finite)
    with pytest.raises(CalculationError):
        kpss_test(series, lags=1)

    def broken(*args: object, **kwargs: object) -> None:
        raise ZeroDivisionError("boom")

    monkeypatch.setattr(ar, "kpss_test", broken)
    with pytest.raises(CalculationError) as excinfo:
        kpss_test(series, lags=1)
    assert isinstance(excinfo.value.__cause__, ZeroDivisionError)


@pytest.mark.parametrize(
    ("options", "required"),
    [
        ({}, 30),
        ({"season_length": 48}, 288),  # six seasons to test for a seasonal difference
        ({"season_length": 3}, 33),  # 30 values left for KPSS after a seasonal difference
        ({"season_length": 48, "seasonal_differencing": 1}, 78),
    ],
    ids=["no_season", "tested_daily_season", "tested_short_season", "seasonal_difference"],
)
def test_difference_orders_need_enough_values_for_their_tests(
    make_series: MakeSeries, options: dict[str, int], required: int
) -> None:
    difference_orders(make_series(required), **options)
    with pytest.raises(InsufficientDataError) as excinfo:
        difference_orders(make_series(required - 1), **options)
    assert excinfo.value.required == required


@pytest.mark.parametrize(
    "options",
    [
        {"season_length": 2},  # Orchid's seasonal test fires for every series at length 2
        {"seasonal_differencing": 1},
        {"season_length": 48, "seasonal_differencing": 0},
        {"season_length": 48, "seasonal_differencing": 2},
        {"season_length": 1, "seasonal_differencing": 1},
        {"season_length": 48.0},
        {"season_length": 48, "seasonal_differencing": True},
    ],
    ids=[
        "tested_season_of_2",
        "difference_without_length",
        "length_without_difference",
        "two_seasonal_differences",
        "season_of_1",
        "float_season",
        "bool_difference",
    ],
)
def test_difference_orders_validate_the_season_as_auto_arima_does(
    make_series: MakeSeries, options: dict[str, object]
) -> None:
    with pytest.raises(InvalidInputError) as excinfo:
        AutoARIMA(**options)  # type: ignore[arg-type]
    assert excinfo.type is InvalidInputError
    with pytest.raises(InvalidInputError) as excinfo:
        difference_orders(make_series(300), **options)  # type: ignore[arg-type]
    assert excinfo.type is InvalidInputError


@pytest.mark.parametrize("scale", [1.0, 1e-7])  # Orchid's tests have absolute floors
def test_white_noise_needs_no_difference_and_a_random_walk_a_regular_one(
    make_series: MakeSeries, scale: float
) -> None:
    noise = make_series(288, level=0.0, amplitude=0.0, phi=0.0, noise=scale)
    walk = make_series(288, level=0.0, amplitude=0.0, phi=1.0, noise=scale)
    assert difference_orders(noise) == DifferenceOrders(regular=0, seasonal=0)
    assert difference_orders(noise, season_length=48) == DifferenceOrders(regular=0, seasonal=0)
    assert difference_orders(walk) == DifferenceOrders(regular=1, seasonal=0)


def test_difference_orders_respect_an_explicit_seasonal_difference(
    make_series: MakeSeries,
) -> None:
    noise = make_series(288, level=0.0, amplitude=0.0, phi=0.0)
    daily = make_series(288)
    assert difference_orders(noise, season_length=48, seasonal_differencing=1).seasonal == 1
    assert difference_orders(daily, season_length=48).seasonal == 1
    assert difference_orders(daily, seasonal_differencing=0).seasonal == 0


def test_an_exact_daily_repeat_needs_only_its_seasonal_difference() -> None:
    # The seasonal difference is identically zero, which AutoARIMA then rejects as constant.
    repeat = HalfHourlySeries(START, [float(t % 48) for t in range(78)])
    orders = difference_orders(repeat, season_length=48, seasonal_differencing=1)
    assert orders == DifferenceOrders(regular=0, seasonal=1)


def test_ar1_spectrum_matches_the_closed_form(make_series: MakeSeries) -> None:
    series = make_series(2000, level=0.0, amplitude=0.0, phi=0.6, noise=1.0, seed=1)
    spectrum = ar_spectrum(series, order=1, nfft=16)
    phi, _, sigma2 = sm.burg(series.complete_values(), 1)
    assert (spectrum.order, spectrum.coefficients) == (1, tuple(phi))
    assert spectrum.innovation_variance == sigma2
    assert spectrum.coefficients[0] == pytest.approx(0.6, abs=0.05)  # Orchid's sign convention
    assert spectrum.frequencies == pytest.approx([k / 16 for k in range(9)])
    # One-sided: each frequency strictly between 0 and 0.5 also carries its negative's power.
    weights = [1.0, *[2.0] * 7, 1.0]
    expected = [
        weight * sigma2 / abs(1 - phi[0] * cmath.exp(-2j * math.pi * f)) ** 2
        for weight, f in zip(weights, spectrum.frequencies, strict=True)
    ]
    assert spectrum.density == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("transform", "factor"),
    [
        (lambda value: value * 1e-7, 1e-14),
        (lambda value: value * 1e3, 1e6),
        (lambda value: -value, 1.0),
        (lambda value: value + 1e3, 1.0),
    ],
    ids=["x1e-7", "x1e3", "negated", "offset"],
)
def test_ar_spectrum_density_scales_with_the_square_of_the_units(
    make_series: MakeSeries, transform: Callable[[float], float], factor: float
) -> None:
    values = make_series(500, level=0.0).complete_values()
    expected = ar_spectrum(HalfHourlySeries(START, values), order=4, nfft=64)
    result = ar_spectrum(HalfHourlySeries(START, [transform(v) for v in values]), order=4, nfft=64)
    assert result.coefficients == pytest.approx(expected.coefficients, rel=1e-9)
    assert result.innovation_variance == pytest.approx(
        factor * expected.innovation_variance, rel=1e-9
    )
    assert result.density == pytest.approx([factor * d for d in expected.density], rel=1e-9)


def test_a_daily_cycle_peaks_at_one_cycle_per_48_half_hours(make_series: MakeSeries) -> None:
    spectrum = ar_spectrum(make_series(672, seed=3), order=48)
    assert len(spectrum.frequencies) == len(spectrum.density) == 673
    peak = max(range(len(spectrum.density)), key=spectrum.density.__getitem__)
    assert spectrum.frequencies[peak] == pytest.approx(1 / 48)


def test_ar_spectrum_validates_order_and_nfft(make_series: MakeSeries) -> None:
    series = make_series(20)
    assert ar_spectrum(series, order=9, nfft=2).frequencies == (0.0, 0.5)
    ar_spectrum(make_series(4), order=1)
    with pytest.raises(InsufficientDataError):
        ar_spectrum(make_series(3), order=1)
    invalid: list[dict[str, object]] = [
        {"order": 0},
        {"order": 10},  # beyond n // 2 - 1
        {"order": 2.0},
        {"order": True},
        {"order": 1, "nfft": 1},
        {"order": 1, "nfft": 64.0},
    ]
    for options in invalid:
        with pytest.raises(InvalidInputError) as excinfo:
            ar_spectrum(series, **options)  # type: ignore[arg-type]
        assert excinfo.type is InvalidInputError


@pytest.mark.parametrize(
    ("values", "order"),
    [
        ([(-1.0) ** t for t in range(40)], 2),
        ([50 + 10 * math.sin(2 * math.pi * t / 48) for t in range(672)], 48),
        ([math.cos(2 * math.pi * t / 48) for t in range(672)], 24),
    ],
    ids=["no_innovations", "rounding_innovations", "explosive"],
)
def test_ar_spectrum_rejects_an_exactly_predictable_series(values: list[float], order: int) -> None:
    # Once Burg's recursion predicts a series exactly it runs on rounding errors, which can
    # leave no innovations, innovations of rounding size or an explosive model.
    with pytest.raises(DegenerateDataError):
        ar_spectrum(HalfHourlySeries(START, values), order=order)


def test_ar_spectrum_accepts_poles_near_the_unit_circle() -> None:
    # An AR(2) daily cycle whose poles have modulus 0.99.
    phi = (2 * 0.99 * math.cos(2 * math.pi / 48), -(0.99**2))
    rng = random.Random(0)
    values = [0.0, 0.0]
    for _ in range(2000):
        values.append(phi[0] * values[-1] + phi[1] * values[-2] + rng.gauss(0.0, 1.0))
    spectrum = ar_spectrum(HalfHourlySeries(START, values), order=2)
    assert spectrum.coefficients == pytest.approx(phi, abs=0.02)
