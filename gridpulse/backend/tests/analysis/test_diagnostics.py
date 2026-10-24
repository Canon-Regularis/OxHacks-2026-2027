import math
import random
import statistics
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from gridpulse.analysis.algorithms.diagnostics import (
    Correlogram,
    CrossCorrelogram,
    autocorrelation,
    cross_correlation,
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
from gridpulse.analysis.algorithms.orchid import autoregressive as ar
from gridpulse.analysis.algorithms.orchid import spectral as sp
from gridpulse.analysis.algorithms.series import HALF_HOUR, HalfHourlySeries

START = datetime(2026, 10, 1, tzinfo=UTC)
Z_95 = statistics.NormalDist().inv_cdf(0.975)
MakeSeries = Callable[..., HalfHourlySeries]
MakeValues = Callable[..., list[float]]

VARYING = HalfHourlySeries(START, [math.sin(i) for i in range(20)])
CORRELOGRAMS: dict[str, Callable[..., Correlogram | CrossCorrelogram]] = {
    "acf": autocorrelation,
    "pacf": partial_autocorrelation,
    "ccf_response": lambda series, **options: cross_correlation(series, VARYING, **options),
    "ccf_driver": lambda series, **options: cross_correlation(VARYING, series, **options),
}


def _demeaned(series: HalfHourlySeries) -> list[float]:
    values = series.complete_values()
    mean = statistics.fmean(values)
    return [value - mean for value in values]


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
