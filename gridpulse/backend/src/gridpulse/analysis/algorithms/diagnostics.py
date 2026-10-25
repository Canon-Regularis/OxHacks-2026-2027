"""Correlograms, residual and stationarity diagnostics and AR spectra backed by Orchid
(P0 #7, #8, #14; P1 #16, #17).

Each result uses the whole input, so it is retrospective. The correlations and test
statistics do not depend on the data's units, so inputs are rescaled before Orchid sees them
to keep its absolute floors (1e-12 on variances, ridges of 1e-12 and 1e-10) from distorting
them: by their range (centred, for the unit-root tests), except for ARCH-LM, which regresses
squares and needs them of order one. Burg's method standardises the series itself.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Literal

from gridpulse.analysis.algorithms._checks import (
    check_difference_tests,
    finite_values,
    orchid_errors,
    require_fraction,
    require_int,
    require_variation,
    scaled_by_range,
)
from gridpulse.analysis.algorithms.errors import (
    CalculationError,
    DegenerateDataError,
    InsufficientDataError,
    InvalidInputError,
)
from gridpulse.analysis.algorithms.orchid import autoregressive as ar
from gridpulse.analysis.algorithms.orchid import smoothing as sm
from gridpulse.analysis.algorithms.orchid import spectral as sp
from gridpulse.analysis.algorithms.series import HalfHourlySeries

_MIN_OBSERVATIONS = 30  # fewer make the tests' asymptotic distributions unreliable
_MAX_REGRESSION_LAGS = 48  # a day; regressions on lags cost O(n * lags^2) in pure Python


@dataclass(frozen=True)
class Correlogram:
    """Correlations at lags 1..K and the half-widths of a zero-centred significance band.

    ``method`` names the band's standard error: "bartlett" or "white-noise".
    """

    lags: tuple[int, ...]
    values: tuple[float, ...]
    band: tuple[float, ...]
    alpha: float
    method: str


@dataclass(frozen=True)
class CrossCorrelogram:
    """Cross-correlations at lags -K..K; the value at lag k is corr(response[t + k], driver[t]).

    k > 0 means the driver leads the response by k half-hours. ``band`` holds z / sqrt(n) at
    every lag, a white-noise band that overstates significance between autocorrelated series.
    """

    lags: tuple[int, ...]
    values: tuple[float, ...]
    band: tuple[float, ...]
    alpha: float


@dataclass(frozen=True)
class LjungBox:
    """Q statistics at lags 1..K with ``df = lag - model_df``; ``None`` p-values mean df <= 0."""

    lags: tuple[int, ...]
    statistics: tuple[float, ...]
    p_values: tuple[float | None, ...]
    df: tuple[int, ...]


@dataclass(frozen=True)
class JarqueBera:
    statistic: float
    p_value: float
    skew: float
    excess_kurtosis: float


@dataclass(frozen=True)
class ArchLM:
    """Engle's test: squared residuals regressed on their previous ``lags`` values."""

    statistic: float
    p_value: float
    lags: int


@dataclass(frozen=True)
class ResidualDiagnostics:
    """P-values below about 1e-16 are reported as 0.0, because Orchid computes 1 - CDF."""

    ljung_box: LjungBox
    durbin_watson: float
    jarque_bera: JarqueBera
    arch_lm: ArchLM


@dataclass(frozen=True)
class UnitRootTest:
    """An ADF or KPSS statistic and its asymptotic critical values by significance level.

    ``rejected_at`` holds the levels at which the null is rejected, in ascending order.
    ``lags`` counts ADF's lagged differences or the autocovariances in KPSS's long-run
    variance; ``nobs`` counts the observations behind the statistic. There is no p-value:
    Orchid interpolates one between the three or four tabulated levels and clamps it to
    [0.005, 0.55], so an ADF statistic of +5 would report 0.55.
    """

    test: Literal["adf", "kpss"]
    null_hypothesis: str
    statistic: float
    critical_values: Mapping[float, float]
    rejected_at: tuple[float, ...]
    lags: int
    nobs: int
    regression: Literal["c", "ct"]


@dataclass(frozen=True)
class DifferenceOrders:
    regular: int
    seasonal: int


@dataclass(frozen=True)
class ArSpectrum:
    """One-sided spectral density of an AR(``order``) model fitted by Burg's method.

    Deviations from the mean follow x[t] = sum over k of coefficients[k - 1] * x[t - k] + e[t],
    with innovations e of variance ``innovation_variance``. ``density`` is in squared data
    units per cycle per half-hour at ``frequencies`` from 0 to 0.5 cycles per half-hour.
    Values strictly between 0 and 0.5 include the power at the matching negative frequency,
    so the density integrates over [0, 0.5] to the model's variance.
    """

    order: int
    coefficients: tuple[float, ...]
    innovation_variance: float
    frequencies: tuple[float, ...]
    density: tuple[float, ...]


def autocorrelation(series: HalfHourlySeries, *, max_lag: int, alpha: float = 0.05) -> Correlogram:
    """Sample autocorrelations; the band at lag k assumes an MA(k - 1) process (Bartlett)."""
    values = _at_least("series", series.complete_values(), 2)
    require_int("max_lag", max_lag, minimum=1, maximum=len(values) - 1)
    alpha, z = _alpha_and_z(alpha)
    scaled = _scaled("series", values)
    with orchid_errors("acf"):
        result = ar.acf(scaled, nlags=max_lag, alpha=alpha)
    correlations, band = _checked("acf", result, z, max_lag + 1)
    return Correlogram(tuple(range(1, max_lag + 1)), correlations[1:], band[1:], alpha, "bartlett")


def partial_autocorrelation(
    series: HalfHourlySeries, *, max_lag: int, alpha: float = 0.05
) -> Correlogram:
    """Partial autocorrelations by Levinson-Durbin on biased autocovariances; white-noise band."""
    values = _at_least("series", series.complete_values(), 4)
    require_int("max_lag", max_lag, minimum=1, maximum=len(values) // 2 - 1)
    alpha, z = _alpha_and_z(alpha)
    scaled = _scaled("series", values)
    with orchid_errors("pacf"):
        result = ar.pacf(scaled, nlags=max_lag, method="ld", alpha=alpha)
    correlations, band = _checked("pacf", result, z, max_lag + 1)
    return Correlogram(
        tuple(range(1, max_lag + 1)), correlations[1:], band[1:], alpha, "white-noise"
    )


def cross_correlation(
    response: HalfHourlySeries, driver: HalfHourlySeries, *, max_lag: int, alpha: float = 0.05
) -> CrossCorrelogram:
    """``response`` and ``driver`` must cover the same periods; see ``CrossCorrelogram``."""
    if response.start != driver.start or len(response) != len(driver):
        raise InvalidInputError(
            f"response ({len(response)} from {response.start.isoformat()}) and driver "
            f"({len(driver)} from {driver.start.isoformat()}) must cover the same periods"
        )
    response_values = _at_least("response", response.complete_values(), 2)
    driver_values = driver.complete_values()
    require_int("max_lag", max_lag, minimum=1, maximum=len(response_values) - 1)
    alpha, z = _alpha_and_z(alpha)
    scaled_response = _scaled("response", response_values)
    scaled_driver = _scaled("driver", driver_values)
    with orchid_errors("ccf"):
        result = ar.ccf(scaled_response, scaled_driver, nlags=max_lag, alpha=alpha)
    correlations, band = _checked("ccf", result, z, 2 * max_lag + 1)
    return CrossCorrelogram(tuple(range(-max_lag, max_lag + 1)), correlations, band, alpha)


def residual_diagnostics(
    residuals: Sequence[float], *, ljung_box_lags: int, model_df: int = 0, arch_lags: int = 12
) -> ResidualDiagnostics:
    """Ljung-Box, Durbin-Watson, Jarque-Bera and ARCH-LM statistics of model residuals.

    ``model_df``, the fitted AR + MA order, reduces the Ljung-Box degrees of freedom.
    ``arch_lags`` is at most 48 and leaves the auxiliary regression a residual degree of freedom.
    """
    values = _at_least("residuals", finite_values("residuals", residuals), _MIN_OBSERVATIONS)
    n = len(values)
    require_int("ljung_box_lags", ljung_box_lags, minimum=1, maximum=n - 1)
    require_int("model_df", model_df, minimum=0, maximum=ljung_box_lags - 1)
    require_int("arch_lags", arch_lags, minimum=1, maximum=min(_MAX_REGRESSION_LAGS, n // 2 - 1))
    scaled = _scaled("residuals", values)
    arch_input = _arch_scaled(values, arch_lags)
    with orchid_errors("residual diagnostics"):
        lb = ar.ljung_box(scaled, lags=ljung_box_lags, model_df=model_df)
        dw = ar.durbin_watson(scaled)
        jb = ar.jarque_bera(scaled)
        arch = ar.heteroskedasticity_test(arch_input, lags=arch_lags)
    estimates = (*lb.stat_lb, dw, jb.statistic, jb.skew, jb.kurtosis, arch.statistic)
    p_values = (*(p for p in lb.pvalue_lb if p is not None), jb.pvalue, arch.pvalue)
    if not all(map(math.isfinite, (*estimates, *p_values))):
        raise CalculationError("Orchid residual diagnostics returned non-finite values")
    return ResidualDiagnostics(
        LjungBox(tuple(lb.lags), tuple(lb.stat_lb), tuple(lb.pvalue_lb), tuple(lb.df)),
        dw,
        JarqueBera(jb.statistic, jb.pvalue, jb.skew, jb.kurtosis),
        ArchLM(arch.statistic, arch.pvalue, arch.lags),
    )


def adf_test(
    series: HalfHourlySeries, *, lags: int, regression: Literal["c", "ct"] = "c"
) -> UnitRootTest:
    """Augmented Dickey-Fuller test of a unit root, with a constant ("c") or a constant and a
    linear trend ("ct") and ``lags`` lagged differences in the test regression.

    Needs 30 values. ``lags`` is fixed, because Orchid's AIC selection compares fits on
    different samples; it is at most 48 and at most (n - 5) // 2, or (n - 6) // 2 with a trend.
    """
    _check_regression(regression)
    values = _at_least("series", series.complete_values(), _MIN_OBSERVATIONS)
    trend = regression == "ct"
    # Orchid regresses n - lags - 1 rows on lags + 2 + trend columns and needs two more rows
    # than columns; that also keeps lags within its cap of n // 2 - 2.
    largest = min(_MAX_REGRESSION_LAGS, (len(values) - 5 - trend) // 2)
    require_int("lags", lags, minimum=0, maximum=largest)
    scaled = _centred("series", values)
    with orchid_errors("adf_test"):
        # None turns off lag selection, as in Orchid's own tests; its annotation omits it.
        result = ar.adf_test(scaled, maxlag=lags, autolag=None, regression=regression)  # type: ignore[arg-type]
    return _unit_root_test("adf", "unit root", result, regression)


def kpss_test(
    series: HalfHourlySeries, *, lags: int, regression: Literal["c", "ct"] = "c"
) -> UnitRootTest:
    """KPSS test of stationarity around a constant ("c") or a linear trend ("ct").

    Needs 30 values. The long-run variance weights autocovariances up to ``lags`` with
    Bartlett weights. ``lags`` is at most n // 5, because the critical values assume far
    fewer lags than values: from n - 2 lags the statistic is (lags + 1) / 2n whatever the
    data. Pass at least 48 for half-hourly data, so at least 240 values: Orchid's default of
    7 to 10 lags (for four weeks to 92 days) understates the long-run variance of such
    persistent series, so the test rejects stationarity too often.
    """
    _check_regression(regression)
    values = _at_least("series", series.complete_values(), _MIN_OBSERVATIONS)
    require_int("lags", lags, minimum=0, maximum=len(values) // 5)
    scaled = _centred("series", values)
    with orchid_errors("kpss_test"):
        result = ar.kpss_test(scaled, regression=regression, lags=lags)
    null_hypothesis = "level stationary" if regression == "c" else "trend stationary"
    return _unit_root_test("kpss", null_hypothesis, result, regression)


def difference_orders(
    series: HalfHourlySeries,
    *,
    season_length: int | None = None,
    seasonal_differencing: int | None = None,
) -> DifferenceOrders:
    """The regular and seasonal differences, each 0 or 1, that ``AutoARIMA`` applies.

    ``seasonal_differencing=None`` tests for a seasonal difference when ``season_length`` is
    given. Orchid's test (STL seasonal strength of at least 0.64) fires for every series at
    season length 2 and often for white noise over fewer than six seasons, so testing needs
    a season length of at least 3 and six seasons. It also fires on integrated series
    without a seasonal cycle, so give ``seasonal_differencing`` when it is known. The regular
    difference is then tested on the seasonally differenced series, which needs 30 values,
    by KPSS at the 5% level with Orchid's default lags; see ``kpss_test`` for why they can
    over-difference persistent series.
    """
    m = season_length
    check_difference_tests(seasonal_differencing, m)
    required = _MIN_OBSERVATIONS + (m or 0)
    if m is not None and seasonal_differencing is None:
        required = max(required, 6 * m)
    values = _at_least("series", series.complete_values(), required)
    require_variation("series", values)
    _, unit = scaled_by_range(values, values)  # Orchid's tests have absolute floors
    with orchid_errors("difference tests"):
        seasonal = seasonal_differencing
        if seasonal is None:
            seasonal = ar.nsdiffs(unit, m=m, max_D=1) if m is not None else 0
        period = (m or 1) if seasonal else 1
        regular = ar.ndiffs(ar.diff(unit, d=0, D=seasonal, m=period), max_d=1)
    return DifferenceOrders(regular, seasonal)


def ar_spectrum(series: HalfHourlySeries, *, order: int, nfft: int = 1344) -> ArSpectrum:
    """The spectrum at the ``nfft // 2 + 1`` frequencies k / nfft; see ``ArSpectrum``.

    ``order`` is at most n // 2 - 1. Each spectral peak takes two coefficients, so an order of
    48 can resolve every harmonic of a daily cycle. The default ``nfft`` is a multiple of 336,
    which puts daily and weekly harmonics on the grid.
    """
    values = _at_least("series", series.complete_values(), 4)
    require_int("order", order, minimum=1, maximum=len(values) // 2 - 1)
    require_int("nfft", nfft, minimum=2)
    spread = _spread("series", values)  # Burg would silently fit a flat model to a constant series
    with orchid_errors("AR spectrum"):
        coefficients, _, innovation_variance = sm.burg(values, order)
        frequencies, density = sp.ar_psd(
            coefficients, innovation_variance, nfft=nfft, fs=1.0, onesided=True
        )
    if not all(map(math.isfinite, (*coefficients, innovation_variance, *density))):
        raise CalculationError("Orchid's AR spectrum has non-finite values")
    # Burg's model is stable and has innovations unless the series is exactly predictable,
    # when its recursion runs on rounding errors. Squaring the range could overflow.
    if innovation_variance / spread / spread <= 1e-12 or not _stable(coefficients):
        raise DegenerateDataError(
            f"the series is exactly predictable from its last {order} values, so it has no "
            "spectral density"
        )
    return ArSpectrum(
        order, tuple(coefficients), innovation_variance, tuple(frequencies), tuple(density)
    )


def _at_least(name: str, values: list[float], minimum: int) -> list[float]:
    if len(values) < minimum:
        raise InsufficientDataError(minimum, len(values), what=f"values in {name}")
    return values


def _alpha_and_z(alpha: float) -> tuple[float, float]:
    alpha = require_fraction("alpha", alpha)
    z = ar.norm_ppf(1.0 - alpha / 2.0)
    if not math.isfinite(z):
        raise InvalidInputError(f"alpha is too small for a finite critical value, got {alpha!r}")
    return alpha, z


def _scaled(name: str, values: Sequence[float]) -> list[float]:
    spread = _spread(name, values)
    return [value / spread for value in values]


def _centred(name: str, values: Sequence[float]) -> list[float]:
    """``values`` less their midrange, divided by their range.

    The ADF and KPSS statistics do not change, but Orchid's absolute ridges on the test
    regressions would otherwise shift them with the offset: ADF's by 1e-4 relative at an
    offset of 3e4 times the range, and by 1% at 3e5 times.
    """
    spread = _spread(name, values)
    centre = min(values) + spread / 2  # the mean could overflow
    return [(value - centre) / spread for value in values]


def _spread(name: str, values: Sequence[float]) -> float:
    """The range of ``values``; ``DegenerateDataError`` if it is at most 1e-12 times their
    magnitude (or 1), the test by which Orchid's Burg fit treats a series as constant."""
    high, low = max(values), min(values)
    spread = high - low
    if not math.isfinite(spread):
        raise CalculationError(f"the range of {name} overflows a float")
    if spread <= 1e-12 * max(1.0, abs(high), abs(low)):
        raise DegenerateDataError(f"no variation in {name}")
    return spread


def _check_regression(regression: object) -> None:
    if regression not in ("c", "ct"):
        raise InvalidInputError(f"regression must be 'c' or 'ct', got {regression!r}")


def _unit_root_test(
    test: Literal["adf", "kpss"],
    null_hypothesis: str,
    result: ar.UnitRootTestResult,
    regression: Literal["c", "ct"],
) -> UnitRootTest:
    statistic = result.statistic
    if result.pvalue is None or not math.isfinite(statistic):  # None: "insufficient data"
        raise CalculationError(f"Orchid {test}_test returned no finite statistic")
    # Orchid annotates the keys as strings; they are the float significance levels.
    critical = {float(level): value for level, value in sorted(result.critical_values.items())}
    # ADF rejects a unit root below a critical value; KPSS rejects stationarity above one.
    rejected_at = tuple(
        level
        for level, value in critical.items()
        if (statistic < value if test == "adf" else statistic > value)
    )
    return UnitRootTest(
        test=test,
        null_hypothesis=null_hypothesis,
        statistic=statistic,
        critical_values=critical,
        rejected_at=rejected_at,
        lags=result.usedlag,
        nobs=result.nobs,
        regression=regression,
    )


def _arch_scaled(values: Sequence[float], lags: int) -> list[float]:
    """``values`` divided by the median magnitude of the ARCH-LM regression's response.

    The test regresses squares on lagged squares. A range set by one outlier would shrink
    the other squares towards Orchid's 1e-10 ridge and 1e-12 floor, so this scale makes
    them of order one, and nearly constant squares are rejected.
    """
    window = values[lags:]
    typical = median(abs(value) for value in window)
    if not typical:  # most of the window is exactly zero: use its root mean square
        typical = math.hypot(*window) / math.sqrt(len(window))
    if not math.isfinite(typical):
        raise CalculationError("the residuals overflow a float")
    if not typical:
        raise DegenerateDataError("the ARCH-LM regression's squared residuals are all zero")
    scaled = [value / typical for value in values]
    squares = [value * value for value in scaled[lags:]]
    centre = sum(squares) / len(squares)
    # Far above Orchid's 1e-10 ridge, far below realistic windows (about 2 per residual).
    if sum((square - centre) * (square - centre) for square in squares) <= 1e-4:
        raise DegenerateDataError("the squared residuals vary too little for ARCH-LM")
    return scaled


def _checked(
    source: str, result: ar.AcfResult, z: float, length: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Orchid's values and band half-widths z * se, if there are ``length`` finite ones of each."""
    values = tuple(result.values)
    band = tuple(z * se for se in result.se or ())
    if len(values) != length or len(band) != length:
        raise CalculationError(
            f"Orchid {source} returned {len(values)} values and {len(band)} standard errors "
            f"where {length} of each were expected"
        )
    if not all(map(math.isfinite, values + band)):
        raise CalculationError(f"Orchid {source} returned non-finite values")
    return values, band


def _stable(coefficients: Sequence[float]) -> bool:
    """Whether every pole lies inside the unit circle: the step-down recursion's reflection
    coefficients all lie strictly between -1 and 1 (Schur-Cohn)."""
    phi = list(coefficients)
    while phi:
        k = phi[-1]
        if not abs(k) < 1:
            return False
        phi = [(phi[j] + k * phi[-2 - j]) / (1 - k * k) for j in range(len(phi) - 1)]
    return True
