"""Correlograms and residual diagnostics backed by Orchid (P0 #7, #8, #14).

Every statistic here is scale-invariant, so inputs are rescaled before Orchid sees them to
keep its absolute floors (1e-12 on variances, a 1e-10 ridge) from distorting the result:
by their range, except for ARCH-LM, which regresses squares and needs them of order one.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

from gridpulse.analysis.algorithms._checks import (
    finite_values,
    orchid_errors,
    require_fraction,
    require_int,
)
from gridpulse.analysis.algorithms.errors import (
    CalculationError,
    DegenerateDataError,
    InsufficientDataError,
    InvalidInputError,
)
from gridpulse.analysis.algorithms.orchid import autoregressive as ar
from gridpulse.analysis.algorithms.series import HalfHourlySeries

_MIN_RESIDUALS = 30
_MAX_ARCH_LAGS = 48  # the auxiliary regression costs O(n * lags^2)


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
    values = _at_least("residuals", finite_values("residuals", residuals), _MIN_RESIDUALS)
    n = len(values)
    require_int("ljung_box_lags", ljung_box_lags, minimum=1, maximum=n - 1)
    require_int("model_df", model_df, minimum=0, maximum=ljung_box_lags - 1)
    require_int("arch_lags", arch_lags, minimum=1, maximum=min(_MAX_ARCH_LAGS, n // 2 - 1))
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
    high, low = max(values), min(values)
    spread = high - low
    if not math.isfinite(spread):
        raise CalculationError(f"the range of {name} overflows a float")
    if spread <= 1e-12 * max(1.0, abs(high), abs(low)):
        raise DegenerateDataError(f"no variation in {name}")
    return [value / spread for value in values]


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
