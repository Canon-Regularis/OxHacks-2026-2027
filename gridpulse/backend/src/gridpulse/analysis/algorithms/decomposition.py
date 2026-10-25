"""Seasonal decompositions, component strengths and centred smoothers backed by Orchid
(P1 #18, #19).

Everything here is two-sided: each output depends on later values as well as earlier ones,
so use these for charts and retrospective analysis, never as causal features. Wherever
Orchid's absolute floors (1e-12 on trend values, variances and residual scales) would make
a result depend on the data's units, values are rescaled before Orchid sees them.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Literal

from gridpulse.analysis.algorithms._checks import orchid_errors, require_int
from gridpulse.analysis.algorithms.diagnostics import _scaled
from gridpulse.analysis.algorithms.errors import (
    CalculationError,
    InsufficientDataError,
    InvalidInputError,
)
from gridpulse.analysis.algorithms.forecasting import _finite
from gridpulse.analysis.algorithms.orchid import forecasting_baseline as fb
from gridpulse.analysis.algorithms.orchid import smoothing as sm
from gridpulse.analysis.algorithms.series import HalfHourlySeries
from gridpulse.analysis.algorithms.signals import _result, _runs


@dataclass(frozen=True)
class Decomposition:
    """Trend, seasonal and remainder components on the input's periods.

    Additive components sum to the series; multiplicative ones multiply to it, so their
    seasonal and remainder are ratios. ``seasonal_indices[j]`` is the seasonal effect at
    phase ``j`` (for STL, whose seasonal can change between cycles, its mean). Period ``t``
    has phase ``t % period``: phases count elapsed half-hours from the first value, so across
    a GB clock change they shift an hour against the local clock.
    """

    method: Literal["classical", "stl"]
    model: Literal["additive", "multiplicative"]
    period: int
    trend: HalfHourlySeries
    seasonal: HalfHourlySeries
    remainder: HalfHourlySeries
    seasonal_indices: tuple[float, ...]


@dataclass(frozen=True)
class ComponentStrength:
    """Wang, Smith and Hyndman's strengths from 0 to 1: one minus the remainder's variance over
    the variance of the remainder plus the trend (or seasonal) component."""

    trend: float
    seasonal: float


def classical_decomposition(
    series: HalfHourlySeries,
    *,
    period: int,
    model: Literal["additive", "multiplicative"] = "additive",
) -> Decomposition:
    """Moving-average trend, per-phase mean seasonal and the remainder.

    The trend is a centred moving average over one period (2 x period when it is even), so
    it and the remainder are ``None`` within ``period // 2`` of either end. Needs two cycles,
    so that every phase has detrended values. A multiplicative model needs positive values,
    the smallest above 1e-12 of the largest.
    """
    require_int("period", period, minimum=2)
    if model not in ("additive", "multiplicative"):
        raise InvalidInputError(f"model must be 'additive' or 'multiplicative', got {model!r}")
    values = _complete(series, period, cycles=2)
    scale = 1.0  # Orchid's additive decomposition has no floors
    if model == "multiplicative":
        # Orchid drops the ratio to any trend value at or below 1e-12. Divided by the largest
        # value, every trend value, a weighted mean of values, is at least the smallest.
        scale = max(values)
        if min(values) <= 1e-12 * scale:
            raise InvalidInputError(
                "a multiplicative decomposition needs positive values above 1e-12 of the largest"
            )
    scaled = [value / scale for value in values]
    with orchid_errors("classical_decomposition"):
        result = fb.classical_decomposition(scaled, period, model=model)
    half = period // 2
    return Decomposition(
        method="classical",
        model=model,
        period=period,
        trend=_component(series, result.trend, half, scale),
        seasonal=_component(series, result.seasonal, 0, 1.0),
        remainder=_component(series, result.residual, half, 1.0),
        seasonal_indices=_finite(result.seasonal_indices),
    )


def stl_decomposition(
    series: HalfHourlySeries,
    *,
    period: int,
    seasonal_window: int = 7,
    trend_window: int | None = None,
) -> Decomposition:
    """Orchid's simplified, robust STL: twice, LOWESS each phase's detrended values across
    cycles for the seasonal, then the deseasonalised series for the trend.

    The seasonal LOWESS spans ``seasonal_window`` cycles and the trend LOWESS ``trend_window``
    periods, by default Cleveland's smallest odd number at least
    ``1.5 * period / (1 - 1.5 / seasonal_window)`` (93 for period 48 and 7 cycles). Needs
    ``seasonal_window`` complete cycles and ``trend_window`` values. Without Cleveland's
    low-pass filter, the seasonal is centred over the whole series rather than each cycle.

    Each LOWESS re-weights once against outliers and shares the weakness described in
    ``lowess``: an outlier tens of standard deviations high can stay in the seasonal instead
    of the remainder, mainly in series of many cycles. Longer seasonal windows resist more.
    """
    require_int("period", period, minimum=2)
    _require_odd("seasonal_window", seasonal_window, minimum=5)
    if trend_window is None:
        trend_window = _default_trend_window(period, seasonal_window)
    else:
        _require_odd("trend_window", trend_window, minimum=5)
    values = _complete(series, period, cycles=seasonal_window)
    n = len(values)
    if trend_window > n:
        raise InsufficientDataError(trend_window, n, what="values for the trend window")
    offset, scale = _unit_range(values)
    with orchid_errors("stl_decomposition"):
        result = sm.stl_decomposition(
            [(value - offset) / scale for value in values],
            period,
            # Orchid's LOWESS fits the ceil(frac * len) nearest points. Each phase has n // period
            # or one more values, so with seasonal_window cycles these give exactly the windows.
            seasonal_frac=(seasonal_window - 0.5) / (n // period + 0.5),
            trend_frac=(trend_window - 0.5) / n,
            iterations=2,
            robust=True,
        )
    return Decomposition(
        method="stl",
        model="additive",
        period=period,
        trend=_component(series, result.trend, 0, scale, offset),
        seasonal=_component(series, result.seasonal, 0, scale),
        remainder=_component(series, result.resid, 0, scale),
        seasonal_indices=tuple(
            scale * fmean(result.seasonal[phase::period]) for phase in range(period)
        ),
    )


def component_strength(series: HalfHourlySeries, *, period: int) -> ComponentStrength:
    """Trend and seasonal strengths of an additive classical decomposition.

    Needs two cycles and some variation. The trend is a moving average over one period, so
    at short periods it follows the noise: white noise has a trend strength near 0.63 at
    period 2 but 0.02 at period 48. A strength is 0 where the component plus remainder has
    a standard deviation below 1e-6 of the range, so a pure cycle has no trend strength.
    """
    require_int("period", period, minimum=2)
    # Orchid's 1e-12 variance floors are absolute, so it sees values divided by their range.
    # Orchid's STL-based strengths (autoregressive) are biased: white noise scores about 0.4
    # at period 48 and 1 at period 2.
    scaled = _scaled("series", _complete(series, period, cycles=2))
    with orchid_errors("component strength"):
        strengths = (fb.trend_strength(scaled, period), fb.seasonal_strength(scaled, period))
    return ComponentStrength(*_finite(strengths))


def median_filter(series: HalfHourlySeries, window: int) -> HalfHourlySeries:
    """The median of each centred ``window``; it removes spikes up to ``window // 2`` periods
    wide, which a mean would smear over the window."""
    _require_odd("window", window)
    return _centred(series, window, lambda run: sm.median_filter(run, window))


def savitzky_golay(series: HalfHourlySeries, window: int, degree: int) -> HalfHourlySeries:
    """The centre of the least-squares polynomial of ``degree`` fitted to each centred
    ``window``, which keeps peaks that a moving average flattens.

    ``degree`` is 0 to 3 and below ``window - 1``, so that the fit smooths. An odd degree
    gives the same values as the even degree below it; degree 0 is the centred mean.
    """
    _require_odd("window", window)
    require_int("degree", degree, minimum=0, maximum=min(3, window - 2))
    return _centred(series, window, lambda run: sm.savitzky_golay(run, window, degree))


def lowess(
    series: HalfHourlySeries, window: int, *, robustness_iterations: int = 3
) -> HalfHourlySeries:
    """LOWESS: the centre of a tricube-weighted line fitted to the ``window`` nearest periods,
    refitted ``robustness_iterations`` times (R's ``iter``) with bisquare weights that
    discount outliers.

    Re-weighting uses the run's median absolute residual, so it makes every value depend on
    its whole run. Where re-weighting leaves a window no weight, Orchid, like R, keeps the raw
    value: an outlier that also discredits the neighbours whose fits it pulled survives
    intact, and returns at every odd-numbered re-weighting, because it then fits exactly.
    A 7-period window can pass an outlier of 50 standard deviations; wider ones resist more.
    """
    _require_odd("window", window, minimum=5)
    require_int("robustness_iterations", robustness_iterations, minimum=0)

    def smooth(run: list[float]) -> list[float]:
        # Orchid stops re-weighting once the median absolute residual is at most 1e-12.
        offset, scale = _unit_range(run)
        fitted = sm.lowess(
            [(value - offset) / scale for value in run],
            frac=(window - 0.5) / len(run),  # Orchid fits the ceil(frac * len) nearest points
            iterations=robustness_iterations + 1,  # Orchid also counts the first fit
        )
        return [offset + scale * value for value in fitted]

    return _centred(series, window, smooth)


def _complete(series: HalfHourlySeries, period: int, *, cycles: int) -> list[float]:
    """The series' values, which must be complete and span ``cycles`` cycles of ``period``."""
    values = series.complete_values()
    if len(values) < cycles * period:
        raise InsufficientDataError(cycles * period, len(values))
    return values


def _require_odd(name: str, value: object, *, minimum: int = 3) -> None:
    # Orchid cannot centre an even window: it widens or shifts it by one period. LOWESS
    # windows need 5: at 3, both neighbours get zero tricube weight, leaving the raw values.
    if require_int(name, value, minimum=minimum) % 2 == 0:
        raise InvalidInputError(f"{name} must be odd, got {value}")


def _default_trend_window(period: int, seasonal_window: int) -> int:
    """Cleveland's: the smallest odd integer >= 1.5 * period / (1 - 1.5 / seasonal_window)."""
    ceiling = -(-3 * period * seasonal_window // (2 * seasonal_window - 3))  # in exact integers
    return ceiling + 1 - ceiling % 2


def _unit_range(values: Sequence[float]) -> tuple[float, float]:
    """The offset and scale that map ``values`` onto [0, 1], or onto 0 if they are constant."""
    low = min(values)
    spread = max(values) - low
    if not math.isfinite(spread):
        raise CalculationError("the range of the values overflows a float")
    return low, spread or 1.0


def _component(
    series: HalfHourlySeries,
    values: Sequence[float | None],
    half: int,
    scale: float,
    offset: float = 0.0,
) -> HalfHourlySeries:
    """Orchid's ``values`` as ``offset + scale * value``, missing within ``half`` of either end."""
    n = len(series)
    out: list[float | None] = [None] * n
    for t in range(half, n - half):
        value = values[t]
        if value is None:
            raise CalculationError("Orchid left a component value missing")
        out[t] = offset + scale * value
    return _result(series, out)


def _centred(
    series: HalfHourlySeries, window: int, smooth: Callable[[list[float]], list[float]]
) -> HalfHourlySeries:
    """``smooth`` on each run of present values, ``None`` within ``window // 2`` of a run's
    ends, where Orchid shrinks or shifts the window, and in runs shorter than ``window``."""
    half = window // 2
    out: list[float | None] = [None] * len(series)
    for first, run in _runs(series.values):
        if len(run) >= window:
            with orchid_errors("smoother"):
                smoothed = smooth(run)
            out[first + half : first + len(run) - half] = smoothed[half:-half]
    return _result(series, out)
