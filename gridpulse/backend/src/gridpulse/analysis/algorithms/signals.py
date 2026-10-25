"""Causal rolling features backed by Orchid (P0 #3, #4, #5; P1 #19).

Each function returns a series with its input's start and length. ``output[t]`` depends
only on inputs up to ``t`` and is ``None`` while its window is incomplete or touches a
missing period. Windows and lags count elapsed half-hours on the UTC grid, so across a GB
clock change lag 48 is 24 hours earlier rather than the same clock time.
"""

import math
from collections.abc import Callable, Iterator, Sequence
from typing import cast

from gridpulse.analysis.algorithms._checks import (
    orchid_errors,
    require_int,
    require_real,
    require_weights,
)
from gridpulse.analysis.algorithms.errors import CalculationError, InvalidInputError
from gridpulse.analysis.algorithms.orchid import forecasting_baseline as fb
from gridpulse.analysis.algorithms.orchid import smoothing as sm
from gridpulse.analysis.algorithms.orchid import spectral as sp
from gridpulse.analysis.algorithms.series import HalfHourlySeries


def trailing_mean(series: HalfHourlySeries, window: int) -> HalfHourlySeries:
    require_int("window", window, minimum=1)
    return _per_run(
        series, window - 1, lambda run: sm.simple_moving_average(run, window, center=False)
    )


def trailing_weighted_mean(series: HalfHourlySeries, weights: Sequence[float]) -> HalfHourlySeries:
    """Weights run oldest to newest: the last one applies to the newest value."""
    checked = list(require_weights(weights))
    return _per_run(series, len(checked) - 1, lambda run: sm.weighted_moving_average(run, checked))


def trailing_savitzky_golay(series: HalfHourlySeries, window: int, degree: int) -> HalfHourlySeries:
    """The newest point of the least-squares polynomial of ``degree`` (0 to 3) fitted to each
    trailing ``window``, of more than ``degree + 1`` values so that the fit smooths.

    Degree 0 is the trailing mean; higher degrees follow trends with less lag but more noise.
    """
    require_int("degree", degree, minimum=0, maximum=3)
    require_int("window", window, minimum=degree + 2)
    # The newest fitted value weights the window by the newest row of the hat matrix, which
    # is symmetric, so that row is the fit to the newest unit vector: one solve, not one per
    # weight.
    design = fb.trend_features(window, degree, 1 - window)
    with orchid_errors("least squares"):
        weights = fb.matvec(design, fb.lstsq(design, [0.0] * (window - 1) + [1.0]))
    taps = weights[::-1]  # lfilter's first tap multiplies the newest value
    return _per_run(series, window - 1, lambda run: sp.lfilter(taps, [1.0], run))


def exponential_moving_average(
    series: HalfHourlySeries,
    *,
    span: int | None = None,
    alpha: float | None = None,
    min_periods: int | None = None,
) -> HalfHourlySeries:
    """EWMA (the pandas ``adjust=False`` recursion), restarted on each run of present values.

    Give exactly one of ``span`` (``alpha = 2 / (span + 1)``) or ``alpha``. Each run emits
    from its ``min_periods``-th value: by default the span, or ``round(2 / alpha - 1)``. The
    run's first value keeps weight ``(1 - alpha) ** i`` at index ``i`` (about 14% at the first
    default output for span 48); pass about ``3 * span`` when that matters. Unlike pandas,
    nothing carries across a gap.
    """
    if (span is None) == (alpha is None):
        raise InvalidInputError("give exactly one of span and alpha")
    if span is not None:
        smoothing = 2 / (require_int("span", span, minimum=1) + 1)
        default = span
    else:
        smoothing = require_real("alpha", alpha)
        if not 0 < smoothing <= 1:
            raise InvalidInputError(f"alpha must be in (0, 1], got {alpha!r}")
        # Any default past the series emits nothing; the cap stops a tiny alpha overflowing.
        default = round(min(2 / smoothing - 1, len(series) + 1))
    periods = default if min_periods is None else require_int("min_periods", min_periods, minimum=1)
    return _per_run(
        series, periods - 1, lambda run: sm.exponential_moving_average(run, alpha=smoothing)
    )


def rolling_variance(series: HalfHourlySeries, window: int, *, ddof: int = 1) -> HalfHourlySeries:
    """Sample (``ddof=1``) or population (``ddof=0``) variance of each window; see ``rolling_std``."""
    require_int("ddof", ddof, minimum=0, maximum=1)
    return _rolling(series, window, lambda values: fb.variance(values, ddof), minimum=ddof + 1)


def rolling_std(series: HalfHourlySeries, window: int, *, ddof: int = 1) -> HalfHourlySeries:
    """Sample (``ddof=1``) or population (``ddof=0``) standard deviation of each window.

    A constant window can give a tiny non-zero value (about 1.5e-17 for six 0.1s), so test for
    constant windows relative to their scale before dividing by this.
    """
    require_int("ddof", ddof, minimum=0, maximum=1)
    return _rolling(series, window, lambda values: fb.std(values, ddof), minimum=ddof + 1)


def rolling_median(series: HalfHourlySeries, window: int) -> HalfHourlySeries:
    return _rolling(series, window, fb.median)


def rolling_quantile(series: HalfHourlySeries, window: int, q: float) -> HalfHourlySeries:
    """Type 7 quantile: linear interpolation, the R and NumPy default."""
    probability = require_real("q", q)
    if not 0 <= probability <= 1:
        raise InvalidInputError(f"q must be between 0 and 1, got {q!r}")
    return _rolling(series, window, lambda values: fb.quantile(values, probability))


def rolling_rms(series: HalfHourlySeries, window: int) -> HalfHourlySeries:
    """Root mean square about zero, so it reflects the level as well as the spread."""
    return _rolling(series, window, sp.rms)


def difference(series: HalfHourlySeries, *, lag: int = 1, order: int = 1) -> HalfHourlySeries:
    """The ``order``-th difference at ``lag`` over rows ``t, t - lag, ..., t - order * lag``.

    Lag 48 gives daily and lag 336 weekly seasonal differences.
    """
    require_int("lag", lag, minimum=1)
    require_int("order", order, minimum=1)
    return _per_run(series, order, lambda run: sm.difference(run, lag=1, order=order), stride=lag)


def _per_run(
    series: HalfHourlySeries,
    warm_up: int,
    compute: Callable[[list[float]], list[float]],
    *,
    stride: int = 1,
) -> HalfHourlySeries:
    """``compute`` on each maximal run of present values in each residue class mod ``stride``.

    ``compute(run)`` returns values aligned to the end of ``run``; the first ``warm_up``
    positions of every run are ``None``.
    """
    out: list[float | None] = [None] * len(series)
    for residue in range(min(stride, len(series))):
        column = series.values[residue::stride]
        masked: list[float | None] = [None] * len(column)
        for first, run in _runs(column):
            if len(run) > warm_up:
                with orchid_errors("feature calculation"):
                    computed = compute(run)
                masked[first + warm_up : first + len(run)] = computed[warm_up - len(run) :]
        out[residue::stride] = masked
    return _result(series, out)


def _rolling(
    series: HalfHourlySeries,
    window: int,
    statistic: Callable[[list[float]], float],
    *,
    minimum: int = 1,
) -> HalfHourlySeries:
    require_int("window", window, minimum=minimum)
    out: list[float | None] = [None] * len(series)
    last_missing = -1
    for t, value in enumerate(series.values):
        if value is None:
            last_missing = t
        elif t - last_missing >= window:
            recent = cast(list[float], list(series.values[t + 1 - window : t + 1]))
            with orchid_errors("rolling statistic"):
                out[t] = statistic(recent)
    return _result(series, out)


def _runs(values: Sequence[float | None]) -> Iterator[tuple[int, list[float]]]:
    """Maximal runs of present values, each with the index of its first value."""
    run: list[float] = []
    for index, value in enumerate([*values, None]):
        if value is not None:
            run.append(value)
        elif run:
            yield index - len(run), run
            run = []


def _result(series: HalfHourlySeries, values: list[float | None]) -> HalfHourlySeries:
    if any(value is not None and not math.isfinite(value) for value in values):
        raise CalculationError("Orchid returned a non-finite value")
    return HalfHourlySeries(series.start, values)
