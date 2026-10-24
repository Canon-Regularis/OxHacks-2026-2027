"""
Time-Series Smoothing & Forecasting Template
============================================
Pure-Python, zero-dependency toolkit for SMOOTHING and FORECASTING a univariate
time series.  Where the sibling ``forecasting/baseline`` module owns the trivial
benchmarks and ``tracking`` owns online state-estimation (Kalman & friends), this
module is about the classical *smoothing* view of a series: extract the signal
(level / trend / season) from the noise, and project it forward.

It spans the field:

  * Moving-average & kernel smoothers -- simple / weighted / cumulative /
    exponential / centered moving averages, the robust median filter, Gaussian
    and Savitzky-Golay (local-polynomial) filters, LOWESS (locally-weighted
    regression) and the Hodrick-Prescott trend filter.
  * The exponential-smoothing (ETS) family -- simple exponential smoothing,
    Holt's linear (and damped) trend, and Holt-Winters seasonal smoothing
    (additive & multiplicative), with automatic maximum-likelihood-style
    optimisation of the smoothing parameters by a from-scratch Nelder-Mead, plus
    AIC/AICc model selection and an ``auto_ets`` chooser.
  * Decomposition -- classical additive / multiplicative decomposition and a
    simplified STL (seasonal-trend via LOWESS).
  * Autocorrelation & classical structure -- ACF / PACF (Durbin-Levinson),
    autoregressive fitting by Yule-Walker and Burg, AR forecasting, and the
    Ljung-Box white-noise test.
  * Forecasting infrastructure -- point forecasts with prediction intervals,
    a full accuracy metric suite (MAE / RMSE / MAPE / sMAPE / MASE / ...),
    rolling-origin (expanding & sliding) back-testing / cross-validation, plus
    differencing, Box-Cox and train/test utilities.

Everything is built from the standard library alone (``math``, ``random``,
``itertools``), so it drops straight into a competition / hackathon / interview /
research laptop -- just copy this one file.  Hand a function a ``list`` of floats
and ask for the smoother, decomposition or forecast you want.

Series utilities
    difference / integrate / seasonal_difference   -- (de)trending transforms
    box_cox / inv_box_cox                           -- variance stabilisation
    train_test_split / rolling_windows              -- evaluation helpers

Smoothers (return a smoothed series of the same length)
    simple_moving_average / weighted_moving_average / cumulative_moving_average
    exponential_moving_average / centered_moving_average
    median_filter / gaussian_filter / savitzky_golay / lowess / hodrick_prescott

Exponential smoothing (return a ForecastResult)
    exponential_smoothing   -- the unified ETS engine (trend/damped/seasonal)
    ses / holt / holt_winters / theta_method        -- convenience wrappers
    auto_ets                -- pick the best ETS configuration by AICc

Decomposition
    classical_decomposition -- trend (MA) + seasonal indices + remainder
    stl_decomposition       -- seasonal-trend decomposition via LOWESS

Autocorrelation & AR
    autocovariance / autocorrelation / partial_autocorrelation
    yule_walker / burg / ar_forecast / ljung_box

Forecasting / evaluation
    forecast                -- dispatch to any method in FORECASTERS by name
    forecast_metrics / mae / rmse / mape / smape / mase / ...
    rolling_origin_backtest / time_series_cv
    SAMPLE_SERIES / get_series                       -- built-in demo data

References
    Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3e).
    Cleveland et al. (1990), STL.  Hodrick & Prescott (1997).
    Savitzky & Golay (1964).  Box, Jenkins, Reinsel, *Time Series Analysis*.
"""

import math
import random
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

Series = List[float]
Vector = List[float]
Matrix = List[List[float]]

_EPS = 1e-12
_INF = float("inf")

__all__ = [
    # series utilities
    "difference", "integrate", "seasonal_difference", "box_cox", "inv_box_cox",
    "train_test_split", "rolling_windows",
    # smoothers
    "simple_moving_average", "weighted_moving_average", "cumulative_moving_average",
    "exponential_moving_average", "centered_moving_average", "median_filter",
    "gaussian_filter", "savitzky_golay", "lowess", "hodrick_prescott",
    # exponential smoothing
    "ForecastResult", "exponential_smoothing", "ses", "holt", "holt_winters",
    "theta_method", "auto_ets",
    # decomposition
    "Decomposition", "classical_decomposition", "stl_decomposition",
    # autocorrelation / AR
    "autocovariance", "autocorrelation", "partial_autocorrelation",
    "yule_walker", "burg", "ar_forecast", "ljung_box",
    # forecasting / evaluation
    "forecast", "FORECASTERS", "forecast_metrics", "mae", "mse", "rmse",
    "mape", "smape", "mase", "me", "mpe", "rmsse",
    "rolling_origin_backtest", "time_series_cv",
    "SAMPLE_SERIES", "get_series",
    # payloads
    "build_forecast_payload", "build_decomposition_payload",
]


# ===========================================================================
# 1. SMALL NUMERIC / STATISTICS HELPERS  (self-contained)
# ===========================================================================
def _as_floats(y: Sequence[float]) -> Series:
    out = [float(v) for v in y]
    if not all(math.isfinite(v) for v in out):
        raise ValueError("series contains a non-finite value (NaN or infinity)")
    return out


def _mean(y: Sequence[float]) -> float:
    n = len(y)
    if n == 0:
        raise ValueError("mean of an empty series")
    return math.fsum(y) / n


def _var(y: Sequence[float], ddof: int = 0) -> float:
    n = len(y)
    if n - ddof <= 0:
        return 0.0
    mu = _mean(y)
    return math.fsum((v - mu) ** 2 for v in y) / (n - ddof)


def _std(y: Sequence[float], ddof: int = 0) -> float:
    return math.sqrt(_var(y, ddof))


def _median(y: Sequence[float]) -> float:
    s = sorted(y)
    n = len(s)
    if n == 0:
        raise ValueError("median of an empty series")
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def _solve(A: Matrix, b: Vector) -> Vector:
    """Solve the linear system ``A x = b`` by Gaussian elimination with partial
    pivoting.  ``A`` is copied; raises on a singular system."""
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-15:
            raise ValueError("singular linear system")
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(n):
            if r != col:
                f = M[r][col] / pv
                if f != 0.0:
                    for k in range(col, n + 1):
                        M[r][k] -= f * M[col][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def _solve_symmetric_penta(d0: Vector, d1: Vector, d2: Vector,
                           rhs: Vector) -> Vector:
    """Solve ``A x = rhs`` for a symmetric *pentadiagonal* positive-definite
    ``A`` given its three upper bands -- ``d0`` (main diagonal, length ``n``),
    ``d1`` (first super-diagonal, length ``n-1``) and ``d2`` (second
    super-diagonal, length ``n-2``) -- via an ``L D L^T`` factorisation in
    ``O(n)`` time and space.  Used by :func:`hodrick_prescott`, whose system is
    pentadiagonal; a dense Gaussian solve would be ``O(n^3)``.  Falls back to
    the dense :func:`_solve` if a non-positive pivot appears (should not happen
    for the strictly positive-definite HP matrix)."""
    n = len(d0)
    dvec = [0.0] * n
    l1 = [0.0] * n                                    # L[i][i-1]
    l2 = [0.0] * n                                    # L[i][i-2]
    for i in range(n):
        a_im1 = d1[i - 1] if i >= 1 else 0.0          # A[i][i-1] (symmetry)
        a_im2 = d2[i - 2] if i >= 2 else 0.0          # A[i][i-2] (symmetry)
        if i >= 2:
            l2[i] = a_im2 / dvec[i - 2]
        if i >= 1:
            corr = l2[i] * l1[i - 1] * dvec[i - 2] if i >= 2 else 0.0
            l1[i] = (a_im1 - corr) / dvec[i - 1]
        piv = d0[i]
        if i >= 1:
            piv -= l1[i] * l1[i] * dvec[i - 1]
        if i >= 2:
            piv -= l2[i] * l2[i] * dvec[i - 2]
        if piv <= _EPS:                               # not positive-definite
            A = [[0.0] * n for _ in range(n)]
            for r in range(n):
                A[r][r] = d0[r]
                if r + 1 < n:
                    A[r][r + 1] = A[r + 1][r] = d1[r]
                if r + 2 < n:
                    A[r][r + 2] = A[r + 2][r] = d2[r]
            return _solve(A, rhs)
        dvec[i] = piv
    z = [0.0] * n                                     # forward solve L z = rhs
    for i in range(n):
        zi = rhs[i]
        if i >= 1:
            zi -= l1[i] * z[i - 1]
        if i >= 2:
            zi -= l2[i] * z[i - 2]
        z[i] = zi
    w = [z[i] / dvec[i] for i in range(n)]            # diagonal solve D w = z
    x = [0.0] * n                                     # back solve L^T x = w
    for i in range(n - 1, -1, -1):
        xi = w[i]
        if i + 1 < n:
            xi -= l1[i + 1] * x[i + 1]
        if i + 2 < n:
            xi -= l2[i + 2] * x[i + 2]
        x[i] = xi
    return x


def _polyfit(x: Sequence[float], y: Sequence[float], degree: int) -> Vector:
    """Least-squares polynomial fit via the normal equations; returns
    coefficients ``[c0, c1, ..., c_degree]`` for ``c0 + c1 x + ...``."""
    degree = min(degree, len(x) - 1)
    if degree < 0:
        return [_mean(y)] if y else [0.0]
    m = degree + 1
    # Vandermonde normal equations  (X^T X) c = X^T y
    powers = [[xi ** p for p in range(m)] for xi in x]
    ata = [[math.fsum(powers[i][a] * powers[i][b] for i in range(len(x)))
            for b in range(m)] for a in range(m)]
    aty = [math.fsum(powers[i][a] * y[i] for i in range(len(x))) for a in range(m)]
    try:
        return _solve(ata, aty)
    except ValueError:
        return [_mean(y)] + [0.0] * (m - 1)


def _nelder_mead(f: Callable[[Vector], float], x0: Vector, *,
                 step: float = 0.1, max_iter: int = 2000, tol: float = 1e-10
                 ) -> Tuple[Vector, float]:
    """Minimise ``f`` from ``x0`` with the Nelder-Mead simplex method.  Used to
    fit smoothing / model parameters when no closed form exists."""
    n = len(x0)
    if n == 0:
        return [], f([])
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    simplex = [list(x0)]
    for i in range(n):
        pt = list(x0)
        pt[i] += step if pt[i] == 0 else step * (1.0 + abs(pt[i]))
        simplex.append(pt)
    fval = [f(p) for p in simplex]
    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: fval[i])
        simplex = [simplex[i] for i in order]
        fval = [fval[i] for i in order]
        if abs(fval[-1] - fval[0]) <= tol * (abs(fval[0]) + abs(fval[-1]) + tol):
            break
        centroid = [math.fsum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        xr = [centroid[j] + alpha * (centroid[j] - simplex[-1][j]) for j in range(n)]
        fr = f(xr)
        if fval[0] <= fr < fval[-2]:
            simplex[-1], fval[-1] = xr, fr
        elif fr < fval[0]:
            xe = [centroid[j] + gamma * (xr[j] - centroid[j]) for j in range(n)]
            fe = f(xe)
            simplex[-1], fval[-1] = (xe, fe) if fe < fr else (xr, fr)
        else:
            xc = [centroid[j] + rho * (simplex[-1][j] - centroid[j]) for j in range(n)]
            fc = f(xc)
            if fc < fval[-1]:
                simplex[-1], fval[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j])
                                  for j in range(n)]
                    fval[i] = f(simplex[i])
    best = min(range(n + 1), key=lambda i: fval[i])
    return simplex[best], fval[best]


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _is_constant(y: Sequence[float]) -> bool:
    """Is the series effectively constant?  A SCALE-RELATIVE test -- the range must
    be negligible against the data's own magnitude -- so genuinely-varying but
    small-magnitude data (e.g. sensor readings ~1e-6) is NOT mistaken for a
    constant.  Autocorrelation / AR use this instead of an absolute variance floor,
    which would collapse small-scale data to spurious white noise."""
    if not y:
        return True
    lo, hi = min(y), max(y)
    return hi - lo <= 1e-12 * max(1.0, abs(lo), abs(hi))


def _z_score(level: float) -> float:
    """Two-sided normal quantile for a central probability ``level`` (e.g. 0.95
    -> 1.96), via a rational approximation of the inverse error function."""
    if not 0.0 < level < 1.0:
        raise ValueError("confidence level must be in (0, 1)")
    p = 0.5 + level / 2.0                         # upper tail point
    # Acklam's inverse-normal-CDF approximation
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


# ===========================================================================
# 2. SERIES UTILITIES
# ===========================================================================
def difference(y: Sequence[float], lag: int = 1, order: int = 1) -> Series:
    """``order``-th differences at the given ``lag``.  ``difference(y)`` is the
    ordinary first difference; each application shortens the series by ``lag``."""
    if lag < 1 or order < 0:
        raise ValueError("lag must be >= 1 and order >= 0")
    out = _as_floats(y)
    for _ in range(order):
        if len(out) <= lag:
            raise ValueError("series too short to difference at lag %d" % lag)
        out = [out[i] - out[i - lag] for i in range(lag, len(out))]
    return out


def integrate(diffs: Sequence[float], initial: Sequence[float], lag: int = 1) -> Series:
    """Inverse of a single ``difference(y, lag)``: rebuild the level series from
    its differences and the first ``lag`` original values."""
    if len(initial) < lag:
        raise ValueError("need at least `lag` initial values to integrate")
    out = list(_as_floats(initial))
    for d in diffs:
        out.append(out[-lag] + float(d))
    return out


def seasonal_difference(y: Sequence[float], m: int) -> Series:
    """Seasonal difference ``y_t - y_{t-m}`` (removes a seasonal pattern of
    period ``m``)."""
    return difference(y, lag=m, order=1)


def box_cox(y: Sequence[float], lam: float) -> Series:
    """Box-Cox variance-stabilising transform (requires strictly positive data;
    ``lam == 0`` gives the log transform)."""
    yy = _as_floats(y)
    if any(v <= 0 for v in yy):
        raise ValueError("Box-Cox requires strictly positive values")
    if abs(lam) < _EPS:
        return [math.log(v) for v in yy]
    return [(v ** lam - 1.0) / lam for v in yy]


def inv_box_cox(z: Sequence[float], lam: float) -> Series:
    """Inverse Box-Cox transform."""
    zz = _as_floats(z)
    if abs(lam) < _EPS:
        return [math.exp(v) for v in zz]
    return [math.copysign(abs(lam * v + 1.0) ** (1.0 / lam), lam * v + 1.0)
            if (lam * v + 1.0) > 0 else 0.0 for v in zz]


def train_test_split(y: Sequence[float], test_size: int) -> Tuple[Series, Series]:
    """Split a series into a leading train part and a trailing test part of
    length ``test_size``."""
    yy = _as_floats(y)
    if not 0 < test_size < len(yy):
        raise ValueError("test_size must be in (0, len(series))")
    return yy[:-test_size], yy[-test_size:]


def rolling_windows(y: Sequence[float], window: int) -> List[Series]:
    """All contiguous windows of length ``window`` (a list of lists)."""
    yy = _as_floats(y)
    if window < 1 or window > len(yy):
        raise ValueError("window must be in [1, len(series)]")
    return [yy[i:i + window] for i in range(len(yy) - window + 1)]


# ===========================================================================
# 3. MOVING-AVERAGE & KERNEL SMOOTHERS
# ===========================================================================
def simple_moving_average(y: Sequence[float], window: int, *, center: bool = False
                          ) -> Series:
    """Simple moving average.  With ``center=False`` (trailing) point ``t`` is the
    mean of the last ``window`` values (the first ``window-1`` shrink the window);
    with ``center=True`` the window is centred on ``t``."""
    yy = _as_floats(y)
    n = len(yy)
    if window < 1 or window > n:
        raise ValueError("window must be in [1, len(series)]")
    out = [0.0] * n
    if not center:
        for t in range(n):
            lo = max(0, t - window + 1)
            out[t] = _mean(yy[lo:t + 1])
    else:
        half = window // 2
        for t in range(n):
            lo, hi = max(0, t - half), min(n, t + half + 1)
            out[t] = _mean(yy[lo:hi])
    return out


def weighted_moving_average(y: Sequence[float], weights: Sequence[float]) -> Series:
    """Trailing weighted moving average; ``weights`` are applied oldest-to-newest
    within each window and internally normalised."""
    yy = _as_floats(y)
    w = _as_floats(weights)
    k = len(w)
    n = len(yy)
    if k < 1 or k > n:
        raise ValueError("need 1 <= len(weights) <= len(series)")
    if math.fsum(w) == 0:
        raise ValueError("weights must not sum to zero")
    out = [0.0] * n
    for t in range(n):
        if t + 1 < k:
            out[t] = _mean(yy[:t + 1])                    # shrink at the start
        else:
            win = yy[t - k + 1:t + 1]
            sw = math.fsum(w)
            out[t] = math.fsum(win[i] * w[i] for i in range(k)) / sw
    return out


def cumulative_moving_average(y: Sequence[float]) -> Series:
    """Cumulative (expanding-window) mean."""
    yy = _as_floats(y)
    out, run = [], 0.0
    for i, v in enumerate(yy):
        run += v
        out.append(run / (i + 1))
    return out


def exponential_moving_average(y: Sequence[float], alpha: Optional[float] = None,
                               *, span: Optional[int] = None) -> Series:
    """Exponential moving average with smoothing factor ``alpha`` in (0, 1].
    Alternatively give a ``span`` (``alpha = 2/(span+1)``)."""
    yy = _as_floats(y)
    if alpha is None:
        if span is None or span < 1:
            raise ValueError("provide alpha in (0,1] or span >= 1")
        alpha = 2.0 / (span + 1.0)
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    out = []
    s = yy[0] if yy else 0.0
    for v in yy:
        s = alpha * v + (1 - alpha) * s
        out.append(s)
    return out


def centered_moving_average(y: Sequence[float], m: int) -> Series:
    """Centred moving average of order ``m`` -- the standard trend estimator used
    in classical decomposition.  For even ``m`` a ``2 x m`` moving average is used
    so the result is symmetric; endpoints (where the window overruns) are None."""
    yy = _as_floats(y)
    n = len(yy)
    if m < 1 or m > n:
        raise ValueError("m must be in [1, len(series)]")
    out: List[Optional[float]] = [None] * n
    if m % 2 == 1:
        half = m // 2
        for t in range(half, n - half):
            out[t] = _mean(yy[t - half:t + half + 1])
    else:
        half = m // 2
        for t in range(half, n - half):
            # 2xm: half weight on the two end points of the (m+1)-span window
            window = yy[t - half:t + half + 1]
            s = 0.5 * window[0] + 0.5 * window[-1] + math.fsum(window[1:-1])
            out[t] = s / m
    return out


def median_filter(y: Sequence[float], window: int) -> Series:
    """Robust median filter -- replaces each point with the median of a centred
    window; excellent for removing spikes / impulse noise."""
    yy = _as_floats(y)
    n = len(yy)
    if window < 1 or window > n:
        raise ValueError("window must be in [1, len(series)]")
    half = window // 2
    return [_median(yy[max(0, t - half):min(n, t + half + 1)]) for t in range(n)]


def gaussian_filter(y: Sequence[float], sigma: float, *, truncate: float = 3.0
                    ) -> Series:
    """Gaussian-kernel smoother with bandwidth ``sigma`` (edges use the available,
    renormalised part of the kernel)."""
    yy = _as_floats(y)
    n = len(yy)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = max(1, int(truncate * sigma + 0.5))
    kernel = [math.exp(-0.5 * (k / sigma) ** 2) for k in range(-radius, radius + 1)]
    out = [0.0] * n
    for t in range(n):
        num = den = 0.0
        for k in range(-radius, radius + 1):
            j = t + k
            if 0 <= j < n:
                w = kernel[k + radius]
                num += w * yy[j]
                den += w
        out[t] = num / den if den > 0 else yy[t]
    return out


def savitzky_golay(y: Sequence[float], window: int, poly: int) -> Series:
    """Savitzky-Golay filter: fit a degree-``poly`` polynomial to each centred
    window of length ``window`` and take its value at the centre.  Preserves
    peaks/derivatives far better than a moving average.  Edges shrink the window."""
    yy = _as_floats(y)
    n = len(yy)
    if window < 1:
        raise ValueError("window must be >= 1")
    if poly < 0:
        raise ValueError("poly degree must be >= 0")
    half = window // 2
    out = [0.0] * n
    for t in range(n):
        lo, hi = max(0, t - half), min(n, t + half + 1)
        xs = [j - t for j in range(lo, hi)]           # centre the window at 0
        coef = _polyfit(xs, yy[lo:hi], poly)
        out[t] = coef[0]                              # value of the fit at x = 0
    return out


def lowess(y: Sequence[float], frac: float = 0.3, *, iterations: int = 3,
           x: Optional[Sequence[float]] = None) -> Series:
    """LOWESS -- locally-weighted scatterplot smoothing (Cleveland 1979).  A
    robust local linear regression with tricube weights; ``iterations`` > 1 adds
    bisquare re-weighting to resist outliers.  ``frac`` is the span (fraction of
    points in each local fit)."""
    yy = _as_floats(y)
    n = len(yy)
    if n == 0:
        return []
    if not 0 < frac <= 1:
        raise ValueError("frac must be in (0, 1]")
    xs = list(range(n)) if x is None else _as_floats(x)
    r = max(2, int(math.ceil(frac * n)))
    # Each point's r nearest neighbours and window half-width depend only on xs
    # (not on y or the robustness weights), so precompute them ONCE rather than
    # re-sorting inside every bisquare iteration.  For the default uniform integer
    # grid (x is None) the r nearest are a CONTIGUOUS index window, so this is
    # O(n*r) instead of O(n^2 log n) -- e.g. lowess/STL on n>=1000 drop from
    # seconds to milliseconds.  The window is [i - r//2, i - r//2 + r) clamped to
    # [0, n): this reproduces the general path's stable-sort tie-break EXACTLY (on
    # an equal-distance tie the sort keeps the smaller index, i.e. the left point),
    # and since every regression sum below uses math.fsum (order-independent) the
    # output is bit-identical.  A caller-supplied x keeps the exact general path.
    nbrs = []
    hs = []
    if x is None:
        for i in range(n):
            a = i - r // 2
            if a < 0:
                a = 0
            elif a + r > n:
                a = n - r
            nbrs.append(range(a, a + r))
            hs.append(float(max(i - a, a + r - 1 - i)) or 1.0)
    else:
        for i in range(n):
            order = sorted(range(n), key=lambda j: abs(xs[j] - xs[i]))[:r]
            nbrs.append(order)
            hs.append(max(abs(xs[j] - xs[i]) for j in order) or 1.0)
    fitted = [0.0] * n
    robustness = [1.0] * n
    for _ in range(max(1, iterations)):
        for i in range(n):
            order = nbrs[i]
            h = hs[i]
            w = []
            for j in order:
                u = abs(xs[j] - xs[i]) / h
                tri = (1 - u ** 3) ** 3 if u < 1 else 0.0
                w.append(tri * robustness[j])
            sw = math.fsum(w)
            if sw <= _EPS:
                fitted[i] = yy[i]
                continue
            swx = math.fsum(w[k] * xs[order[k]] for k in range(r))
            swy = math.fsum(w[k] * yy[order[k]] for k in range(r))
            swxx = math.fsum(w[k] * xs[order[k]] ** 2 for k in range(r))
            swxy = math.fsum(w[k] * xs[order[k]] * yy[order[k]] for k in range(r))
            denom = sw * swxx - swx * swx
            if abs(denom) < _EPS:
                fitted[i] = swy / sw
            else:
                b1 = (sw * swxy - swx * swy) / denom
                b0 = (swy - b1 * swx) / sw
                fitted[i] = b0 + b1 * xs[i]
        resid = [yy[i] - fitted[i] for i in range(n)]
        s = _median([abs(e) for e in resid])
        if s <= _EPS:
            break
        robustness = [(1 - (e / (6 * s)) ** 2) ** 2 if abs(e) < 6 * s else 0.0
                      for e in resid]
    return fitted


def hodrick_prescott(y: Sequence[float], lam: float = 1600.0) -> Series:
    """Hodrick-Prescott trend filter: the trend ``tau`` minimising
    ``sum (y-tau)^2 + lam * sum (tau_{t+1}-2 tau_t + tau_{t-1})^2``.  ``lam``
    controls smoothness (1600 is the classic quarterly-data value).  Solves the
    exact ``(I + lam D'D) tau = y`` system.  ``I + lam D'D`` is symmetric
    *pentadiagonal*, so it is factored in ``O(n)`` with a banded ``L D L^T``
    solve (a dense Gaussian solve would be ``O(n^3)`` -- ~18s at ``n=1000``).

    The system matrix is ill-conditioned (condition number ~``1 + 16*lam``), so a
    large DC offset in ``y`` costs precision.  Since the HP filter preserves
    constants exactly (``HP(y) = HP(y - mu) + mu``), the mean is removed before the
    solve and added back -- mirroring the demean-first handling in ``_var`` /
    ``autocovariance`` -- which keeps a series carrying a big baseline (prices,
    Kelvin temperatures, ...) accurate to full precision (~3000x better at a 1e6
    offset) while leaving normal-scale results unchanged to rounding."""
    yy = _as_floats(y)
    n = len(yy)
    if lam < 0:
        raise ValueError("lam must be non-negative")
    if n < 3:
        return list(yy)
    mu = _mean(yy)                                    # demean for conditioning
    yc = [v - mu for v in yy]
    # Accumulate only the three upper bands of A = I + lam * D'D.  D is the
    # (n-2) x n second-difference operator; each of its rows contributes the
    # 3x3 outer product lam * coef coef^T over columns (k, k+1, k+2).
    d0 = [1.0] * n                                    # main diagonal
    d1 = [0.0] * (n - 1)                              # first super-diagonal
    d2 = [0.0] * (n - 2)                              # second super-diagonal
    coef = (1.0, -2.0, 1.0)
    for k in range(n - 2):
        idx = (k, k + 1, k + 2)
        for a in range(3):
            for b in range(3):
                i, j = idx[a], idx[b]
                if j < i:                             # keep the upper triangle
                    continue
                v = lam * coef[a] * coef[b]
                if j == i:
                    d0[i] += v
                elif j == i + 1:
                    d1[i] += v
                else:
                    d2[i] += v
    tau = _solve_symmetric_penta(d0, d1, d2, yc)
    return [t + mu for t in tau]                      # restore the offset


# ===========================================================================
# 4. EXPONENTIAL SMOOTHING  (ETS family)
# ===========================================================================
@dataclass
class ForecastResult:
    """Everything a forecast produces.  ``fitted`` are the in-sample one-step-ahead
    predictions and ``residuals = actual - fitted`` the full in-sample errors (both
    length ``n``); ``forecast`` holds the ``h`` future point forecasts with optional
    ``lower`` / ``upper`` prediction-interval bounds.

    ``sse``, ``sigma``, ``aic`` and ``aicc`` are measured over the *effective
    sample* -- the ``n - burn`` residuals AFTER the warm-up transient (whose length
    is exposed as ``state['burn']`` / ``state['nobs_effective']``) -- because the
    first ``burn`` one-step errors reflect the crude heuristic initial states rather
    than the model's fit, and including them both biases model selection (seasonal
    models carry a longer warm-up) and inflates the prediction intervals.  Hence
    ``sse == sum(e*e for e in residuals[burn:])`` -- NOT over the full residual
    series -- and ``sigma`` is the matching residual scale (the ETS engine reports
    ``sqrt(sse / nobs_effective)``; :func:`ar_forecast` reports the estimated AR
    innovation standard deviation)."""
    method: str
    fitted: Series
    residuals: Series
    forecast: Series
    lower: Optional[Series] = None
    upper: Optional[Series] = None
    params: Dict[str, float] = field(default_factory=dict)
    sse: float = 0.0
    sigma: float = 0.0
    aic: Optional[float] = None
    aicc: Optional[float] = None
    state: Dict[str, Any] = field(default_factory=dict)


def _ets_burn(trend: Optional[str], seasonal: Optional[str], m: int) -> int:
    if seasonal:
        return m
    return 1 if trend else 0


def _ets_init(y: Series, trend: Optional[str], seasonal: Optional[str], m: int
              ) -> Tuple[float, float, Optional[List[float]]]:
    n = len(y)
    if seasonal:
        if n < 2 * m:
            raise ValueError("need at least two full seasons (2*m points) for a "
                             "seasonal model")
        level = _mean(y[:m])
        b = (_mean(y[m:2 * m]) - _mean(y[:m])) / m if trend else 0.0
        if seasonal == "add":
            s = [y[i] - level for i in range(m)]
        else:                                         # multiplicative
            if level == 0:
                raise ValueError("multiplicative seasonality needs non-zero level")
            s = [y[i] / level for i in range(m)]
        return level, b, s
    level = y[0]
    b = (y[1] - y[0]) if (trend and n >= 2) else 0.0
    return level, b, None


def _ets_run(y: Series, alpha: float, beta: float, gamma: float, phi: float,
             trend: Optional[str], seasonal: Optional[str], m: int
             ) -> Tuple[Series, float, float, Optional[List[float]]]:
    """Run the ETS recursion; returns the one-step fitted series and the final
    (level, trend, seasonal) state."""
    level, b, s = _ets_init(y, trend, seasonal, m)
    fitted: Series = []
    for t in range(len(y)):
        season = (s[t % m] if seasonal else (0.0 if seasonal != "mul" else 1.0))
        trend_term = phi * b if trend else 0.0
        base = level + trend_term
        if seasonal == "add":
            yhat = base + season
        elif seasonal == "mul":
            yhat = base * season
        else:
            yhat = base
        fitted.append(yhat)
        yt = y[t]
        if seasonal == "add":
            new_level = alpha * (yt - season) + (1 - alpha) * base
        elif seasonal == "mul":
            new_level = alpha * (yt / season if season != 0 else yt) + (1 - alpha) * base
        else:
            new_level = alpha * yt + (1 - alpha) * base
        new_b = (beta * (new_level - level) + (1 - beta) * trend_term) if trend else 0.0
        if seasonal == "add":
            s[t % m] = gamma * (yt - new_level) + (1 - gamma) * season
        elif seasonal == "mul":
            s[t % m] = gamma * (yt / new_level if new_level != 0 else season) + \
                       (1 - gamma) * season
        level, b = new_level, new_b
    return fitted, level, b, s


def _ets_project(level: float, b: float, s: Optional[List[float]], phi: float,
                 trend: Optional[str], seasonal: Optional[str], m: int,
                 n: int, h: int) -> Series:
    out: Series = []
    for k in range(1, h + 1):
        if trend:
            damp = math.fsum(phi ** i for i in range(1, k + 1))
            base = level + damp * b
        else:
            base = level
        if seasonal:
            season = s[(n - 1 + k) % m]
            out.append(base + season if seasonal == "add" else base * season)
        else:
            out.append(base)
    return out


def exponential_smoothing(y: Sequence[float], *, trend: Optional[str] = None,
                          damped: bool = False, seasonal: Optional[str] = None,
                          seasonal_periods: Optional[int] = None,
                          alpha: Optional[float] = None, beta: Optional[float] = None,
                          gamma: Optional[float] = None, phi: Optional[float] = None,
                          optimize: bool = True, h: int = 0, level: float = 0.95
                          ) -> ForecastResult:
    """The unified exponential-smoothing (ETS) engine.

    ``trend``    -- None or ``'add'`` (Holt linear trend).
    ``damped``   -- damp the trend (only with a trend); adds parameter ``phi``.
    ``seasonal`` -- None, ``'add'`` (additive) or ``'mul'`` (multiplicative).
    ``seasonal_periods`` (``m``) -- required when ``seasonal`` is set.
    Any of ``alpha/beta/gamma/phi`` may be fixed; the rest are optimised by
    minimising the in-sample one-step SSE (Nelder-Mead) when ``optimize`` is True.
    ``h`` future points are forecast, with ``level`` prediction intervals.
    """
    y = _as_floats(y)
    n = len(y)
    if n < 2:
        raise ValueError("need at least 2 observations")
    if trend not in (None, "add"):
        raise ValueError("trend must be None or 'add'")
    if seasonal not in (None, "add", "mul"):
        raise ValueError("seasonal must be None, 'add' or 'mul'")
    if damped and not trend:
        raise ValueError("damped requires a trend")
    m = int(seasonal_periods or 0)
    if seasonal and m < 2:
        raise ValueError("seasonal models need seasonal_periods >= 2")
    # a fixed smoothing parameter is only meaningful for the component it drives;
    # reject an inapplicable one rather than silently ignoring it.
    if beta is not None and not trend:
        raise ValueError("beta is only valid with a trend")
    if gamma is not None and not seasonal:
        raise ValueError("gamma is only valid with a seasonal component")
    if phi is not None and not damped:
        raise ValueError("phi is only valid with damped=True")
    # any caller-fixed parameter must lie in its admissible range (mirrors the
    # box the optimiser enforces on the free ones); reject nonsense like alpha=5.
    for _nm, _val, _excl0 in (("alpha", alpha, True), ("beta", beta, False),
                              ("gamma", gamma, False), ("phi", phi, True)):
        if _val is not None and not ((0.0 < _val <= 1.0) if _excl0 else (0.0 <= _val <= 1.0)):
            raise ValueError("%s must be in %s" % (_nm, "(0, 1]" if _excl0 else "[0, 1]"))

    burn = _ets_burn(trend, seasonal, m)

    # which parameters are free (to optimise) vs fixed by the caller
    spec = [("alpha", alpha, True)]
    if trend:
        spec.append(("beta", beta, True))
    if seasonal:
        spec.append(("gamma", gamma, True))
    if damped:
        spec.append(("phi", phi, True))
    free = [(name, default) for (name, val, default) in spec if val is None]
    fixed = {name: val for (name, val, _) in spec if val is not None}

    def unpack(vec: Vector) -> Dict[str, float]:
        p = dict(fixed)
        for (name, _), v in zip(free, vec):
            lo, hi = (0.80, 0.999) if name == "phi" else (1e-4, 0.9999)
            p[name] = _clip(v, lo, hi)
        p.setdefault("beta", 0.0)
        p.setdefault("gamma", 0.0)
        p.setdefault("phi", 1.0)
        return p

    def sse_of(vec: Vector) -> float:
        p = unpack(vec)
        # penalise the optimiser for straying outside the admissible box
        pen = 0.0
        for (name, _), v in zip(free, vec):
            lo, hi = (0.80, 0.999) if name == "phi" else (1e-4, 0.9999)
            if v < lo:
                pen += (lo - v) ** 2
            elif v > hi:
                pen += (v - hi) ** 2
        try:
            fitted, *_ = _ets_run(y, p["alpha"], p["beta"], p["gamma"], p["phi"],
                                  trend, seasonal, m)
        except (ValueError, ZeroDivisionError, OverflowError):
            return _INF
        sse = math.fsum((y[t] - fitted[t]) ** 2 for t in range(burn, n))
        if sse != sse or sse == _INF:
            return _INF
        return sse + 1e6 * pen

    if optimize and free:
        x0 = [0.3 if name != "phi" else 0.95 for (name, _) in free]
        best, _ = _nelder_mead(sse_of, x0, step=0.15, max_iter=1500)
        params = unpack(best)
    else:
        params = unpack([0.3 if name != "phi" else 0.95 for (name, _) in free])

    fitted, lvl, b, s = _ets_run(y, params["alpha"], params["beta"],
                                 params["gamma"], params["phi"], trend, seasonal, m)
    residuals = [y[t] - fitted[t] for t in range(n)]
    sse = math.fsum(residuals[t] ** 2 for t in range(burn, n))
    k_eff = n - burn
    sigma = math.sqrt(sse / k_eff) if k_eff > 0 else 0.0

    fc = _ets_project(lvl, b, s, params["phi"], trend, seasonal, m, n, h) if h > 0 else []
    lower = upper = None
    if h > 0 and sigma > 0:
        z = _z_score(level)
        # simple residual-based interval, widening with the horizon
        lower = [fc[k] - z * sigma * math.sqrt(k + 1) for k in range(h)]
        upper = [fc[k] + z * sigma * math.sqrt(k + 1) for k in range(h)]

    n_params = len(spec) + 1 + (1 if trend else 0) + (m if seasonal else 0)  # +states
    # AIC must be on a CONSISTENT (log-likelihood) scale for EVERY configuration so
    # auto_ets can rank them against each other: compute it whenever there are
    # residuals, and add the finite-sample AICc correction only when its
    # denominator is positive.  (Previously a heavily-parameterised seasonal model
    # on a short series left aic=None, and auto_ets then fell back to raw SSE --
    # incomparable to another model's negative log-scale AIC.)
    aic = aicc = None
    if k_eff > 0:
        aic = -_INF if sse <= 0 else k_eff * math.log(sse / k_eff) + 2 * n_params
        denom = k_eff - n_params - 1
        aicc = aic if (aic == -_INF or denom <= 0) else \
            aic + 2 * n_params * (n_params + 1) / denom

    name = "ETS(%s,%s,%s)" % ("A", ("Ad" if damped else "A") if trend else "N",
                              {"add": "A", "mul": "M", None: "N"}[seasonal])
    return ForecastResult(
        method=name, fitted=fitted, residuals=residuals, forecast=fc,
        lower=lower, upper=upper, params={k: params[k] for k, _, _ in spec},
        sse=sse, sigma=sigma, aic=aic, aicc=aicc,
        state={"level": lvl, "trend": b, "seasonal": s,
               "burn": burn, "nobs_effective": k_eff})


def ses(y: Sequence[float], *, alpha: Optional[float] = None, h: int = 1,
        optimize: bool = True, level: float = 0.95) -> ForecastResult:
    """Simple exponential smoothing (no trend, no season)."""
    return exponential_smoothing(y, alpha=alpha, h=h, optimize=optimize, level=level)


def holt(y: Sequence[float], *, alpha: Optional[float] = None,
         beta: Optional[float] = None, damped: bool = False, h: int = 1,
         optimize: bool = True, level: float = 0.95) -> ForecastResult:
    """Holt's linear trend method (double exponential smoothing), optionally
    damped."""
    return exponential_smoothing(y, trend="add", damped=damped, alpha=alpha,
                                 beta=beta, h=h, optimize=optimize, level=level)


def holt_winters(y: Sequence[float], seasonal_periods: int, *,
                 seasonal: str = "add", damped: bool = False,
                 alpha: Optional[float] = None, beta: Optional[float] = None,
                 gamma: Optional[float] = None, h: int = 1, optimize: bool = True,
                 level: float = 0.95) -> ForecastResult:
    """Holt-Winters triple exponential smoothing (additive or multiplicative
    seasonality, optionally damped trend)."""
    return exponential_smoothing(y, trend="add", damped=damped, seasonal=seasonal,
                                 seasonal_periods=seasonal_periods, alpha=alpha,
                                 beta=beta, gamma=gamma, h=h, optimize=optimize,
                                 level=level)


def theta_method(y: Sequence[float], *, h: int = 1,
                 seasonal_periods: Optional[int] = None, level: float = 0.95
                 ) -> ForecastResult:
    """The Theta method (Assimakopoulos & Nikolopoulos 2000) -- winner of the M3
    competition.  Combines simple exponential smoothing of the series with the
    long-term linear (drift) trend; optionally deseasonalises first."""
    y = _as_floats(y)
    n = len(y)
    if n < 3:
        raise ValueError("theta method needs at least 3 observations")
    work = y
    seas: Optional[List[float]] = None
    model = "multiplicative"
    # seasonal_periods in {None, 1} means "no seasonality" -> the plain Theta path.
    # Deseasonalise MULTIPLICATIVELY only for strictly-positive data, else
    # ADDITIVELY, so a series with zeros/negatives does not crash.
    if seasonal_periods and seasonal_periods >= 2 and n >= 2 * seasonal_periods:
        model = "multiplicative" if all(v > 0 for v in y) else "additive"
        dec = classical_decomposition(y, seasonal_periods, model=model)
        seas = dec.seasonal
        if model == "multiplicative":
            work = [y[t] / seas[t] if seas[t] != 0 else y[t] for t in range(n)]
        else:
            work = [y[t] - seas[t] for t in range(n)]
    # theta=0 line: ordinary least-squares trend; theta=2 line: SES of the data
    coef = _polyfit(list(range(n)), work, 1)
    b0, b1 = coef[0], coef[1]
    s = ses(work, h=h, optimize=True)
    drift = [b0 + b1 * (n + k) for k in range(h)]
    ses_fc = s.forecast
    fc = [0.5 * (drift[k] + ses_fc[k]) for k in range(h)]
    if seas is not None:
        base = [seas[(n - seasonal_periods + (k % seasonal_periods)) % n] for k in range(h)]
        fc = ([fc[k] * base[k] for k in range(h)] if model == "multiplicative"
              else [fc[k] + base[k] for k in range(h)])
    fitted = [0.5 * ((b0 + b1 * t) + s.fitted[t]) for t in range(n)]
    if seas is not None:
        fitted = ([fitted[t] * seas[t] for t in range(n)] if model == "multiplicative"
                  else [fitted[t] + seas[t] for t in range(n)])
    residuals = [y[t] - fitted[t] for t in range(n)]
    sigma = _std(residuals[1:], ddof=1) if n > 2 else 0.0
    lower = upper = None
    if sigma > 0:
        z = _z_score(level)
        lower = [fc[k] - z * sigma * math.sqrt(k + 1) for k in range(h)]
        upper = [fc[k] + z * sigma * math.sqrt(k + 1) for k in range(h)]
    return ForecastResult(method="Theta", fitted=fitted, residuals=residuals,
                          forecast=fc, lower=lower, upper=upper,
                          params={"b0": b0, "b1": b1, "alpha": s.params.get("alpha", 0.0)},
                          sse=math.fsum(e * e for e in residuals), sigma=sigma)


def auto_ets(y: Sequence[float], *, seasonal_periods: Optional[int] = None,
             h: int = 1, level: float = 0.95) -> ForecastResult:
    """Automatically choose the best ETS configuration (trend? damped? seasonal
    add/mul?) by minimising AICc, refit, and forecast."""
    y = _as_floats(y)
    configs: List[Tuple[Optional[str], bool, Optional[str]]] = [
        (None, False, None), ("add", False, None), ("add", True, None)]
    if seasonal_periods and len(y) >= 2 * seasonal_periods:
        for tr, dp in [(None, False), ("add", False), ("add", True)]:
            configs.append((tr, dp, "add"))
            if all(v > 0 for v in y):
                configs.append((tr, dp, "mul"))
    best: Optional[ForecastResult] = None
    best_ic = _INF
    for trend, damped, seasonal in configs:
        try:
            res = exponential_smoothing(
                y, trend=trend, damped=damped, seasonal=seasonal,
                seasonal_periods=seasonal_periods, h=h, level=level)
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        ic = res.aicc if res.aicc is not None else (res.aic if res.aic is not None
                                                    else res.sse)
        if ic is not None and ic < best_ic:
            best_ic, best = ic, res
    if best is None:
        return ses(y, h=h, level=level)
    return best


# ===========================================================================
# 5. DECOMPOSITION
# ===========================================================================
@dataclass
class Decomposition:
    """A time series split into ``observed = trend (+|*) seasonal (+|*) resid``."""
    observed: Series
    trend: List[Optional[float]]
    seasonal: Series
    resid: List[Optional[float]]
    model: str
    period: int

    def seasonally_adjusted(self) -> Series:
        if self.model == "additive":
            return [self.observed[t] - self.seasonal[t] for t in range(len(self.observed))]
        return [self.observed[t] / self.seasonal[t] if self.seasonal[t] else self.observed[t]
                for t in range(len(self.observed))]


def classical_decomposition(y: Sequence[float], period: int, *,
                            model: str = "additive") -> Decomposition:
    """Classical seasonal decomposition: a centred moving average estimates the
    trend, the detrended series is averaged by season to get seasonal indices
    (normalised to sum to 0 / average to 1), and the remainder is the residual."""
    yy = _as_floats(y)
    n = len(yy)
    if model not in ("additive", "multiplicative"):
        raise ValueError("model must be 'additive' or 'multiplicative'")
    if period < 2 or n < 2 * period:
        raise ValueError("need period >= 2 and at least two full periods")
    if model == "multiplicative" and any(v <= 0 for v in yy):
        raise ValueError("multiplicative decomposition needs positive values")
    trend = centered_moving_average(yy, period)
    detr: List[Optional[float]] = []
    for t in range(n):
        if trend[t] is None:
            detr.append(None)
        elif model == "additive":
            detr.append(yy[t] - trend[t])
        else:
            detr.append(yy[t] / trend[t] if trend[t] else None)
    # average detrended values by season index
    by_season: List[List[float]] = [[] for _ in range(period)]
    for t in range(n):
        if detr[t] is not None:
            by_season[t % period].append(detr[t])
    base = 0.0 if model == "additive" else 1.0
    idx = [(_mean(vals) if vals else base) for vals in by_season]
    if model == "additive":                          # normalise to sum 0
        adj = _mean(idx)
        idx = [v - adj for v in idx]
    else:                                             # normalise to mean 1
        adj = _mean(idx)
        idx = [v / adj if adj else 1.0 for v in idx]
    seasonal = [idx[t % period] for t in range(n)]
    resid: List[Optional[float]] = []
    for t in range(n):
        if trend[t] is None:
            resid.append(None)
        elif model == "additive":
            resid.append(yy[t] - trend[t] - seasonal[t])
        else:
            d = trend[t] * seasonal[t]
            resid.append(yy[t] / d if d else None)
    return Decomposition(observed=yy, trend=trend, seasonal=seasonal, resid=resid,
                         model=model, period=period)


def stl_decomposition(y: Sequence[float], period: int, *,
                      seasonal_frac: float = 0.75, trend_frac: float = 0.5,
                      iterations: int = 2, robust: bool = True) -> Decomposition:
    """A simplified STL (Seasonal-Trend decomposition using LOESS): iterate
    between LOWESS-smoothing each seasonal subseries and LOWESS-smoothing the
    deseasonalised trend.  Additive model.  Endpoint-friendly.

    With ``robust=True`` (the default) each inner LOWESS pass runs a bisquare
    re-weighting step so that gross outliers are down-weighted and pushed into
    the residual instead of corrupting the trend and seasonal components; set
    ``robust=False`` for a faster, classical-decomposition-like fit."""
    yy = _as_floats(y)
    n = len(yy)
    if period < 2 or n < 2 * period:
        raise ValueError("need period >= 2 and at least two full periods")
    inner = 2 if robust else 1
    # SEED the trend with a centred moving average (constant-filled at the ends).
    # Starting from a zero trend, the first detrend is a no-op, so each trending
    # seasonal sub-series is smoothed as-is and the TREND leaks into the seasonal
    # (the trend estimate then collapses to the global mean).  A real initial
    # trend makes the sub-series roughly flat so the seasonal is recovered cleanly.
    cma = centered_moving_average(yy, period)
    fi = next(i for i in range(n) if cma[i] is not None)
    li = max(i for i in range(n) if cma[i] is not None)
    trend = [cma[i] if cma[i] is not None else (cma[fi] if i < fi else cma[li])
             for i in range(n)]
    seasonal = [0.0] * n
    for _ in range(max(1, iterations)):
        detr = [yy[t] - trend[t] for t in range(n)]
        # smooth each seasonal sub-series (all points of a given phase) with LOWESS
        for phase in range(period):
            idxs = list(range(phase, n, period))
            sub = [detr[i] for i in idxs]
            sm = lowess(sub, frac=min(1.0, seasonal_frac), iterations=inner) if len(sub) > 2 else sub
            for k, i in enumerate(idxs):
                seasonal[i] = sm[k]
        # centre the seasonal component (mean 0)
        mu = _mean(seasonal)
        seasonal = [s - mu for s in seasonal]
        deseason = [yy[t] - seasonal[t] for t in range(n)]
        trend = lowess(deseason, frac=min(1.0, trend_frac), iterations=inner)
    resid = [yy[t] - trend[t] - seasonal[t] for t in range(n)]
    return Decomposition(observed=yy, trend=list(trend), seasonal=seasonal,
                         resid=list(resid), model="additive", period=period)


# ===========================================================================
# 6. AUTOCORRELATION & AUTOREGRESSION
# ===========================================================================
def autocovariance(y: Sequence[float], nlags: int) -> Vector:
    """Biased sample autocovariances ``gamma_0 .. gamma_nlags`` (divisor ``n``)."""
    yy = _as_floats(y)
    n = len(yy)
    if nlags < 0 or nlags >= n:
        raise ValueError("need 0 <= nlags < len(series)")
    mu = _mean(yy)
    dev = [v - mu for v in yy]
    return [math.fsum(dev[t] * dev[t + k] for t in range(n - k)) / n
            for k in range(nlags + 1)]


def autocorrelation(y: Sequence[float], nlags: int) -> Vector:
    """Sample autocorrelation function ``rho_0=1 .. rho_nlags``.  Scale-invariant:
    only an effectively-constant series collapses to white noise."""
    g = autocovariance(y, nlags)
    if _is_constant(_as_floats(y)):
        return [1.0] + [0.0] * nlags
    return [gk / g[0] for gk in g]


def partial_autocorrelation(y: Sequence[float], nlags: int) -> Vector:
    """Partial autocorrelation function via the Durbin-Levinson recursion
    (``pacf[0]=1``)."""
    if nlags < 0:
        raise ValueError("nlags must be >= 0")
    acf = autocorrelation(y, nlags)
    pacf = [1.0]
    if nlags == 0:
        return pacf
    phi = [[0.0] * (nlags + 1) for _ in range(nlags + 1)]
    phi[1][1] = acf[1]
    pacf.append(acf[1])
    for k in range(2, nlags + 1):
        num = acf[k] - math.fsum(phi[k - 1][j] * acf[k - j] for j in range(1, k))
        den = 1.0 - math.fsum(phi[k - 1][j] * acf[j] for j in range(1, k))
        pk = num / den if abs(den) > _EPS else 0.0
        phi[k][k] = pk
        for j in range(1, k):
            phi[k][j] = phi[k - 1][j] - pk * phi[k - 1][k - j]
        pacf.append(pk)
    return pacf


def yule_walker(y: Sequence[float], order: int) -> Tuple[Vector, float, float]:
    """Fit an AR(``order``) model by the Yule-Walker equations.  Returns
    ``(phi, intercept, sigma2)`` where ``phi`` are the AR coefficients, the
    intercept makes the model mean-correct, and ``sigma2`` is the noise variance."""
    yy = _as_floats(y)
    if order < 1 or order >= len(yy):
        raise ValueError("need 1 <= order < len(series)")
    g = autocovariance(yy, order)
    if _is_constant(yy):                  # effectively-constant series: AR is
        return [0.0] * order, _mean(yy), 0.0   # degenerate -> flat at the mean
    # Solve the Yule-Walker system on the CORRELATION matrix (unit diagonal), not
    # the covariance matrix: an AR fit is scale-free, but a covariance matrix on
    # tiny-magnitude data would have entries below _solve's absolute pivot floor
    # and be wrongly rejected as singular.  rho = g / g[0] makes it scale-invariant
    # and well-conditioned.
    rho = [gk / g[0] for gk in g]
    R = [[rho[abs(i - j)] for j in range(order)] for i in range(order)]
    r = [rho[k + 1] for k in range(order)]
    phi = _solve(R, r)
    sigma2 = g[0] * (1.0 - math.fsum(phi[k] * rho[k + 1] for k in range(order)))
    mu = _mean(yy)
    intercept = mu * (1.0 - math.fsum(phi))
    return phi, intercept, max(0.0, sigma2)


def burg(y: Sequence[float], order: int) -> Tuple[Vector, float, float]:
    """Fit an AR(``order``) model by Burg's method (minimises forward+backward
    prediction error; often more accurate than Yule-Walker on short series).
    Returns ``(phi, intercept, sigma2)``."""
    yy = _as_floats(y)
    n = len(yy)
    if order < 1 or order >= n:
        raise ValueError("need 1 <= order < len(series)")
    mu = _mean(yy)
    if _is_constant(yy):                  # degenerate -> flat at the mean (and
        return [0.0] * order, mu, 0.0     # avoids dividing by sd == 0 below)
    # normalise to unit scale before the recursion: AR coefficients are scale-free,
    # but dk (a sum of squares) hits the absolute _EPS threshold on tiny-magnitude
    # data and mu_k collapses to 0 (a silent all-zero fit).  Rescale sigma2 at the end.
    sd = _std(yy)
    x = [(v - mu) / sd for v in yy]
    f = list(x)
    bb = list(x)
    a = [0.0] * (order + 1)
    a[0] = 1.0
    dk = math.fsum(2.0 * f[i] * f[i] for i in range(n)) - f[0] * f[0] - bb[n - 1] * bb[n - 1]
    sigma2 = math.fsum(v * v for v in x) / n
    for k in range(order):
        num = math.fsum(f[i] * bb[i - 1] for i in range(k + 1, n))
        mu_k = -2.0 * num / dk if abs(dk) > _EPS else 0.0
        for i in range(0, (k + 1) // 2 + 1):
            j = k + 1 - i
            ai, aj = a[i], a[j]
            a[i] = ai + mu_k * aj
            if i != j:
                a[j] = aj + mu_k * ai
        sigma2 *= (1.0 - mu_k * mu_k)
        fp = list(f)
        for i in range(n - 1, k, -1):
            f[i] = fp[i] + mu_k * bb[i - 1]
            bb[i] = bb[i - 1] + mu_k * fp[i]
        dk = (1.0 - mu_k * mu_k) * dk - f[k + 1] * f[k + 1] - bb[n - 1] * bb[n - 1]
    phi = [-a[k + 1] for k in range(order)]
    intercept = mu * (1.0 - math.fsum(phi))
    return phi, intercept, max(0.0, sigma2 * sd * sd)   # undo the unit-scale normalisation


def ar_forecast(y: Sequence[float], order: int, h: int, *, method: str = "yule_walker"
                ) -> ForecastResult:
    """Fit an AR(``order``) model (``method`` = ``'yule_walker'`` or ``'burg'``)
    and forecast ``h`` steps by recursive substitution."""
    yy = _as_floats(y)
    if method == "yule_walker":
        phi, c, sigma2 = yule_walker(yy, order)
    elif method == "burg":
        phi, c, sigma2 = burg(yy, order)
    else:
        raise ValueError("method must be 'yule_walker' or 'burg'")
    n = len(yy)
    fitted = [yy[0]] * order + [
        c + math.fsum(phi[k] * yy[t - 1 - k] for k in range(order))
        for t in range(order, n)]
    residuals = [yy[t] - fitted[t] for t in range(n)]
    hist = list(yy)
    fc: Series = []
    for _ in range(h):
        nxt = c + math.fsum(phi[k] * hist[-1 - k] for k in range(order))
        fc.append(nxt)
        hist.append(nxt)
    sigma = math.sqrt(sigma2)
    lower = upper = None
    if h > 0 and sigma > 0:
        z = _z_score(0.95)
        lower = [fc[k] - z * sigma * math.sqrt(k + 1) for k in range(h)]
        upper = [fc[k] + z * sigma * math.sqrt(k + 1) for k in range(h)]
    return ForecastResult(method="AR(%d)" % order, fitted=fitted, residuals=residuals,
                          forecast=fc, lower=lower, upper=upper,
                          params={"intercept": c, **{"phi%d" % (k + 1): phi[k]
                                                     for k in range(order)}},
                          sse=math.fsum(residuals[t] ** 2 for t in range(order, n)),
                          sigma=sigma,
                          state={"burn": order, "nobs_effective": n - order})


def ljung_box(y: Sequence[float], lags: int = 10, *, dof: int = 0) -> Tuple[float, float]:
    """Ljung-Box test for residual autocorrelation (white-noise test).  Returns
    the ``Q`` statistic and an approximate p-value from the chi-squared survival
    function with ``lags - dof`` degrees of freedom."""
    yy = _as_floats(y)
    n = len(yy)
    if lags < 1 or lags >= n:
        raise ValueError("need 1 <= lags < len(series)")
    rho = autocorrelation(yy, lags)
    q = n * (n + 2) * math.fsum(rho[k] ** 2 / (n - k) for k in range(1, lags + 1))
    df = max(1, lags - dof)
    return q, _chi2_sf(q, df)


def _chi2_sf(x: float, k: int) -> float:
    """Survival function (1 - CDF) of the chi-squared distribution with ``k``
    degrees of freedom, via the regularised upper incomplete gamma function."""
    if x <= 0:
        return 1.0
    return _gammaincc(k / 2.0, x / 2.0)


def _gammaincc(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) (Numerical Recipes style)."""
    if x < 0 or a <= 0:
        return 1.0
    if x < a + 1.0:                                   # series expansion for P, then Q=1-P
        ap = a
        s = 1.0 / a
        d = s
        for _ in range(500):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-14:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b = x + 1.0 - a                                   # continued fraction for Q
    c = 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


# ===========================================================================
# 7. ACCURACY METRICS
# ===========================================================================
def _check_pair(actual: Sequence[float], predicted: Sequence[float]) -> Tuple[Series, Series]:
    a, p = _as_floats(actual), _as_floats(predicted)
    if len(a) != len(p):
        raise ValueError("actual and predicted must have the same length")
    if not a:
        raise ValueError("empty series")
    return a, p


def me(actual: Sequence[float], predicted: Sequence[float]) -> float:
    a, p = _check_pair(actual, predicted)
    return math.fsum(a[i] - p[i] for i in range(len(a))) / len(a)


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    a, p = _check_pair(actual, predicted)
    return math.fsum(abs(a[i] - p[i]) for i in range(len(a))) / len(a)


def mse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    a, p = _check_pair(actual, predicted)
    return math.fsum((a[i] - p[i]) ** 2 for i in range(len(a))) / len(a)


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    return math.sqrt(mse(actual, predicted))


def mpe(actual: Sequence[float], predicted: Sequence[float]) -> float:
    a, p = _check_pair(actual, predicted)
    terms = [(a[i] - p[i]) / a[i] for i in range(len(a)) if a[i] != 0]
    return 100.0 * math.fsum(terms) / len(terms) if terms else _INF


def mape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute percentage error (%).  Undefined where actual is zero;
    those points are skipped."""
    a, p = _check_pair(actual, predicted)
    terms = [abs((a[i] - p[i]) / a[i]) for i in range(len(a)) if a[i] != 0]
    return 100.0 * math.fsum(terms) / len(terms) if terms else _INF


def smape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Symmetric MAPE (%) in the [0, 200] convention."""
    a, p = _check_pair(actual, predicted)
    terms = []
    for i in range(len(a)):
        denom = abs(a[i]) + abs(p[i])
        if denom > 0:
            terms.append(2.0 * abs(a[i] - p[i]) / denom)
    return 100.0 * math.fsum(terms) / len(terms) if terms else 0.0


def mase(actual: Sequence[float], predicted: Sequence[float],
         train: Sequence[float], *, m: int = 1) -> float:
    """Mean absolute scaled error: MAE of the forecast divided by the in-sample
    MAE of a (seasonal, period ``m``) naive forecast on ``train``.  Scale-free and
    the recommended default for comparing across series."""
    a, p = _check_pair(actual, predicted)
    tr = _as_floats(train)
    if len(tr) <= m:
        raise ValueError("training series too short for the naive scale")
    scale = math.fsum(abs(tr[i] - tr[i - m]) for i in range(m, len(tr))) / (len(tr) - m)
    ref = math.fsum(abs(t) for t in tr) / len(tr)      # PURELY relative floor: MASE
    if scale <= _EPS * ref:                             # is scale-free, so a tiny-
        return _INF                                    # but-non-zero scale is fine;
    return mae(a, p) / scale                            # only ~constant train -> inf


def rmsse(actual: Sequence[float], predicted: Sequence[float],
          train: Sequence[float], *, m: int = 1) -> float:
    """Root mean squared scaled error (the M5-competition metric)."""
    a, p = _check_pair(actual, predicted)
    tr = _as_floats(train)
    if len(tr) <= m:
        raise ValueError("training series too short for the naive scale")
    scale = math.fsum((tr[i] - tr[i - m]) ** 2 for i in range(m, len(tr))) / (len(tr) - m)
    ref = math.fsum(t * t for t in tr) / len(tr)       # PURELY relative floor
    if scale <= _EPS * ref:                             # (squared scale) -> RMSSE
        return _INF                                    # stays scale-invariant
    return math.sqrt(mse(a, p) / scale)


def forecast_metrics(actual: Sequence[float], predicted: Sequence[float], *,
                     train: Optional[Sequence[float]] = None, m: int = 1
                     ) -> Dict[str, float]:
    """A dict of the common accuracy metrics.  ``train`` (and ``m``) enable the
    scaled metrics MASE / RMSSE."""
    out = {
        "me": me(actual, predicted),
        "mae": mae(actual, predicted),
        "mse": mse(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mpe": mpe(actual, predicted),
        "mape": mape(actual, predicted),
        "smape": smape(actual, predicted),
    }
    if train is not None:
        out["mase"] = mase(actual, predicted, train, m=m)
        out["rmsse"] = rmsse(actual, predicted, train, m=m)
    return out


# ===========================================================================
# 8. FORECASTER REGISTRY & DISPATCH
# ===========================================================================
def _naive(train: Series, h: int, **kw: Any) -> Series:
    return [train[-1]] * h


def _seasonal_naive(train: Series, h: int, *, m: int = 1, **kw: Any) -> Series:
    if len(train) < m:
        raise ValueError("series shorter than one season")
    return [train[-m + (k % m)] for k in range(h)]


def _drift(train: Series, h: int, **kw: Any) -> Series:
    n = len(train)
    if n < 2:
        return [train[-1]] * h
    slope = (train[-1] - train[0]) / (n - 1)
    return [train[-1] + (k + 1) * slope for k in range(h)]


def _mean_fc(train: Series, h: int, **kw: Any) -> Series:
    return [_mean(train)] * h


def _holt_winters_fc(train: Series, h: int, **kw: Any) -> Series:
    # Holt-Winters is inherently seasonal, so it needs a period; surface a clear
    # message instead of a cryptic "<lambda>() missing 1 required argument" when a
    # caller reaches it via forecast()/back-testing without m.
    m = kw.get("m")
    if not m:
        raise ValueError("forecast(method='holt_winters') requires a seasonal period, "
                         "e.g. forecast(y, 'holt_winters', h, m=12)")
    return holt_winters(train, m, seasonal=kw.get("seasonal", "add"), h=h).forecast


FORECASTERS: Dict[str, Callable[..., Series]] = {
    "naive": _naive,
    "seasonal_naive": _seasonal_naive,
    "drift": _drift,
    "mean": _mean_fc,
    "ses": lambda tr, h, **kw: ses(tr, h=h).forecast,
    "holt": lambda tr, h, **kw: holt(tr, h=h, damped=kw.get("damped", False)).forecast,
    "holt_damped": lambda tr, h, **kw: holt(tr, h=h, damped=True).forecast,
    "holt_winters": _holt_winters_fc,
    "theta": lambda tr, h, **kw: theta_method(tr, h=h,
        seasonal_periods=kw.get("m")).forecast,
    "auto_ets": lambda tr, h, **kw: auto_ets(tr, seasonal_periods=kw.get("m"), h=h).forecast,
    "ar": lambda tr, h, **kw: ar_forecast(tr, kw.get("order", 1), h,
        method=kw.get("ar_method", "yule_walker")).forecast,
}


def forecast(y: Sequence[float], method: str, h: int, **kwargs: Any) -> Series:
    """Dispatch to any forecaster in ``FORECASTERS`` by name and return the
    ``h``-step point forecasts.  Extra keyword args (``m``, ``order``, ...) are
    passed through."""
    if method not in FORECASTERS:
        raise ValueError("unknown method %r (have %s)"
                         % (method, ", ".join(sorted(FORECASTERS))))
    yy = _as_floats(y)
    if h < 1:
        raise ValueError("h must be >= 1")
    return FORECASTERS[method](yy, h, **kwargs)


# ===========================================================================
# 9. BACK-TESTING / CROSS-VALIDATION
# ===========================================================================
def rolling_origin_backtest(y: Sequence[float], method: str, *, h: int = 1,
                            initial: Optional[int] = None, step: int = 1,
                            mode: str = "expanding", window: Optional[int] = None,
                            m: int = 1, **kwargs: Any) -> Dict[str, Any]:
    """Rolling-origin evaluation (a.k.a. time-series cross-validation).  Starting
    from an ``initial`` training length, repeatedly forecast ``h`` steps, score
    against the held-out actuals, then advance the origin by ``step``.  ``mode``
    is ``'expanding'`` (growing window) or ``'sliding'`` (fixed ``window``)."""
    yy = _as_floats(y)
    n = len(yy)
    if method not in FORECASTERS:                       # raise a catchable ValueError
        raise ValueError("unknown method %r (have %s)"  # (mirrors forecast()); a bare
                         % (method, ", ".join(sorted(FORECASTERS))))  # KeyError would
    if mode not in ("expanding", "sliding"):            # escape time_series_cv's guard
        raise ValueError("mode must be 'expanding' or 'sliding'")
    if initial is None:
        initial = max(2 * m, n // 2, 3)
    if window is None:
        window = initial
    if initial < 1 or initial + h > n:
        raise ValueError("initial training length leaves no room for h-step tests")
    origins = list(range(initial, n - h + 1, step))
    all_actual: Series = []
    all_pred: Series = []
    per_origin: List[Dict[str, Any]] = []
    for origin in origins:
        train = yy[:origin] if mode == "expanding" else yy[max(0, origin - window):origin]
        actual = yy[origin:origin + h]
        try:
            pred = FORECASTERS[method](train, h, m=m, **kwargs)
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        pred = list(pred)[:h]
        if len(pred) < h:
            continue
        all_actual.extend(actual)
        all_pred.extend(pred)
        per_origin.append({"origin": origin, "actual": actual, "pred": pred,
                           "rmse": rmse(actual, pred)})
    if not all_actual:
        raise ValueError("no valid backtest origins produced forecasts")
    metrics = forecast_metrics(all_actual, all_pred, train=yy, m=m)
    return {"method": method, "h": h, "mode": mode, "n_origins": len(per_origin),
            "metrics": metrics, "per_origin": per_origin}


def time_series_cv(y: Sequence[float], methods: Sequence[str], *, h: int = 1,
                   initial: Optional[int] = None, step: int = 1, m: int = 1,
                   metric: str = "rmse", **kwargs: Any) -> Dict[str, float]:
    """Compare several forecasters by rolling-origin back-test and return
    ``{method: score}`` on the chosen ``metric`` (lower is better)."""
    if isinstance(methods, str):
        # a bare str is a valid Sequence[str] (of characters), so ``for method in
        # methods`` would silently iterate the letters and score gibberish; catch
        # the singular/plural mix-up with rolling_origin_backtest(method=...).
        raise TypeError("methods must be a list of method names, e.g. [%r], not the "
                        "bare string %r" % (methods, methods))
    scores: Dict[str, float] = {}
    for method in methods:
        try:
            bt = rolling_origin_backtest(y, method, h=h, initial=initial, step=step,
                                         m=m, **kwargs)
            scores[method] = bt["metrics"].get(metric, _INF)
        except (ValueError, ZeroDivisionError, OverflowError):
            scores[method] = _INF
    return scores


# ===========================================================================
# 10. BUILT-IN SAMPLE SERIES
# ===========================================================================
def _airline(n: int = 96, m: int = 12, seed: int = 0) -> Series:
    """A multiplicative trend+seasonal series in the spirit of the classic airline
    passengers data."""
    rng = random.Random(seed)
    out = []
    for t in range(n):
        trend = 100.0 + 2.5 * t
        season = 1.0 + 0.35 * math.sin(2 * math.pi * (t % m) / m)
        noise = 1.0 + rng.gauss(0, 0.03)
        out.append(trend * season * noise)
    return out


def _trend_series(n: int = 60, seed: int = 1) -> Series:
    rng = random.Random(seed)
    return [10.0 + 1.5 * t + rng.gauss(0, 3.0) for t in range(n)]


def _seasonal_series(n: int = 72, m: int = 12, seed: int = 2) -> Series:
    rng = random.Random(seed)
    return [50.0 + 12.0 * math.sin(2 * math.pi * (t % m) / m) + rng.gauss(0, 2.0)
            for t in range(n)]


def _random_walk(n: int = 100, seed: int = 3) -> Series:
    rng = random.Random(seed)
    out, s = [], 0.0
    for _ in range(n):
        s += rng.gauss(0, 1.0)
        out.append(s)
    return out


def _noisy_sine(n: int = 120, period: int = 20, seed: int = 4) -> Series:
    rng = random.Random(seed)
    return [5.0 * math.sin(2 * math.pi * t / period) + rng.gauss(0, 0.5) for t in range(n)]


SAMPLE_SERIES: Dict[str, Callable[[], Series]] = {
    "airline": _airline,
    "trend": _trend_series,
    "seasonal": _seasonal_series,
    "random_walk": _random_walk,
    "noisy_sine": _noisy_sine,
}


def get_series(name: str) -> Series:
    """Instantiate a built-in demo series by name (see ``SAMPLE_SERIES``)."""
    if name not in SAMPLE_SERIES:
        raise ValueError("unknown series %r (have %s)"
                         % (name, ", ".join(sorted(SAMPLE_SERIES))))
    return SAMPLE_SERIES[name]()


# ===========================================================================
# 11. JSON PAYLOADS  (strict-JSON for dashboards)
# ===========================================================================
def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float):
        if obj != obj or obj in (_INF, -_INF):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def build_forecast_payload(result: ForecastResult) -> Dict[str, Any]:
    """Strict-JSON summary of a ForecastResult for a dashboard / API."""
    return _json_safe({
        "method": result.method,
        "params": result.params,
        "forecast": result.forecast,
        "lower": result.lower,
        "upper": result.upper,
        "fitted": result.fitted,
        "residuals": result.residuals,
        "sse": result.sse,
        "sigma": result.sigma,
        "aic": result.aic,
        "aicc": result.aicc,
    })


def build_decomposition_payload(dec: Decomposition) -> Dict[str, Any]:
    """Strict-JSON summary of a Decomposition."""
    return _json_safe({
        "model": dec.model,
        "period": dec.period,
        "observed": dec.observed,
        "trend": dec.trend,
        "seasonal": dec.seasonal,
        "resid": dec.resid,
        "seasonally_adjusted": dec.seasonally_adjusted(),
    })


# ===========================================================================
# 12. DEMO
# ===========================================================================
def _demo() -> None:
    sep = "=" * 70
    print(sep); print("TIME-SERIES SMOOTHING & FORECASTING TEMPLATE DEMO"); print(sep)

    print("\n[1] Smoothers on a noisy sine (last 5 values):")
    ns = get_series("noisy_sine")
    print("    raw           :", [round(v, 2) for v in ns[-5:]])
    print("    SMA(5)        :", [round(v, 2) for v in simple_moving_average(ns, 5)[-5:]])
    print("    EMA(a=0.3)    :", [round(v, 2) for v in exponential_moving_average(ns, 0.3)[-5:]])
    print("    Savitzky-Golay:", [round(v, 2) for v in savitzky_golay(ns, 7, 2)[-5:]])
    print("    LOWESS        :", [round(v, 2) for v in lowess(ns, 0.3)[-5:]])

    print("\n[2] Simple exponential smoothing (auto alpha) on a random walk:")
    rw = get_series("random_walk")
    r = ses(rw, h=3)
    print("    alpha=%.3f  forecast(3)=%s" % (r.params["alpha"], [round(v, 2) for v in r.forecast]))

    print("\n[3] Holt's damped trend on a trending series:")
    tr = get_series("trend")
    r = holt(tr, damped=True, h=4)
    print("    params=%s" % {k: round(v, 3) for k, v in r.params.items()})
    print("    forecast(4)=%s" % [round(v, 2) for v in r.forecast])

    print("\n[4] Holt-Winters (multiplicative) on the airline series:")
    air = get_series("airline")
    r = holt_winters(air, 12, seasonal="mul", h=12)
    print("    params=%s" % {k: round(v, 3) for k, v in r.params.items()})
    print("    12-step forecast[:4]=%s" % [round(v, 1) for v in r.forecast[:4]])
    print("    AICc=%.1f" % (r.aicc if r.aicc is not None else float("nan")))

    print("\n[5] auto_ets picks a model by AICc:")
    print("    chosen:", auto_ets(air, seasonal_periods=12, h=1).method)

    print("\n[6] Classical decomposition of the airline series:")
    dec = classical_decomposition(air, 12, model="multiplicative")
    print("    seasonal indices:", [round(v, 3) for v in dec.seasonal[:12]])

    print("\n[7] ACF / PACF of the seasonal series (first 5 lags):")
    ss = get_series("seasonal")
    print("    ACF :", [round(v, 3) for v in autocorrelation(ss, 5)])
    print("    PACF:", [round(v, 3) for v in partial_autocorrelation(ss, 5)])

    print("\n[8] AR(2) forecast + Ljung-Box residual test:")
    r = ar_forecast(ss, 2, 4, method="burg")
    q, pval = ljung_box(r.residuals[2:], lags=8)
    print("    forecast(4)=%s  Ljung-Box Q=%.2f p=%.3f" %
          ([round(v, 2) for v in r.forecast], q, pval))

    print("\n[9] Rolling-origin back-test comparing methods (RMSE, h=6):")
    scores = time_series_cv(air, ["naive", "seasonal_naive", "drift", "ses",
                                  "holt_winters"], h=6, m=12, seasonal="mul")
    for method, s in sorted(scores.items(), key=lambda kv: kv[1]):
        print("    %-16s RMSE=%.2f" % (method, s))

    print("\n" + sep); print("demo OK"); print(sep)


if __name__ == "__main__":
    _demo()
