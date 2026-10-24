"""
Forecasting -- Baselines: a self-contained, zero-dependency toolkit
===================================================================

**Every strong forecast starts from a strong baseline.**  Before an LSTM or a
gradient-boosted lag model earns its keep it has to beat the classical reference
forecasters -- naive, seasonal-naive, drift, exponential smoothing, the Theta
method, regression-on-time, autoregression -- scored with the right metric under
honest rolling-origin backtesting.  This module is that whole reference toolbox,
built from scratch on the Python standard library alone.  No NumPy, no pandas, no
statsmodels / sktime / darts / prophet: the linear algebra (Gaussian-pivot solve,
Householder QR least-squares, Cholesky), the exponential-smoothing recursions and
their optimiser, the ACF/PACF (Levinson-Durbin), the decomposition and every
metric are implemented by hand.  Copy ``forecasting_baseline_template.py`` into a
competition / hackathon / interview / embedded sandbox and go.

Layers
------
1.  **Linear algebra** -- matmul / transpose / solve (partial pivoting), ordinary
    and ridge least-squares by BOTH normal equations AND Householder QR, Cholesky.
2.  **Time-series statistics** -- mean/var/median/quantile, autocovariance, ``acf``
    / ``pacf`` (Levinson-Durbin), Ljung-Box, seasonality detection & strength,
    ``norm_ppf`` / ``t_ppf`` quantiles, moving averages, OLS trend line.
3.  **Transforms** -- log / Box-Cox (+ Guerrero-lite lambda) with exact inverses,
    (seasonal) differencing + integration, standardise / min-max scalers.
4.  **Forecaster interface** -- a ``BaseForecaster`` ABC (fit / predict / forecast /
    fitted / resid / predict_interval) and a uniform ``ForecastResult``.
5.  **Naive baselines** -- naive, seasonal-naive, drift, mean, (weighted) moving
    average, random-walk-with-drift, Naive2 (the M4 MASE reference).
6.  **Exponential smoothing (ETS)** -- SES, Holt (linear / damped / exponential),
    Holt-Winters additive & multiplicative, a general engine with from-scratch
    Nelder-Mead / golden-section fitting, and ``auto_ets`` model selection.
7.  **Decomposition & Theta** -- classical additive / multiplicative decomposition,
    (de)seasonalise, and the Theta method (M3 winner, ``thetaf`` closed form).
8.  **Regression forecasters** -- deterministic trend (poly), seasonal dummies,
    Fourier terms, and a general time-series linear regression.
9.  **Autoregression & intermittent demand** -- AR(p) via OLS and Yule-Walker with
    AIC/BIC order selection; Croston, SBA and TSB.
10. **Evaluation** -- ME/MAE/MSE/RMSE/MAPE/sMAPE/WAPE/MASE/RMSSE/MPE/R2, pinball
    loss, interval coverage / width / MSIS; rolling-origin cross-validation.
11. **Combination** -- mean / median / trimmed / weighted / inverse-error ensembles.
12. **Datasets** -- deterministic synthetic series (trend, season, AR, intermittent).

Conventions
-----------
* A time series ``y`` is a plain ``list`` of floats, equally spaced, no gaps.
* Forecast horizons ``h`` count 1..h steps past the end of ``y``.
* Everything is deterministic given its inputs; only multi-start optimisation and
  simulation intervals draw randomness, always via an explicit ``seed``.
* Seasonal period is ``m`` (e.g. 12 monthly, 4 quarterly, 7 daily-weekly).
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple)

__version__ = "1.0.0"

Vector = List[float]
Matrix = List[List[float]]

_EPS = 1e-12
_TINY = 1e-8
_BIG = 1e18

__all__ = [
    # exceptions / types
    "ForecastError", "LinAlgError", "ForecastResult", "DecompositionResult",
    # linear algebra
    "zeros_vec", "zeros_mat", "eye", "transpose", "matmul", "matvec", "dot",
    "solve", "cholesky", "lstsq", "lstsq_qr", "ridge_lstsq", "qr_householder",
    # statistics
    "mean", "variance", "std", "median", "quantile", "autocovariance", "acf",
    "pacf", "ljung_box", "norm_ppf", "norm_cdf", "t_ppf", "ols_line",
    "moving_average", "seasonality_test", "detect_seasonal_period",
    "seasonal_strength", "trend_strength",
    # transforms
    "log_transform", "inv_log_transform", "boxcox", "inv_boxcox",
    "boxcox_lambda", "difference", "integrate", "seasonal_difference",
    "inv_seasonal_difference", "StandardScaler", "MinMaxScaler",
    # base
    "BaseForecaster",
    # naive
    "Naive", "SeasonalNaive", "Drift", "MeanForecast", "MovingAverageForecaster",
    "WeightedMovingAverage", "Naive2",
    # ets
    "SESFit", "ses_fit", "SimpleExpSmoothing", "Holt", "HoltWinters",
    "ETSForecaster", "auto_ets",
    # decomposition + theta
    "classical_decomposition", "seasonal_indices", "deseasonalise",
    "reseasonalise", "seasonal_adjust", "Theta", "theta_forecast",
    # regression
    "trend_features", "seasonal_dummy_features", "fourier_features",
    "LinearRegressionForecaster", "TrendForecaster", "SeasonalDummyForecaster",
    "FourierForecaster",
    # ar + intermittent
    "ARForecaster", "yule_walker", "Croston", "TSB",
    # metrics
    "me", "mae", "mse", "rmse", "mape", "smape", "wape", "mase", "rmsse",
    "mpe", "r2_score", "median_ae", "pinball_loss", "coverage", "interval_width",
    "msis", "evaluate",
    # cross-validation
    "train_test_split", "rolling_origin_splits", "cross_validate", "backtest",
    # combination
    "combine_mean", "combine_median", "combine_weighted", "combine_trimmed",
    "inverse_error_weights",
    # datasets
    "make_trend_seasonal", "make_ar_series", "make_intermittent",
]


# ===========================================================================
# 0. EXCEPTIONS & VALIDATION HELPERS
# ===========================================================================
class ForecastError(ValueError):
    """Invalid input to a forecasting routine (bad series, horizon, config)."""


class LinAlgError(ForecastError):
    """A singular or non-conformable matrix was encountered."""


def _is_finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (math.isnan(x) or math.isinf(x))


def _as_floats(x: Sequence[Any], name: str = "y") -> Vector:
    try:
        out = [float(v) for v in x]
    except (TypeError, ValueError):
        raise ForecastError("%s must be a sequence of numbers." % name)
    return out


def _check_y(y: Sequence[Any], min_len: int = 1, name: str = "y",
             positive: bool = False) -> Vector:
    out = _as_floats(y, name)
    if len(out) < min_len:
        raise ForecastError("%s needs at least %d observations (got %d)."
                            % (name, min_len, len(out)))
    if any(not _is_finite(v) for v in out):
        raise ForecastError("%s contains non-finite values." % name)
    if positive and any(v <= 0.0 for v in out):
        raise ForecastError("%s must be strictly positive for this method." % name)
    return out


def _check_h(h: int) -> int:
    if not isinstance(h, int) or h < 1:
        raise ForecastError("horizon h must be an integer >= 1 (got %r)." % (h,))
    return h


def _check_m(m: int, n: int) -> int:
    if not isinstance(m, int) or m < 1:
        raise ForecastError("seasonal period m must be an integer >= 1 (got %r)." % (m,))
    return m


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if abs(b) > _EPS:
        return a / b
    return default


def _mul_safe_div(a: float, b: float) -> float:
    """Sign-preserving guarded division for multiplicative denominators."""
    if abs(b) > _EPS:
        return a / b
    return a / (_EPS if b >= 0 else -_EPS)


# ===========================================================================
# 1. LINEAR ALGEBRA (from scratch)
# ===========================================================================
def zeros_vec(n: int) -> Vector:
    if n < 0:
        raise LinAlgError("zeros_vec length must be >= 0.")
    return [0.0] * n


def zeros_mat(r: int, c: int) -> Matrix:
    if r < 0 or c < 0:
        raise LinAlgError("zeros_mat dimensions must be >= 0.")
    return [[0.0] * c for _ in range(r)]


def eye(n: int) -> Matrix:
    m = zeros_mat(n, n)
    for i in range(n):
        m[i][i] = 1.0
    return m


def transpose(A: Matrix) -> Matrix:
    if not A:
        return []
    c = len(A[0])
    if any(len(row) != c for row in A):
        raise LinAlgError("transpose: ragged matrix.")
    return [list(col) for col in zip(*A)]


def matmul(A: Matrix, B: Matrix) -> Matrix:
    if not A or not B:
        raise LinAlgError("matmul: empty matrix.")
    if len(A[0]) != len(B):
        raise LinAlgError("matmul: inner dimensions %d != %d." % (len(A[0]), len(B)))
    Bt = transpose(B)
    return [[math.fsum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def matvec(A: Matrix, x: Vector) -> Vector:
    if not A:
        return []
    if len(A[0]) != len(x):
        raise LinAlgError("matvec: dimension mismatch.")
    return [math.fsum(a * xi for a, xi in zip(row, x)) for row in A]


def dot(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise LinAlgError("dot: length mismatch.")
    return math.fsum(ai * bi for ai, bi in zip(a, b))


def solve(A: Matrix, b: Vector) -> Vector:
    """Solve the square system ``A x = b`` by Gaussian elimination with partial
    pivoting.  Raises ``LinAlgError`` if ``A`` is singular."""
    n = len(A)
    if n == 0 or any(len(row) != n for row in A) or len(b) != n:
        raise LinAlgError("solve: A must be square and conform with b.")
    M = [list(row) + [b[i]] for i, row in enumerate(A)]
    scale = max((abs(v) for row in A for v in row), default=1.0) or 1.0
    tol = _EPS * scale
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[p][k]) <= tol:
            raise LinAlgError("solve: singular matrix.")
        if p != k:
            M[k], M[p] = M[p], M[k]
        pivot = M[k][k]
        for i in range(k + 1, n):
            f = M[i][k] / pivot
            if f != 0.0:
                for j in range(k, n + 1):
                    M[i][j] -= f * M[k][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n] - math.fsum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / M[i][i]
    return x


def cholesky(A: Matrix, jitter: float = 0.0) -> Matrix:
    """Lower-triangular Cholesky factor ``L`` with ``A = L Lᵀ`` (adaptive jitter)."""
    n = len(A)
    for attempt in range(7):
        add = jitter if attempt == 0 else jitter + 1e-12 * (10 ** attempt)
        L = zeros_mat(n, n)
        ok = True
        for i in range(n):
            for j in range(i + 1):
                s = A[i][j] - math.fsum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    s += add
                    if s <= 0.0:
                        ok = False
                        break
                    L[i][j] = math.sqrt(s)
                else:
                    L[i][j] = s / L[j][j]
            if not ok:
                break
        if ok:
            return L
    raise LinAlgError("cholesky: matrix is not positive definite.")


def qr_householder(A: Matrix) -> Tuple[Matrix, Matrix]:
    """Reduced Householder QR: returns ``(Q, R)`` with ``A = Q R``, ``R`` upper
    triangular, ``Q`` with orthonormal columns."""
    n = len(A)
    if n == 0:
        return [], []
    p = len(A[0])
    R = [list(row) for row in A]
    Q = eye(n)
    for k in range(min(n - 1, p)):
        x = [R[i][k] for i in range(k, n)]
        normx = math.sqrt(math.fsum(v * v for v in x))
        if normx <= _EPS:
            continue
        alpha = -normx if x[0] >= 0 else normx
        v = list(x)
        v[0] -= alpha
        vnorm = math.sqrt(math.fsum(t * t for t in v))
        if vnorm <= _EPS:
            continue
        v = [t / vnorm for t in v]
        # apply H = I - 2 v vᵀ to R[k:, :] and Q[:, k:]
        for j in range(p):
            s = math.fsum(v[i] * R[k + i][j] for i in range(n - k))
            for i in range(n - k):
                R[k + i][j] -= 2.0 * v[i] * s
        for j in range(n):
            s = math.fsum(v[i] * Q[j][k + i] for i in range(n - k))
            for i in range(n - k):
                Q[j][k + i] -= 2.0 * v[i] * s
    kk = min(n, p)                          # economy QR: Q is n x kk, R is kk x p
    Qr = [[Q[i][j] for j in range(kk)] for i in range(n)]
    Rr = [[R[i][j] for j in range(p)] for i in range(kk)]
    return Qr, Rr


def lstsq_qr(X: Matrix, y: Vector) -> Vector:
    """Least-squares ``min ‖Xβ − y‖`` via numerically-stable Householder QR.

    Applies each reflector to ``y`` in place and never forms the ``n×n`` ``Q``, so
    it is O(n·p²) time and O(n·p) memory -- linear in the series length for a fixed
    feature count (forming ``Q`` would make every regression / OLS-AR fit O(n²))."""
    n = len(X)
    if n == 0:
        return []
    p = len(X[0])
    if len(y) != n:
        raise LinAlgError("lstsq_qr: X rows != len(y).")
    R = [list(row) for row in X]            # becomes upper-triangular in its first p rows
    yv = list(y)                            # accumulates Qᵀy under the reflectors
    for k in range(min(n, p)):
        normx = math.sqrt(math.fsum(R[i][k] * R[i][k] for i in range(k, n)))
        if normx <= _EPS:
            continue
        alpha = -normx if R[k][k] >= 0 else normx
        v = [R[k][k] - alpha] + [R[i][k] for i in range(k + 1, n)]
        vnorm = math.sqrt(math.fsum(t * t for t in v))
        if vnorm <= _EPS:
            continue
        v = [t / vnorm for t in v]
        for j in range(k, p):               # reflect the remaining R columns
            s = math.fsum(v[i] * R[k + i][j] for i in range(n - k))
            for i in range(n - k):
                R[k + i][j] -= 2.0 * v[i] * s
        sy = math.fsum(v[i] * yv[k + i] for i in range(n - k))   # ...and y
        for i in range(n - k):
            yv[k + i] -= 2.0 * v[i] * sy
    beta = [0.0] * p                        # extra (free) variables stay 0
    for i in range(min(n, p) - 1, -1, -1):
        if abs(R[i][i]) <= _EPS:
            continue
        s = yv[i] - math.fsum(R[i][j] * beta[j] for j in range(i + 1, p))
        beta[i] = s / R[i][i]
    return beta


def ridge_lstsq(X: Matrix, y: Vector, ridge: float = 0.0,
                penalise_intercept: bool = False) -> Vector:
    """Least-squares via the (optionally ridged) normal equations ``(XᵀX+λI)β=Xᵀy``."""
    n = len(X)
    if n == 0:
        return []
    p = len(X[0])
    if len(y) != n:
        raise LinAlgError("ridge_lstsq: X rows != len(y).")
    Xt = transpose(X)
    XtX = [[math.fsum(Xt[i][t] * Xt[j][t] for t in range(n)) for j in range(p)]
           for i in range(p)]
    Xty = [math.fsum(Xt[i][t] * y[t] for t in range(n)) for i in range(p)]
    if ridge > 0.0:
        for i in range(p):
            if penalise_intercept or i > 0:
                XtX[i][i] += ridge
    return solve(XtX, Xty)


def lstsq(X: Matrix, y: Vector, ridge: float = 0.0, method: str = "qr") -> Vector:
    """Ordinary/ridge least squares.  ``method='qr'`` (stable) or ``'normal'``."""
    if ridge > 0.0 or method == "normal":
        return ridge_lstsq(X, y, ridge=ridge)
    return lstsq_qr(X, y)


# ===========================================================================
# 2. TIME-SERIES STATISTICS
# ===========================================================================
def mean(x: Vector) -> float:
    if not x:
        raise ForecastError("mean of empty sequence.")
    return math.fsum(x) / len(x)


def _degenerate_variance(g0: float, y: Vector) -> bool:
    """True if ``y`` is (numerically) constant, tested SCALE-INVARIANTLY.

    Autocorrelation / R² / AR estimation are all invariant to rescaling ``y``, so
    the "zero variance" guard must compare the variance ``g0`` to the size of the
    series' own deviations, not to an absolute constant -- otherwise a genuinely
    varying but tiny-amplitude series (e.g. returns ~1e-6) is wrongly treated as
    constant.  (This is the recurring 'absolute threshold on a scale-dependent
    quantity' class.)"""
    if not y:
        return True
    mu = math.fsum(y) / len(y)
    scale = max((v - mu) ** 2 for v in y)
    return g0 <= _EPS * scale


def variance(x: Vector, ddof: int = 0) -> float:
    n = len(x)
    if n - ddof <= 0:
        return 0.0
    mu = mean(x)
    return math.fsum((v - mu) ** 2 for v in x) / (n - ddof)


def std(x: Vector, ddof: int = 0) -> float:
    return math.sqrt(max(variance(x, ddof), 0.0))


def median(x: Vector) -> float:
    if not x:
        raise ForecastError("median of empty sequence.")
    s = sorted(x)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def quantile(x: Vector, q: float) -> float:
    """Linear-interpolation quantile (type 7, the R/NumPy default)."""
    if not x:
        raise ForecastError("quantile of empty sequence.")
    if not 0.0 <= q <= 1.0:
        raise ForecastError("q must lie in [0, 1].")
    s = sorted(x)
    n = len(s)
    if n == 1:
        return s[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    frac = pos - lo
    if lo + 1 < n:
        return s[lo] + frac * (s[lo + 1] - s[lo])
    return s[lo]


def autocovariance(y: Vector, max_lag: int, adjusted: bool = False,
                   demean: bool = True) -> Vector:
    """Sample autocovariances ``gamma_0..gamma_max_lag`` (biased by default)."""
    n = len(y)
    mu = mean(y) if demean else 0.0
    d = [v - mu for v in y]
    out = []
    for k in range(max_lag + 1):
        s = math.fsum(d[t] * d[t - k] for t in range(k, n))
        out.append(s / ((n - k) if adjusted else n))
    return out


def acf(y: Vector, nlags: int) -> Vector:
    """Autocorrelation function ``[1, r_1, ..., r_nlags]``."""
    y = _check_y(y, 1)
    g = autocovariance(y, nlags)
    if _degenerate_variance(g[0], y):
        return [1.0] + [0.0] * nlags
    return [gk / g[0] for gk in g]


def pacf(y: Vector, nlags: int) -> Vector:
    """Partial autocorrelation ``[1, a_1, ..., a_nlags]`` via Levinson-Durbin."""
    y = _check_y(y, 1)
    g = autocovariance(y, nlags)
    out = [1.0]
    if _degenerate_variance(g[0], y):
        return out + [0.0] * nlags
    a = [0.0] * (nlags + 1)
    E = g[0]
    e_tol = _EPS * g[0]                     # relative to the series' own variance (scale-invariant)
    for k in range(1, nlags + 1):
        acc = g[k] - math.fsum(a[j] * g[k - j] for j in range(1, k))
        kref = acc / E if E > e_tol else 0.0
        new = list(a)
        new[k] = kref
        for j in range(1, k):
            new[j] = a[j] - kref * a[k - j]
        a = new
        E *= (1.0 - kref * kref)
        out.append(kref)
        if E <= e_tol:
            out.extend([0.0] * (nlags - k))
            break
    return out


def ljung_box(y: Vector, lags: int = 10) -> Tuple[float, int]:
    """Ljung-Box Q statistic and its degrees of freedom (test for autocorrelation)."""
    y = _check_y(y, 2)
    n = len(y)
    r = acf(y, lags)[1:]
    q = n * (n + 2) * math.fsum((r[k] ** 2) / (n - k - 1) for k in range(lags)
                                if n - k - 1 > 0)
    return q, lags


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_cdf(x: float) -> float:
    """Standard-normal CDF."""
    return _norm_cdf(x)


# Acklam's rational approximation to the inverse normal CDF (rel. err < 1.2e-9).
_ACKLAM_A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
_ACKLAM_B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
_ACKLAM_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
_ACKLAM_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (probit) via Acklam's approximation + Halley step."""
    p = min(max(p, _EPS), 1.0 - _EPS)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q +
               _ACKLAM_C[3]) * q + _ACKLAM_C[4]) * q + _ACKLAM_C[5]) / \
            ((((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q + _ACKLAM_D[2]) * q +
              _ACKLAM_D[3]) * q + 1.0)
    elif p <= phigh:
        q = p - 0.5
        r = q * q
        x = (((((_ACKLAM_A[0] * r + _ACKLAM_A[1]) * r + _ACKLAM_A[2]) * r +
               _ACKLAM_A[3]) * r + _ACKLAM_A[4]) * r + _ACKLAM_A[5]) * q / \
            (((((_ACKLAM_B[0] * r + _ACKLAM_B[1]) * r + _ACKLAM_B[2]) * r +
               _ACKLAM_B[3]) * r + _ACKLAM_B[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q +
                _ACKLAM_C[3]) * q + _ACKLAM_C[4]) * q + _ACKLAM_C[5]) / \
            ((((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q + _ACKLAM_D[2]) * q +
              _ACKLAM_D[3]) * q + 1.0)
    # one Halley refinement
    e = _norm_cdf(x) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    x = x - u / (1.0 + x * u / 2.0)
    return x


def _betacf(a: float, b: float, x: float) -> float:
    """Lentz's continued fraction for the incomplete beta function."""
    fpmin = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for mm in range(1, 300):
        m2 = 2 * mm
        aa = mm * (b - mm) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + mm) * (qab + mm) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function ``I_x(a, b)`` (from scratch)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _t_cdf(t: float, df: float) -> float:
    """Student-t CDF via the incomplete beta function."""
    ib = 0.5 * _betai(0.5 * df, 0.5, df / (df + t * t))
    return ib if t <= 0.0 else 1.0 - ib


def t_ppf(p: float, df: float) -> float:
    """Student-t quantile: a Cornish-Fisher initial guess refined by Newton's method
    against the exact t-CDF (accurate for all ``df``, including df = 1, 2)."""
    if df <= 0:
        raise ForecastError("t_ppf: df must be > 0.")
    p = min(max(p, _EPS), 1.0 - _EPS)
    if df > 1e6:
        return norm_ppf(p)
    z = norm_ppf(p)
    z2 = z * z
    x = z + (z2 * z + z) / (4.0 * df) + (5.0 * z2 * z2 * z + 16.0 * z2 * z
                                         + 3.0 * z) / (96.0 * df * df)
    log_c = (math.lgamma(0.5 * (df + 1.0)) - math.lgamma(0.5 * df)
             - 0.5 * math.log(df * math.pi))
    for _ in range(80):                                    # Newton on the exact CDF
        pdf = math.exp(log_c - 0.5 * (df + 1.0) * math.log(1.0 + x * x / df))
        if pdf < 1e-300:
            break
        step = (_t_cdf(x, df) - p) / pdf
        x -= step
        if abs(step) < 1e-12 * (abs(x) + 1e-12):
            break
    return x


def _z_value(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ForecastError("confidence level must be in (0, 1).")
    return norm_ppf(0.5 + level / 2.0)


def ols_line(y: Vector, t0: int = 0) -> Tuple[float, float]:
    """OLS straight line ``a + b·t`` of ``y`` on ``t = t0, t0+1, ...``  ->  ``(a, b)``."""
    n = len(y)
    if n == 0:
        raise ForecastError("ols_line: empty series.")
    if n == 1:
        return y[0], 0.0
    tbar = t0 + (n - 1) / 2.0
    ybar = mean(y)
    sxx = math.fsum((t0 + i - tbar) ** 2 for i in range(n))
    sxy = math.fsum((t0 + i - tbar) * (y[i] - ybar) for i in range(n))
    b = sxy / sxx if sxx > _EPS else 0.0
    a = ybar - b * tbar
    return a, b


def moving_average(y: Vector, order: int, centered: bool = True) -> List[Optional[float]]:
    """Order-``order`` moving average (2×order for even ``order`` when centered).

    Returns ``None`` where the window overhangs an end (never NaN)."""
    n = len(y)
    out: List[Optional[float]] = [None] * n
    if order <= 1:
        return [float(v) for v in y]
    if order > n:
        return out
    if order % 2 == 1:
        k = (order - 1) // 2
        for t in range(k, n - k):
            out[t] = math.fsum(y[t - k:t + k + 1]) / order
    else:
        k = order // 2
        for t in range(k, n - k):
            s = 0.5 * y[t - k] + math.fsum(y[t - k + 1:t + k]) + 0.5 * y[t + k]
            out[t] = s / order
    return out


def seasonality_test(y: Vector, m: int, conf: float = 0.90) -> bool:
    """The ``thetaf`` seasonality gate: is the lag-``m`` autocorrelation significant?"""
    y = _check_y(y, 1)
    n = len(y)
    if m <= 1 or n <= 2 * m:
        return False
    g = autocovariance(y, m)
    if _degenerate_variance(g[0], y):
        return False
    r = [gk / g[0] for gk in g]
    stat = math.sqrt((1.0 + 2.0 * math.fsum(r[k] ** 2 for k in range(1, m))) / n)
    if stat < _EPS:
        return False
    z = norm_ppf(0.5 + conf / 2.0)
    return abs(r[m]) / stat > z


def detect_seasonal_period(y: Vector, candidates: Sequence[int] = (2, 3, 4, 6, 7, 12,
                           24, 52), max_lag: Optional[int] = None) -> int:
    """Pick the candidate period with the strongest significant ACF peak (1 = none)."""
    y = _check_y(y, 3)
    n = len(y)
    ml = max_lag if max_lag is not None else min(n - 1, max(candidates))
    if ml < 2:
        return 1
    r = acf(y, min(ml + 1, n - 1))              # +1 lag so the peak test can see r[m+1]
    best_m, best_r = 1, 0.0
    for m in candidates:
        if m < 2 or m > ml or n <= 2 * m:
            continue
        left = r[m - 1]
        right = r[m + 1] if m + 1 < len(r) else -1.0
        # A seasonal period shows a POSITIVE LOCAL PEAK in the ACF (r[m] stands above
        # its neighbours).  A monotone-decaying autocorrelation -- an AR process -- has
        # r[m] < r[m-1] and is filtered out here, and an anti-phase half-period has
        # r[m] < 0; both would otherwise be mistaken for seasonality.
        if (r[m] > 0.0 and r[m] >= left and r[m] >= right
                and seasonality_test(y, m) and r[m] > best_r):
            best_m, best_r = m, r[m]
    return best_m


def seasonal_strength(y: Vector, m: int) -> float:
    """Strength of seasonality in [0, 1] (Wang-Smith-Hyndman), 0 if none."""
    y = _check_y(y, 1)
    if m <= 1 or len(y) < 2 * m:
        return 0.0
    dec = classical_decomposition(y, m)
    detr = [y[t] - (dec.trend[t] if dec.trend[t] is not None else 0.0)
            for t in range(len(y)) if dec.trend[t] is not None]
    resid = [dec.residual[t] for t in range(len(y)) if dec.residual[t] is not None]
    if not resid:
        return 0.0
    var_r = variance(resid)
    var_dt = variance(detr) if len(detr) > 1 else 0.0
    if var_dt < _EPS:
        return 0.0
    return max(0.0, 1.0 - var_r / var_dt)


def trend_strength(y: Vector, m: int = 1) -> float:
    """Strength of trend in [0, 1] (deseasonalised residual vs detrended-adjusted)."""
    y = _check_y(y, 3)
    if m > 1 and len(y) >= 2 * m:
        dec = classical_decomposition(y, m)
        adj = [y[t] - dec.seasonal[t] for t in range(len(y))]
    else:
        adj = list(y)
    trend = moving_average(adj, m if m > 1 else 3)
    dtr, res = [], []
    for t in range(len(y)):
        if trend[t] is not None:
            dtr.append(trend[t])
            res.append(adj[t] - trend[t])
    if len(res) < 2:
        return 0.0
    detrended_var = variance([adj[t] for t in range(len(y)) if trend[t] is not None])
    if detrended_var < _EPS:
        return 0.0
    return max(0.0, 1.0 - variance(res) / detrended_var)


# ===========================================================================
# 3. TRANSFORMS (each invertible)
# ===========================================================================
def log_transform(y: Vector) -> Vector:
    y = _check_y(y, 1, positive=True)
    return [math.log(v) for v in y]


def inv_log_transform(z: Vector) -> Vector:
    return [math.exp(v) for v in z]


def boxcox(y: Vector, lam: float) -> Vector:
    """Box-Cox transform with parameter ``lam`` (``lam=0`` -> log)."""
    y = _check_y(y, 1, positive=True)
    if abs(lam) < _EPS:
        return [math.log(v) for v in y]
    return [(v ** lam - 1.0) / lam for v in y]


def inv_boxcox(z: Vector, lam: float) -> Vector:
    """Inverse Box-Cox."""
    if abs(lam) < _EPS:
        return [math.exp(v) for v in z]
    return [max(_EPS, (lam * v + 1.0)) ** (1.0 / lam) for v in z]


def boxcox_lambda(y: Vector, m: int = 1, bounds: Tuple[float, float] = (-1.0, 2.0),
                  n_grid: int = 61) -> float:
    """Guerrero-lite: pick ``lam`` minimising the coefficient of variation of the
    per-group standard deviation (groups = seasonal cycles, or contiguous blocks)."""
    y = _check_y(y, 2, positive=True)
    n = len(y)
    g = m if m > 1 else max(2, int(round(math.sqrt(n))))
    groups = [y[i:i + g] for i in range(0, n, g) if len(y[i:i + g]) >= 2]
    if len(groups) < 2:
        return 1.0

    def dispersion(lam: float) -> float:
        ratios = []
        for grp in groups:
            mu = mean(grp)
            sd = std(grp, ddof=1)
            if mu > _EPS:
                ratios.append(sd / (mu ** (1.0 - lam)))
        if len(ratios) < 2:
            return _BIG
        mr = mean(ratios)
        return _safe_div(std(ratios, ddof=1), mr, _BIG)

    lo, hi = bounds
    best_lam, best = 1.0, _BIG
    for i in range(n_grid):
        lam = lo + (hi - lo) * i / (n_grid - 1)
        d = dispersion(lam)
        if d < best:
            best, best_lam = d, lam
    return best_lam


def difference(y: Vector, d: int = 1) -> Vector:
    """``d``-th order first differencing."""
    out = _check_y(y, 1)
    for _ in range(d):
        out = [out[t] - out[t - 1] for t in range(1, len(out))]
    return out


def integrate(diffs: Vector, initial: Sequence[float]) -> Vector:
    """Invert :func:`difference`.  ``initial`` = the first ``d`` values of the
    original series, so that ``integrate(difference(y, d), y[:d]) == y``.

    The ``d`` cumulative sums are anchored by the *per-level* initial conditions
    (the ``k``-th forward difference of ``initial`` at position 0), which are
    reconstructed here from the raw first ``d`` values."""
    init = list(initial)
    d = len(init)
    anchors = []
    cur = init
    for _ in range(d):
        anchors.append(cur[0])
        cur = [cur[t] - cur[t - 1] for t in range(1, len(cur))]
    out = list(diffs)
    for a in reversed(anchors):
        cum = [a]
        for v in out:
            cum.append(cum[-1] + v)
        out = cum
    return out


def seasonal_difference(y: Vector, m: int) -> Vector:
    """Seasonal differencing ``y_t − y_{t−m}``."""
    y = _check_y(y, m + 1)
    return [y[t] - y[t - m] for t in range(m, len(y))]


def inv_seasonal_difference(diffs: Vector, initial: Sequence[float], m: int) -> Vector:
    """Invert :func:`seasonal_difference`.  ``initial`` = the first ``m`` values."""
    if len(initial) != m:
        raise ForecastError("inv_seasonal_difference: initial must have length m.")
    out = list(initial)
    for v in diffs:
        out.append(out[-m] + v)
    return out


@dataclass
class StandardScaler:
    """Zero-mean unit-variance scaler (fit/transform/inverse_transform)."""
    mean_: float = 0.0
    std_: float = 1.0

    def fit(self, y: Vector) -> "StandardScaler":
        y = _check_y(y, 1)
        self.mean_ = mean(y)
        self.std_ = std(y) or 1.0
        return self

    def transform(self, y: Vector) -> Vector:
        return [(v - self.mean_) / self.std_ for v in y]

    def inverse_transform(self, z: Vector) -> Vector:
        return [v * self.std_ + self.mean_ for v in z]

    def fit_transform(self, y: Vector) -> Vector:
        return self.fit(y).transform(y)


@dataclass
class MinMaxScaler:
    """Scale to ``[0, 1]`` (fit/transform/inverse_transform)."""
    min_: float = 0.0
    range_: float = 1.0

    def fit(self, y: Vector) -> "MinMaxScaler":
        y = _check_y(y, 1)
        self.min_ = min(y)
        self.range_ = (max(y) - self.min_) or 1.0
        return self

    def transform(self, y: Vector) -> Vector:
        return [(v - self.min_) / self.range_ for v in y]

    def inverse_transform(self, z: Vector) -> Vector:
        return [v * self.range_ + self.min_ for v in z]

    def fit_transform(self, y: Vector) -> Vector:
        return self.fit(y).transform(y)


# ===========================================================================
# 4. FORECASTER INTERFACE
# ===========================================================================
@dataclass
class ForecastResult:
    """Uniform output of every forecaster's :meth:`forecast`."""
    mean: Vector
    lower: Optional[Vector] = None
    upper: Optional[Vector] = None
    level: Optional[float] = None
    method: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    fitted: Optional[Vector] = None
    residuals: Optional[Vector] = None
    sigma2: Optional[float] = None


class BaseForecaster(ABC):
    """Abstract base: ``fit(y)`` then ``predict(h)`` / ``forecast(h, level)``.

    Subclasses implement ``_fit`` (store state, set ``self.fitted_``) and
    ``_predict`` (point forecasts).  The base provides fitted values, residuals
    and residual-based prediction intervals."""

    def __init__(self) -> None:
        self.y_: Vector = []
        self.n_: int = 0
        self.fitted_: Vector = []
        self.resid_: Vector = []
        self._is_fit = False
        self.method: str = self.__class__.__name__

    # -- required hooks --
    @abstractmethod
    def _fit(self, y: Vector) -> None: ...

    @abstractmethod
    def _predict(self, h: int) -> Vector: ...

    # -- public API --
    def fit(self, y: Sequence[float]) -> "BaseForecaster":
        self.y_ = _check_y(y, self._min_obs())
        self.n_ = len(self.y_)
        self._fit(self.y_)
        if not self.fitted_:
            self.fitted_ = list(self.y_)
        self.resid_ = [self.y_[t] - self.fitted_[t] for t in range(self.n_)]
        self._is_fit = True
        return self

    def predict(self, h: int) -> Vector:
        self._require_fit()
        return self._predict(_check_h(h))

    def forecast(self, h: int, level: Optional[float] = None) -> ForecastResult:
        self._require_fit()
        h = _check_h(h)
        mean_ = self._predict(h)
        lo = up = None
        if level is not None:
            lo, up = self.predict_interval(h, level)
        return ForecastResult(mean=mean_, lower=lo, upper=up, level=level,
                              method=self.method, fitted=list(self.fitted_),
                              residuals=list(self.resid_), sigma2=self._sigma2())

    def fitted_values(self) -> Vector:
        self._require_fit()
        return list(self.fitted_)

    def resid(self) -> Vector:
        self._require_fit()
        return list(self.resid_)

    def predict_interval(self, h: int, level: float = 0.95) -> Tuple[Vector, Vector]:
        """Default interval: Gaussian residual std, widened by ``sqrt(step)``."""
        self._require_fit()
        h = _check_h(h)
        pt = self._predict(h)
        s = math.sqrt(self._sigma2())
        z = _z_value(level)
        lo = [pt[i] - z * s * math.sqrt(i + 1) for i in range(h)]
        up = [pt[i] + z * s * math.sqrt(i + 1) for i in range(h)]
        return lo, up

    # -- helpers subclasses may override --
    def _min_obs(self) -> int:
        return 1

    def _sigma2(self) -> float:
        r = [self.resid_[t] for t in range(len(self.resid_))]
        if len(r) <= 1:
            return 0.0
        return math.fsum(e * e for e in r) / len(r)

    def _require_fit(self) -> None:
        if not self._is_fit:
            raise ForecastError("forecaster is not fitted; call fit(y) first.")


# ===========================================================================
# 5. NAIVE & SIMPLE BASELINES
# ===========================================================================
class Naive(BaseForecaster):
    """Naive forecast: ``ŷ_{n+h} = y_n`` (random walk)."""

    def _fit(self, y: Vector) -> None:
        self.fitted_ = [y[0]] + list(y[:-1])          # one-step lag

    def _predict(self, h: int) -> Vector:
        return [self.y_[-1]] * h

    def predict_interval(self, h: int, level: float = 0.95) -> Tuple[Vector, Vector]:
        pt = self._predict(h)
        s = math.sqrt(self._sigma2())
        z = _z_value(level)
        lo = [pt[i] - z * s * math.sqrt(i + 1) for i in range(h)]
        up = [pt[i] + z * s * math.sqrt(i + 1) for i in range(h)]
        return lo, up


class SeasonalNaive(BaseForecaster):
    """Seasonal naive: ``ŷ_{n+h} = y_{n+h-m(k+1)}`` (last value of the same season)."""

    def __init__(self, m: int) -> None:
        super().__init__()
        self.m = m
        self.method = "SeasonalNaive(m=%d)" % m

    def _min_obs(self) -> int:
        return max(1, self.m)

    def _fit(self, y: Vector) -> None:
        m = self.m
        self.fitted_ = [y[t - m] if t >= m else y[t] for t in range(len(y))]

    def _predict(self, h: int) -> Vector:
        m, y = self.m, self.y_
        n = len(y)
        return [y[n - m + ((i) % m)] for i in range(h)]

    def predict_interval(self, h: int, level: float = 0.95) -> Tuple[Vector, Vector]:
        pt = self._predict(h)
        s = math.sqrt(self._sigma2())
        z = _z_value(level)
        m = self.m
        # variance grows by the number of completed seasonal cycles ahead + 1
        lo, up = [], []
        for i in range(h):
            k = i // m
            w = math.sqrt(k + 1)
            lo.append(pt[i] - z * s * w)
            up.append(pt[i] + z * s * w)
        return lo, up


class Drift(BaseForecaster):
    """Drift: extrapolate the straight line through the first and last points,
    ``ŷ_{n+h} = y_n + h·(y_n − y_1)/(n − 1)``."""

    def _min_obs(self) -> int:
        return 2

    def _fit(self, y: Vector) -> None:
        n = len(y)
        self.slope_ = (y[-1] - y[0]) / (n - 1)
        self.fitted_ = [y[0]] + [y[t - 1] + self.slope_ for t in range(1, n)]

    def _predict(self, h: int) -> Vector:
        return [self.y_[-1] + (i + 1) * self.slope_ for i in range(h)]


class MeanForecast(BaseForecaster):
    """Mean forecast: ``ŷ_{n+h} = ȳ`` for all ``h``."""

    def _fit(self, y: Vector) -> None:
        self.mean_ = mean(y)
        self.fitted_ = [self.mean_] * len(y)

    def _predict(self, h: int) -> Vector:
        return [self.mean_] * h

    def predict_interval(self, h: int, level: float = 0.95) -> Tuple[Vector, Vector]:
        pt = self._predict(h)
        n = self.n_
        s = std(self.y_, ddof=1) if n > 1 else 0.0
        se = s * math.sqrt(1.0 + 1.0 / n)                # mean forecast SE
        z = _z_value(level)
        return [v - z * se for v in pt], [v + z * se for v in pt]


class MovingAverageForecaster(BaseForecaster):
    """Flat forecast equal to the mean of the last ``window`` observations."""

    def __init__(self, window: int = 3) -> None:
        super().__init__()
        if window < 1:
            raise ForecastError("window must be >= 1.")
        self.window = window
        self.method = "MovingAverage(window=%d)" % window

    def _fit(self, y: Vector) -> None:
        w = self.window
        n = len(y)
        fitted = []
        for t in range(n):
            if t == 0:
                fitted.append(y[0])
            else:
                lo = max(0, t - w)
                fitted.append(mean(y[lo:t]))
        self.fitted_ = fitted
        self.level_ = mean(y[-w:]) if n >= 1 else y[-1]

    def _predict(self, h: int) -> Vector:
        return [self.level_] * h


class WeightedMovingAverage(BaseForecaster):
    """Flat forecast from a weighted average of the last ``len(weights)`` points
    (weights need not be normalised; most-recent-last)."""

    def __init__(self, weights: Sequence[float]) -> None:
        super().__init__()
        w = _as_floats(weights, "weights")
        if not w or math.fsum(w) == 0.0:
            raise ForecastError("weights must be non-empty and not sum to zero.")
        s = math.fsum(w)
        self.weights = [wi / s for wi in w]
        self.method = "WeightedMovingAverage(k=%d)" % len(w)

    def _min_obs(self) -> int:
        return len(self.weights)

    def _fit(self, y: Vector) -> None:
        k = len(self.weights)
        n = len(y)
        fitted = [y[0]] * min(k, n)
        for t in range(k, n):
            fitted.append(math.fsum(self.weights[j] * y[t - k + j] for j in range(k)))
        self.fitted_ = fitted[:n]
        self.level_ = math.fsum(self.weights[j] * y[n - k + j] for j in range(k))

    def _predict(self, h: int) -> Vector:
        return [self.level_] * h


class Naive2(BaseForecaster):
    """Naive2: seasonally-adjust (classical decomposition), naive-forecast the
    adjusted series, then re-seasonalise.  The M4 MASE / OWA reference model."""

    def __init__(self, m: int = 1, model: str = "multiplicative") -> None:
        super().__init__()
        self.m = m
        self.model = _norm_model(model)
        self.method = "Naive2(m=%d)" % m

    def _fit(self, y: Vector) -> None:
        m = self.m
        self.seasonal_ = None
        if m > 1 and len(y) >= 2 * m and seasonality_test(y, m):
            self.seasonal_ = seasonal_indices(y, m, model=self.model)
            adj = deseasonalise(y, self.seasonal_, model=self.model, start_phase=0)
        else:
            adj = list(y)
        self.adj_ = adj
        self.mase_scale = _mase_denominator(y, m)
        # fitted: naive on adjusted then reseasonalise
        fit_adj = [adj[0]] + list(adj[:-1])
        if self.seasonal_ is not None:
            self.fitted_ = reseasonalise(fit_adj, self.seasonal_,
                                         model=self.model, start_phase=0)
        else:
            self.fitted_ = fit_adj

    def _predict(self, h: int) -> Vector:
        base = [self.adj_[-1]] * h
        if self.seasonal_ is not None:
            return reseasonalise(base, self.seasonal_, model=self.model,
                                 start_phase=self.n_ % self.m)
        return base


# ===========================================================================
# 6. EXPONENTIAL SMOOTHING (ETS)
# ===========================================================================
def _golden_section(f: Callable[[float], float], a: float, b: float,
                    tol: float = 1e-8, max_iter: int = 200) -> Tuple[float, float]:
    """Golden-section minimiser of a unimodal ``f`` on ``[a, b]``."""
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    invphi2 = (3.0 - math.sqrt(5.0)) / 2.0
    c = a + invphi2 * (b - a)
    d = a + invphi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = a + invphi2 * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)
    x = 0.5 * (a + b)
    return x, f(x)


def _nelder_mead(f: Callable[[Vector], float], x0: Vector, step: Vector,
                 tol: float = 1e-10, max_iter: int = 2000) -> Tuple[Vector, float]:
    """Downhill-simplex minimiser (reflect / expand / contract / shrink)."""
    n = len(x0)
    if n == 0:
        return [], f([])
    simplex = [list(x0)]
    for j in range(n):
        p = list(x0)
        p[j] += step[j] if step[j] != 0 else 0.05
        simplex.append(p)
    fvals = [f(p) for p in simplex]
    ra, rex, rc, rsh = 1.0, 2.0, 0.5, 0.5
    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if abs(fvals[-1] - fvals[0]) <= tol * (abs(fvals[0]) + abs(fvals[-1]) + _EPS):
            break
        centroid = [math.fsum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        xr = [centroid[j] + ra * (centroid[j] - simplex[-1][j]) for j in range(n)]
        fr = f(xr)
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[0]:
            xe = [centroid[j] + rex * (xr[j] - centroid[j]) for j in range(n)]
            fe = f(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        else:
            xc = [centroid[j] + rc * (simplex[-1][j] - centroid[j]) for j in range(n)]
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + rsh * (simplex[i][j] - simplex[0][j])
                                  for j in range(n)]
                    fvals[i] = f(simplex[i])
    best = min(range(n + 1), key=lambda i: fvals[i])
    return simplex[best], fvals[best]


@dataclass
class SESFit:
    alpha: float
    l0: float
    level: float
    fitted: Vector
    residuals: Vector
    sse: float
    sigma2: float


def _ses_run(y: Vector, alpha: float, l0: float) -> Tuple[Vector, Vector, float]:
    lev = l0
    fitted, resid = [], []
    for t in range(len(y)):
        fitted.append(lev)
        e = y[t] - lev
        resid.append(e)
        lev = lev + alpha * e
    sse = math.fsum(e * e for e in resid)
    return fitted, resid, lev


def _ses_optimal_l0(y: Vector, alpha: float) -> float:
    """Closed-form profiled initial level for fixed ``alpha`` (SSE is quadratic in l0)."""
    n = len(y)
    num = den = 0.0
    p = 0.0                              # particular part with l0 = 0
    for t in range(n):
        c = (1.0 - alpha) ** t          # homogeneous coefficient of l0 in ŷ_t = l_{t-1}
        num += (y[t] - p) * c
        den += c * c
        p = p + alpha * (y[t] - p)       # update particular level (l0=0 branch)
    return _safe_div(num, den, y[0])


def ses_fit(y: Vector, alpha: Optional[float] = None,
            alpha_bounds: Tuple[float, float] = (1e-4, 0.9999),
            init: str = "optimal", dof: int = 1) -> SESFit:
    """Fit Simple Exponential Smoothing = ETS(A,N,N)."""
    y = _check_y(y, 1)
    n = len(y)
    if n == 1:
        return SESFit(alpha_bounds[0], y[0], y[0], [y[0]], [0.0], 0.0, 0.0)

    def l0_for(a: float) -> float:
        if init == "first":
            return y[0]
        if init == "mean":
            return mean(y)
        return _ses_optimal_l0(y, a)

    def sse_for(a: float) -> float:
        _, resid, _ = _ses_run(y, a, l0_for(a))
        return math.fsum(e * e for e in resid)

    if alpha is None:
        a_star, _ = _golden_section(sse_for, alpha_bounds[0], alpha_bounds[1])
    else:
        a_star = alpha
    l0 = l0_for(a_star)
    fitted, resid, level = _ses_run(y, a_star, l0)
    sse = math.fsum(e * e for e in resid)
    sigma2 = sse / (n - dof) if n > dof else (sse / n if n > 0 else 0.0)
    return SESFit(a_star, l0, level, fitted, resid, sse, sigma2)


def _norm_model(model: str) -> str:
    mstr = str(model).lower()
    if mstr in ("additive", "add", "a"):
        return "additive"
    if mstr in ("multiplicative", "mul", "mult", "m"):
        return "multiplicative"
    raise ForecastError("model must be 'additive' or 'multiplicative'.")


def _damped_sum(phi: float, h: int) -> float:
    """``phi + phi² + ... + phi^h`` (the multi-step damping factor); ``h`` if phi≈1."""
    if abs(1.0 - phi) < _EPS:
        return float(h)
    return phi * (1.0 - phi ** h) / (1.0 - phi)


@dataclass
class ETSState:
    alpha: float
    beta: float
    gamma: float
    phi: float
    l0: float
    b0: float
    s0: Vector


class ETSForecaster(BaseForecaster):
    """General exponential smoothing (ETS) engine.

    ``trend``   : None | 'add' | 'mul'      (mul = multiplicative/exponential trend)
    ``seasonal``: None | 'add' | 'mul'
    Damping (``damped=True``) applies ``phi`` to the trend.  Free smoothing
    parameters are estimated by minimising the one-step in-sample SSE (golden
    section for pure SES, else multi-start Nelder-Mead).  Additive-component
    models get analytic prediction intervals; multiplicative ones widen the
    residual std by ``sqrt(step)``."""

    def __init__(self, trend: Optional[str] = None, damped: bool = False,
                 seasonal: Optional[str] = None, seasonal_periods: Optional[int] = None,
                 alpha: Optional[float] = None, beta: Optional[float] = None,
                 gamma: Optional[float] = None, phi: Optional[float] = None,
                 n_restarts: int = 6, seed: int = 0) -> None:
        super().__init__()
        self.trend = _norm_comp(trend)
        self.seasonal = _norm_comp(seasonal)
        self.damped = damped and self.trend is not None
        self.m = seasonal_periods if self.seasonal else 1
        self.alpha, self.beta, self.gamma, self.phi = alpha, beta, gamma, phi
        self.n_restarts = n_restarts
        self.seed = seed
        if self.seasonal and (not seasonal_periods or seasonal_periods < 2):
            raise ForecastError("seasonal ETS needs seasonal_periods >= 2.")
        self.method = "ETS(%s,%s,%s)" % (
            "A", ({None: "N", "add": "A", "mul": "M"}[self.trend]) +
            ("d" if self.damped else ""),
            {None: "N", "add": "A", "mul": "M"}[self.seasonal])

    def _min_obs(self) -> int:
        if self.seasonal:
            return max(self.m + 1, 2)
        return 2 if self.trend else 1

    def _fit(self, y: Vector) -> None:
        if self.seasonal == "mul" or self.trend == "mul":
            if min(y) <= 0:
                raise ForecastError("multiplicative ETS requires strictly positive y.")
        state0 = self._init_state(y)
        free = self._free_mask()
        theta0, unpack = self._pack(state0, free)

        def objective(theta: Vector) -> float:
            st = unpack(theta)
            res = self._run(y, st)
            if res is None:
                return _BIG
            _, resid, _, _, _ = res
            sse = math.fsum(e * e for e in resid)
            return sse if _is_finite(sse) else _BIG

        if not theta0:
            best_theta = []
        elif len(theta0) == 1 and self._only_alpha_free(free):
            a, _ = _golden_section(lambda a: objective([a]), _TINY, 1.0 - _TINY)
            best_theta = [a]
        else:
            rng = random.Random(self.seed)
            starts = [theta0]
            grid = [0.1, 0.3, 0.5, 0.7, 0.9]
            for _ in range(max(0, self.n_restarts - 1)):
                starts.append([rng.choice(grid) if 0 <= v <= 1 else v for v in theta0])
            best_theta, best = theta0, _BIG
            for s in starts:
                cand, val = _nelder_mead(objective, s,
                                         [0.1] * len(s), tol=1e-10, max_iter=1500)
                if val < best:
                    best, best_theta = val, cand
        self.state_ = unpack(best_theta) if best_theta or theta0 == [] else state0
        run = self._run(y, self.state_)
        if run is None:
            raise ForecastError("ETS fit produced non-finite states.")
        fitted, resid, ln, bn, seas = run
        self.fitted_ = fitted
        self.level_n_, self.trend_n_, self.season_last_ = ln, bn, seas
        self._sse_ = math.fsum(e * e for e in resid)

    def _predict(self, h: int) -> Vector:
        ln, bn = self.level_n_, self.trend_n_
        seas = self.season_last_
        m = self.m
        out = []
        for i in range(1, h + 1):
            if self.trend is None:
                tt = ln
            elif self.trend == "add":
                tt = ln + _damped_sum(self.state_.phi, i) * bn
            else:  # mul
                exps = _damped_sum(self.state_.phi, i)
                tt = ln * (_pow_clamped(bn, exps))
            if self.seasonal == "add":
                out.append(tt + seas[(i - 1) % m])
            elif self.seasonal == "mul":
                out.append(tt * seas[(i - 1) % m])
            else:
                out.append(tt)
        return out

    def predict_interval(self, h: int, level: float = 0.95) -> Tuple[Vector, Vector]:
        pt = self._predict(h)
        z = _z_value(level)
        n0 = self.n_ - (self.m if self.seasonal else 1)
        sigma2 = self._sse_ / max(1, n0)
        if self.seasonal != "mul" and self.trend != "mul":
            lo, up = [], []
            for i in range(1, h + 1):
                vh = self._var_ratio(i)
                se = math.sqrt(max(sigma2 * vh, 0.0))
                lo.append(pt[i - 1] - z * se)
                up.append(pt[i - 1] + z * se)
            return lo, up
        s = math.sqrt(sigma2)
        return ([pt[i] - z * s * math.sqrt(i + 1) for i in range(h)],
                [pt[i] + z * s * math.sqrt(i + 1) for i in range(h)])

    # -- analytic additive variance ratios v_h (Hyndman et al. 2008) --
    def _var_ratio(self, h: int) -> float:
        a = self.state_.alpha
        b = self.state_.alpha * self.state_.beta if self.trend else 0.0  # combined beta
        g = self.state_.gamma if self.seasonal else 0.0
        phi = self.state_.phi if self.damped else 1.0
        m = self.m
        k = (h - 1) // m
        v = 1.0
        if self.trend and not self.damped:
            v += (h - 1) * (a * a + a * b * h + (b * b * h / 6.0) * (2 * h - 1))
        elif self.trend and self.damped:
            v += a * a * (h - 1)
            if abs(1.0 - phi) > _EPS:
                v += (b * phi * h / (1.0 - phi) ** 2) * (2 * a * (1.0 - phi) + b * phi)
                v -= (b * phi * (1.0 - phi ** h) /
                      ((1.0 - phi) ** 2 * (1.0 - phi * phi))) * \
                     (2 * a * (1.0 - phi * phi) + b * phi * (1.0 + 2 * phi - phi ** h))
        else:
            v += a * a * (h - 1)
        if self.seasonal:
            v += g * k * (2 * a + g + (b * m * (k + 1) if self.trend else 0.0))
        return max(v, 1.0)

    # -- state init (classical-decomposition heuristic) --
    def _init_state(self, y: Vector) -> ETSState:
        m = self.m
        n = len(y)
        s0: Vector = []
        if self.seasonal and n >= 2 * m:
            idx = seasonal_indices(y, m, model=("multiplicative"
                                   if self.seasonal == "mul" else "additive"))
            s0 = list(idx)
            adj = deseasonalise(y, s0, model=("multiplicative"
                                if self.seasonal == "mul" else "additive"))
        elif self.seasonal:
            base = mean(y[:m]) if n >= m else mean(y)
            if self.seasonal == "add":
                s0 = _seasonal_norm_add([y[i] - base for i in range(min(m, n))] +
                                        [0.0] * max(0, m - n))
            else:
                s0 = _seasonal_norm_mul([_mul_safe_div(y[i], base)
                                         for i in range(min(m, n))] + [1.0] * max(0, m - n))
            adj = list(y)
        else:
            adj = list(y)
        k = min(len(adj), 10)
        a, b = ols_line(adj[:k])
        l0 = a
        if self.trend == "add":
            b0 = b
        elif self.trend == "mul":
            b0 = max(1.0 + b / max(abs(l0), _EPS), _TINY)
        else:
            b0 = 0.0
        return ETSState(
            alpha=self.alpha if self.alpha is not None else 0.3,
            beta=self.beta if self.beta is not None else 0.1,
            gamma=self.gamma if self.gamma is not None else 0.1,
            phi=self.phi if self.phi is not None else (0.98 if self.damped else 1.0),
            l0=l0, b0=b0, s0=s0)

    # -- the master recursion; returns None on infeasible states --
    def _run(self, y: Vector, st: ETSState):
        m = self.m
        n = len(y)
        alpha, beta, gamma, phi = st.alpha, st.beta, st.gamma, st.phi
        if not self.damped:
            phi = 1.0
        lev, tr = st.l0, st.b0
        seas = list(st.s0) if self.seasonal else []
        fitted, resid = [], []
        for t in range(n):
            # trend term (pre-update level+trend)
            if self.trend is None:
                base = lev
            elif self.trend == "add":
                base = lev + phi * tr
            else:
                base = lev * _pow_clamped(tr, phi)
            s_tm = seas[t % m] if self.seasonal else None
            if self.seasonal == "add":
                yhat = base + s_tm
            elif self.seasonal == "mul":
                yhat = base * s_tm
            else:
                yhat = base
            if not _is_finite(yhat):
                return None
            e = y[t] - yhat
            fitted.append(yhat)
            resid.append(e)
            # deseasonalised observation for the level update
            if self.seasonal == "add":
                deseason = y[t] - s_tm
            elif self.seasonal == "mul":
                if abs(s_tm) < _EPS:
                    return None
                deseason = y[t] / s_tm
            else:
                deseason = y[t]
            new_lev = alpha * deseason + (1.0 - alpha) * base
            if self.trend == "add":
                new_tr = beta * (new_lev - lev) + (1.0 - beta) * phi * tr
            elif self.trend == "mul":
                if abs(lev) < _EPS:
                    return None
                new_tr = beta * (new_lev / lev) + (1.0 - beta) * _pow_clamped(tr, phi)
            else:
                new_tr = tr
            if self.seasonal == "add":
                seas[t % m] = gamma * (y[t] - base) + (1.0 - gamma) * s_tm
            elif self.seasonal == "mul":
                if abs(base) < _EPS:
                    return None
                seas[t % m] = gamma * (y[t] / base) + (1.0 - gamma) * s_tm
            lev, tr = new_lev, new_tr
            if not (_is_finite(lev) and _is_finite(tr)):
                return None
        last_season = []
        if self.seasonal:
            # order the last full cycle so index (i-1)%m aligns with h=1..
            last_season = [seas[(n + j) % m] for j in range(m)]
        return fitted, resid, lev, tr, last_season

    # -- parameter packing / bounds --
    def _free_mask(self) -> Dict[str, bool]:
        return {
            "alpha": self.alpha is None,
            "beta": self.beta is None and self.trend is not None,
            "gamma": self.gamma is None and self.seasonal is not None,
            "phi": self.phi is None and self.damped,
        }

    def _only_alpha_free(self, free: Dict[str, bool]) -> bool:
        return free["alpha"] and not (free["beta"] or free["gamma"] or free["phi"])

    def _pack(self, st: ETSState, free: Dict[str, bool]):
        order = [k for k in ("alpha", "beta", "gamma", "phi") if free[k]]
        theta0 = [getattr(st, k) for k in order]

        def unpack(theta: Vector) -> ETSState:
            vals = dict(alpha=st.alpha, beta=st.beta, gamma=st.gamma, phi=st.phi)
            for name, v in zip(order, theta):
                if name == "phi":
                    vals[name] = min(0.999, max(0.8, v))
                else:
                    vals[name] = min(1.0 - _TINY, max(_TINY, v))
            return ETSState(vals["alpha"], vals["beta"], vals["gamma"], vals["phi"],
                            st.l0, st.b0, st.s0)

        return theta0, unpack

    def _sigma2(self) -> float:
        n0 = max(1, self.n_ - (self.m if self.seasonal else 1))
        return self._sse_ / n0


def _norm_comp(c: Optional[str]) -> Optional[str]:
    if c is None:
        return None
    cs = str(c).lower()
    if cs in ("add", "additive", "a"):
        return "add"
    if cs in ("mul", "multiplicative", "m"):
        return "mul"
    raise ForecastError("component must be None, 'add' or 'mul'.")


def _pow_clamped(base: float, exp: float) -> float:
    if base <= 0.0:
        return _EPS
    le = exp * math.log(base)
    le = min(50.0, max(-50.0, le))
    return math.exp(le)


def _seasonal_norm_add(s: Vector) -> Vector:
    mu = mean(s)
    return [v - mu for v in s]


def _seasonal_norm_mul(s: Vector) -> Vector:
    mu = mean(s)
    if abs(mu) < _EPS:
        return [1.0] * len(s)
    return [v / mu for v in s]


# -- convenience wrappers --
def SimpleExpSmoothing(alpha: Optional[float] = None) -> ETSForecaster:
    """SES = ETS(A,N,N)."""
    return ETSForecaster(trend=None, seasonal=None, alpha=alpha)


def Holt(damped: bool = False, exponential: bool = False,
         alpha: Optional[float] = None, beta: Optional[float] = None,
         phi: Optional[float] = None) -> ETSForecaster:
    """Holt's linear (or damped / exponential) trend method."""
    return ETSForecaster(trend="mul" if exponential else "add", damped=damped,
                         alpha=alpha, beta=beta, phi=phi)


def HoltWinters(seasonal_periods: int, seasonal: str = "add", trend: str = "add",
                damped: bool = False, **kw: Any) -> ETSForecaster:
    """Holt-Winters seasonal method (additive or multiplicative seasonality)."""
    return ETSForecaster(trend=trend, seasonal=seasonal, damped=damped,
                         seasonal_periods=seasonal_periods, **kw)


def auto_ets(y: Vector, seasonal_periods: Optional[int] = None,
             allow_multiplicative: Optional[bool] = None,
             damped_candidates: Sequence[bool] = (False, True),
             criterion: str = "aicc", seed: int = 0) -> ETSForecaster:
    """Fit every admissible ETS combination and keep the best by AIC/AICc/BIC."""
    if criterion not in ("aic", "aicc", "bic"):
        raise ForecastError("criterion must be 'aic', 'aicc' or 'bic' (got %r)."
                            % (criterion,))
    y = _check_y(y, 2)
    n = len(y)
    m = seasonal_periods
    pos = min(y) > 0
    if allow_multiplicative is None:
        allow_multiplicative = pos
    trends = [None, "add"] + (["mul"] if (allow_multiplicative and pos) else [])
    seasons = [None]
    if m and m >= 2 and n >= 2 * m:
        seasons += ["add"] + (["mul"] if (allow_multiplicative and pos) else [])
    best_model, best_ic = None, _BIG
    for tr in trends:
        for damp in damped_candidates:
            if damp and tr is None:
                continue
            for sea in seasons:
                try:
                    f = ETSForecaster(trend=tr, damped=damp, seasonal=sea,
                                      seasonal_periods=m if sea else None, seed=seed)
                    f.fit(y)
                except ForecastError:
                    continue
                k = (1 + (1 if tr else 0) + (1 if damp else 0) +
                     (1 if sea else 0))
                # Every candidate's SSE is over all n one-step residuals, so score the
                # likelihood on the SAME n for all: a per-candidate effective sample
                # (n-m vs n-1) makes the AIC/BIC comparison non-scale-invariant.
                n0 = n
                sigma2 = max(f._sse_ / n0, _EPS)
                ll = -0.5 * n0 * (math.log(2 * math.pi) + 1 + math.log(sigma2))
                aic = -2 * ll + 2 * k
                if criterion == "bic":
                    ic = -2 * ll + k * math.log(n0)
                elif criterion == "aic":
                    ic = aic
                else:
                    ic = aic + (2 * k * (k + 1) / (n0 - k - 1)) if n0 - k - 1 > 0 else _BIG
                if ic < best_ic:
                    best_ic, best_model = ic, f
    if best_model is None:
        return ETSForecaster().fit(y)
    return best_model


# ===========================================================================
# 7. DECOMPOSITION & THETA
# ===========================================================================
@dataclass
class DecompositionResult:
    observed: Vector
    trend: List[Optional[float]]
    seasonal: Vector
    residual: List[Optional[float]]
    seasonal_indices: Vector
    period: int
    model: str
    method: str = "classical"


def classical_decomposition(y: Vector, m: int, model: str = "additive",
                            normalize: bool = True) -> DecompositionResult:
    """Textbook additive / multiplicative decomposition via centered moving average."""
    y = _check_y(y, 1)
    model = _norm_model(model)
    n = len(y)
    if m <= 1:
        seas = [0.0] * n if model == "additive" else [1.0] * n
        res = [0.0] * n if model == "additive" else [1.0] * n
        return DecompositionResult(list(y), list(y), seas, [r for r in res],
                                   [0.0] if model == "additive" else [1.0], 1, model)
    trend = moving_average(y, m, centered=True)
    detr: List[Optional[float]] = [None] * n
    for t in range(n):
        if trend[t] is None:
            continue
        if model == "additive":
            detr[t] = y[t] - trend[t]
        else:
            detr[t] = _safe_div(y[t], trend[t], None) if abs(trend[t]) > _EPS else None
    raw = []
    for j in range(m):
        vals = [detr[t] for t in range(n) if t % m == j and detr[t] is not None]
        if vals:
            raw.append(mean(vals))
        else:
            raw.append(0.0 if model == "additive" else 1.0)
    if normalize:
        idx = _seasonal_norm_add(raw) if model == "additive" else _seasonal_norm_mul(raw)
    else:
        idx = raw
    seasonal = [idx[t % m] for t in range(n)]
    residual: List[Optional[float]] = [None] * n
    for t in range(n):
        if trend[t] is None:
            continue
        if model == "additive":
            residual[t] = y[t] - trend[t] - seasonal[t]
        else:
            denom = trend[t] * seasonal[t]
            residual[t] = _safe_div(y[t], denom, None) if abs(denom) > _EPS else None
    return DecompositionResult(list(y), trend, seasonal, residual, idx, m, model)


def seasonal_indices(y: Vector, m: int, model: str = "additive") -> Vector:
    """Just the length-``m`` normalised seasonal indices (phase 0..m-1)."""
    return classical_decomposition(y, m, model=model).seasonal_indices


def deseasonalise(y: Vector, indices: Vector, model: str = "additive",
                  start_phase: int = 0) -> Vector:
    """Strip a known seasonal pattern from ``y``."""
    model = _norm_model(model)
    m = len(indices)
    out = []
    for t in range(len(y)):
        s = indices[(start_phase + t) % m]
        if model == "additive":
            out.append(y[t] - s)
        else:
            out.append(y[t] / s if abs(s) > _EPS else y[t])
    return out


def reseasonalise(values: Vector, indices: Vector, model: str = "additive",
                  start_phase: int = 0) -> Vector:
    """Put a seasonal pattern back onto ``values`` (inverse of :func:`deseasonalise`)."""
    model = _norm_model(model)
    m = len(indices)
    out = []
    for t in range(len(values)):
        s = indices[(start_phase + t) % m]
        out.append(values[t] + s if model == "additive" else values[t] * s)
    return out


def seasonal_adjust(y: Vector, m: int, model: str = "additive") -> Vector:
    """Seasonally-adjust the in-sample series."""
    idx = seasonal_indices(y, m, model=model)
    return deseasonalise(y, idx, model=model, start_phase=0)


class Theta(BaseForecaster):
    """The Theta method (Assimakopoulos & Nikolopoulos 2000), reproducing
    ``forecast::thetaf``: optionally deseasonalise, blend a linear-trend line with
    an SES of the doubled-curvature line, then re-seasonalise."""

    def __init__(self, m: int = 1, theta: float = 2.0,
                 seasonal: Any = "auto", seasonal_model: str = "multiplicative",
                 alpha: Optional[float] = None) -> None:
        super().__init__()
        self.m = m
        self.theta = theta
        self.seasonal = seasonal
        self.seasonal_model = _norm_model(seasonal_model)
        self.alpha = alpha
        self.method = "Theta(theta=%.1f)" % theta

    def _fit(self, y: Vector) -> None:
        m = self.m
        n = len(y)
        if self.seasonal == "auto":
            is_seasonal = seasonality_test(y, m) if m > 1 else False
        else:
            is_seasonal = bool(self.seasonal) and m > 1
        self.indices_ = None
        if is_seasonal:
            idx = seasonal_indices(y, m, model=self.seasonal_model)
            # Multiplicative seasonality is only defined for positive series: a
            # non-positive index would invert the (re)seasonalised interval bounds
            # (lower > upper).  Fall back to non-seasonal Theta in that case.
            if self.seasonal_model == "multiplicative" and any(v <= 1e-10 for v in idx):
                is_seasonal = False
            else:
                self.indices_ = idx
        x = (deseasonalise(y, self.indices_, model=self.seasonal_model)
             if self.indices_ is not None else list(y))
        fit = ses_fit(x, alpha=self.alpha, dof=2)
        self.level_ = fit.level
        self.alpha_ = fit.alpha
        self.sigma2_ = fit.sigma2
        a, b = ols_line(x, t0=0)
        self.slope_ = b
        self.x_ = x
        self.is_seasonal_ = is_seasonal
        fitted_x = fit.fitted
        if self.indices_ is not None:
            self.fitted_ = reseasonalise(fitted_x, self.indices_,
                                         model=self.seasonal_model, start_phase=0)
        else:
            self.fitted_ = fitted_x

    def _predict(self, h: int) -> Vector:
        alpha = self.alpha_
        n = self.n_
        drift_w = (self.theta - 1.0) / self.theta
        transient = (1.0 - (1.0 - alpha) ** n) / alpha if alpha > _EPS else n
        mean_x = [self.level_ + drift_w * self.slope_ * (i + transient)
                  for i in range(h)]
        if self.indices_ is not None:
            return reseasonalise(mean_x, self.indices_, model=self.seasonal_model,
                                 start_phase=n % self.m)
        return mean_x

    def predict_interval(self, h: int, level: float = 0.95) -> Tuple[Vector, Vector]:
        pt_x = [self.level_ + ((self.theta - 1.0) / self.theta) * self.slope_ *
                (i + ((1.0 - (1.0 - self.alpha_) ** self.n_) / self.alpha_
                 if self.alpha_ > _EPS else self.n_)) for i in range(h)]
        z = _z_value(level)
        lo_x, up_x = [], []
        for i in range(h):
            se = math.sqrt(max(self.sigma2_ * (1.0 + i * self.alpha_ ** 2), 0.0))
            lo_x.append(pt_x[i] - z * se)
            up_x.append(pt_x[i] + z * se)
        if self.indices_ is not None:
            sp = self.n_ % self.m
            lo = reseasonalise(lo_x, self.indices_, model=self.seasonal_model, start_phase=sp)
            up = reseasonalise(up_x, self.indices_, model=self.seasonal_model, start_phase=sp)
            return lo, up
        return lo_x, up_x


def theta_forecast(y: Vector, h: int, m: int = 1, theta: float = 2.0,
                   level: Optional[float] = None) -> ForecastResult:
    """Functional Theta forecast."""
    return Theta(m=m, theta=theta).fit(y).forecast(h, level=level)


# ===========================================================================
# 8. REGRESSION FORECASTERS
# ===========================================================================
def trend_features(n: int, degree: int = 1, start: int = 0) -> Matrix:
    """Design matrix ``[1, t, t², ..., t^degree]`` for ``t = start..start+n-1``."""
    return [[float(t) ** d for d in range(degree + 1)]
            for t in range(start, start + n)]


def seasonal_dummy_features(n: int, m: int, start_phase: int = 0) -> Matrix:
    """``m-1`` seasonal dummy columns (drop the last to avoid collinearity)."""
    rows = []
    for t in range(n):
        phase = (start_phase + t) % m
        rows.append([1.0 if phase == j else 0.0 for j in range(m - 1)])
    return rows


def fourier_features(n: int, m: int, K: int, start: int = 0) -> Matrix:
    """``2K`` Fourier columns ``sin/cos(2πk t/m)`` for ``k = 1..K``."""
    if K > m // 2:
        raise ForecastError("K must be <= m//2 for Fourier terms.")
    rows = []
    for t in range(start, start + n):
        row = []
        for k in range(1, K + 1):
            ang = 2.0 * math.pi * k * t / m
            # At the Nyquist frequency (m even, k=m/2) sin(pi*t)=0 for all integer t,
            # so keep only the cosine term (matches forecast::fourier).
            if not (m % 2 == 0 and k == m // 2):
                row.append(math.sin(ang))
            row.append(math.cos(ang))
        rows.append(row)
    return rows


class LinearRegressionForecaster(BaseForecaster):
    """General time-series linear regression on user-supplied deterministic features.

    ``feature_fn(n, start)`` must return an ``n×p`` design matrix for time indices
    ``start..start+n-1``; the same function builds future rows for the horizon."""

    def __init__(self, feature_fn: Callable[[int, int], Matrix],
                 ridge: float = 0.0) -> None:
        super().__init__()
        self.feature_fn = feature_fn
        self.ridge = ridge
        self.method = "LinearRegression"

    def _min_obs(self) -> int:
        return 2

    def _fit(self, y: Vector) -> None:
        n = len(y)
        X = self.feature_fn(n, 0)
        self.beta_ = lstsq(X, y, ridge=self.ridge)
        self.fitted_ = matvec(X, self.beta_)

    def _predict(self, h: int) -> Vector:
        Xf = self.feature_fn(h, self.n_)
        return matvec(Xf, self.beta_)


class TrendForecaster(LinearRegressionForecaster):
    """Deterministic polynomial-trend regression."""

    def __init__(self, degree: int = 1, ridge: float = 0.0) -> None:
        super().__init__(lambda n, s: trend_features(n, degree, s), ridge=ridge)
        self.method = "Trend(degree=%d)" % degree


class SeasonalDummyForecaster(LinearRegressionForecaster):
    """Polynomial trend + seasonal dummy regression."""

    def __init__(self, m: int, degree: int = 1, ridge: float = 0.0) -> None:
        self.m = m
        self.degree = degree

        def feat(n: int, s: int) -> Matrix:
            tr = trend_features(n, degree, s)
            du = seasonal_dummy_features(n, m, start_phase=s % m)
            return [tr[i] + du[i] for i in range(n)]

        super().__init__(feat, ridge=ridge)
        self.method = "SeasonalDummy(m=%d,deg=%d)" % (m, degree)

    def _min_obs(self) -> int:
        return self.m + self.degree + 1


class FourierForecaster(LinearRegressionForecaster):
    """Polynomial trend + Fourier-term seasonal regression (harmonic regression)."""

    def __init__(self, m: int, K: int = 1, degree: int = 1, ridge: float = 0.0) -> None:
        self.m, self.K, self.degree = m, K, degree

        def feat(n: int, s: int) -> Matrix:
            tr = trend_features(n, degree, s)
            fo = fourier_features(n, m, K, start=s)
            return [tr[i] + fo[i] for i in range(n)]

        super().__init__(feat, ridge=ridge)
        self.method = "Fourier(m=%d,K=%d,deg=%d)" % (m, K, degree)

    def _min_obs(self) -> int:
        return 2 * self.K + self.degree + 2


# ===========================================================================
# 9. AUTOREGRESSION & INTERMITTENT DEMAND
# ===========================================================================
def yule_walker(y: Vector, p: int) -> Tuple[Vector, float]:
    """Yule-Walker AR(p) estimation via Levinson-Durbin -> ``(phi, sigma2)``."""
    y = _check_y(y, p + 1)
    g = autocovariance(y, p, adjusted=False)
    a = [0.0] * (p + 1)
    E = g[0]
    if _degenerate_variance(E, y):
        return [0.0] * p, 0.0
    e_tol = _EPS * g[0]                     # relative innovation-variance floor (scale-invariant)
    for m_ in range(1, p + 1):
        acc = g[m_] - math.fsum(a[j] * g[m_ - j] for j in range(1, m_))
        k = acc / E if E > e_tol else 0.0
        new = list(a)
        new[m_] = k
        for j in range(1, m_):
            new[j] = a[j] - k * a[m_ - j]
        a = new
        E *= (1.0 - k * k)
        if E <= e_tol:
            E = e_tol
            break
    return a[1:p + 1], E


def _build_lag_matrix(y: Vector, p: int, intercept: bool = True) -> Tuple[Matrix, Vector]:
    n = len(y)
    X, target = [], []
    for t in range(p, n):
        row = ([1.0] if intercept else []) + [y[t - i] for i in range(1, p + 1)]
        X.append(row)
        target.append(y[t])
    return X, target


class ARForecaster(BaseForecaster):
    """Autoregressive AR(p) forecaster (OLS or Yule-Walker), with recursive
    multi-step forecasts and optional AIC/BIC order selection."""

    def __init__(self, p: int = 1, method: str = "ols", include_mean: bool = True,
                 ridge: float = 0.0, ic: Optional[str] = None,
                 max_p: Optional[int] = None) -> None:
        super().__init__()
        if method not in ("ols", "yule_walker"):
            raise ForecastError("AR method must be 'ols' or 'yule_walker' (got %r)."
                                % (method,))
        if ic is not None and ic not in ("aic", "bic"):
            raise ForecastError("AR ic must be None, 'aic' or 'bic' (got %r)." % (ic,))
        self.p = p
        self.fit_method = method
        self.include_mean = include_mean
        self.ridge = ridge
        self.ic = ic
        self.max_p = max_p
        self.method = "AR(%d)" % p

    def _min_obs(self) -> int:
        # Order selection caps p to what the series supports, so it only needs a
        # handful of points; a fixed order needs an over-determined OLS (n >= 2p+2).
        if self.ic is not None:
            return 4
        return 2 * self.p + 2

    def _fit(self, y: Vector) -> None:
        if self.ic is not None:
            self.p = self._select_order(y)
        p = self.p
        self.phi_, self.c_, self.mu_, self.sigma2_ = self._fit_p(y, p)
        n = len(y)
        fitted = list(y[:p])
        for t in range(p, n):
            fitted.append(self.c_ + math.fsum(self.phi_[i] * y[t - 1 - i]
                                              for i in range(p)))
        self.fitted_ = fitted
        self.method = "AR(%d)" % p

    def _fit_p(self, y: Vector, p: int) -> Tuple[Vector, float, float, float]:
        n = len(y)
        if self.fit_method == "yule_walker":
            phi, sigma2 = yule_walker(y, p)
            mu = mean(y) if self.include_mean else 0.0
            c = mu * (1.0 - math.fsum(phi))
            return phi, c, mu, sigma2
        X, target = _build_lag_matrix(y, p, intercept=self.include_mean)
        beta = lstsq(X, target, ridge=self.ridge)
        if self.include_mean:
            c, phi = beta[0], beta[1:]
        else:
            c, phi = 0.0, beta
        pred = matvec(X, beta)
        resid = [target[i] - pred[i] for i in range(len(target))]
        m_used = len(target)
        sigma2 = math.fsum(e * e for e in resid) / m_used if m_used else 0.0
        sphi = math.fsum(phi)
        mu = c / (1.0 - sphi) if abs(1.0 - sphi) > _EPS else mean(y)
        return phi, c, mu, sigma2

    def _select_order(self, y: Vector) -> int:
        n = len(y)
        max_p = self.max_p or min(int(10 * math.log10(n)) + 1, n // 2)
        max_p = max(1, min(max_p, (n - 2) // 2))    # keep every candidate OLS over-determined
        # Score EVERY candidate order on the SAME trailing window [max_p, n): comparing
        # information criteria across different effective sample sizes is both invalid and
        # not scale-invariant (the m*log(sigma2) scale-shift would differ per order).
        m0 = n - max_p
        best_p, best_ic = 1, _BIG
        for p in range(1, max_p + 1):
            try:
                phi, c, _, _ = self._fit_p(y, p)
            except (LinAlgError, ForecastError):
                continue
            sse = math.fsum((y[t] - c - math.fsum(phi[i] * y[t - 1 - i]
                             for i in range(p))) ** 2 for t in range(max_p, n))
            sigma2 = max(sse / m0, _EPS)
            k = p + (1 if self.include_mean else 0)
            if self.ic == "bic":
                val = m0 * math.log(sigma2) + k * math.log(m0)
            else:
                val = m0 * math.log(sigma2) + 2 * k
            if val < best_ic:
                best_ic, best_p = val, p
        return best_p

    def _predict(self, h: int) -> Vector:
        p = self.p
        hist = list(self.y_)
        out = []
        for _ in range(h):
            yhat = self.c_ + math.fsum(self.phi_[i] * hist[-1 - i] for i in range(p))
            out.append(yhat)
            hist.append(yhat)
        return out

    def predict_interval(self, h: int, level: float = 0.95) -> Tuple[Vector, Vector]:
        pt = self._predict(h)
        psi = _ar_ma_weights(self.phi_, h)
        z = _z_value(level)
        lo, up = [], []
        for i in range(h):
            var = self.sigma2_ * math.fsum(psi[j] ** 2 for j in range(i + 1))
            se = math.sqrt(max(var, 0.0))
            lo.append(pt[i] - z * se)
            up.append(pt[i] + z * se)
        return lo, up

    def _sigma2(self) -> float:
        return self.sigma2_


def _ar_ma_weights(phi: Vector, n_weights: int) -> Vector:
    p = len(phi)
    psi = [1.0]
    for j in range(1, n_weights):
        psi.append(math.fsum(phi[i] * psi[j - 1 - i]
                             for i in range(min(j, p)) if j - 1 - i >= 0))
    return psi


def _nonzero_decomposition(y: Vector) -> Tuple[Vector, Vector, List[int]]:
    sizes, intervals, indices = [], [], []
    counter = 0
    for t in range(len(y)):
        counter += 1
        if y[t] != 0.0:
            sizes.append(y[t])
            intervals.append(float(counter))
            indices.append(t)
            counter = 0
    return sizes, intervals, indices


class Croston(BaseForecaster):
    """Croston's method for intermittent demand (and the SBA / SBJ bias-corrected
    variants): separate exponential smoothing of demand sizes and inter-arrival
    intervals; the forecast is a flat per-period rate."""

    def __init__(self, alpha: float = 0.1, variant: str = "croston",
                 init: str = "mean") -> None:
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ForecastError("Croston alpha must be in (0, 1).")
        self.alpha = alpha
        self.variant = variant.lower()
        self.init = init
        self.method = "Croston" if self.variant == "croston" else self.variant.upper()

    def _fit(self, y: Vector) -> None:
        if any(v < 0 for v in y):
            raise ForecastError("Croston requires non-negative demand.")
        a = self.alpha
        sizes, intervals, indices = _nonzero_decomposition(y)
        n = len(y)
        if not sizes:
            self.rate_ = 0.0
            self.fitted_ = [0.0] * n
            return
        if self.init == "first":
            z, q = sizes[0], intervals[0]
        elif self.init == "naive":
            z, q = sizes[0], 1.0
        else:
            z, q = mean(sizes), mean(intervals)
        # per-period fitted rate + smoothing at each demand epoch
        fitted = [0.0] * n
        rate = self._rate(z, q)
        demand_ptr = 0
        for t in range(n):
            fitted[t] = rate
            if y[t] != 0.0:
                z = a * y[t] + (1.0 - a) * z
                q = a * intervals[demand_ptr] + (1.0 - a) * q
                demand_ptr += 1
                rate = self._rate(z, q)
        self.z_, self.q_ = z, q
        self.rate_ = self._rate(z, q)
        self.fitted_ = fitted

    def _rate(self, z: float, q: float) -> float:
        base = _safe_div(z, q, 0.0)
        if self.variant == "sba":
            return (1.0 - self.alpha / 2.0) * base
        if self.variant == "sbj":
            denom = 2.0 - self.alpha
            return (1.0 - self.alpha / denom) * base if denom > _EPS else base
        return base

    def _predict(self, h: int) -> Vector:
        return [self.rate_] * h


class TSB(BaseForecaster):
    """Teunter-Syntetos-Babai method: smooth the demand *probability* every period
    (handling obsolescence) and the demand size on demand epochs; forecast = p·z."""

    def __init__(self, alpha: float = 0.1, beta: float = 0.1,
                 init: str = "mean") -> None:
        super().__init__()
        if not (0.0 < alpha < 1.0 and 0.0 < beta < 1.0):
            raise ForecastError("TSB alpha, beta must be in (0, 1).")
        self.alpha, self.beta, self.init = alpha, beta, init
        self.method = "TSB"

    def _fit(self, y: Vector) -> None:
        if any(v < 0 for v in y):
            raise ForecastError("TSB requires non-negative demand.")
        a, b = self.alpha, self.beta
        sizes, intervals, indices = _nonzero_decomposition(y)
        n = len(y)
        J = len(sizes)
        if self.init == "first" and J > 0:
            p = 1.0 / intervals[0]
            z = sizes[0]
        else:
            p = J / n if n else 0.0
            z = mean(sizes) if sizes else 0.0
        fitted = [0.0] * n
        for t in range(n):
            fitted[t] = p * z                # one-step-ahead forecast for period t
            if y[t] != 0.0:
                p = a * 1.0 + (1.0 - a) * p
                z = b * y[t] + (1.0 - b) * z
            else:
                p = (1.0 - a) * p
            p = min(1.0, max(0.0, p))
        self.p_, self.z_ = p, z
        self.rate_ = p * z
        self.fitted_ = fitted

    def _predict(self, h: int) -> Vector:
        return [self.rate_] * h


# ===========================================================================
# 10. EVALUATION METRICS
# ===========================================================================
def _err(y: Vector, yhat: Vector) -> Vector:
    if len(y) != len(yhat):
        raise ForecastError("metric: length mismatch (%d vs %d)." % (len(y), len(yhat)))
    if not y:
        raise ForecastError("metric: empty input.")
    return [y[i] - yhat[i] for i in range(len(y))]


def me(y: Vector, yhat: Vector) -> float:
    return mean(_err(y, yhat))


def mae(y: Vector, yhat: Vector) -> float:
    return mean([abs(e) for e in _err(y, yhat)])


def mse(y: Vector, yhat: Vector) -> float:
    return mean([e * e for e in _err(y, yhat)])


def rmse(y: Vector, yhat: Vector) -> float:
    return math.sqrt(mse(y, yhat))


def mape(y: Vector, yhat: Vector) -> float:
    e = _err(y, yhat)
    terms = [abs(e[i]) / abs(y[i]) for i in range(len(y)) if abs(y[i]) > _EPS]
    return 100.0 * mean(terms) if terms else float("inf")


def smape(y: Vector, yhat: Vector) -> float:
    e = _err(y, yhat)
    terms = []
    for i in range(len(y)):
        denom = abs(y[i]) + abs(yhat[i])
        terms.append(200.0 * abs(e[i]) / denom if denom > _EPS else 0.0)
    return mean(terms)


def wape(y: Vector, yhat: Vector) -> float:
    e = _err(y, yhat)
    denom = math.fsum(abs(v) for v in y)
    return 100.0 * _safe_div(math.fsum(abs(v) for v in e), denom, float("inf"))


def _mase_denominator(y: Vector, m: int) -> float:
    """In-sample seasonal-naive MAE (falls back to lag-1 when the series is short)."""
    n = len(y)
    lag = m if (m >= 1 and n > m) else 1
    diffs = [abs(y[t] - y[t - lag]) for t in range(lag, n)]
    if not diffs:
        return _EPS
    return math.fsum(diffs) / len(diffs)


def mase(y: Vector, yhat: Vector, y_train: Vector, m: int = 1) -> float:
    """Mean Absolute Scaled Error: MAE divided by the in-sample seasonal-naive MAE."""
    scale = _mase_denominator(_check_y(y_train, 2), m)
    return _safe_div(mae(y, yhat), scale, float("inf"))


def rmsse(y: Vector, yhat: Vector, y_train: Vector, m: int = 1) -> float:
    """Root Mean Squared Scaled Error (the M5 metric)."""
    yt = _check_y(y_train, 2)
    n = len(yt)
    lag = m if (m >= 1 and n > m) else 1
    denom = math.fsum((yt[t] - yt[t - lag]) ** 2 for t in range(lag, n)) / max(1, n - lag)
    return math.sqrt(_safe_div(mse(y, yhat), denom, float("inf")))


def mpe(y: Vector, yhat: Vector) -> float:
    e = _err(y, yhat)
    terms = [e[i] / y[i] for i in range(len(y)) if abs(y[i]) > _EPS]
    return 100.0 * mean(terms) if terms else float("inf")


def r2_score(y: Vector, yhat: Vector) -> float:
    e = _err(y, yhat)
    ybar = mean(y)
    ss_res = math.fsum(v * v for v in e)
    ss_tot = math.fsum((v - ybar) ** 2 for v in y)
    if _degenerate_variance(ss_tot / len(y), y):        # scale-invariant zero-variance test
        return 0.0
    return 1.0 - ss_res / ss_tot                         # ss_tot > 0 guaranteed above


def median_ae(y: Vector, yhat: Vector) -> float:
    return median([abs(e) for e in _err(y, yhat)])


def pinball_loss(y: Vector, yhat_q: Vector, q: float) -> float:
    """Quantile (pinball) loss at quantile level ``q``."""
    if not 0.0 < q < 1.0:
        raise ForecastError("pinball q must be in (0, 1).")
    if len(y) != len(yhat_q):
        raise ForecastError("pinball: length mismatch.")
    terms = []
    for i in range(len(y)):
        d = y[i] - yhat_q[i]
        terms.append(q * d if d >= 0 else (q - 1.0) * d)
    return mean(terms)


def coverage(y: Vector, lower: Vector, upper: Vector) -> float:
    """Empirical coverage: fraction of ``y`` inside ``[lower, upper]``."""
    if not (len(y) == len(lower) == len(upper)):
        raise ForecastError("coverage: length mismatch.")
    return mean([1.0 if lower[i] <= y[i] <= upper[i] else 0.0 for i in range(len(y))])


def interval_width(lower: Vector, upper: Vector) -> float:
    if len(lower) != len(upper):
        raise ForecastError("interval_width: length mismatch.")
    return mean([upper[i] - lower[i] for i in range(len(lower))])


def msis(y: Vector, lower: Vector, upper: Vector, y_train: Vector,
         level: float = 0.95, m: int = 1) -> float:
    """Mean Scaled Interval Score (M4): interval sharpness + calibration penalty,
    scaled by the in-sample seasonal-naive MAE."""
    if not (len(y) == len(lower) == len(upper)):
        raise ForecastError("msis: length mismatch.")
    a = 1.0 - level
    scale = _mase_denominator(_check_y(y_train, 2), m)
    terms = []
    for i in range(len(y)):
        w = upper[i] - lower[i]
        pen = 0.0
        if y[i] < lower[i]:
            pen += (2.0 / a) * (lower[i] - y[i])
        if y[i] > upper[i]:
            pen += (2.0 / a) * (y[i] - upper[i])
        terms.append(w + pen)
    return _safe_div(mean(terms), scale, float("inf"))


def evaluate(y: Vector, yhat: Vector, y_train: Optional[Vector] = None,
             m: int = 1) -> Dict[str, float]:
    """Compute a standard panel of point-forecast metrics at once."""
    out = {
        "ME": me(y, yhat), "MAE": mae(y, yhat), "MSE": mse(y, yhat),
        "RMSE": rmse(y, yhat), "MAPE": mape(y, yhat), "sMAPE": smape(y, yhat),
        "WAPE": wape(y, yhat), "MPE": mpe(y, yhat), "R2": r2_score(y, yhat),
        "MedianAE": median_ae(y, yhat),
    }
    if y_train is not None:
        out["MASE"] = mase(y, yhat, y_train, m)
        out["RMSSE"] = rmsse(y, yhat, y_train, m)
    return out


# ===========================================================================
# 11. CROSS-VALIDATION / BACKTESTING
# ===========================================================================
def train_test_split(y: Vector, h: int) -> Tuple[Vector, Vector]:
    """Hold out the final ``h`` observations as the test set."""
    y = _check_y(y, h + 1)
    return y[:-h], y[-h:]


def rolling_origin_splits(n: int, h: int, initial: int, step: int = 1,
                          expanding: bool = True) -> List[Tuple[int, int, int]]:
    """Generate ``(train_start, train_end, test_end)`` index triples for rolling
    origin evaluation (``expanding`` window or fixed sliding window)."""
    if initial < 1 or h < 1 or step < 1:
        raise ForecastError("rolling_origin_splits: bad parameters.")
    splits = []
    origin = initial
    while origin + h <= n:
        train_start = 0 if expanding else max(0, origin - initial)
        splits.append((train_start, origin, origin + h))
        origin += step
    return splits


def cross_validate(y: Vector, make_forecaster: Callable[[], BaseForecaster],
                   h: int, initial: Optional[int] = None, step: int = 1,
                   expanding: bool = True, m: int = 1,
                   metrics: Sequence[str] = ("MAE", "RMSE", "sMAPE", "MASE")
                   ) -> Dict[str, float]:
    """Rolling-origin cross-validation: refit a fresh forecaster on each training
    window and average the requested metrics over all folds/horizons."""
    y = _check_y(y, 3)
    n = len(y)
    init = initial if initial is not None else max(2 * m, n // 2, 3)
    splits = rolling_origin_splits(n, h, init, step, expanding)
    if not splits:
        raise ForecastError("cross_validate: no folds (initial/h too large).")
    per_metric: Dict[str, List[float]] = {k: [] for k in metrics}
    for ts, te, tend in splits:
        train, test = y[ts:te], y[te:tend]
        f = make_forecaster()
        f.fit(train)
        pred = f.predict(len(test))
        scores = evaluate(test, pred, y_train=train, m=m)
        for k in metrics:
            v = scores.get(k)
            if v is not None and _is_finite(v):
                per_metric[k].append(v)
    return {k: (mean(v) if v else float("nan")) for k, v in per_metric.items()}


def backtest(y: Vector, make_forecaster: Callable[[], BaseForecaster], h: int,
             initial: Optional[int] = None, step: int = 1, expanding: bool = True
             ) -> List[ForecastResult]:
    """Return the per-fold :class:`ForecastResult`s from a rolling-origin backtest."""
    y = _check_y(y, 3)
    n = len(y)
    init = initial if initial is not None else max(n // 2, 3)
    out = []
    for ts, te, tend in rolling_origin_splits(n, h, init, step, expanding):
        f = make_forecaster()
        f.fit(y[ts:te])
        out.append(f.forecast(tend - te))
    return out


# ===========================================================================
# 12. FORECAST COMBINATION / ENSEMBLES
# ===========================================================================
def _check_forecasts(forecasts: Sequence[Vector]) -> int:
    if not forecasts:
        raise ForecastError("no forecasts to combine.")
    h = len(forecasts[0])
    if any(len(f) != h for f in forecasts):
        raise ForecastError("combine: forecasts have differing lengths.")
    return h


def combine_mean(forecasts: Sequence[Vector]) -> Vector:
    """Simple average of several forecasts (a very strong ensemble baseline)."""
    h = _check_forecasts(forecasts)
    return [mean([f[i] for f in forecasts]) for i in range(h)]


def combine_median(forecasts: Sequence[Vector]) -> Vector:
    h = _check_forecasts(forecasts)
    return [median([f[i] for f in forecasts]) for i in range(h)]


def combine_trimmed(forecasts: Sequence[Vector], trim: float = 0.1) -> Vector:
    """Trimmed-mean combination (drop the top/bottom ``trim`` fraction per horizon)."""
    h = _check_forecasts(forecasts)
    k = len(forecasts)
    cut = int(math.floor(trim * k))
    out = []
    for i in range(h):
        col = sorted(f[i] for f in forecasts)
        kept = col[cut:k - cut] if k - 2 * cut > 0 else col
        out.append(mean(kept))
    return out


def combine_weighted(forecasts: Sequence[Vector], weights: Sequence[float]) -> Vector:
    h = _check_forecasts(forecasts)
    w = _as_floats(weights, "weights")
    if len(w) != len(forecasts):
        raise ForecastError("combine_weighted: one weight per forecast required.")
    s = math.fsum(w)
    if abs(s) < _EPS:
        raise ForecastError("weights sum to zero.")
    w = [wi / s for wi in w]
    return [math.fsum(w[j] * forecasts[j][i] for j in range(len(w))) for i in range(h)]


def inverse_error_weights(errors: Sequence[float], power: float = 1.0) -> Vector:
    """Weights proportional to ``1 / error^power`` (lower validation error -> more weight)."""
    e = _as_floats(errors, "errors")
    inv = [1.0 / (abs(v) ** power) if abs(v) > _EPS else _BIG for v in e]
    s = math.fsum(inv)
    return [v / s for v in inv]


# ===========================================================================
# 13. SYNTHETIC DATASETS (deterministic)
# ===========================================================================
def make_trend_seasonal(n: int, slope: float = 0.5, m: int = 12,
                        amplitude: float = 10.0, level: float = 100.0,
                        noise: float = 0.0, seed: int = 0) -> Vector:
    """A trend + sinusoidal-seasonal series with optional Gaussian noise."""
    rng = random.Random(seed)
    out = []
    for t in range(n):
        val = level + slope * t + amplitude * math.sin(2.0 * math.pi * t / m)
        if noise:
            val += rng.gauss(0.0, noise)
        out.append(val)
    return out


def make_ar_series(n: int, phi: Sequence[float] = (0.6,), c: float = 0.0,
                   sigma: float = 1.0, seed: int = 0, burn: int = 50) -> Vector:
    """Simulate a stationary AR(p) process."""
    rng = random.Random(seed)
    p = len(phi)
    hist = [0.0] * p
    out = []
    for _ in range(n + burn):
        val = c + math.fsum(phi[i] * hist[-1 - i] for i in range(p)) + rng.gauss(0, sigma)
        hist.append(val)
        out.append(val)
    return out[burn:]


def make_intermittent(n: int, p_demand: float = 0.3, mean_size: float = 5.0,
                      seed: int = 0) -> Vector:
    """Simulate an intermittent-demand series (Bernoulli occurrence × Poisson-ish size)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if rng.random() < p_demand:
            out.append(float(max(1, int(round(rng.gauss(mean_size, mean_size / 3.0))))))
        else:
            out.append(0.0)
    return out


# ===========================================================================
# 14. DEMO
# ===========================================================================
def _demo() -> None:
    print("=" * 70)
    print("  FORECASTING BASELINES (pure Python) - demo")
    print("=" * 70)

    y = make_trend_seasonal(60, slope=0.8, m=12, amplitude=15.0, level=100.0,
                            noise=3.0, seed=1)
    train, test = train_test_split(y, 12)

    print("\n-- classical decomposition (m=12) --")
    dec = classical_decomposition(train, 12, model="additive")
    defined = [t for t in range(len(train)) if dec.trend[t] is not None]
    print("   trend defined on %d/%d points, seasonal indices sum = %.3f"
          % (len(defined), len(train), math.fsum(dec.seasonal_indices)))

    print("-- baselines on a 12-step holdout (lower sMAPE = better) --")
    models = {
        "Naive": Naive(),
        "SeasonalNaive": SeasonalNaive(12),
        "Drift": Drift(),
        "Theta": Theta(m=12),
        "Holt-Winters": HoltWinters(12, seasonal="add", trend="add"),
        "auto_ets": None,
    }
    for name, mdl in models.items():
        if name == "auto_ets":
            mdl = auto_ets(train, seasonal_periods=12)
        else:
            mdl.fit(train)
        pred = mdl.predict(12)
        print("   %-14s sMAPE=%6.2f  MASE=%5.2f"
              % (name, smape(test, pred), mase(test, pred, train, 12)))

    print("-- AR order selection + forecast --")
    ar = ARForecaster(p=1, ic="aic", max_p=8).fit(train)
    fr = ar.forecast(12, level=0.95)
    print("   selected %s  RMSE=%.2f  95%% PI width=%.1f"
          % (ar.method, rmse(test, fr.mean), interval_width(fr.lower, fr.upper)))

    print("-- Theta prediction interval coverage --")
    tf = Theta(m=12).fit(train).forecast(12, level=0.90)
    print("   90%% interval covers %.0f%% of the holdout"
          % (100.0 * coverage(test, tf.lower, tf.upper)))

    print("-- intermittent demand (Croston / SBA / TSB) --")
    di = make_intermittent(80, p_demand=0.25, mean_size=6.0, seed=2)
    for nm, mdl in [("Croston", Croston(0.1)), ("SBA", Croston(0.1, "sba")),
                    ("TSB", TSB(0.1, 0.1))]:
        r = mdl.fit(di).predict(1)[0]
        print("   %-8s per-period rate = %.3f" % (nm, r))

    print("-- rolling-origin cross-validation (expanding window) --")
    cv = cross_validate(y, lambda: Theta(m=12), h=6, initial=24, step=6, m=12)
    print("   Theta CV: " + "  ".join("%s=%.2f" % (k, v) for k, v in cv.items()))

    print("-- forecast combination --")
    fc = [Naive().fit(train).predict(12), SeasonalNaive(12).fit(train).predict(12),
          Theta(m=12).fit(train).predict(12)]
    ens = combine_mean(fc)
    print("   ensemble sMAPE=%.2f  (vs best single above)" % smape(test, ens))

    print("\n" + "=" * 70)
    print("  demo OK")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
