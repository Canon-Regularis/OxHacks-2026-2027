"""
Autoregressive & ARIMA Forecasting -- a self-contained, zero-dependency toolkit
for the *autoregressive* side of time-series forecasting: the ARIMA / SARIMA /
state-space family and its multivariate and volatility cousins, implemented from
scratch on the Python standard library.

Where the sibling *baseline* module covers the classical benchmark forecasters
(naive, exponential smoothing, Theta, decomposition), this module owns the deeper
*autoregressive modelling machinery* that those baselines are measured against:

  * **Box-Jenkins ARIMA / SARIMA / SARIMAX** -- AR, MA, ARMA, ARIMA and seasonal
    models, estimated by Yule-Walker (Levinson-Durbin), Burg, Hannan-Rissanen,
    conditional sum of squares (CSS) and **exact maximum likelihood via a Kalman
    filter**, with analytic (psi-weight) and simulation prediction intervals.
  * **State space & Kalman** -- a general linear-Gaussian state-space model with a
    Kalman filter + RTS smoother, the state-space form of ARIMA for exact
    likelihood, and structural (unobserved-components) local-level / local-trend
    models.
  * **Vector autoregression (VAR)** -- multivariate OLS estimation, lag selection,
    forecasting, orthogonalised impulse responses, forecast-error-variance
    decomposition and Granger causality.
  * **Volatility (GARCH)** -- GARCH / GJR-GARCH conditional-variance models by MLE.
  * **Diagnostics** -- ACF / PACF, Ljung-Box, Durbin-Watson, Jarque-Bera, ADF and
    KPSS unit-root tests, information criteria and automatic order selection
    (a Hyndman-Khandakar stepwise ``auto_arima``).
  * **Evaluation** -- a full forecast-accuracy metric suite, rolling-origin
    backtesting, the Diebold-Mariano test, forecast combination and a small model
    registry, plus strict-JSON competition payloads.

Everything -- the linear algebra (LU, Cholesky, QR, least squares, Levinson-Durbin,
discrete Lyapunov), the special functions (normal / t / chi-square / F via error,
gamma and beta functions), the polynomial root finder, the optimisers and the RNG
-- is implemented on ``math``, ``cmath``, ``random``, ``itertools`` and
``statistics`` only, so it drops straight into a competition / hackathon / embedded
sandbox: just copy this one file.

Contents
--------
  0.  Scaffold: version, constants, type aliases
  1.  Special functions + deterministic RNG (erf/norm/t/chi2/F, gammainc, betainc)
  2.  Compensated statistics helpers
  3.  Dense linear algebra (LU, Cholesky, QR, solve, inv, logdet, Lyapunov)
  4.  Polynomials & roots (lag polynomials, Durand-Kerner, companion, spectral radius)
  5.  Levinson-Durbin + reflection-coefficient parameter maps
  6.  Deterministic optimisers (Nelder-Mead, finite-difference Hessian)
  7.  Core result records + strict-JSON helper
  8.  Information criteria (AIC / AICc / BIC / HQIC)
  9.  Time-series kernels: autocovariance / ACF / PACF / CCF, Ljung-Box, ...
 10.  Transforms: Box-Cox, standardisation, detrending
 11.  Differencing (one ordered differencer with exact inversion)
 12.  Decomposition: moving average, classical & STL-lite seasonal decomposition
 13.  Unit-root & seasonality: ADF, KPSS, ndiffs, nsdiffs, period detection
 14.  Psi/pi weights + forecast-error variance + prediction intervals
 15.  State space: Kalman filter / smoother, ARIMA state-space form
 16.  Classical ARIMA estimators (Yule-Walker, Burg, Hannan-Rissanen, CSS)
 17.  ARIMA / SARIMA / SARIMAX model + results (exact-ML via Kalman)
 18.  Exponential smoothing & Theta (for self-contained baselines / ensembles)
 19.  Structural unobserved-components models
 20.  Vector autoregression (VAR) + impulse response + Granger causality
 21.  Volatility: GARCH / GJR-GARCH
 22.  Automatic order selection (auto_arima, grid search)
 23.  Baseline forecasters + model registry
 24.  Forecast-accuracy metrics & scoring
 25.  Backtesting, Diebold-Mariano, ensembling
 26.  Series simulators
 27.  Payloads + demo + __all__

Every model option is keyword-only.  The demo is byte-deterministic across
``PYTHONHASHSEED``.
"""

from __future__ import annotations

import bisect
import cmath
import itertools
import math
import random
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Sequence, Tuple)

# ===========================================================================
# 0. SCAFFOLD
# ===========================================================================
__version__ = "1.0.0"

_EPS = 1e-12
_INF = float("inf")
_NEG_INF = float("-inf")
_LOG2PI = math.log(2.0 * math.pi)

Vector = List[float]
Matrix = List[List[float]]


# ===========================================================================
# 1. SPECIAL FUNCTIONS + DETERMINISTIC RNG
# ===========================================================================
def norm_pdf(x: float) -> float:
    """Standard normal density."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Standard normal quantile (inverse CDF), Acklam's rational approximation
    refined by one Halley step -- accurate to ~1e-15 across the tails, which the
    95%/99% prediction bands rely on."""
    if p <= 0.0:
        return _NEG_INF
    if p >= 1.0:
        return _INF
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= phigh:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    # one Halley refinement
    e = norm_cdf(x) - p
    u = e / norm_pdf(x)
    x = x - u / (1.0 + 0.5 * x * u)
    return x


def _gammainc_lower(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x) via series / continued fraction."""
    if x <= 0.0 or a <= 0.0:
        return 0.0
    if x < a + 1.0:                                   # series expansion
        term = 1.0 / a
        total = term
        n = a
        for _ in range(300):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # continued fraction (Lentz) for Q, then P = 1 - Q
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 300):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return 1.0 - q


def gammainc(a: float, x: float) -> float:
    """Regularised lower incomplete gamma function P(a, x) in [0, 1]."""
    return _gammainc_lower(a, x)


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b) via Lentz continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta)

    def cf(aa: float, bb: float, xx: float) -> float:
        tiny = 1e-300
        c = 1.0
        d = 1.0 - (aa + bb) * xx / (aa + 1.0)
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        h = d
        for m in range(1, 300):
            m2 = 2 * m
            aterm = m * (bb - m) * xx / ((aa + m2 - 1.0) * (aa + m2))
            d = 1.0 + aterm * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + aterm / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            h *= d * c
            aterm = -(aa + m) * (aa + bb + m) * xx / ((aa + m2) * (aa + m2 + 1.0))
            d = 1.0 + aterm * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + aterm / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-15:
                break
        return h
    if x < (a + 1.0) / (a + b + 2.0):
        return front * cf(a, b, x) / a
    return 1.0 - front * cf(b, a, 1.0 - x) / b


def chi2_cdf(x: float, *, df: float) -> float:
    """Chi-square CDF."""
    if x <= 0.0 or df <= 0.0:
        return 0.0
    return gammainc(df / 2.0, x / 2.0)


def chi2_sf(x: float, *, df: float) -> float:
    """Chi-square survival function 1 - CDF (upper-tail p-value)."""
    return 1.0 - chi2_cdf(x, df=df)


def _bisect_ppf(cdf: Callable[[float], float], p: float, lo: float, hi: float,
                *, iters: int = 100) -> float:
    """Deterministic bisection inverse of a monotone CDF on [lo, hi]."""
    if p <= 0.0:
        return lo
    if p >= 1.0:
        return hi
    flo = cdf(lo) - p
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = cdf(mid) - p
        if abs(fm) < 1e-14 or (hi - lo) < 1e-14:
            return mid
        if (fm > 0.0) == (flo > 0.0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def chi2_ppf(p: float, *, df: float) -> float:
    """Chi-square quantile (inverse CDF)."""
    hi = max(1.0, df + 12.0 * math.sqrt(2.0 * df) + 20.0)
    return _bisect_ppf(lambda x: chi2_cdf(x, df=df), p, 0.0, hi)


def t_cdf(x: float, *, df: float) -> float:
    """Student-t CDF via the regularised incomplete beta function."""
    if df <= 0.0:
        return float("nan")
    if df > 1e6:                                      # numerically normal
        return norm_cdf(x)
    xx = df / (df + x * x)
    ib = 0.5 * betainc(df / 2.0, 0.5, xx)
    return 1.0 - ib if x > 0.0 else ib


def t_ppf(p: float, *, df: float) -> float:
    """Student-t quantile (inverse CDF)."""
    if df > 1e6:
        return norm_ppf(p)
    bound = max(10.0, 60.0 / max(df, 1.0) + 10.0)
    return _bisect_ppf(lambda x: t_cdf(x, df=df), p, -bound, bound)


def f_sf(x: float, *, dfn: float, dfd: float) -> float:
    """Upper-tail p-value of the F distribution."""
    if x <= 0.0:
        return 1.0
    xx = dfd / (dfd + dfn * x)
    return betainc(dfd / 2.0, dfn / 2.0, xx)


class DeterministicRNG:
    """A single seeded pseudo-random generator so every stochastic path (optimiser
    restarts, simulation, bootstrap) is reproducible and independent of Python's
    hash seed.  Wraps ``random.Random`` and caches a spare Box-Muller normal."""

    __slots__ = ("_rng", "_spare")

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._spare: Optional[float] = None

    def random(self) -> float:
        return self._rng.random()

    def uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self._rng.random()

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def gauss(self) -> float:
        if self._spare is not None:
            v = self._spare
            self._spare = None
            return v
        u1 = max(self._rng.random(), 1e-300)
        u2 = self._rng.random()
        r = math.sqrt(-2.0 * math.log(u1))
        self._spare = r * math.sin(2.0 * math.pi * u2)
        return r * math.cos(2.0 * math.pi * u2)

    def sample_normals(self, n: int) -> List[float]:
        return [self.gauss() for _ in range(n)]

    def choice_index(self, n: int) -> int:
        return self._rng.randrange(n)


# ===========================================================================
# 2. COMPENSATED STATISTICS HELPERS
# ===========================================================================
def _kahan_sum(xs: Iterable[float]) -> float:
    """Compensated (Kahan) summation, fixed input order -> deterministic."""
    total = 0.0
    comp = 0.0
    for x in xs:
        y = x - comp
        t = total + y
        comp = (t - total) - y
        total = t
    return total


def _mean(xs: Sequence[float]) -> float:
    n = len(xs)
    return _kahan_sum(xs) / n if n else 0.0


def _var(xs: Sequence[float], *, ddof: int = 0) -> float:
    n = len(xs)
    if n - ddof <= 0:
        return 0.0
    mu = _mean(xs)
    return _kahan_sum((x - mu) ** 2 for x in xs) / (n - ddof)


def _std(xs: Sequence[float], *, ddof: int = 0) -> float:
    return math.sqrt(max(_var(xs, ddof=ddof), 0.0))


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def _safe_div(a: float, b: float, *, default: float = 0.0) -> float:
    """Divide, flooring the magnitude of the denominator at ``_EPS``."""
    if abs(b) < _EPS:
        if a == 0.0:
            return default
        b = _EPS if b >= 0.0 else -_EPS
    return a / b


def _quantile(xs: Sequence[float], q: float) -> float:
    """Type-7 (linear interpolation) empirical quantile."""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    if n == 1:
        return s[0]
    q = min(max(q, 0.0), 1.0)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _clean(y: Sequence[float]) -> List[float]:
    """Coerce a series to a list of floats, rejecting NaN/inf up front."""
    out = [float(v) for v in y]
    for v in out:
        if math.isnan(v) or math.isinf(v):
            raise ValueError("series contains NaN/inf; clean or interpolate first")
    return out


# ===========================================================================
# 3. DENSE LINEAR ALGEBRA
# ===========================================================================
def zeros(r: int, c: int) -> Matrix:
    return [[0.0] * c for _ in range(r)]


def eye(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def transpose(A: Matrix) -> Matrix:
    if not A:
        return []
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def matmul(A: Matrix, B: Matrix) -> Matrix:
    n, k, m = len(A), len(B), (len(B[0]) if B else 0)
    C = zeros(n, m)
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for t in range(k):
            a = Ai[t]
            if a == 0.0:
                continue
            Bt = B[t]
            for j in range(m):
                Ci[j] += a * Bt[j]
    return C


def matvec(A: Matrix, x: Vector) -> Vector:
    return [_kahan_sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def outer(a: Vector, b: Vector) -> Matrix:
    return [[ai * bj for bj in b] for ai in a]


def mat_add(A: Matrix, B: Matrix) -> Matrix:
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def mat_sub(A: Matrix, B: Matrix) -> Matrix:
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def mat_scale(A: Matrix, s: float) -> Matrix:
    return [[s * v for v in row] for row in A]


def kron(A: Matrix, B: Matrix) -> Matrix:
    ra, ca = len(A), len(A[0])
    rb, cb = len(B), len(B[0])
    C = zeros(ra * rb, ca * cb)
    for i in range(ra):
        for j in range(ca):
            aij = A[i][j]
            for k in range(rb):
                for l in range(cb):
                    C[i * rb + k][j * cb + l] = aij * B[k][l]
    return C


def frobenius_norm(A: Matrix) -> float:
    return math.sqrt(_kahan_sum(v * v for row in A for v in row))


def lu_decompose(A: Matrix) -> Tuple[Matrix, List[int], int]:
    """LU decomposition with partial pivoting.  Returns (LU, pivots, sign)."""
    n = len(A)
    LU = [row[:] for row in A]
    piv = list(range(n))
    sign = 1
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(LU[i][k]))
        if abs(LU[p][k]) < _EPS:
            continue
        if p != k:
            LU[k], LU[p] = LU[p], LU[k]
            piv[k], piv[p] = piv[p], piv[k]
            sign = -sign
        pivot = LU[k][k]
        for i in range(k + 1, n):
            f = LU[i][k] / pivot
            LU[i][k] = f
            for j in range(k + 1, n):
                LU[i][j] -= f * LU[k][j]
    return LU, piv, sign


def solve(A: Matrix, b: Vector, *, ridge: float = 0.0) -> Vector:
    """Solve ``A x = b`` via LU; add a small ridge if near-singular."""
    n = len(A)
    if n == 0:
        return []
    M = [row[:] for row in A]
    if ridge > 0.0:
        for i in range(n):
            M[i][i] += ridge
    LU, piv, _ = lu_decompose(M)
    diag_min = min(abs(LU[i][i]) for i in range(n))
    if diag_min < _EPS and ridge == 0.0:               # retry with adaptive ridge
        tr = sum(abs(A[i][i]) for i in range(n)) / n + 1.0
        return solve(A, b, ridge=1e-10 * tr)
    y = [b[piv[i]] for i in range(n)]
    for i in range(n):
        for j in range(i):
            y[i] -= LU[i][j] * y[j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= LU[i][j] * x[j]
        x[i] = s / (LU[i][i] if abs(LU[i][i]) > _EPS else _EPS)
    return x


def logdet(A: Matrix) -> float:
    """Log |det(A)| via LU (never forms the determinant)."""
    n = len(A)
    LU, _, _ = lu_decompose(A)
    return _kahan_sum(math.log(abs(LU[i][i])) if abs(LU[i][i]) > _EPS
                      else math.log(_EPS) for i in range(n))


def inv(A: Matrix, *, ridge: float = 0.0) -> Matrix:
    n = len(A)
    cols = [solve(A, [1.0 if i == j else 0.0 for i in range(n)], ridge=ridge)
            for j in range(n)]
    return transpose(cols)


def cholesky(A: Matrix, *, jitter: float = 0.0) -> Optional[Matrix]:
    """Lower Cholesky factor L (A = L L').  Retries with growing jitter on the
    diagonal; returns ``None`` if it cannot be made positive definite."""
    n = len(A)
    add = jitter
    for _ in range(8):
        L = zeros(n, n)
        ok = True
        for i in range(n):
            for j in range(i + 1):
                s = A[i][j] + (add if i == j else 0.0)
                s -= _kahan_sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    if s <= _EPS:
                        ok = False
                        break
                    L[i][j] = math.sqrt(s)
                else:
                    L[i][j] = s / L[j][j]
            if not ok:
                break
        if ok:
            return L
        add = add * 10.0 if add > 0.0 else 1e-10 * (abs(A[0][0]) + 1.0)
    return None


def chol_solve(L: Matrix, b: Vector) -> Vector:
    """Solve ``L L' x = b`` given the lower factor L."""
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - _kahan_sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - _kahan_sum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
    return x


def is_pos_def(A: Matrix) -> bool:
    return cholesky(A) is not None


def logdet_chol(L: Matrix) -> float:
    return 2.0 * _kahan_sum(math.log(L[i][i]) for i in range(len(L)))


def qr_householder(A: Matrix) -> Tuple[Matrix, Matrix]:
    """Householder QR: returns (Q, R) with A = Q R, Q (m x n), R (n x n)."""
    m = len(A)
    n = len(A[0]) if A else 0
    R = [row[:] for row in A]
    Q = eye(m)
    for k in range(min(m, n)):
        x = [R[i][k] for i in range(k, m)]
        alpha = -math.copysign(math.sqrt(_kahan_sum(v * v for v in x)), x[0] or 1.0)
        if abs(alpha) < _EPS:
            continue
        v = x[:]
        v[0] -= alpha
        vnorm2 = _kahan_sum(t * t for t in v)
        if vnorm2 < _EPS:
            continue
        for j in range(n):                             # R <- (I - 2 vv'/v'v) R
            s = _kahan_sum(v[i] * R[k + i][j] for i in range(m - k))
            f = 2.0 * s / vnorm2
            for i in range(m - k):
                R[k + i][j] -= f * v[i]
        for j in range(m):                             # Q <- Q (I - 2 vv'/v'v)
            s = _kahan_sum(v[i] * Q[j][k + i] for i in range(m - k))
            f = 2.0 * s / vnorm2
            for i in range(m - k):
                Q[j][k + i] -= f * v[i]
    Rn = [R[i][:n] for i in range(n)]
    Qn = [row[:n] for row in Q]
    return Qn, Rn


def qr_lstsq(A: Matrix, b: Vector, *, ridge: float = 0.0) -> Tuple[Vector, int]:
    """Least-squares solve of ``A x ~= b`` via normal equations with a Cholesky /
    ridge fallback.  Returns (coefficients, estimated rank)."""
    m = len(A)
    n = len(A[0]) if A else 0
    if n == 0:
        return [], 0
    AtA = zeros(n, n)
    Atb = [0.0] * n
    for i in range(m):
        Ai = A[i]
        bi = b[i]
        for r in range(n):
            air = Ai[r]
            Atb[r] += air * bi
            row = AtA[r]
            for c in range(r, n):
                row[c] += air * Ai[c]
    for r in range(n):
        for c in range(r):
            AtA[r][c] = AtA[c][r]
        AtA[r][r] += ridge
    L = cholesky(AtA, jitter=1e-12)
    if L is None:
        x = solve(AtA, Atb, ridge=max(ridge, 1e-8))
        return x, n
    x = chol_solve(L, Atb)
    rank = sum(1 for i in range(n) if L[i][i] > 1e-7)
    return x, rank


def spectral_radius(A: Matrix) -> float:
    """Largest eigenvalue modulus of a square matrix, via the roots of its
    characteristic polynomial (Faddeev-LeVerrier -> Durand-Kerner)."""
    n = len(A)
    if n == 0:
        return 0.0
    if n == 1:
        return abs(A[0][0])
    # Faddeev-LeVerrier to get characteristic polynomial coefficients
    M = [row[:] for row in A]
    I = eye(n)
    c = [1.0] * (n + 1)
    Mk = [row[:] for row in I]
    for k in range(1, n + 1):
        AM = matmul(A, Mk) if k > 1 else [row[:] for row in A]
        tr = _kahan_sum(AM[i][i] for i in range(n))
        ck = -tr / k
        c[k] = ck
        Mk = mat_add(AM, mat_scale(I, ck))
    # char poly (descending): z^n + c1 z^{n-1} + ... + cn ; ascending for solver
    asc = list(reversed(c))
    roots = durand_kerner_roots(asc)
    return max((abs(r) for r in roots), default=0.0)


def solve_discrete_lyapunov(T: Matrix, M: Matrix, *, max_iter: int = 200,
                            tol: float = 1e-12) -> Matrix:
    """Solve the discrete Lyapunov equation ``P = T P T' + M`` for symmetric P.
    Uses the vectorised linear system when stable, else squaring iteration."""
    n = len(T)
    if n == 0:
        return []
    if n <= 8 and spectral_radius(T) < 1.0 - 1e-8:
        # small & stable: exact vectorised solve (I - kron(T, T)) vec(P) = vec(M).
        # This is O(n^6); for the large companion states of seasonal ARIMA models the
        # squaring iteration below (O(n^3 log(1/tol))) is used instead.
        K = kron(T, T)
        n2 = n * n
        A = [[(1.0 if i == j else 0.0) - K[i][j] for j in range(n2)]
             for i in range(n2)]
        vecM = [M[i][j] for i in range(n) for j in range(n)]
        vecP = solve(A, vecM, ridge=1e-14)
        P = [[vecP[i * n + j] for j in range(n)] for i in range(n)]
    else:                                              # doubling / squaring
        P = [row[:] for row in M]
        Tk = [row[:] for row in T]
        for _ in range(max_iter):
            P_new = mat_add(P, matmul(matmul(Tk, P), transpose(Tk)))
            Tk = matmul(Tk, Tk)
            if frobenius_norm(mat_sub(P_new, P)) < tol * (frobenius_norm(P) + 1.0):
                P = P_new
                break
            P = P_new
    # symmetrise
    return [[0.5 * (P[i][j] + P[j][i]) for j in range(n)] for i in range(n)]


# ===========================================================================
# 4. POLYNOMIALS & ROOTS
# ===========================================================================
def poly_mul(a: Sequence[float], b: Sequence[float], *, trim: bool = True,
             tol: float = _EPS) -> List[float]:
    """Multiply two lag polynomials given in ASCENDING powers (index j is the
    coefficient of B**j).  Empty inputs are treated as the unit polynomial."""
    if not a:
        a = [1.0]
    if not b:
        b = [1.0]
    c = [0.0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai == 0.0:
            continue
        for j, bj in enumerate(b):
            c[i + j] += ai * bj
    if trim:
        while len(c) > 1 and abs(c[-1]) < tol:
            c.pop()
    return c


def _expand_lag_polynomial(coefs: Sequence[float], *, m: int = 1, d: int = 0,
                           D: int = 0, seasonal: Sequence[float] = (),
                           kind: str = "ar") -> List[float]:
    """Build the full (non-seasonal x seasonal x differencing) lag polynomial in
    ascending powers.  This is the SOLE owner of the AR/MA sign convention:

        AR:  Phi(B) = 1 - sum phi_i B^i        (with (1-B)^d (1-B^m)^D factors)
        MA:  Theta(B) = 1 + sum theta_j B^j    (no differencing factors)
    """
    sgn = -1.0 if kind == "ar" else 1.0
    acc = [1.0] + [sgn * c for c in coefs]
    if seasonal:
        sfac = [1.0] + [0.0] * (len(seasonal) * m)
        for k, sc in enumerate(seasonal, start=1):
            sfac[k * m] = sgn * sc
        acc = poly_mul(acc, sfac, trim=False)
    if kind == "ar":
        for _ in range(d):
            acc = poly_mul(acc, [1.0, -1.0], trim=False)
        if D > 0 and m >= 1:
            seas_diff = [1.0] + [0.0] * (m - 1) + [-1.0]
            for _ in range(D):
                acc = poly_mul(acc, seas_diff, trim=False)
    return acc


def durand_kerner_roots(coefs: Sequence[float], *, max_iter: int = 500,
                        tol: float = 1e-13,
                        seed: complex = 0.4 + 0.9j) -> List[complex]:
    """All complex roots of a real polynomial (ascending powers) via the
    Weierstrass / Durand-Kerner simultaneous iteration.  Deterministic: the n
    starting points are ``seed**k``."""
    c = list(coefs)
    while len(c) > 1 and abs(c[-1]) < _EPS:            # deflate leading (high) zeros
        c.pop()
    lead = 0
    while lead < len(c) - 1 and abs(c[lead]) < _EPS:   # trailing (constant) zeros -> root 0
        lead += 1
    zero_roots = [0.0 + 0.0j] * lead
    c = c[lead:]
    n = len(c) - 1
    if n <= 0:
        return zero_roots
    a = [c[i] / c[-1] for i in range(len(c))]          # monic, ascending
    if n == 1:
        return zero_roots + [complex(-a[0])]
    if n == 2:
        disc = cmath.sqrt(a[1] * a[1] - 4.0 * a[0])
        return zero_roots + [(-a[1] + disc) / 2.0, (-a[1] - disc) / 2.0]

    def horner(z: complex) -> complex:
        r = a[-1]
        for k in range(len(a) - 2, -1, -1):
            r = r * z + a[k]
        return r

    roots = [seed ** k for k in range(n)]
    for _ in range(max_iter):
        max_upd = 0.0
        for i in range(n):
            num = horner(roots[i])
            den = 1.0 + 0.0j
            for j in range(n):
                if j != i:
                    den *= (roots[i] - roots[j])
            if abs(den) < 1e-300:
                den = 1e-300 + 0.0j
            upd = num / den
            roots[i] -= upd
            max_upd = max(max_upd, abs(upd))
        if max_upd < tol:
            break
    return zero_roots + roots


def companion_matrix(coefs: Sequence[float]) -> Matrix:
    """Companion matrix of a monic polynomial given by ``coefs`` (ascending, the
    leading coefficient assumed 1 after normalisation)."""
    a = list(coefs)
    lead = a[-1]
    a = [v / lead for v in a]
    n = len(a) - 1
    C = zeros(n, n)
    for i in range(n):
        C[0][i] = -a[n - 1 - i]
    for i in range(1, n):
        C[i][i - 1] = 1.0
    return C


# ===========================================================================
# 5. LEVINSON-DURBIN + PARAMETER MAPS
# ===========================================================================
def levinson_durbin(gamma: Sequence[float], *, order: int
                    ) -> Tuple[List[float], List[float], List[float]]:
    """Levinson-Durbin recursion.  Given autocovariances gamma_0..gamma_p, return
    (AR coefficients in +phi convention, reflection/PACF coefficients, innovation
    variances v_0..v_p)."""
    g = list(gamma)
    p = order
    if g[0] <= _EPS:
        return [0.0] * p, [0.0] * p, [max(g[0], 0.0)] * (p + 1)
    phi = [0.0] * (p + 1)
    refl = [0.0] * (p + 1)
    v = [0.0] * (p + 1)
    v[0] = g[0]
    prev = [0.0] * (p + 1)
    for k in range(1, p + 1):
        acc = g[k] - _kahan_sum(prev[j] * g[k - j] for j in range(1, k))
        rk = acc / v[k - 1] if v[k - 1] > _EPS else 0.0
        refl[k] = rk
        cur = prev[:]
        cur[k] = rk
        for j in range(1, k):
            cur[j] = prev[j] - rk * prev[k - j]
        v[k] = v[k - 1] * (1.0 - rk * rk)
        prev = cur
        if v[k] <= _EPS:
            v[k] = max(v[k], 0.0)
    return prev[1:p + 1], refl[1:p + 1], v


def _reflection_to_ar(refl: Sequence[float]) -> List[float]:
    """Levinson step-up: reflection coefficients -> AR coefficients (+phi)."""
    phi: List[float] = []
    for k, rk in enumerate(refl):
        newphi = phi[:] + [rk]
        for j in range(k):
            newphi[j] = phi[j] - rk * phi[k - 1 - j]
        phi = newphi
    return phi


def _ar_to_reflection(phi: Sequence[float]) -> List[float]:
    """Levinson step-down: AR coefficients -> reflection coefficients."""
    a = list(phi)
    p = len(a)
    refl = [0.0] * p
    for k in range(p - 1, -1, -1):
        rk = a[k]
        refl[k] = rk
        if abs(1.0 - rk * rk) < _EPS:
            break
        prev = [0.0] * (k)
        for j in range(k):
            prev[j] = (a[j] + rk * a[k - 1 - j]) / (1.0 - rk * rk)
        a = prev
    return refl


def pacf_transform(raw: Sequence[float]) -> List[float]:
    """Map unconstrained reals -> stationary/invertible AR (or MA) coefficients,
    via ``tanh`` reflection coefficients and the Levinson step-up.  Used to make
    likelihood optimisation unconstrained yet always admissible."""
    refl = [math.tanh(r) for r in raw]
    return _reflection_to_ar(refl)


def pacf_untransform(coefs: Sequence[float]) -> List[float]:
    """Inverse of :func:`pacf_transform` (stationary AR -> unconstrained reals)."""
    refl = _ar_to_reflection(coefs)
    out = []
    for r in refl:
        r = min(max(r, -0.999999), 0.999999)
        out.append(math.atanh(r))
    return out


def logit(p: float) -> float:
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return math.log(p / (1.0 - p))


def expit(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def softmax_simplex(raw: Sequence[float], *, cap: float = 1.0 - 1e-6) -> List[float]:
    """Map unconstrained reals to non-negative weights summing to < ``cap``
    (used to keep GARCH persistence inside the stationarity region)."""
    exps = [math.exp(min(r, 50.0)) for r in raw]
    denom = 1.0 + _kahan_sum(exps)
    return [cap * e / denom for e in exps]


# ===========================================================================
# 6. DETERMINISTIC OPTIMISERS
# ===========================================================================
@dataclass
class OptResult:
    """Outcome of a numeric minimisation."""
    x: List[float]
    fun: float
    nit: int
    converged: bool
    message: str = ""


def minimize_nelder_mead(fn: Callable[[List[float]], float], x0: Sequence[float],
                         *, bounds: Optional[Sequence[Tuple[float, float]]] = None,
                         restarts: int = 1, seed: int = 0, maxiter: int = 1000,
                         xatol: float = 1e-8, fatol: float = 1e-8,
                         init_step: float = 0.1) -> OptResult:
    """Nelder-Mead downhill simplex with deterministic multi-start.  Bounds are
    enforced by reflecting infeasible / non-finite evaluations to ``+inf``."""
    n = len(x0)
    rng = DeterministicRNG(seed)

    def clip_ok(x: Sequence[float]) -> bool:
        if bounds is None:
            return True
        return all(bounds[i][0] - 1e-12 <= x[i] <= bounds[i][1] + 1e-12
                   for i in range(n))

    def f(x: Sequence[float]) -> float:
        if not clip_ok(x):
            return _INF
        try:
            v = fn(list(x))
        except (ValueError, OverflowError, ZeroDivisionError):
            return _INF
        if v is None or math.isnan(v) or math.isinf(v):
            return _INF
        return v

    best_x = list(x0)
    best_f = f(x0)
    total_it = 0
    for r in range(max(1, restarts)):
        if r == 0:
            start = list(x0)
        else:
            start = [x0[i] + init_step * (rng.gauss()) for i in range(n)]
        simplex = [list(start)]
        fvals = [f(start)]
        for i in range(n):
            pt = list(start)
            step = init_step * (abs(start[i]) if abs(start[i]) > 1e-6 else 1.0)
            pt[i] += step
            simplex.append(pt)
            fvals.append(f(pt))
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        it = 0
        while it < maxiter:
            it += 1
            order = sorted(range(n + 1), key=lambda k: fvals[k])
            simplex = [simplex[k] for k in order]
            fvals = [fvals[k] for k in order]
            spread = max(abs(fvals[k] - fvals[0]) for k in range(1, n + 1))
            xspread = max(max(abs(simplex[k][j] - simplex[0][j]) for j in range(n))
                          for k in range(1, n + 1)) if n else 0.0
            if spread < fatol and xspread < xatol:
                break
            centroid = [_kahan_sum(simplex[k][j] for k in range(n)) / n
                        for j in range(n)]
            xr = [centroid[j] + alpha * (centroid[j] - simplex[-1][j])
                  for j in range(n)]
            fr = f(xr)
            if fvals[0] <= fr < fvals[-2]:
                simplex[-1], fvals[-1] = xr, fr
            elif fr < fvals[0]:
                xe = [centroid[j] + gamma * (xr[j] - centroid[j]) for j in range(n)]
                fe = f(xe)
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
            else:
                xc = [centroid[j] + rho * (simplex[-1][j] - centroid[j])
                      for j in range(n)]
                fc = f(xc)
                if fc < fvals[-1]:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    for k in range(1, n + 1):
                        simplex[k] = [simplex[0][j] + sigma
                                      * (simplex[k][j] - simplex[0][j])
                                      for j in range(n)]
                        fvals[k] = f(simplex[k])
        total_it += it
        if fvals[0] < best_f:
            best_f, best_x = fvals[0], simplex[0][:]
    return OptResult(best_x, best_f, total_it, best_f < _INF, "nelder-mead")


def _hessian_fd(fn: Callable[[List[float]], float], x: Sequence[float],
                *, eps: float = 1e-4) -> Matrix:
    """Central-difference Hessian, symmetrised."""
    n = len(x)
    x = list(x)
    H = zeros(n, n)
    f0 = fn(x)
    for i in range(n):
        for j in range(i, n):
            hi = eps * (abs(x[i]) + eps)
            hj = eps * (abs(x[j]) + eps)
            xpp = x[:]; xpp[i] += hi; xpp[j] += hj
            xpm = x[:]; xpm[i] += hi; xpm[j] -= hj
            xmp = x[:]; xmp[i] -= hi; xmp[j] += hj
            xmm = x[:]; xmm[i] -= hi; xmm[j] -= hj
            val = (fn(xpp) - fn(xpm) - fn(xmp) + fn(xmm)) / (4.0 * hi * hj)
            H[i][j] = val
            H[j][i] = val
    return H


# ===========================================================================
# 7. CORE RESULT RECORDS + STRICT-JSON HELPER
# ===========================================================================
@dataclass
class ForecastResult:
    """The canonical output of every ``.forecast()``: point forecasts plus,
    optionally, multi-level prediction bands and a quantile fan.

    ``lower`` / ``upper`` are keyed by coverage level (e.g. 0.80, 0.95); each maps
    to a length-``h`` list.  ``quantiles`` maps a probability tau -> length-``h``
    list.  All dict emission is done in ``sorted(keys)`` order for determinism."""
    model: str
    h: int
    mean: List[float]
    index: List[int] = field(default_factory=list)
    lower: Dict[float, List[float]] = field(default_factory=dict)
    upper: Dict[float, List[float]] = field(default_factory=dict)
    quantiles: Dict[float, List[float]] = field(default_factory=dict)
    level: List[float] = field(default_factory=list)
    sigma: List[float] = field(default_factory=list)
    method: str = "analytic"
    transform: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.index:
            self.index = list(range(self.h))
        self.level = sorted(self.level)

    def conf_int(self, level: float = 0.95) -> List[Tuple[float, float]]:
        """Return the (lower, upper) band pairs for a coverage level."""
        lo = self.lower.get(level)
        hi = self.upper.get(level)
        if lo is None or hi is None:
            return [(m, m) for m in self.mean]
        return list(zip(lo, hi))

    def summary(self) -> str:
        rows = ["forecast(%s) h=%d method=%s" % (self.model, self.h, self.method)]
        for i in range(self.h):
            line = "  t+%-3d %12.4f" % (i + 1, self.mean[i])
            for lv in self.level:
                if lv in self.lower:
                    line += "  [%.0f%%: %.3f, %.3f]" % (
                        100 * lv, self.lower[lv][i], self.upper[lv][i])
            rows.append(line)
        return "\n".join(rows)


@dataclass
class ICResult:
    """Information-criteria bundle for a fitted model."""
    aic: float
    aicc: float
    bic: float
    hqic: float
    loglik: float
    k_params: int
    nobs: int
    note: str = ""


@dataclass
class AcfResult:
    lags: List[int]
    values: List[float]
    confint_lower: Optional[List[float]] = None
    confint_upper: Optional[List[float]] = None
    se: Optional[List[float]] = None
    qstat: Optional[List[float]] = None
    pvalues: Optional[List[float]] = None
    nobs: int = 0
    method: str = ""


@dataclass
class LjungBoxResult:
    lags: List[int]
    stat_lb: List[float]
    pvalue_lb: List[Optional[float]]
    stat_bp: List[float]
    pvalue_bp: List[Optional[float]]
    df: List[int]
    model_df: int = 0


@dataclass
class NormalityTestResult:
    statistic: float
    pvalue: float
    skew: float
    kurtosis: float          # excess kurtosis
    n: int
    note: str = ""


@dataclass
class HetTestResult:
    statistic: float
    pvalue: float
    method: str
    dof: int
    lags: int


@dataclass
class UnitRootTestResult:
    test: str
    statistic: float
    pvalue: Optional[float]
    usedlag: int
    nobs: int
    critical_values: Dict[str, float]
    regression: str
    null_hypothesis: str
    stationary: bool
    flags: List[str] = field(default_factory=list)

    def summary(self) -> str:
        cvs = ", ".join("%s=%.3f" % (k, v)
                        for k, v in sorted(self.critical_values.items()))
        return ("%s: stat=%.4f p=%s lag=%d  H0=%s -> %s\n  crit: %s"
                % (self.test, self.statistic,
                   "%.4f" % self.pvalue if self.pvalue is not None else "n/a",
                   self.usedlag, self.null_hypothesis,
                   "stationary" if self.stationary else "not stationary", cvs))


@dataclass
class SeasonalPeriodResult:
    period: int
    strength: float
    method: str
    candidates_scored: List[Tuple[int, float]] = field(default_factory=list)


@dataclass
class DecompositionResult:
    observed: List[float]
    trend: List[Optional[float]]
    seasonal: List[float]
    resid: List[Optional[float]]
    model: str
    period: int
    seasonal_indices: List[float] = field(default_factory=list)

    def seasadj(self) -> List[float]:
        if self.model == "multiplicative":
            return [_safe_div(self.observed[i], self.seasonal[i], default=self.observed[i])
                    for i in range(len(self.observed))]
        return [self.observed[i] - self.seasonal[i] for i in range(len(self.observed))]


def _json_safe(obj: Any) -> Any:
    """Recursively map an object to strict-JSON: inf/nan -> None, floats rounded
    to 6 dp, dict keys stringified, tuples -> lists."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else round(obj, 6)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# ===========================================================================
# 8. INFORMATION CRITERIA
# ===========================================================================
def count_params(*, p: int = 0, q: int = 0, P: int = 0, Q: int = 0, d: int = 0,
                 D: int = 0, m: int = 0, trend: str = "c",
                 include_mean: bool = True, k_exog: int = 0,
                 count_sigma2: bool = True) -> int:
    """Number of free parameters, using the Hyndman convention (the innovation
    variance counts).  A mean term is admissible only when ``d + D == 0``; a
    linear drift ('t' / 'ct') only when ``d + D <= 1``."""
    k = p + q + P + Q + k_exog
    trend_terms = {"n": 0, "c": 1, "t": 1, "ct": 2}.get(trend, 0)
    if "t" in trend and d + D > 1:
        trend_terms -= 1
    if "c" in trend and d + D > 0:
        trend_terms -= 1
    k += max(trend_terms, 0)
    if trend == "n" and include_mean and d + D == 0:
        k += 1
    if count_sigma2:
        k += 1
    return k


def gaussian_loglike(resid: Optional[Sequence[float]] = None, *,
                     sigma2: Optional[float] = None, nobs: Optional[int] = None,
                     ssr: Optional[float] = None,
                     det_terms: Optional[Sequence[float]] = None) -> float:
    """Gaussian log-likelihood in one of three modes: concentrated (``sigma2``
    None -> plug in the ML variance), fixed-scale, or exact via per-obs
    log-determinant terms."""
    if det_terms is not None:                          # exact decomposition
        n = len(det_terms)
        s = _kahan_sum(det_terms)
        if resid is not None and sigma2 is not None:
            q = _kahan_sum(r * r for r in resid) / max(sigma2, _EPS)
            return -0.5 * (n * _LOG2PI + s + q)
        return -0.5 * (n * _LOG2PI + s)
    if resid is not None:
        n = len(resid)
        rss = ssr if ssr is not None else _kahan_sum(r * r for r in resid)
        if sigma2 is None:                             # concentrated
            s2 = max(rss / n, _EPS)
            return -0.5 * n * (_LOG2PI + math.log(s2) + 1.0)
        s2 = max(sigma2, _EPS)
        return -0.5 * (n * (_LOG2PI + math.log(s2)) + rss / s2)
    if ssr is not None and nobs is not None:
        s2 = max((sigma2 if sigma2 is not None else ssr / nobs), _EPS)
        return -0.5 * nobs * (_LOG2PI + math.log(s2) + 1.0)
    return float("nan")


def information_criteria(*, loglik: float, n_params: int, nobs: int,
                         note: str = "") -> ICResult:
    """AIC / AICc / BIC / HQIC from a log-likelihood and parameter count."""
    k, n = n_params, nobs
    aic = -2.0 * loglik + 2.0 * k
    bic = -2.0 * loglik + k * math.log(n) if n > 0 else _INF
    aicc = aic + (2.0 * k * (k + 1) / (n - k - 1)) if (n - k - 1) > 0 else _INF
    hqic = -2.0 * loglik + 2.0 * k * math.log(math.log(n)) \
        if (n > 2 and math.log(n) > 1.0) else _INF
    return ICResult(aic, aicc, bic, hqic, loglik, k, n, note)


@dataclass
class ModelComparison:
    criterion: str
    ranking: List[Tuple[str, float]]
    weights: Dict[str, float]
    best: Optional[str]


def compare_models(candidates: Sequence[Tuple[str, float]], *,
                   criterion: str = "aicc") -> ModelComparison:
    """Rank (label, IC-value) pairs, computing delta-IC and Akaike weights."""
    finite = [(nm, ic) for nm, ic in candidates if math.isfinite(ic)]
    if not finite:
        return ModelComparison(criterion, list(candidates), {}, None)
    ic_min = min(ic for _, ic in finite)
    weights_raw = {nm: math.exp(-0.5 * (ic - ic_min)) for nm, ic in finite}
    tot = _kahan_sum(weights_raw.values())
    weights = {nm: w / tot for nm, w in weights_raw.items()}
    ranking = sorted(candidates, key=lambda t: (t[1], t[0]))
    return ModelComparison(criterion, ranking, weights, ranking[0][0])


# ===========================================================================
# 9. TIME-SERIES KERNELS
# ===========================================================================
def autocovariance(y: Sequence[float], *, nlags: Optional[int] = None,
                   demean: bool = True, adjusted: bool = False) -> List[float]:
    """Sample autocovariances gamma_0..gamma_nlags.  The biased (divide-by-n)
    estimator is the default because it is positive semidefinite."""
    y = _clean(y)
    n = len(y)
    if n == 0:
        raise ValueError("autocovariance requires a non-empty series")
    if nlags is None:
        nlags = min(n - 1, int(10 * math.log10(n)) + 20) if n > 1 else 0
    nlags = min(nlags, n - 1)
    mu = _mean(y) if demean else 0.0
    dev = [v - mu for v in y]
    out = []
    for k in range(nlags + 1):
        s = _kahan_sum(dev[t] * dev[t - k] for t in range(k, n))
        out.append(s / ((n - k) if adjusted else n))
    return out


def acf(y: Sequence[float], *, nlags: Optional[int] = None,
        adjusted: bool = False, alpha: float = 0.05,
        qstat: bool = False) -> AcfResult:
    """Autocorrelation function with Bartlett confidence bands (and optionally the
    Ljung-Box Q statistics)."""
    y = _clean(y)
    n = len(y)
    gam = autocovariance(y, nlags=nlags, adjusted=adjusted)
    g0 = gam[0] if gam[0] > _EPS else _EPS
    r = [g / g0 for g in gam]
    lags = list(range(len(r)))
    z = norm_ppf(1.0 - alpha / 2.0)
    se = [0.0]
    for k in range(1, len(r)):
        var = (1.0 + 2.0 * _kahan_sum(r[j] * r[j] for j in range(1, k))) / n
        se.append(math.sqrt(max(var, 0.0)))
    lo = [r[k] - z * se[k] for k in range(len(r))]
    hi = [r[k] + z * se[k] for k in range(len(r))]
    qs = pv = None
    if qstat:
        qs, pv = [], []
        run = 0.0
        for k in range(1, len(r)):
            run += r[k] * r[k] / (n - k) if n - k > 0 else 0.0
            q = n * (n + 2.0) * run
            qs.append(q)
            pv.append(chi2_sf(q, df=k))
    return AcfResult(lags, r, lo, hi, se, qs, pv, n, "acf")


def pacf(y: Sequence[float], *, nlags: Optional[int] = None,
         method: str = "ld", alpha: float = 0.05) -> AcfResult:
    """Partial autocorrelation function.  ``method='ld'`` uses Levinson-Durbin on
    the sample autocovariances; ``'ols'`` regresses on lagged values."""
    y = _clean(y)
    n = len(y)
    if nlags is None:
        nlags = min(n // 2 - 1, int(10 * math.log10(n)) + 10) if n > 4 else 1
    nlags = max(1, min(nlags, n // 2 - 1 if n > 4 else 1))
    vals = [1.0]
    if method == "ld":
        gam = autocovariance(y, nlags=nlags)
        _, refl, _ = levinson_durbin(gam, order=nlags)
        vals += refl
    else:                                              # OLS regression per order
        mu = _mean(y)
        dev = [v - mu for v in y]
        for k in range(1, nlags + 1):
            X = [[dev[t - j] for j in range(1, k + 1)] for t in range(k, n)]
            yy = [dev[t] for t in range(k, n)]
            coef, _ = qr_lstsq(X, yy, ridge=1e-10)
            vals.append(coef[-1] if coef else 0.0)
    z = norm_ppf(1.0 - alpha / 2.0)
    se = [0.0] + [1.0 / math.sqrt(n)] * (len(vals) - 1)
    lo = [vals[k] - z * se[k] for k in range(len(vals))]
    hi = [vals[k] + z * se[k] for k in range(len(vals))]
    return AcfResult(list(range(len(vals))), vals, lo, hi, se, None, None, n,
                     "pacf-" + method)


def ccf(x: Sequence[float], y: Sequence[float], *, nlags: Optional[int] = None,
        alpha: float = 0.05) -> AcfResult:
    """Cross-correlation function between two equal-length series."""
    x = _clean(x); y = _clean(y)
    n = min(len(x), len(y))
    if n == 0:
        raise ValueError("ccf requires non-empty series")
    x, y = x[:n], y[:n]
    if nlags is None:
        nlags = min(n - 1, int(10 * math.log10(n)) + 10)
    mx, my = _mean(x), _mean(y)
    sx = math.sqrt(max(_kahan_sum((v - mx) ** 2 for v in x), _EPS))
    sy = math.sqrt(max(_kahan_sum((v - my) ** 2 for v in y), _EPS))
    lags = list(range(-nlags, nlags + 1))
    vals = []
    for k in lags:
        if k >= 0:
            s = _kahan_sum((x[t] - mx) * (y[t - k] - my) for t in range(k, n))
        else:
            s = _kahan_sum((x[t + k] - mx) * (y[t] - my) for t in range(-k, n))
        vals.append(s / (sx * sy))
    se = [1.0 / math.sqrt(n)] * len(vals)
    z = norm_ppf(1.0 - alpha / 2.0)
    lo = [v - z * s for v, s in zip(vals, se)]
    hi = [v + z * s for v, s in zip(vals, se)]
    return AcfResult(lags, vals, lo, hi, se, None, None, n, "ccf")


def ljung_box(resid: Sequence[float], *, lags: int = 10, model_df: int = 0,
              boxpierce: bool = True) -> LjungBoxResult:
    """Ljung-Box (and Box-Pierce) portmanteau test for residual autocorrelation.
    Degrees of freedom are corrected by ``model_df`` (fitted AR+MA order)."""
    resid = _clean(resid)
    n = len(resid)
    ac = acf(resid, nlags=lags)
    r = ac.values
    lb_s, bp_s, lb_p, bp_p, dfs = [], [], [], [], []
    run_lb = 0.0
    run_bp = 0.0
    for k in range(1, lags + 1):
        rk = r[k] if k < len(r) else 0.0
        run_lb += rk * rk / (n - k) if (n - k) > 0 else 0.0
        run_bp += rk * rk
        lb = n * (n + 2.0) * run_lb
        bp = n * run_bp
        df = k - model_df
        lb_s.append(lb)
        bp_s.append(bp)
        dfs.append(df)
        lb_p.append(chi2_sf(lb, df=df) if df > 0 else None)
        bp_p.append(chi2_sf(bp, df=df) if df > 0 else None)
    return LjungBoxResult(list(range(1, lags + 1)), lb_s, lb_p,
                          bp_s if boxpierce else [], bp_p if boxpierce else [],
                          dfs, model_df)


def durbin_watson(resid: Sequence[float]) -> float:
    """Durbin-Watson statistic for first-order residual autocorrelation."""
    resid = _clean(resid)
    if len(resid) < 2:
        return 2.0
    num = _kahan_sum((resid[i] - resid[i - 1]) ** 2 for i in range(1, len(resid)))
    den = _kahan_sum(r * r for r in resid)
    return _safe_div(num, den, default=2.0)


def jarque_bera(resid: Sequence[float]) -> NormalityTestResult:
    """Jarque-Bera normality test from residual skewness and excess kurtosis."""
    resid = _clean(resid)
    n = len(resid)
    if n < 2:
        return NormalityTestResult(0.0, 1.0, 0.0, 0.0, n, "too few obs")
    mu = _mean(resid)
    m2 = _kahan_sum((r - mu) ** 2 for r in resid) / n
    m3 = _kahan_sum((r - mu) ** 3 for r in resid) / n
    m4 = _kahan_sum((r - mu) ** 4 for r in resid) / n
    s = m2 if m2 > _EPS else _EPS
    skew = m3 / (s ** 1.5)
    kurt = m4 / (s * s) - 3.0
    jb = n / 6.0 * (skew * skew + kurt * kurt / 4.0)
    return NormalityTestResult(jb, chi2_sf(jb, df=2), skew, kurt, n)


def heteroskedasticity_test(resid: Sequence[float], *, lags: int = 12
                            ) -> HetTestResult:
    """Engle's ARCH-LM test for conditional heteroskedasticity in residuals."""
    resid = _clean(resid)
    e2 = [r * r for r in resid]
    n = len(e2)
    lags = min(lags, n // 2)
    if lags < 1:
        return HetTestResult(0.0, 1.0, "archlm", 0, 0)
    X = [[1.0] + [e2[t - j] for j in range(1, lags + 1)] for t in range(lags, n)]
    yy = [e2[t] for t in range(lags, n)]
    coef, _ = qr_lstsq(X, yy, ridge=1e-10)
    fitted = [_kahan_sum(X[i][j] * coef[j] for j in range(len(coef)))
              for i in range(len(X))]
    ybar = _mean(yy)
    ss_res = _kahan_sum((yy[i] - fitted[i]) ** 2 for i in range(len(yy)))
    ss_tot = _kahan_sum((v - ybar) ** 2 for v in yy)
    r2 = 1.0 - _safe_div(ss_res, ss_tot, default=0.0)
    stat = len(yy) * r2
    return HetTestResult(stat, chi2_sf(stat, df=lags), "archlm", lags, lags)


def diagnose_residuals(resid: Sequence[float], *, model_df: int = 0,
                       lb_lags: Optional[int] = None) -> Dict[str, Any]:
    """Convenience bundle of the standard residual diagnostics."""
    resid = _clean(resid)
    n = len(resid)
    if lb_lags is None:
        lb_lags = min(10, max(1, n // 5))
    lb = ljung_box(resid, lags=lb_lags, model_df=model_df)
    jb = jarque_bera(resid)
    het = heteroskedasticity_test(resid, lags=min(12, max(1, n // 4)))
    last_p = lb.pvalue_lb[-1]
    return {
        "n": n,
        "mean_resid": _mean(resid),
        "sigma2": _var(resid),
        "durbin_watson": durbin_watson(resid),
        "ljung_box_stat": lb.stat_lb[-1] if lb.stat_lb else 0.0,
        "ljung_box_pvalue": last_p,
        "jarque_bera_pvalue": jb.pvalue,
        "arch_pvalue": het.pvalue,
        "white_noise": bool(last_p is not None and last_p > 0.05),
        "normal": jb.pvalue > 0.05,
        "homoskedastic": het.pvalue > 0.05,
    }


# ===========================================================================
# 10. TRANSFORMS
# ===========================================================================
def box_cox(y: Sequence[float], *, lam: float, shift: float = 0.0) -> List[float]:
    """Box-Cox power transform; ``lam`` near 0 -> natural log.  Requires
    ``y + shift > 0``."""
    y = _clean(y)
    out = []
    for v in y:
        base = v + shift
        if base <= 0.0:
            raise ValueError("box_cox requires positive (y + shift)")
        if abs(lam) < 1e-6:
            out.append(math.log(base))
        else:
            out.append((base ** lam - 1.0) / lam)
    return out


def box_cox_inverse(w: Sequence[float], *, lam: float, shift: float = 0.0
                    ) -> List[float]:
    """Inverse Box-Cox transform."""
    out = []
    for v in w:
        if abs(lam) < 1e-6:
            out.append(math.exp(v) - shift)
        else:
            base = lam * v + 1.0
            base = base if base > _EPS else _EPS
            out.append(base ** (1.0 / lam) - shift)
    return out


def guerrero_lambda(y: Sequence[float], *, m: int = 1,
                    bounds: Tuple[float, float] = (-1.0, 2.0)) -> float:
    """Select the Box-Cox lambda by Guerrero's coefficient-of-variation method
    (golden-section search, fixed iteration count for determinism)."""
    y = _clean(y)
    n = len(y)
    m = max(m, 2)
    ngroups = n // m
    if ngroups < 2:
        return 1.0
    groups = [y[i * m:(i + 1) * m] for i in range(ngroups)]

    def cv(lam: float) -> float:
        ratios = []
        for g in groups:
            mu = _mean(g)
            sd = _std(g, ddof=1)
            if mu <= _EPS:
                continue
            ratios.append(sd / (mu ** (1.0 - lam)))
        if len(ratios) < 2:
            return _INF
        return _std(ratios, ddof=1) / max(_mean(ratios), _EPS)

    lo, hi = bounds
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a = hi - gr * (hi - lo)
    b = lo + gr * (hi - lo)
    fa, fb = cv(a), cv(b)
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - gr * (hi - lo)
            fa = cv(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + gr * (hi - lo)
            fb = cv(b)
    return round(0.5 * (lo + hi), 4)


class Standardizer:
    """Centre and scale a series, remembering the parameters for inversion."""

    def __init__(self, *, with_mean: bool = True, with_std: bool = True,
                 ddof: int = 0) -> None:
        self.with_mean = with_mean
        self.with_std = with_std
        self.ddof = ddof
        self.mean_ = 0.0
        self.std_ = 1.0

    def fit(self, y: Sequence[float]) -> "Standardizer":
        y = _clean(y)
        self.mean_ = _mean(y) if self.with_mean else 0.0
        self.std_ = _std(y, ddof=self.ddof) if self.with_std else 1.0
        if self.std_ < _EPS:
            self.std_ = 1.0
        return self

    def transform(self, y: Sequence[float]) -> List[float]:
        return [(v - self.mean_) / self.std_ for v in y]

    def fit_transform(self, y: Sequence[float]) -> List[float]:
        return self.fit(y).transform(y)

    def inverse_transform(self, w: Sequence[float]) -> List[float]:
        return [v * self.std_ + self.mean_ for v in w]


def detrend(y: Sequence[float], *, order: int = 1,
            return_trend: bool = False):
    """Remove a polynomial trend of the given order by least squares."""
    y = _clean(y)
    n = len(y)
    tau = [(2.0 * t / (n - 1) - 1.0) if n > 1 else 0.0 for t in range(n)]
    X = [[tau[t] ** k for k in range(order + 1)] for t in range(n)]
    coef, _ = qr_lstsq(X, y, ridge=1e-12)
    trend = [_kahan_sum(X[t][k] * coef[k] for k in range(order + 1))
             for t in range(n)]
    resid = [y[t] - trend[t] for t in range(n)]
    if return_trend:
        return resid, trend
    return resid


# ===========================================================================
# 11. DIFFERENCING (one ordered differencer with exact inversion)
# ===========================================================================
class Differencer:
    """Apply seasonal-then-regular differencing ``(1-B^m)^D (1-B)^d`` and invert
    it exactly using the stored boundary values.  The single differencing API in
    this module -- every forecaster routes integration through it."""

    def __init__(self, *, d: int = 1, D: int = 0, m: int = 1) -> None:
        self.d = d
        self.D = D
        self.m = m
        self._seasonal_tails: List[List[float]] = []
        self._regular_tails: List[float] = []
        self.n_in_ = 0

    def fit_transform(self, y: Sequence[float]) -> List[float]:
        y = _clean(y)
        self.n_in_ = len(y)
        if len(y) <= self.d + self.D * self.m:
            raise ValueError("series too short for the requested differencing")
        cur = list(y)
        self._seasonal_tails = []
        for _ in range(self.D):
            self._seasonal_tails.append(cur[:self.m])
            cur = [cur[t] - cur[t - self.m] for t in range(self.m, len(cur))]
        self._regular_tails = []
        for _ in range(self.d):
            self._regular_tails.append(cur[0])
            cur = [cur[t] - cur[t - 1] for t in range(1, len(cur))]
        return cur

    def transform(self, y: Sequence[float]) -> List[float]:
        return self.fit_transform(y)

    def inverse_transform(self, w: Sequence[float]) -> List[float]:
        """Undo the differencing (LIFO): regular integrations first, then
        seasonal, replaying the stored boundary blocks."""
        cur = list(w)
        for anchor in reversed(self._regular_tails):    # undo regular diffs
            out = [anchor]
            run = anchor
            for v in cur:
                run += v
                out.append(run)
            cur = out
        for tail in reversed(self._seasonal_tails):      # undo seasonal diffs
            out = list(tail)
            for t in range(len(cur)):
                out.append(cur[t] + out[t])
            cur = out
        return cur

    def forecast_integrate(self, base: Sequence[float], fdiff: Sequence[float]
                           ) -> List[float]:
        """Integrate a forecast of the differenced series back onto levels, given
        the ORIGINAL series ``base`` (needed for the lag anchors)."""
        base = list(base)
        history = base[:]
        out = []
        d, D, m = self.d, self.D, self.m
        for step in range(len(fdiff)):
            val = fdiff[step]
            ext = history + out
            n = len(ext)
            # invert regular differencing: add back d previous-level increments
            if d >= 1:
                # w_t (fully differenced) -> undo (1-B)^d and (1-B^m)^D by the
                # ARIMA recursion: y_t = w_t + sum of binomial-lag terms
                val += _integration_lag_sum(ext, d=d, D=D, m=m)
            elif D >= 1:
                val += _integration_lag_sum(ext, d=0, D=D, m=m)
            out.append(val)
        return out


def _binom_int(n: int, k: int) -> int:
    return math.comb(n, k)


def _integration_terms(d: int, D: int, m: int) -> List[Tuple[int, float]]:
    """The lag polynomial 1 - (1-B)^d (1-B^m)^D as (lag, coefficient) pairs so
    that y_t = w_t + sum coef * y_{t-lag}."""
    poly = _expand_lag_polynomial([], m=m, d=d, D=D, kind="ar")   # (1-B)^d(1-B^m)^D
    terms = []
    for lag in range(1, len(poly)):
        c = poly[lag]
        if abs(c) > _EPS:
            terms.append((lag, -c))          # move to RHS: y_t = w_t - sum poly*y
    return terms


def _integration_lag_sum(history: Sequence[float], *, d: int, D: int, m: int
                         ) -> float:
    terms = _integration_terms(d, D, m)
    n = len(history)
    return _kahan_sum(c * history[n - lag] for lag, c in terms if lag <= n)


def diff(y: Sequence[float], *, d: int = 1, D: int = 0, m: int = 1) -> List[float]:
    """Stateless differencing wrapper."""
    return Differencer(d=d, D=D, m=m).fit_transform(y)


def integrate(w: Sequence[float], history: Sequence[float], *, d: int = 1,
              D: int = 0, m: int = 1) -> List[float]:
    """Integrate a differenced forecast ``w`` back to levels given the original
    series ``history``."""
    return Differencer(d=d, D=D, m=m).forecast_integrate(history, w)


def fourier_terms(n: int, *, m: int, k: int, t0: int = 0) -> Matrix:
    """Fourier (harmonic) seasonal regressors: cos/sin(2*pi*j*t/m), j=1..k."""
    k = min(k, m // 2)
    out = []
    for t in range(t0, t0 + n):
        row = []
        for j in range(1, k + 1):
            ang = 2.0 * math.pi * j * t / m
            row.append(math.cos(ang))
            if j != m / 2.0:
                row.append(math.sin(ang))
        out.append(row)
    return out


# ===========================================================================
# 12. DECOMPOSITION
# ===========================================================================
def moving_average(y: Sequence[float], *, window: int, center: bool = True
                   ) -> List[Optional[float]]:
    """Simple (optionally centred) moving average; even windows use a 2xwindow
    weighted average (half weight on the endpoints), as in classical decomposition."""
    y = _clean(y)
    n = len(y)
    out: List[Optional[float]] = [None] * n
    if window <= 0 or window > n:
        return out
    if center and window % 2 == 0:                     # 2xW centred average
        half = window // 2
        for t in range(half, n - half):
            s = 0.5 * y[t - half] + 0.5 * y[t + half]
            s += _kahan_sum(y[t + j] for j in range(-half + 1, half))
            out[t] = s / window
    else:
        half = window // 2
        for t in range(n):
            lo = t - half
            hi = t + half + (0 if window % 2 == 1 else -1)
            if not center:
                lo, hi = t - window + 1, t
            if lo < 0 or hi >= n:
                continue
            out[t] = _kahan_sum(y[lo:hi + 1]) / (hi - lo + 1)
    return out


def seasonal_decompose(y: Sequence[float], *, model: str = "additive",
                       period: int) -> DecompositionResult:
    """Classical seasonal decomposition: a centred moving-average trend, seasonal
    indices from the detrended series, and a remainder."""
    y = _clean(y)
    n = len(y)
    m = period
    if n < 2 * m:
        raise ValueError("need at least two full periods to decompose")
    trend = moving_average(y, window=m, center=True)
    if model == "multiplicative":
        detr = [(_safe_div(y[t], trend[t], default=1.0) if trend[t] is not None
                 else None) for t in range(n)]
    else:
        detr = [(y[t] - trend[t] if trend[t] is not None else None)
                for t in range(n)]
    # average seasonal component per phase
    phase_vals: List[List[float]] = [[] for _ in range(m)]
    for t in range(n):
        if detr[t] is not None:
            phase_vals[t % m].append(detr[t])
    seas_idx = [(_mean(v) if v else (1.0 if model == "multiplicative" else 0.0))
                for v in phase_vals]
    if model == "multiplicative":
        adj = m / _kahan_sum(seas_idx) if _kahan_sum(seas_idx) > _EPS else 1.0
        seas_idx = [s * adj for s in seas_idx]
    else:
        adj = _mean(seas_idx)
        seas_idx = [s - adj for s in seas_idx]
    seasonal = [seas_idx[t % m] for t in range(n)]
    if model == "multiplicative":
        resid = [(_safe_div(y[t], trend[t] * seasonal[t], default=1.0)
                  if trend[t] is not None else None) for t in range(n)]
    else:
        resid = [(y[t] - trend[t] - seasonal[t] if trend[t] is not None else None)
                 for t in range(n)]
    return DecompositionResult(y, trend, seasonal, resid, model, m, seas_idx)


def _loess_point(x: Sequence[float], y: Sequence[float], x0: float, *,
                 span_pts: int, degree: int = 1, sorted_x: bool = False) -> float:
    """Local (tricube-weighted) polynomial regression evaluated at ``x0``: solve
    the weighted normal equations (X' W X) b = X' W y and return the intercept.
    ``sorted_x`` (x ascending) restricts the neighbour search to a contiguous window
    -- O(span) instead of sorting all n points, which made STL O(n^2 log n)."""
    n = len(x)
    span_pts = min(span_pts, n)
    if sorted_x:
        # the span_pts nearest neighbours of x0 in an ascending x lie within
        # span_pts index positions of its insertion point on either side.
        pos = bisect.bisect_left(x, x0)
        lo, hi = max(0, pos - span_pts), min(n, pos + span_pts)
        dist = sorted(range(lo, hi), key=lambda i: abs(x[i] - x0))[:span_pts]
    else:
        dist = sorted(range(n), key=lambda i: abs(x[i] - x0))[:span_pts]
    dmax = max(abs(x[i] - x0) for i in dist) or 1.0
    AtA = zeros(degree + 1, degree + 1)
    Atb = [0.0] * (degree + 1)
    for i in dist:
        u = abs(x[i] - x0) / dmax
        wr = (1.0 - u ** 3) ** 3 if u < 1.0 else 0.0
        row = [(x[i] - x0) ** k for k in range(degree + 1)]
        for a in range(degree + 1):
            Atb[a] += wr * row[a] * y[i]
            for b in range(degree + 1):
                AtA[a][b] += wr * row[a] * row[b]
    coef = solve(AtA, Atb, ridge=1e-10)
    return coef[0]


def stl_decompose(y: Sequence[float], *, period: int, seasonal_span: int = 7,
                  trend_span: Optional[int] = None) -> DecompositionResult:
    """A compact STL-style loess decomposition: iteratively loess-smooth the
    cycle-subseries for the seasonal, subtract, loess-smooth for the trend."""
    y = _clean(y)
    n = len(y)
    m = period
    if trend_span is None:
        trend_span = int(1.5 * m / (1.0 - 1.5 / max(seasonal_span, 2))) | 1
        trend_span = min(max(trend_span, m + 1), n)
    seasonal = [0.0] * n
    trend = [0.0] * n
    for _ in range(2):                                 # inner iterations
        detr = [y[t] - trend[t] for t in range(n)]
        # smooth each cycle-subseries
        for phase in range(m):
            idx = list(range(phase, n, m))
            xs = [float(i) for i in idx]
            ys = [detr[i] for i in idx]
            sp = min(max(seasonal_span, 3), len(idx))
            for i in idx:
                seasonal[i] = _loess_point(xs, ys, float(i), span_pts=sp,
                                           degree=1, sorted_x=True)
        smean = _mean(seasonal)
        seasonal = [s - smean for s in seasonal]        # de-mean seasonal
        deseas = [y[t] - seasonal[t] for t in range(n)]
        xs = [float(t) for t in range(n)]
        sp = min(max(trend_span, 3), n)
        trend = [_loess_point(xs, deseas, float(t), span_pts=sp, degree=1,
                              sorted_x=True)
                 for t in range(n)]
    resid = [y[t] - trend[t] - seasonal[t] for t in range(n)]
    seas_idx = [_mean([seasonal[i] for i in range(phase, n, m)])
                for phase in range(m)]
    return DecompositionResult(y, list(trend), seasonal, list(resid),
                               "additive", m, seas_idx)


def seasonal_strength(y: Sequence[float], *, m: int) -> float:
    """Seasonal strength in [0, 1]: 1 - Var(resid)/Var(resid+seasonal)."""
    y = _clean(y)
    if m < 2 or len(y) < 2 * m:
        return 0.0
    dec = stl_decompose(y, period=m)
    resid = [r for r in dec.resid]
    deseas = [resid[i] + dec.seasonal[i] for i in range(len(resid))]
    vr = _var(resid)
    vd = _var(deseas)
    return max(0.0, min(1.0, 1.0 - _safe_div(vr, vd, default=1.0)))


def trend_strength(y: Sequence[float], *, m: int = 1) -> float:
    """Trend strength in [0, 1]."""
    y = _clean(y)
    if len(y) < max(2 * m, 4):
        return 0.0
    dec = stl_decompose(y, period=max(m, 2))
    resid = [r for r in dec.resid]
    detr = [resid[i] + dec.trend[i] for i in range(len(resid))]
    return max(0.0, min(1.0, 1.0 - _safe_div(_var(resid), _var(detr), default=1.0)))


# ===========================================================================
# 13. UNIT-ROOT & SEASONALITY TESTS
# ===========================================================================
# Asymptotic Dickey-Fuller tau critical values (MacKinnon), by regression type.
_ADF_CRIT = {
    "n": {0.01: -2.566, 0.05: -1.941, 0.10: -1.617},
    "c": {0.01: -3.430, 0.05: -2.861, 0.10: -2.567},
    "ct": {0.01: -3.960, 0.05: -3.410, 0.10: -3.120},
}
# KPSS critical values (Kwiatkowski et al. 1992), right-tailed.
_KPSS_CRIT = {
    "c": {0.10: 0.347, 0.05: 0.463, 0.025: 0.574, 0.01: 0.739},
    "ct": {0.10: 0.119, 0.05: 0.146, 0.025: 0.176, 0.01: 0.216},
}


def _interp_pvalue(stat: float, table: Dict[float, float], *,
                   left_tail: bool) -> Tuple[float, List[str]]:
    """Approximate a p-value by linear interpolation in a (p -> critical) table."""
    pts = sorted(table.items(), key=lambda kv: kv[1])   # by critical value
    flags: List[str] = []
    crits = [c for _, c in pts]
    ps = [p for p, _ in pts]
    if left_tail:                                       # ADF: reject if stat < crit
        if stat <= crits[0]:
            return ps[0] * (crits[0] / stat if stat < 0 else 1.0) if False else max(ps[0] / 2.0, 0.0005), ["extrapolated"]
        if stat >= crits[-1]:
            return min(ps[-1] + (1.0 - ps[-1]) * 0.5, 0.99), ["extrapolated"]
        for i in range(len(crits) - 1):
            if crits[i] <= stat <= crits[i + 1]:
                w = (stat - crits[i]) / (crits[i + 1] - crits[i])
                return ps[i] + w * (ps[i + 1] - ps[i]), flags
    else:                                               # KPSS: reject if stat > crit
        if stat <= crits[0]:
            return min(ps[0] + (1.0 - ps[0]) * 0.5, 0.99), ["extrapolated"]
        if stat >= crits[-1]:
            return max(ps[-1] / 2.0, 0.001), ["extrapolated"]
        for i in range(len(crits) - 1):
            if crits[i] <= stat <= crits[i + 1]:
                w = (stat - crits[i]) / (crits[i + 1] - crits[i])
                return ps[i] + w * (ps[i + 1] - ps[i]), flags
    return 0.5, ["extrapolated"]


def adf_test(y: Sequence[float], *, maxlag: Optional[int] = None,
             regression: str = "c", autolag: str = "aic") -> UnitRootTestResult:
    """Augmented Dickey-Fuller test.  Regresses Delta y_t on y_{t-1}, deterministic
    terms and ``k`` lagged differences; the tau statistic is the t-stat on the
    y_{t-1} coefficient.  ``autolag`` selects ``k`` by AIC on the common sample."""
    y = _clean(y)
    n = len(y)
    if maxlag is None:
        maxlag = int(math.ceil(12.0 * (n / 100.0) ** 0.25))
    maxlag = max(0, min(maxlag, n // 2 - 2))
    dy = [y[t] - y[t - 1] for t in range(1, n)]

    def build(k: int):
        start = k + 1
        rows, yy = [], []
        for t in range(start, n):
            row = [y[t - 1]]
            for j in range(1, k + 1):
                row.append(dy[t - 1 - j])              # dy index offset
            if "c" in regression:
                row.append(1.0)
            if "t" in regression:
                row.append(float(t))
            rows.append(row)
            yy.append(y[t] - y[t - 1])
        return rows, yy

    best = None
    lags = range(maxlag + 1) if autolag else [maxlag]
    for k in lags:
        X, yy = build(k)
        if len(yy) <= len(X[0]) + 1:
            continue
        coef, _ = qr_lstsq(X, yy, ridge=1e-12)
        fitted = [_kahan_sum(X[i][j] * coef[j] for j in range(len(coef)))
                  for i in range(len(X))]
        resid = [yy[i] - fitted[i] for i in range(len(yy))]
        rss = _kahan_sum(r * r for r in resid)
        nobs = len(yy)
        kp = len(coef)
        s2 = rss / max(nobs - kp, 1)
        ll = -0.5 * nobs * (_LOG2PI + math.log(max(rss / nobs, _EPS)) + 1.0)
        aic = -2.0 * ll + 2.0 * kp
        if best is None or aic < best[0]:
            best = (aic, k, coef, X, yy, resid, s2)
    if best is None:
        return UnitRootTestResult("ADF", 0.0, None, 0, 0, dict(_ADF_CRIT[regression]),
                                  regression, "unit root (non-stationary)", False,
                                  ["insufficient data"])
    _, k, coef, X, yy, resid, s2 = best
    # standard error of the y_{t-1} coefficient
    AtA = zeros(len(coef), len(coef))
    for i in range(len(X)):
        for a in range(len(coef)):
            for b in range(len(coef)):
                AtA[a][b] += X[i][a] * X[i][b]
    cov = inv(AtA, ridge=1e-12)
    se = math.sqrt(max(s2 * cov[0][0], _EPS))
    tau = coef[0] / se
    crit = dict(_ADF_CRIT[regression])
    pval, flags = _interp_pvalue(tau, crit, left_tail=True)
    stationary = tau < crit[0.05]
    return UnitRootTestResult("ADF", tau, pval, k, len(yy), crit, regression,
                              "unit root (non-stationary)", stationary, flags)


def kpss_test(y: Sequence[float], *, regression: str = "c",
              lags: Optional[int] = None) -> UnitRootTestResult:
    """KPSS stationarity test (null: stationary).  Large statistic -> reject."""
    y = _clean(y)
    n = len(y)
    if regression == "ct":
        tt = [(2.0 * t / (n - 1) - 1.0) for t in range(n)]
        X = [[1.0, tt[t]] for t in range(n)]
        coef, _ = qr_lstsq(X, y, ridge=1e-12)
        resid = [y[t] - (coef[0] + coef[1] * tt[t]) for t in range(n)]
    else:
        mu = _mean(y)
        resid = [v - mu for v in y]
    S = list(itertools.accumulate(resid))
    if lags is None:
        lags = int(4.0 * (n / 100.0) ** 0.25)
    lags = max(0, min(lags, n - 1))
    s2 = _kahan_sum(r * r for r in resid) / n
    for l in range(1, lags + 1):                       # Bartlett long-run variance
        w = 1.0 - l / (lags + 1.0)
        cov = _kahan_sum(resid[t] * resid[t - l] for t in range(l, n)) / n
        s2 += 2.0 * w * cov
    s2 = max(s2, _EPS)
    stat = _kahan_sum(s * s for s in S) / (n * n * s2)
    crit = dict(_KPSS_CRIT.get(regression, _KPSS_CRIT["c"]))
    pval, flags = _interp_pvalue(stat, crit, left_tail=False)
    stationary = stat < crit[0.05]
    return UnitRootTestResult("KPSS", stat, pval, lags, n, crit, regression,
                              "stationary", stationary, flags)


def ndiffs(y: Sequence[float], *, test: str = "kpss", alpha: float = 0.05,
           max_d: int = 2, regression: str = "c") -> int:
    """Number of regular differences needed for stationarity (KPSS by default:
    difference while the stationarity null is rejected)."""
    y = _clean(y)
    d = 0
    cur = list(y)
    for _ in range(max_d):
        if len(cur) < 10:
            break
        try:
            if test == "kpss":
                res = kpss_test(cur, regression=regression)
                need = (res.pvalue is not None and res.pvalue < alpha)
            else:
                res = adf_test(cur, regression=regression)
                need = not (res.pvalue is not None and res.pvalue < alpha)
        except (ValueError, ZeroDivisionError):
            break
        if not need:
            break
        cur = [cur[t] - cur[t - 1] for t in range(1, len(cur))]
        d += 1
    return d


def nsdiffs(y: Sequence[float], *, m: int, max_D: int = 1,
            threshold: float = 0.64) -> int:
    """Number of seasonal differences, via a seasonal-strength heuristic."""
    y = _clean(y)
    if m < 2:
        return 0
    D = 0
    cur = list(y)
    for _ in range(max_D):
        if len(cur) < 2 * m:
            break
        if seasonal_strength(cur, m=m) < threshold:
            break
        cur = [cur[t] - cur[t - m] for t in range(m, len(cur))]
        D += 1
    return D


def detect_seasonal_period(y: Sequence[float], *,
                           candidates: Optional[Sequence[int]] = None,
                           max_period: Optional[int] = None,
                           min_cycles: int = 2,
                           strength_threshold: float = 0.3) -> SeasonalPeriodResult:
    """Detect the dominant seasonal period from the autocorrelation peaks and the
    STL seasonal strength."""
    y = _clean(y)
    n = len(y)
    if max_period is None:
        max_period = n // min_cycles
    if candidates is None:
        candidates = [m for m in (2, 3, 4, 6, 7, 12, 24, 52) if m <= max_period]
        candidates = list(candidates) or [2]
    # ACF of the DETRENDED series so a smooth trend does not inflate small lags
    detr = detrend(y, order=1)
    ac = acf(detr, nlags=min(max_period, n - 1))
    scored: List[Tuple[int, float]] = []
    strengths: Dict[int, float] = {}
    peaks: Dict[int, float] = {}
    for m in candidates:
        if m >= len(ac.values) - 1 or n < min_cycles * m:
            continue
        strength = seasonal_strength(y, m=m) if n >= 2 * m else 0.0
        strengths[m] = strength
        # ACF-peak bonus: reward lags that are a local maximum (true seasonality)
        peak = ac.values[m] if (ac.values[m] > ac.values[m - 1]
                                 and ac.values[m] >= ac.values[m + 1]) else 0.0
        peaks[m] = peak
        score = strength + 0.3 * max(peak, 0.0)
        scored.append((m, score))
    if not scored:
        return SeasonalPeriodResult(1, 0.0, "acf", [])
    scored.sort(key=lambda t: (-t[1], -strengths.get(t[0], 0.0), t[0]))
    best_m, best_score = scored[0]
    sig = 1.96 / math.sqrt(n)                          # ACF significance band
    # A genuine seasonal lag stands out as a LOCAL ACF maximum (peak > sig); a
    # monotonically decaying ACF (a plain AR process) has no such peak even though
    # seasonal_strength -- driven by an STL fit that overfits tiny periods like
    # m=2 -- can look large.  Requiring the peak avoids reporting AR autocorrelation
    # as seasonality.
    if (strengths.get(best_m, 0.0) < strength_threshold
            or abs(ac.values[best_m]) < sig
            or peaks.get(best_m, 0.0) <= sig):
        return SeasonalPeriodResult(1, best_score, "acf", scored)
    return SeasonalPeriodResult(best_m, best_score, "acf", scored)


# ===========================================================================
# 14. PSI/PI WEIGHTS + FORECAST-ERROR VARIANCE + INTERVALS
# ===========================================================================
def psi_weights(*, phi: Sequence[float] = (), theta: Sequence[float] = (),
                seasonal_phi: Sequence[float] = (),
                seasonal_theta: Sequence[float] = (), d: int = 0, D: int = 0,
                m: int = 1, n: int = 100) -> List[float]:
    """MA(inf) weights psi_0..psi_{n-1} of the (possibly integrated, seasonal)
    process.  The AR polynomial INCLUDES the differencing factors so forecast
    variance for integrated series grows correctly."""
    ar = _expand_lag_polynomial(phi, m=m, d=d, D=D, seasonal=seasonal_phi, kind="ar")
    ma = _expand_lag_polynomial(theta, m=m, seasonal=seasonal_theta, kind="ma")
    psi = [0.0] * n
    psi[0] = 1.0
    for j in range(1, n):
        val = ma[j] if j < len(ma) else 0.0
        val -= _kahan_sum(ar[i] * psi[j - i]
                          for i in range(1, min(len(ar), j + 1)))
        psi[j] = val
    return psi


def pi_weights(*, phi: Sequence[float] = (), theta: Sequence[float] = (),
               seasonal_phi: Sequence[float] = (),
               seasonal_theta: Sequence[float] = (), m: int = 1, n: int = 100
               ) -> List[float]:
    """AR(inf) (inverse) weights pi_0..pi_{n-1} (requires invertibility)."""
    ar = _expand_lag_polynomial(phi, m=m, seasonal=seasonal_phi, kind="ar")
    ma = _expand_lag_polynomial(theta, m=m, seasonal=seasonal_theta, kind="ma")
    pi = [0.0] * n
    pi[0] = 1.0
    for j in range(1, n):
        val = ar[j] if j < len(ar) else 0.0
        val -= _kahan_sum(ma[i] * pi[j - i]
                          for i in range(1, min(len(ma), j + 1)))
        pi[j] = val
    return pi


def forecast_error_variance(psi: Sequence[float], *, sigma2: float, horizon: int
                            ) -> List[float]:
    """Per-horizon forecast-error variance sigma2 * sum_{j<h} psi_j^2."""
    sigma2 = max(sigma2, _EPS)
    out = []
    run = 0.0
    for h in range(horizon):
        pj = psi[h] if h < len(psi) else (psi[-1] if psi else 0.0)
        run += pj * pj
        out.append(sigma2 * run)
    return out


def _check_level(level: float) -> float:
    """Validate a prediction-interval confidence level.  It must be a FRACTION in
    (0, 1): passing a percentage (``95``) or an out-of-range value would otherwise
    silently yield infinite- or negative-width intervals via ``norm_ppf`` of an
    argument outside (0, 1)."""
    if not (isinstance(level, (int, float)) and 0.0 < level < 1.0):
        raise ValueError("confidence level must be a fraction in (0, 1), got %r"
                         % (level,))
    return level


def _level_z(level: float) -> float:
    """Two-sided normal quantile for a (validated) confidence level."""
    return norm_ppf(0.5 + _check_level(level) / 2.0)


def analytic_forecast_intervals(mean: Sequence[float], *, psi: Sequence[float],
                                sigma2: float, model: str = "ARIMA",
                                levels: Sequence[float] = (0.80, 0.95)
                                ) -> ForecastResult:
    """Assemble a :class:`ForecastResult` with Gaussian analytic bands from the
    point forecast, psi-weights and innovation variance."""
    h = len(mean)
    var = forecast_error_variance(psi, sigma2=sigma2, horizon=h)
    sigma = [math.sqrt(max(v, 0.0)) for v in var]
    lower: Dict[float, List[float]] = {}
    upper: Dict[float, List[float]] = {}
    for lv in levels:
        z = _level_z(lv)
        lower[lv] = [mean[i] - z * sigma[i] for i in range(h)]
        upper[lv] = [mean[i] + z * sigma[i] for i in range(h)]
    return ForecastResult(model=model, h=h, mean=list(mean), lower=lower,
                          upper=upper, level=list(levels), sigma=sigma,
                          method="analytic")


def bootstrap_forecast_intervals(paths: Sequence[Sequence[float]], *,
                                 mean: Optional[Sequence[float]] = None,
                                 model: str = "ARIMA",
                                 levels: Sequence[float] = (0.80, 0.95)
                                 ) -> ForecastResult:
    """Empirical prediction intervals from simulated forecast paths (correctly
    asymmetric after nonlinear transforms)."""
    h = len(paths[0])
    cols = [[p[i] for p in paths] for i in range(h)]
    pt = list(mean) if mean is not None else [_median(c) for c in cols]
    lower: Dict[float, List[float]] = {}
    upper: Dict[float, List[float]] = {}
    for lv in levels:
        a = (1.0 - _check_level(lv)) / 2.0
        lower[lv] = [_quantile(cols[i], a) for i in range(h)]
        upper[lv] = [_quantile(cols[i], 1.0 - a) for i in range(h)]
    sigma = [_std(c) for c in cols]
    return ForecastResult(model=model, h=h, mean=pt, lower=lower, upper=upper,
                          level=list(levels), sigma=sigma, method="simulation")


# ===========================================================================
# 15. STATE SPACE: KALMAN FILTER / SMOOTHER + ARIMA STATE-SPACE FORM
# ===========================================================================
@dataclass
class StateSpaceModel:
    """A univariate-observation linear-Gaussian state-space model:

        y_t = Z alpha_t + eps_t,     eps_t ~ N(0, H)
        alpha_{t+1} = T alpha_t + R eta_t,   eta_t ~ N(0, Q)

    with ``Z`` a length-r row, ``T``/``Q`` r x r, ``R`` r x g, and scalar ``H``."""
    T: Matrix
    Z: Vector
    Q: Matrix
    H: float = 0.0
    R: Optional[Matrix] = None
    a1: Optional[Vector] = None
    P1: Optional[Matrix] = None

    def __post_init__(self) -> None:
        r = len(self.T)
        if self.R is None:
            self.R = eye(r)
        if self.a1 is None:
            self.a1 = [0.0] * r
        if self.P1 is None:
            self.P1 = self.stationary_cov()

    def rqr(self) -> Matrix:
        """R Q R'."""
        return matmul(matmul(self.R, self.Q), transpose(self.R))

    def stationary_cov(self) -> Matrix:
        """Stationary state covariance (diffuse fallback for unit-root states)."""
        r = len(self.T)
        if spectral_radius(self.T) < 1.0 - 1e-8:
            return solve_discrete_lyapunov(self.T, self.rqr())
        P = mat_scale(eye(r), 1e6)                      # approximate diffuse prior
        return P


@dataclass
class KalmanFilterOutput:
    filtered_state: List[Vector]
    predicted_state: List[Vector]
    forecast_error: List[float]
    forecast_error_var: List[float]
    loglike: float
    scale: float
    nobs: int


def kalman_filter(model: StateSpaceModel, y: Sequence[float], *,
                  concentrate_scale: bool = False) -> KalmanFilterOutput:
    """Run the Kalman filter, returning filtered states and the Gaussian
    log-likelihood.  With ``concentrate_scale`` the innovation scale is
    concentrated out (``Q`` and ``H`` are treated as sigma2-relative)."""
    T, Z, H = model.T, model.Z, model.H
    RQR = model.rqr()
    r = len(T)
    a = list(model.a1)
    P = [row[:] for row in model.P1]
    n = len(y)
    filt, pred = [], []
    v_list, F_list = [], []
    loglike = 0.0
    sum_v2F = 0.0
    n_used = 0
    for t in range(n):
        yt = y[t]
        pred.append(list(a))
        if yt is None or (isinstance(yt, float) and math.isnan(yt)):
            # missing: prediction step only
            a = matvec(T, a)
            P = mat_add(matmul(matmul(T, P), transpose(T)), RQR)
            filt.append(list(a))
            v_list.append(0.0)
            F_list.append(_INF)
            continue
        Pa = [_kahan_sum(P[i][j] * Z[j] for j in range(r)) for i in range(r)]
        F = _kahan_sum(Z[i] * Pa[i] for i in range(r)) + H
        F = max(F, _EPS)
        v = yt - _kahan_sum(Z[i] * a[i] for i in range(r))
        v_list.append(v)
        F_list.append(F)
        # filtered state a_{t|t} = a + P Z' v / F
        af = [a[i] + Pa[i] * v / F for i in range(r)]
        filt.append(af)
        # gain K = T P Z' / F ; a_{t+1} = T a + K v
        TPa = matvec(T, Pa)
        K = [TPa[i] / F for i in range(r)]
        a_new = [_kahan_sum(T[i][j] * a[j] for j in range(r)) + K[i] * v
                 for i in range(r)]
        # P_{t+1} = T P T' + RQR - K F K'
        TPT = matmul(matmul(T, P), transpose(T))
        P = [[TPT[i][j] + RQR[i][j] - K[i] * F * K[j] for j in range(r)]
             for i in range(r)]
        a = a_new
        loglike += -0.5 * (_LOG2PI + math.log(F) + v * v / F)
        sum_v2F += v * v / F
        n_used += 1
    if concentrate_scale and n_used > 0:
        scale = max(sum_v2F / n_used, _EPS)
        # concentrated log-likelihood
        logF = _kahan_sum(math.log(F_list[t]) for t in range(n)
                          if math.isfinite(F_list[t]))
        loglike = -0.5 * (n_used * (_LOG2PI + 1.0) + logF + n_used * math.log(scale))
    else:
        scale = 1.0
    return KalmanFilterOutput(filt, pred, v_list, F_list, loglike, scale, n_used)


def arima_ssm(*, phi: Sequence[float] = (), theta: Sequence[float] = (),
              sigma2: float = 1.0) -> StateSpaceModel:
    """State-space (Hamilton companion) form of a stationary ARMA(p, q) process."""
    p, q = len(phi), len(theta)
    r = max(p, q + 1)
    phi_f = list(phi) + [0.0] * (r - p)
    theta_f = [1.0] + list(theta) + [0.0] * (r - q - 1)
    T = zeros(r, r)
    for i in range(r):
        T[i][0] = phi_f[i]
    for i in range(r - 1):
        T[i][i + 1] = 1.0
    R = [[theta_f[i]] for i in range(r)]
    Z = [1.0] + [0.0] * (r - 1)
    Q = [[sigma2]]
    return StateSpaceModel(T=T, Z=Z, Q=Q, H=0.0, R=R)


def _exact_gaussian_llf(phi_full: Sequence[float], theta_full: Sequence[float],
                        xc: Sequence[float], *, fallback: Tuple[float, int]
                        ) -> Tuple[float, int]:
    """Exact full-sample Gaussian log-likelihood of a zero-mean ARMA series via the
    Kalman filter (innovation scale concentrated out), returning ``(loglike, nobs)``
    with ``nobs = len(xc)`` for *every* order.  The conditional CSS likelihood is
    evaluated on only ``len(xc) - max(p, q)`` observations, so its value -- and any
    information criterion derived from it -- is not comparable across orders.  Falls
    back to the supplied conditional ``(loglike, nobs)`` when the companion form is
    non-stationary or the filter is numerically degenerate."""
    try:
        ss = arima_ssm(phi=phi_full, theta=theta_full, sigma2=1.0)
        kf = kalman_filter(ss, xc, concentrate_scale=True)
        if math.isfinite(kf.loglike) and kf.nobs > 0:
            return kf.loglike, kf.nobs
    except (ValueError, ZeroDivisionError, OverflowError):
        pass
    return fallback


def kalman_smoother(model: StateSpaceModel, y: Sequence[float]
                    ) -> List[Vector]:
    """RTS smoother: smoothed state means (a light companion to the filter)."""
    filt = kalman_filter(model, y)
    T = model.T
    r = len(T)
    n = len(y)
    # re-run storing predicted covariances
    a = list(model.a1)
    P = [row[:] for row in model.P1]
    RQR = model.rqr()
    pred_a, pred_P, filt_a, filt_P = [], [], [], []
    Z, H = model.Z, model.H
    for t in range(n):
        pred_a.append(list(a)); pred_P.append([row[:] for row in P])
        yt = y[t]
        if yt is None or (isinstance(yt, float) and math.isnan(yt)):
            a = matvec(T, a); P = mat_add(matmul(matmul(T, P), transpose(T)), RQR)
            filt_a.append(list(a)); filt_P.append([row[:] for row in P])
            continue
        Pa = [_kahan_sum(P[i][j] * Z[j] for j in range(r)) for i in range(r)]
        F = max(_kahan_sum(Z[i] * Pa[i] for i in range(r)) + H, _EPS)
        v = yt - _kahan_sum(Z[i] * a[i] for i in range(r))
        af = [a[i] + Pa[i] * v / F for i in range(r)]
        Pf = [[P[i][j] - Pa[i] * Pa[j] / F for j in range(r)] for i in range(r)]
        filt_a.append(af); filt_P.append(Pf)
        a = matvec(T, af)
        P = mat_add(matmul(matmul(T, Pf), transpose(T)), RQR)
    smoothed = [list(v) for v in filt_a]
    for t in range(n - 2, -1, -1):
        Pp = pred_P[t + 1]
        Pp_inv = inv(Pp, ridge=1e-12)
        J = matmul(matmul(filt_P[t], transpose(T)), Pp_inv)
        diff = [smoothed[t + 1][i] - pred_a[t + 1][i] for i in range(r)]
        smoothed[t] = [filt_a[t][i] + _kahan_sum(J[i][j] * diff[j]
                       for j in range(r)) for i in range(r)]
    return smoothed


def simulate_ssm(model: StateSpaceModel, n: int, *, seed: int = 0) -> List[float]:
    """Simulate an observation series from a state-space model."""
    rng = DeterministicRNG(seed)
    T, Z, H = model.T, model.Z, model.H
    r = len(T)
    Lq = cholesky(model.Q, jitter=1e-12) or eye(len(model.Q))
    a = list(model.a1)
    out = []
    for _ in range(n):
        y = _kahan_sum(Z[i] * a[i] for i in range(r))
        if H > 0.0:
            y += math.sqrt(H) * rng.gauss()
        out.append(y)
        eta = matvec(Lq, [rng.gauss() for _ in range(len(model.Q))])
        Reta = matvec(model.R, eta)
        a = [_kahan_sum(T[i][j] * a[j] for j in range(r)) + Reta[i]
             for i in range(r)]
    return out


# ===========================================================================
# 16. CLASSICAL ARIMA ESTIMATORS
# ===========================================================================
@dataclass
class ARFitResult:
    phi: List[float]
    sigma2: float
    mu: float
    order: int
    method: str
    theta: List[float] = field(default_factory=list)
    seasonal_phi: List[float] = field(default_factory=list)
    seasonal_theta: List[float] = field(default_factory=list)


@dataclass
class ARMAFitResult:
    phi: List[float]
    theta: List[float]
    sigma2: float
    mu: float
    method: str
    seasonal_phi: List[float] = field(default_factory=list)
    seasonal_theta: List[float] = field(default_factory=list)
    loglik: float = 0.0
    converged: bool = True


def yule_walker(y: Sequence[float], *, order: int, adjusted: bool = False
                ) -> ARFitResult:
    """Fit an AR(p) model by the Yule-Walker equations (Levinson-Durbin)."""
    y = _clean(y)
    mu = _mean(y)
    gam = autocovariance(y, nlags=order, adjusted=adjusted)
    phi, _, v = levinson_durbin(gam, order=order)
    sigma2 = max(v[order], _EPS)
    return ARFitResult(phi, sigma2, mu, order, "yule-walker")


def burg(y: Sequence[float], *, order: int, demean: bool = True) -> ARFitResult:
    """Fit an AR(p) model by Burg's method (minimises forward+backward prediction
    error; better on short series than Yule-Walker).  Uses the +phi convention
    x_t = sum phi_i x_{t-i} + e_t."""
    y = _clean(y)
    mu = _mean(y) if demean else 0.0
    x = [v - mu for v in y]
    n = len(x)
    f = x[:]                                            # forward prediction errors
    b = x[:]                                            # backward prediction errors
    a: List[float] = []                                 # AR coefficients (+phi)
    for m in range(1, order + 1):
        num = _kahan_sum(f[t] * b[t - 1] for t in range(m, n))
        den = _kahan_sum(f[t] * f[t] for t in range(m, n)) + \
            _kahan_sum(b[t - 1] * b[t - 1] for t in range(m, n))
        k = 2.0 * num / den if den > _EPS else 0.0
        new_a = a[:] + [k]
        for i in range(m - 1):
            new_a[i] = a[i] - k * a[m - 2 - i]
        a = new_a
        fnew = f[:]
        bnew = b[:]
        for t in range(m, n):
            fnew[t] = f[t] - k * b[t - 1]
            bnew[t] = b[t - 1] - k * f[t]
        f, b = fnew, bnew
    phi = list(a)
    res = _arma_residuals_css(x, [1.0] + [-p for p in phi], [1.0])[0]
    sigma2 = max(_kahan_sum(r * r for r in res) / max(len(res), 1), _EPS)
    return ARFitResult(phi, sigma2, mu, order, "burg")


def _arma_residuals_css(x: Sequence[float], ar_poly: Sequence[float],
                        ma_poly: Sequence[float]) -> Tuple[List[float], int]:
    """Conditional ARMA residuals: given the FULL (expanded, leading-1) AR and MA
    lag polynomials, e_t = sum_i ar[i] x_{t-i} - sum_{j>=1} ma[j] e_{t-j}, with
    pre-sample residuals set to zero.  Returns (residuals, start index)."""
    n = len(x)
    p = len(ar_poly) - 1
    q = len(ma_poly) - 1
    start = p
    e = [0.0] * n
    for t in range(start, n):
        ar_part = _kahan_sum(ar_poly[i] * x[t - i]
                             for i in range(min(p + 1, t + 1)))
        ma_part = _kahan_sum(ma_poly[j] * e[t - j]
                             for j in range(1, min(q + 1, t + 1)))
        e[t] = ar_part - ma_part
    return e[start:], start


def hannan_rissanen(y: Sequence[float], *, order: Tuple[int, int],
                    seasonal_order: Tuple[int, int] = (0, 0), m: int = 1
                    ) -> ARMAFitResult:
    """Hannan-Rissanen two-stage estimation of an ARMA(p, q): fit a long AR to
    approximate the innovations, then regress on lagged values and innovations."""
    y = _clean(y)
    p, q = order
    P, Q = seasonal_order
    mu = _mean(y)
    x = [v - mu for v in y]
    n = len(x)
    long_p = min(max(int(math.sqrt(n)), (p + q + P * m + Q * m) + 5), n // 2)
    ar = yule_walker(x, order=long_p)
    innov = _arma_residuals_css(x, [1.0] + [-c for c in ar.phi], [1.0])[0]
    innov = [0.0] * long_p + innov                     # align back to x length
    lags_ar = list(range(1, p + 1)) + [k * m for k in range(1, P + 1)]
    lags_ma = list(range(1, q + 1)) + [k * m for k in range(1, Q + 1)]
    start = max(lags_ar + lags_ma + [1])
    rows, target = [], []
    for t in range(start, n):
        row = [x[t - l] for l in lags_ar] + [innov[t - l] for l in lags_ma]
        rows.append(row)
        target.append(x[t])
    coef, _ = qr_lstsq(rows, target, ridge=1e-8)
    phi = coef[:p]
    seasonal_phi = coef[p:p + P]
    theta = coef[p + P:p + P + q]
    seasonal_theta = coef[p + P + q:p + P + q + Q]
    res = _arma_residuals_css(
        x, _expand_lag_polynomial(phi, m=m, seasonal=seasonal_phi, kind="ar"),
        _expand_lag_polynomial(theta, m=m, seasonal=seasonal_theta, kind="ma"))[0]
    sigma2 = max(_kahan_sum(r * r for r in res) / max(len(res), 1), _EPS)
    return ARMAFitResult(list(phi), list(theta), sigma2, mu, "hannan-rissanen",
                         list(seasonal_phi), list(seasonal_theta))


def css_estimate(y: Sequence[float], *, order: Tuple[int, int, int],
                 seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
                 include_mean: bool = True, start_params: Optional[Sequence[float]] = None,
                 enforce_stationarity: bool = True,
                 enforce_invertibility: bool = True, maxiter: int = 1000,
                 tol: float = 1e-8, seed: int = 0) -> ARMAFitResult:
    """Conditional-sum-of-squares estimation of a (seasonal) ARMA on an ALREADY
    DIFFERENCED series.  Parameters are kept stationary/invertible during the
    optimisation via the reflection-coefficient (``pacf_transform``) map."""
    x = _clean(y)
    p, d, q = order
    P, D, Q, m = seasonal_order
    mu0 = _mean(x) if include_mean else 0.0

    def unpack(vec: Sequence[float]):
        i = 0
        rphi = vec[i:i + p]; i += p
        rtheta = vec[i:i + q]; i += q
        rsphi = vec[i:i + P]; i += P
        rstheta = vec[i:i + Q]; i += Q
        mu = vec[i] if include_mean else 0.0
        phi = pacf_transform(rphi) if enforce_stationarity else list(rphi)
        theta = pacf_transform(rtheta) if enforce_invertibility else list(rtheta)
        sphi = pacf_transform(rsphi) if enforce_stationarity else list(rsphi)
        stheta = pacf_transform(rstheta) if enforce_invertibility else list(rstheta)
        return phi, theta, sphi, stheta, mu

    def ssr(vec: Sequence[float]) -> float:
        phi, theta, sphi, stheta, mu = unpack(vec)
        xc = [v - mu for v in x]
        ar = _expand_lag_polynomial(phi, m=m, seasonal=sphi, kind="ar")
        ma = _expand_lag_polynomial(theta, m=m, seasonal=stheta, kind="ma")
        e, _ = _arma_residuals_css(xc, ar, ma)
        if not e:
            return _INF
        return _kahan_sum(r * r for r in e)

    nfree = p + q + P + Q + (1 if include_mean else 0)
    if start_params is not None and len(start_params) == nfree:
        x0 = list(start_params)
    else:
        # warm start: Yule-Walker for the AR block (mapped to raw reflection
        # coordinates) so a single simplex restart converges quickly and well
        x0 = [0.0] * (p + q + P + Q)
        if p >= 1 and len(x) > p + 2:
            try:
                phi_yw = yule_walker([v - mu0 for v in x], order=p).phi
                if enforce_stationarity:
                    raw = pacf_untransform(phi_yw)
                else:
                    raw = phi_yw
                for i in range(p):
                    x0[i] = max(min(raw[i], 5.0), -5.0)
            except (ValueError, ZeroDivisionError, OverflowError):
                pass
        if include_mean:
            x0.append(mu0)
    if nfree == 0:
        vec: List[float] = []
    else:
        restarts = 2 if (q + Q) else 1                  # MA needs a little more care
        res = minimize_nelder_mead(ssr, x0, restarts=restarts, seed=seed,
                                   maxiter=maxiter, xatol=tol, fatol=tol)
        vec = res.x
    phi, theta, sphi, stheta, mu = unpack(vec) if nfree else ([], [], [], [], mu0)
    xc = [v - mu for v in x]
    ar = _expand_lag_polynomial(phi, m=m, seasonal=sphi, kind="ar")
    ma = _expand_lag_polynomial(theta, m=m, seasonal=stheta, kind="ma")
    e, start = _arma_residuals_css(xc, ar, ma)
    nobs = len(e)
    sigma2 = max(_kahan_sum(r * r for r in e) / max(nobs, 1), _EPS)
    ll = gaussian_loglike(e)
    return ARMAFitResult(list(phi), list(theta), sigma2, mu, "css",
                         list(sphi), list(stheta), ll, True)


# ===========================================================================
# 17. ARIMA / SARIMA MODEL + RESULTS
# ===========================================================================
def _arma_forecast_diff(x_hist: Sequence[float], e_hist: Sequence[float],
                        ar: Sequence[float], ma: Sequence[float], h: int
                        ) -> List[float]:
    """Recursively forecast the (mean-removed, differenced) ARMA series ``h`` steps.
    ``ar``/``ma`` are the expanded leading-1 lag polynomials."""
    p = len(ar) - 1
    q = len(ma) - 1
    x = list(x_hist)
    e = list(e_hist)
    n = len(x)
    fc = []
    for k in range(1, h + 1):
        t = n + k - 1
        val = -_kahan_sum(ar[i] * x[t - i] for i in range(1, min(p + 1, t + 1)))
        val += _kahan_sum(ma[j] * (e[t - j] if t - j < len(e) else 0.0)
                          for j in range(1, min(q + 1, t + 1)) if t - j < len(e))
        x.append(val)
        e.append(0.0)
        fc.append(val)
    return fc


@dataclass
class ARIMAResults:
    """A fitted ARIMA / SARIMA / SARIMAX model."""
    model: "ARIMA"
    phi: List[float]
    theta: List[float]
    seasonal_phi: List[float]
    seasonal_theta: List[float]
    mu: float
    sigma2: float
    llf: float
    aic: float
    bic: float
    aicc: float
    hqic: float
    resid: List[float]
    fittedvalues: List[float]
    nobs: int
    k_params: int
    d: int
    D: int
    m: int
    method: str
    converged: bool = True
    bse: List[float] = field(default_factory=list)
    param_names: List[str] = field(default_factory=list)
    params: List[float] = field(default_factory=list)

    def forecast(self, h: int = 1, *, level: Sequence[float] = (0.80, 0.95),
                 method: str = "analytic", nsim: int = 1000, seed: int = 0
                 ) -> ForecastResult:
        """Forecast ``h`` steps with prediction intervals (analytic psi-weight bands
        by default, or empirical bands from simulated paths)."""
        if h <= 0:
            return ForecastResult(model="ARIMA", h=0, mean=[])
        md = self.model
        w = self._w                                     # this fit's differenced series
        x = [v - self.mu for v in w]
        ar = _expand_lag_polynomial(self.phi, m=self.m, seasonal=self.seasonal_phi,
                                    kind="ar")
        ma = _expand_lag_polynomial(self.theta, m=self.m,
                                    seasonal=self.seasonal_theta, kind="ma")
        e_full = self._e_full
        fdiff_x = _arma_forecast_diff(x, e_full, ar, ma, h)
        fdiff = [v + self.mu for v in fdiff_x]          # add mean back
        # integrate back to the original level
        mean = md._differencer.forecast_integrate(md._endog, fdiff)
        psi = psi_weights(phi=self.phi, theta=self.theta,
                          seasonal_phi=self.seasonal_phi,
                          seasonal_theta=self.seasonal_theta, d=self.d, D=self.D,
                          m=self.m, n=h + 1)
        if method == "simulation":
            paths = md._simulate_paths(self, h, n_paths=nsim, seed=seed)
            fr = bootstrap_forecast_intervals(paths, mean=mean, model=md.name,
                                              levels=level)
        else:
            fr = analytic_forecast_intervals(mean, psi=psi, sigma2=self.sigma2,
                                             model=md.name, levels=level)
        fr.index = list(range(self.nobs_orig(), self.nobs_orig() + h))
        fr.extra["psi"] = psi[:h]
        return fr

    def nobs_orig(self) -> int:
        return len(self.model._endog)

    def predict(self) -> List[float]:
        """In-sample one-step-ahead predictions (fitted values)."""
        return list(self.fittedvalues)

    def simulate(self, nsimulations: int, *, seed: int = 0) -> List[float]:
        """Simulate a future path continuing the fitted process."""
        paths = self.model._simulate_paths(self, nsimulations, n_paths=1,
                                           seed=seed)
        return paths[0]

    def summary(self, *, digits: int = 4) -> str:
        md = self.model
        lines = ["%s  method=%s  nobs=%d" % (md.name, self.method, self.nobs)]
        lines.append("  order=(%d,%d,%d)  seasonal=(%d,%d,%d,%d)" % (
            len(self.phi), self.d, len(self.theta),
            len(self.seasonal_phi), self.D, len(self.seasonal_theta), self.m))
        lines.append("  loglik=%.*f  aic=%.*f  bic=%.*f  aicc=%.*f" % (
            digits, self.llf, digits, self.aic, digits, self.bic,
            digits, self.aicc))
        lines.append("  sigma2=%.*f" % (digits, self.sigma2))
        for i, nm in enumerate(self.param_names):
            se = self.bse[i] if i < len(self.bse) and self.bse else float("nan")
            lines.append("    %-14s %*.*f  (se %.*f)" % (
                nm, digits + 6, digits, self.params[i], digits, se))
        return "\n".join(lines)


class ARIMA:
    """Box-Jenkins ARIMA / seasonal ARIMA model with several estimators.

    ``order=(p, d, q)`` and ``seasonal_order=(P, D, Q, m)``.  ``trend`` selects the
    deterministic term: ``'n'`` none, ``'c'`` constant/drift (admissible when
    ``d+D <= 1``).  Estimation methods: ``'yule-walker'`` and ``'burg'`` (pure AR),
    ``'hr'`` (Hannan-Rissanen), ``'css'`` (conditional sum of squares), ``'mle'``
    (exact likelihood via the Kalman filter) and ``'css-mle'`` (CSS start refined
    by exact MLE, the default)."""

    def __init__(self, endog: Sequence[float], *, order: Tuple[int, int, int] = (1, 0, 0),
                 seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
                 trend: str = "c", boxcox_lambda: Optional[float] = None,
                 name: str = "ARIMA") -> None:
        self._endog = _clean(endog)
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.boxcox_lambda = boxcox_lambda
        self.name = name
        p, d, q = order
        P, D, Q, m = seasonal_order
        need = p + q + d + D * m + P * m + Q * m + 2
        if len(self._endog) < need:
            raise ValueError("series too short (%d) for order %s x %s"
                             % (len(self._endog), order, seasonal_order))
        self._differencer = Differencer(d=d, D=D, m=m)
        self._w: List[float] = []
        self._e_full: List[float] = []

    def fit(self, *, method: str = "css-mle", enforce_stationarity: bool = True,
            enforce_invertibility: bool = True, maxiter: int = 1000,
            tol: float = 1e-8, seed: int = 0) -> ARIMAResults:
        p, d, q = self.order
        P, D, Q, m = self.seasonal_order
        y = self._endog
        if self.boxcox_lambda is not None:
            y = box_cox(y, lam=self.boxcox_lambda)
        w = self._differencer.fit_transform(y)
        self._w = w
        include_mean = (self.trend != "n") and (d + D <= 1)

        if method in ("yule-walker", "burg") and q == 0 and P == 0 and Q == 0:
            fit = (yule_walker(w, order=p) if method == "yule-walker"
                   else burg(w, order=p))
            phi, theta = fit.phi, []
            sphi, stheta, mu = [], [], (fit.mu if include_mean else 0.0)
        elif method == "hr":
            fit = hannan_rissanen(w, order=(p, q), seasonal_order=(P, Q), m=m)
            phi, theta = fit.phi, fit.theta
            sphi, stheta = fit.seasonal_phi, fit.seasonal_theta
            mu = fit.mu if include_mean else 0.0
        else:                                           # css / mle / css-mle
            fit = css_estimate(w, order=(p, d, q),
                               seasonal_order=(P, D, Q, m),
                               include_mean=include_mean,
                               enforce_stationarity=enforce_stationarity,
                               enforce_invertibility=enforce_invertibility,
                               maxiter=maxiter, tol=tol, seed=seed)
            phi, theta = fit.phi, fit.theta
            sphi, stheta, mu = fit.seasonal_phi, fit.seasonal_theta, fit.mu
            if method in ("mle", "css-mle"):
                phi, theta, sphi, stheta, mu = self._refine_mle(
                    w, phi, theta, sphi, stheta, mu, include_mean,
                    enforce_stationarity, enforce_invertibility, seed)

        # residuals / fitted values on the original scale
        xc = [v - mu for v in w]
        ar = _expand_lag_polynomial(phi, m=m, seasonal=sphi, kind="ar")
        ma = _expand_lag_polynomial(theta, m=m, seasonal=stheta, kind="ma")
        e, start = _arma_residuals_css(xc, ar, ma)
        e_full = [0.0] * start + e
        self._e_full = e_full
        n_resid = len(e)
        sigma2 = max(_kahan_sum(r * r for r in e) / max(n_resid, 1), _EPS)
        off = d + D * m
        fitted = [y[off + start + k] - e[k] for k in range(n_resid)]

        k_params = count_params(p=p, q=q, P=P, Q=Q, d=d, D=D, m=m,
                                trend=self.trend, include_mean=include_mean)
        # Information criteria on the EXACT full-sample (Kalman) log-likelihood.
        # The conditional CSS likelihood is evaluated on only len(w) - order
        # observations, so its value -- and every IC derived from it -- is not
        # comparable across orders: dropping the first max(p, q) points
        # systematically inflates the log-likelihood of higher-order models, which
        # made auto_arima and the AICc leaderboard drift to the largest order.
        # Score every candidate on the SAME full differenced sample instead.
        llf, nobs = _exact_gaussian_llf([-c for c in ar[1:]], list(ma[1:]), xc,
                                        fallback=(gaussian_loglike(e), n_resid))
        ic = information_criteria(loglik=llf, n_params=k_params, nobs=nobs)
        names = (["ar.L%d" % i for i in range(1, p + 1)]
                 + ["ma.L%d" % j for j in range(1, q + 1)]
                 + ["ar.S.L%d" % (k * m) for k in range(1, P + 1)]
                 + ["ma.S.L%d" % (k * m) for k in range(1, Q + 1)]
                 + (["const"] if include_mean else []))
        params = list(phi) + list(theta) + list(sphi) + list(stheta) \
            + ([mu] if include_mean else [])
        bse = self._standard_errors(w, phi, theta, sphi, stheta, mu,
                                    include_mean, m) if nobs > k_params else []
        res = ARIMAResults(self, list(phi), list(theta), list(sphi),
                           list(stheta), mu, sigma2, llf, ic.aic, ic.bic,
                           ic.aicc, ic.hqic, list(e), fitted, nobs, k_params,
                           d, D, m, method, fit.converged if hasattr(fit, "converged")
                           else True, bse, names, params)
        # snapshot the fit-specific state onto the result so it is an IMMUTABLE,
        # self-contained view: forecasting must use THIS fit's differenced series and
        # residuals, not whatever the (mutable) model instance holds after a later
        # refit with a different method/params.
        res._w = list(w)
        res._e_full = list(e_full)
        return res

    def _neg_loglike_mle(self, w, include_mean, es, ei, p, q, P, Q, m):
        def nll(vec):
            i = 0
            rphi = vec[i:i + p]; i += p
            rtheta = vec[i:i + q]; i += q
            rsphi = vec[i:i + P]; i += P
            rstheta = vec[i:i + Q]; i += Q
            mu = vec[i] if include_mean else 0.0
            phi = pacf_transform(rphi) if es else list(rphi)
            theta = pacf_transform(rtheta) if ei else list(rtheta)
            sphi = pacf_transform(rsphi) if es else list(rsphi)
            stheta = pacf_transform(rstheta) if ei else list(rstheta)
            xc = [v - mu for v in w]
            # combine seasonal into single long ARMA for the state-space form
            ar = _expand_lag_polynomial(phi, m=m, seasonal=sphi, kind="ar")
            ma = _expand_lag_polynomial(theta, m=m, seasonal=stheta, kind="ma")
            long_phi = [-a for a in ar[1:]]
            long_theta = list(ma[1:])
            try:
                ssm = arima_ssm(phi=long_phi, theta=long_theta, sigma2=1.0)
                kf = kalman_filter(ssm, xc, concentrate_scale=True)
            except (ValueError, OverflowError):
                return _INF
            return -kf.loglike
        return nll

    def _refine_mle(self, w, phi, theta, sphi, stheta, mu, include_mean, es, ei,
                    seed):
        p, d, q = self.order
        P, D, Q, m = self.seasonal_order
        x0 = (list(pacf_untransform(phi) if es and phi else phi)
              + list(pacf_untransform(theta) if ei and theta else theta)
              + list(pacf_untransform(sphi) if es and sphi else sphi)
              + list(pacf_untransform(stheta) if ei and stheta else stheta)
              + ([mu] if include_mean else []))
        nll = self._neg_loglike_mle(w, include_mean, es, ei, p, q, P, Q, m)
        res = minimize_nelder_mead(nll, x0, restarts=1, seed=seed, maxiter=800)
        vec = res.x
        i = 0
        rphi = vec[i:i + p]; i += p
        rtheta = vec[i:i + q]; i += q
        rsphi = vec[i:i + P]; i += P
        rstheta = vec[i:i + Q]; i += Q
        mu2 = vec[i] if include_mean else 0.0
        return ((pacf_transform(rphi) if es else list(rphi)),
                (pacf_transform(rtheta) if ei else list(rtheta)),
                (pacf_transform(rsphi) if es else list(rsphi)),
                (pacf_transform(rstheta) if ei else list(rstheta)), mu2)

    def _standard_errors(self, w, phi, theta, sphi, stheta, mu, include_mean, m):
        p, d, q = self.order
        P, D, Q, mm = self.seasonal_order
        params = list(phi) + list(theta) + list(sphi) + list(stheta) \
            + ([mu] if include_mean else [])

        def nll(vec):
            i = 0
            ph = vec[i:i + p]; i += p
            th = vec[i:i + q]; i += q
            sph = vec[i:i + P]; i += P
            sth = vec[i:i + Q]; i += Q
            mu_ = vec[i] if include_mean else 0.0
            xc = [v - mu_ for v in w]
            ar = _expand_lag_polynomial(ph, m=m, seasonal=sph, kind="ar")
            ma = _expand_lag_polynomial(th, m=m, seasonal=sth, kind="ma")
            e, _ = _arma_residuals_css(xc, ar, ma)
            if not e:
                return _INF
            return -gaussian_loglike(e)
        try:
            H = _hessian_fd(nll, params, eps=1e-4)
            cov = inv(H, ridge=1e-8)
            return [math.sqrt(abs(cov[i][i])) for i in range(len(params))]
        except (ValueError, ZeroDivisionError, OverflowError):
            return []

    def _simulate_paths(self, res: ARIMAResults, h: int, *, n_paths: int,
                        seed: int) -> List[List[float]]:
        """Simulate ``n_paths`` future paths by bootstrapping residuals through the
        ARMA recursion and integrating back to the original scale."""
        rng = DeterministicRNG(seed)
        m = self.m if hasattr(self, "m") else res.m
        ar = _expand_lag_polynomial(res.phi, m=res.m, seasonal=res.seasonal_phi,
                                    kind="ar")
        ma = _expand_lag_polynomial(res.theta, m=res.m,
                                    seasonal=res.seasonal_theta, kind="ma")
        x_hist = [v - res.mu for v in res._w]
        e_hist = res._e_full
        resid_pool = [r for r in res.resid] or [0.0]
        paths = []
        for _ in range(n_paths):
            x = list(x_hist)
            e = list(e_hist)
            fdiff = []
            n = len(x)
            for k in range(1, h + 1):
                t = n + k - 1
                shock = resid_pool[rng.choice_index(len(resid_pool))]
                val = -_kahan_sum(ar[i] * x[t - i]
                                  for i in range(1, min(len(ar), t + 1)))
                val += _kahan_sum(ma[j] * (e[t - j] if t - j < len(e) else 0.0)
                                  for j in range(1, min(len(ma), t + 1))
                                  if t - j < len(e))
                val += shock
                x.append(val)
                e.append(shock)
                fdiff.append(val + res.mu)
            level = self._differencer.forecast_integrate(self._endog, fdiff)
            paths.append(level)
        return paths

    @property
    def m(self) -> int:
        return self.seasonal_order[3]


# ===========================================================================
# 18. EXPONENTIAL SMOOTHING & THETA
# ===========================================================================
@dataclass
class ETSResult:
    """A fitted exponential-smoothing (ETS) model."""
    model: str
    params: Dict[str, float]
    fittedvalues: List[float]
    resid: List[float]
    level: List[float]
    trend: List[float]
    season: List[float]
    sigma2: float
    llf: float
    aic: float
    bic: float
    aicc: float
    nobs: int
    k_params: int
    trend_type: str
    season_type: str
    m: int
    damped: bool
    converged: bool = True

    def summary(self) -> str:
        ps = "  ".join("%s=%.4f" % (k, v) for k, v in sorted(self.params.items()))
        return ("ETS(%s)  nobs=%d\n  %s\n  aic=%.3f bic=%.3f sigma2=%.4f"
                % (self.model, self.nobs, ps, self.aic, self.bic, self.sigma2))


def _ets_recurse(y: Sequence[float], *, alpha: float, beta: float, gamma: float,
                 phi: float, l0: float, b0: float, s0: Sequence[float],
                 trend: str, season: str, m: int):
    """Run the Holt-Winters recursion; return (fitted, level, trend, season)."""
    n = len(y)
    lev = [0.0] * n
    tr = [0.0] * n
    seas = list(s0) + [0.0] * n
    fitted = [0.0] * n
    l_prev = l0
    b_prev = b0
    for t in range(n):
        s_idx = t                                        # seasonal index into seas
        if season == "add":
            base = l_prev + (phi * b_prev if trend != "n" else 0.0)
            fitted[t] = base + seas[s_idx]
            l_t = alpha * (y[t] - seas[s_idx]) + (1.0 - alpha) * base
        elif season == "mul":
            base = l_prev + (phi * b_prev if trend != "n" else 0.0)
            sfac = seas[s_idx] if abs(seas[s_idx]) > _EPS else 1.0
            fitted[t] = base * sfac
            l_t = alpha * (y[t] / sfac) + (1.0 - alpha) * base
        else:
            base = l_prev + (phi * b_prev if trend != "n" else 0.0)
            fitted[t] = base
            l_t = alpha * y[t] + (1.0 - alpha) * base
        if trend != "n":
            b_t = beta * (l_t - l_prev) + (1.0 - beta) * phi * b_prev
        else:
            b_t = 0.0
        if season == "add":
            seas[s_idx + m] = gamma * (y[t] - l_t) + (1.0 - gamma) * seas[s_idx]
        elif season == "mul":
            seas[s_idx + m] = gamma * (y[t] / (l_t if abs(l_t) > _EPS else _EPS)) \
                + (1.0 - gamma) * seas[s_idx]
        lev[t] = l_t
        tr[t] = b_t
        l_prev, b_prev = l_t, b_t
    return fitted, lev, tr, seas


class ExponentialSmoothing:
    """Holt-Winters exponential smoothing with optional trend (damped or not) and
    additive / multiplicative seasonality; smoothing parameters and initial states
    are estimated by minimising the sum of squared errors."""

    def __init__(self, endog: Sequence[float], *, trend: Optional[str] = None,
                 damped_trend: bool = False, seasonal: Optional[str] = None,
                 seasonal_periods: int = 1, name: str = "ETS") -> None:
        self.y = _clean(endog)
        self.trend = trend or "n"                        # 'n'|'add'
        self.damped = damped_trend and self.trend != "n"
        self.season = seasonal or "n"                    # 'n'|'add'|'mul'
        self.m = seasonal_periods if self.season != "n" else 1
        self.name = name
        if self.season != "n" and len(self.y) < 2 * self.m:
            raise ValueError("need two full seasons for seasonal smoothing")

    def _init_states(self):
        y = self.y
        m = self.m
        if self.season != "n":
            dec = seasonal_decompose(
                y, period=m,
                model="multiplicative" if self.season == "mul" else "additive")
            s0 = dec.seasonal_indices[:]
            deseas = dec.seasadj()
            l0 = _mean(deseas[:m])
            b0 = (_mean(deseas[m:2 * m]) - _mean(deseas[:m])) / m \
                if self.trend != "n" else 0.0
        else:
            l0 = _mean(y[:min(len(y), 3)])
            b0 = (y[1] - y[0]) if (self.trend != "n" and len(y) > 1) else 0.0
            s0 = [0.0] * m if self.season != "mul" else [1.0] * m
        return l0, b0, s0

    def fit(self, *, seed: int = 0, opt_restarts: int = 2) -> ETSResult:
        y = self.y
        m = self.m
        has_trend = self.trend != "n"
        has_season = self.season != "n"
        l0, b0, s0 = self._init_states()

        def unpack(vec):
            i = 0
            alpha = expit(vec[i]); i += 1
            beta = expit(vec[i]) if has_trend else 0.0
            i += 1 if has_trend else 0
            gamma = expit(vec[i]) if has_season else 0.0
            i += 1 if has_season else 0
            phi = 0.8 + 0.2 * expit(vec[i]) if self.damped else 1.0
            i += 1 if self.damped else 0
            lev0 = vec[i]; i += 1
            bt0 = vec[i] if has_trend else 0.0
            i += 1 if has_trend else 0
            if has_season:
                sraw = list(vec[i:i + m])
                if self.season == "mul":
                    tot = _kahan_sum(math.exp(v) for v in sraw)
                    s = [m * math.exp(v) / tot for v in sraw]
                else:
                    s = [v - _mean(sraw) for v in sraw]
            else:
                s = s0
            return alpha, beta, gamma, phi, lev0, bt0, s

        def sse(vec):
            alpha, beta, gamma, phi, lev0, bt0, s = unpack(vec)
            fitted, _, _, _ = _ets_recurse(
                y, alpha=alpha, beta=beta, gamma=gamma, phi=phi, l0=lev0, b0=bt0,
                s0=s, trend=self.trend, season=self.season, m=m)
            return _kahan_sum((y[t] - fitted[t]) ** 2 for t in range(len(y)))

        x0 = [logit(0.3)]
        if has_trend:
            x0.append(logit(0.1))
        if has_season:
            x0.append(logit(0.1))
        if self.damped:
            x0.append(0.0)
        x0.append(l0)
        if has_trend:
            x0.append(b0)
        if has_season:
            if self.season == "mul":
                x0 += [math.log(max(v, 1e-6)) for v in s0]
            else:
                x0 += list(s0)
        res = minimize_nelder_mead(sse, x0, restarts=opt_restarts, seed=seed,
                                   maxiter=1200)
        alpha, beta, gamma, phi, lev0, bt0, s = unpack(res.x)
        fitted, lev, tr, seas = _ets_recurse(
            y, alpha=alpha, beta=beta, gamma=gamma, phi=phi, l0=lev0, b0=bt0,
            s0=s, trend=self.trend, season=self.season, m=m)
        resid = [y[t] - fitted[t] for t in range(len(y))]
        n = len(y)
        sigma2 = max(_kahan_sum(r * r for r in resid) / n, _EPS)
        llf = -0.5 * n * (_LOG2PI + math.log(sigma2) + 1.0)
        k = 1 + (1 if has_trend else 0) + (1 if has_season else 0) \
            + (1 if self.damped else 0) + 1 + (1 if has_trend else 0) \
            + (m if has_season else 0) + 1
        ic = information_criteria(loglik=llf, n_params=k, nobs=n)
        self._final = (alpha, beta, gamma, phi, lev, tr, seas)
        code = "%s%s%s" % ("A", {"n": "N", "add": "Ad" if self.damped else "A"}[self.trend],
                           {"n": "N", "add": "A", "mul": "M"}[self.season])
        params = {"alpha": alpha}
        if has_trend:
            params["beta"] = beta
        if has_season:
            params["gamma"] = gamma
        if self.damped:
            params["phi"] = phi
        return ETSResult(code, params, fitted, resid, lev, tr,
                         seas[:n] if has_season else [0.0] * n, sigma2, llf,
                         ic.aic, ic.bic, ic.aicc, n, k, self.trend, self.season,
                         m, self.damped, res.converged)

    def fit_forecast(self, h: int, **kw) -> ForecastResult:
        r = self.fit()
        return self.forecast(r, h, **kw)

    def forecast(self, result: ETSResult, h: int, *,
                 level: Sequence[float] = (0.80, 0.95)) -> ForecastResult:
        alpha, beta, gamma, phi, lev, tr, seas = self._final
        n = len(self.y)
        l_last = lev[-1]
        b_last = tr[-1]
        m = self.m
        mean = []
        for k in range(1, h + 1):
            damp = _kahan_sum(phi ** j for j in range(1, k + 1)) if self.damped \
                else (k if self.trend != "n" else 0.0)
            base = l_last + (damp * b_last if self.trend != "n" else 0.0)
            if self.season == "add":
                base += seas[n + (k - 1) % m] if (n + (k - 1) % m) < len(seas) \
                    else seas[n - m + (k - 1) % m]
            elif self.season == "mul":
                sfac = seas[n - m + (k - 1) % m]
                base *= sfac if abs(sfac) > _EPS else 1.0
            mean.append(base)
        # analytic bands: approximate variance growth alpha-based
        sigma = math.sqrt(result.sigma2)
        lower: Dict[float, List[float]] = {}
        upper: Dict[float, List[float]] = {}
        for lv in level:
            z = _level_z(lv)
            band = []
            for k in range(1, h + 1):
                var_mult = 1.0 + (k - 1) * alpha * alpha
                band.append(z * sigma * math.sqrt(var_mult))
            lower[lv] = [mean[i] - band[i] for i in range(h)]
            upper[lv] = [mean[i] + band[i] for i in range(h)]
        return ForecastResult(model=self.name, h=h, mean=mean, lower=lower,
                              upper=upper, level=list(level), method="analytic")


class SimpleExpSmoothing(ExponentialSmoothing):
    """Simple exponential smoothing (no trend, no seasonality)."""

    def __init__(self, endog: Sequence[float]) -> None:
        super().__init__(endog, trend=None, seasonal=None, name="SES")


class Holt(ExponentialSmoothing):
    """Holt's linear (or damped) trend method."""

    def __init__(self, endog: Sequence[float], *, damped: bool = False) -> None:
        super().__init__(endog, trend="add", damped_trend=damped, name="Holt")


def auto_ets(y: Sequence[float], *, seasonal_periods: int = 1,
             criterion: str = "aicc", seed: int = 0):
    """Select the best exponential-smoothing model over the (A/N x N/A/M) grid by
    an information criterion."""
    y = _clean(y)
    trends = ["n", "add"]
    seasons = ["n"]
    if seasonal_periods >= 2 and len(y) >= 2 * seasonal_periods:
        seasons = ["n", "add", "mul"]
    best = None
    for tr in trends:
        for damp in ([False, True] if tr != "n" else [False]):
            for se in seasons:
                if se == "mul" and min(y) <= 0.0:
                    continue
                try:
                    mdl = ExponentialSmoothing(
                        y, trend=(None if tr == "n" else tr), damped_trend=damp,
                        seasonal=(None if se == "n" else se),
                        seasonal_periods=seasonal_periods)
                    r = mdl.fit(seed=seed)
                except (ValueError, ZeroDivisionError, OverflowError):
                    continue
                ic = {"aic": r.aic, "bic": r.bic, "aicc": r.aicc}[criterion]
                if best is None or ic < best[0]:
                    best = (ic, mdl, r)
    if best is None:
        mdl = SimpleExpSmoothing(y)
        return mdl, mdl.fit(seed=seed)
    return best[1], best[2]


class ThetaModel:
    """The Theta method (Assimakopoulos & Nikolopoulos; M3 winner): decompose into
    two theta-lines, extrapolate the long-run trend with a linear fit and the
    short-run line with SES, then recombine (optionally re-seasonalising)."""

    def __init__(self, endog: Sequence[float], *, seasonal_periods: int = 1,
                 deseasonalize: str = "auto") -> None:
        self.y = _clean(endog)
        self.m = seasonal_periods
        self.deseasonalize = deseasonalize
        self.name = "Theta"

    def fit(self, *, seed: int = 0):
        y = self.y
        n = len(y)
        m = self.m
        self._seasonal = None
        work = list(y)
        do_deseason = (self.deseasonalize in ("auto", True) and m >= 2
                       and n >= 2 * m and seasonal_strength(y, m=m) > 0.3
                       and min(y) > 0.0)
        if do_deseason:
            dec = seasonal_decompose(y, period=m, model="multiplicative")
            self._seasonal = dec.seasonal_indices
            work = [_safe_div(y[t], dec.seasonal[t], default=y[t]) for t in range(n)]
        # linear trend (theta=0 line)
        _, trend = detrend(work, order=1, return_trend=True)
        self._trend = trend
        ses = SimpleExpSmoothing(work).fit(seed=seed)
        self._ses = ses
        self._alpha = ses.params["alpha"]
        self._work_last = work[-1]
        self._n = n
        # slope of the trend line
        b = (trend[-1] - trend[0]) / (n - 1) if n > 1 else 0.0
        self._slope = b
        self._converged = ses.converged
        fitted = [0.5 * ses.fittedvalues[t] + 0.5 * trend[t] for t in range(n)]
        if self._seasonal is not None:
            dec = seasonal_decompose(y, period=m, model="multiplicative")
            fitted = [fitted[t] * dec.seasonal[t] for t in range(n)]
        self._fitted = fitted
        self.resid = [y[t] - fitted[t] for t in range(n)]
        self.sigma2 = max(_var(self.resid), _EPS)
        return self

    def forecast(self, h: int, *, level: Sequence[float] = (0.80, 0.95)
                 ) -> ForecastResult:
        n = self._n
        alpha = self._alpha
        ses_fc = self._ses._final if hasattr(self._ses, "_final") else None
        ses_last = self._ses.fittedvalues[-1] if self._ses.fittedvalues else self._work_last
        # SES flat forecast + half the drift extrapolation
        mean = []
        for k in range(1, h + 1):
            theta2 = ses_last + 0.5 * self._slope * (k + (1.0 - (1.0 - alpha) ** n)
                                                     / max(alpha, _EPS))
            mean.append(theta2)
        if self._seasonal is not None:
            mean = [mean[k] * self._seasonal[(n + k) % self.m] for k in range(h)]
        sigma = math.sqrt(self.sigma2)
        lower: Dict[float, List[float]] = {}
        upper: Dict[float, List[float]] = {}
        for lv in level:
            z = _level_z(lv)
            lower[lv] = [mean[k] - z * sigma * math.sqrt(1.0 + k * alpha * alpha)
                         for k in range(h)]
            upper[lv] = [mean[k] + z * sigma * math.sqrt(1.0 + k * alpha * alpha)
                         for k in range(h)]
        return ForecastResult(model="Theta", h=h, mean=mean, lower=lower,
                              upper=upper, level=list(level), method="analytic")


# ===========================================================================
# 19. STRUCTURAL (UNOBSERVED-COMPONENTS) MODELS
# ===========================================================================
@dataclass
class StructuralResult:
    model: str
    params: Dict[str, float]
    level: List[float]
    trend: List[float]
    fittedvalues: List[float]
    resid: List[float]
    sigma2: float
    llf: float
    aic: float
    bic: float
    nobs: int
    k_params: int
    _ss: StateSpaceModel = None
    converged: bool = True

    def forecast(self, h: int, *, level: Sequence[float] = (0.80, 0.95)
                 ) -> ForecastResult:
        ss = self._ss
        r = len(ss.T)
        a = list(self._a_last)
        P = [row[:] for row in self._P_last]
        RQR = ss.rqr()
        mean, var = [], []
        for _ in range(h):
            a = matvec(ss.T, a)
            P = mat_add(matmul(matmul(ss.T, P), transpose(ss.T)), RQR)
            yhat = _kahan_sum(ss.Z[i] * a[i] for i in range(r))
            fvar = _kahan_sum(ss.Z[i] * _kahan_sum(P[i][j] * ss.Z[j]
                              for j in range(r)) for i in range(r)) + ss.H
            mean.append(yhat)
            var.append(max(fvar, _EPS))
        sigma = [math.sqrt(v) for v in var]
        lower: Dict[float, List[float]] = {}
        upper: Dict[float, List[float]] = {}
        for lv in level:
            z = _level_z(lv)
            lower[lv] = [mean[i] - z * sigma[i] for i in range(h)]
            upper[lv] = [mean[i] + z * sigma[i] for i in range(h)]
        return ForecastResult(model=self.model, h=h, mean=mean, lower=lower,
                              upper=upper, level=list(level), sigma=sigma)

    def summary(self) -> str:
        ps = "  ".join("%s=%.5f" % (k, v) for k, v in sorted(self.params.items()))
        return "%s  nobs=%d\n  %s\n  llf=%.3f aic=%.3f" % (
            self.model, self.nobs, ps, self.llf, self.aic)


def _fit_structural(y: Sequence[float], *, trend: bool, name: str, seed: int
                    ) -> StructuralResult:
    y = _clean(y)
    n = len(y)
    scale = _var(y) + 1.0
    r = 2 if trend else 1

    def build(theta):
        # variances via exp so they stay positive
        sig_eps = math.exp(theta[0]) * scale
        sig_level = math.exp(theta[1]) * scale
        if trend:
            sig_trend = math.exp(theta[2]) * scale
            T = [[1.0, 1.0], [0.0, 1.0]]
            Z = [1.0, 0.0]
            Q = [[sig_level, 0.0], [0.0, sig_trend]]
            R = eye(2)
        else:
            T = [[1.0]]
            Z = [1.0]
            Q = [[sig_level]]
            R = [[1.0]]
        a1 = [y[0]] + ([0.0] if trend else [])
        P1 = mat_scale(eye(r), 1e6 * scale)              # diffuse level init
        return StateSpaceModel(T=T, Z=Z, Q=Q, H=sig_eps, R=R, a1=a1, P1=P1)

    def nll(theta):
        try:
            kf = kalman_filter(build(theta), y)
        except (ValueError, OverflowError):
            return _INF
        return -kf.loglike

    x0 = [-1.0, -1.0] + ([-1.0] if trend else [])
    res = minimize_nelder_mead(nll, x0, restarts=2, seed=seed, maxiter=1000)
    ss = build(res.x)
    kf = kalman_filter(ss, y)
    smoothed = kalman_smoother(ss, y)
    lev = [s[0] for s in smoothed]
    tr = [s[1] for s in smoothed] if trend else [0.0] * n
    fitted = [_kahan_sum(ss.Z[i] * smoothed[t][i] for i in range(r))
              for t in range(n)]
    resid = [y[t] - fitted[t] for t in range(n)]
    sigma2 = max(_var(resid), _EPS)
    k = (3 if trend else 2)
    ic = information_criteria(loglik=kf.loglike, n_params=k, nobs=n)
    out = StructuralResult(name, {"sigma2_eps": ss.H, "sigma2_level": ss.Q[0][0],
                                  **({"sigma2_trend": ss.Q[1][1]} if trend else {})},
                           lev, tr, fitted, resid, sigma2, kf.loglike, ic.aic,
                           ic.bic, n, k, ss, res.converged)
    # store last filtered state for forecasting
    a = list(ss.a1); P = [row[:] for row in ss.P1]; RQR = ss.rqr()
    for t in range(n):
        Pa = [_kahan_sum(P[i][j] * ss.Z[j] for j in range(r)) for i in range(r)]
        F = max(_kahan_sum(ss.Z[i] * Pa[i] for i in range(r)) + ss.H, _EPS)
        v = y[t] - _kahan_sum(ss.Z[i] * a[i] for i in range(r))
        af = [a[i] + Pa[i] * v / F for i in range(r)]
        Pf = [[P[i][j] - Pa[i] * Pa[j] / F for j in range(r)] for i in range(r)]
        a = matvec(ss.T, af)
        P = mat_add(matmul(matmul(ss.T, Pf), transpose(ss.T)), RQR)
    out._a_last = a
    out._P_last = P
    return out


def local_level(y: Sequence[float], *, seed: int = 0) -> StructuralResult:
    """Local-level (random-walk-plus-noise) structural model."""
    return _fit_structural(y, trend=False, name="LocalLevel", seed=seed)


def local_linear_trend(y: Sequence[float], *, seed: int = 0) -> StructuralResult:
    """Local-linear-trend structural model."""
    return _fit_structural(y, trend=True, name="LocalLinearTrend", seed=seed)


# ===========================================================================
# 20. VECTOR AUTOREGRESSION (VAR)
# ===========================================================================
@dataclass
class IRFResult:
    irf: List[Matrix]                  # irf[h] is a k x k matrix
    orthogonalized: bool
    horizon: int
    names: List[str]


@dataclass
class FEVDResult:
    fevd: List[Matrix]                 # fevd[h][i][j] = share of var i from shock j
    horizon: int
    names: List[str]


@dataclass
class GrangerCausalityResult:
    caused: List[str]
    causing: List[str]
    statistic: float
    pvalue: float
    df: Tuple[int, int]
    kind: str


@dataclass
class OrderSelectionResult:
    aic: Dict[int, float]
    bic: Dict[int, float]
    hqic: Dict[int, float]
    fpe: Dict[int, float]
    selected_aic: int
    selected_bic: int


@dataclass
class VARResults:
    coefs: List[Matrix]                # A_1..A_p, each k x k
    intercept: Vector
    sigma_u: Matrix
    resid: List[Vector]
    fittedvalues: List[Vector]
    p: int
    k: int
    names: List[str]
    nobs: int
    llf: float
    aic: float
    bic: float
    hqic: float
    fpe: float
    _model: "VAR"

    def forecast(self, h: int = 1, *, level: Sequence[float] = (0.80, 0.95)
                 ) -> ForecastResult:
        data = self._model.data
        k, p = self.k, self.p
        hist = [row[:] for row in data[-p:]]
        preds = []
        for step in range(h):
            yhat = list(self.intercept)
            for lag in range(1, p + 1):
                prev = (preds[step - lag] if step - lag >= 0
                        else hist[p + step - lag])
                Alag = self.coefs[lag - 1]
                for i in range(k):
                    yhat[i] += _kahan_sum(Alag[i][j] * prev[j] for j in range(k))
            preds.append(yhat)
        # forecast-error covariance via MA(inf)
        Phi = self._ma_rep(h)
        fvar = [[0.0] * k for _ in range(h)]
        cum = [[0.0] * k for _ in range(k)]
        for step in range(h):
            Pi = Phi[step]
            contrib = matmul(matmul(Pi, self.sigma_u), transpose(Pi))
            cum = mat_add(cum, contrib)
            for i in range(k):
                fvar[step][i] = max(cum[i][i], _EPS)
        mean_flat = [preds[s][0] for s in range(h)]     # primary variable series
        lower: Dict[float, List[float]] = {}
        upper: Dict[float, List[float]] = {}
        for lv in level:
            z = _level_z(lv)
            lower[lv] = [preds[s][0] - z * math.sqrt(fvar[s][0]) for s in range(h)]
            upper[lv] = [preds[s][0] + z * math.sqrt(fvar[s][0]) for s in range(h)]
        fr = ForecastResult(model="VAR", h=h, mean=mean_flat, lower=lower,
                            upper=upper, level=list(level))
        fr.extra["forecast_matrix"] = preds
        fr.extra["variables"] = self.names
        return fr

    def _ma_rep(self, h: int) -> List[Matrix]:
        k, p = self.k, self.p
        Phi = [eye(k)]
        for i in range(1, h):
            acc = zeros(k, k)
            for j in range(1, min(i, p) + 1):
                acc = mat_add(acc, matmul(Phi[i - j], self.coefs[j - 1]))
            Phi.append(acc)
        return Phi

    def irf(self, h: int = 10, *, orthogonalized: bool = True) -> IRFResult:
        Phi = self._ma_rep(h + 1)
        if orthogonalized:
            L = cholesky(self.sigma_u, jitter=1e-10) or eye(self.k)
            out = [matmul(Phi[i], L) for i in range(h + 1)]
        else:
            out = Phi[:h + 1]
        return IRFResult(out, orthogonalized, h, self.names)

    def fevd(self, h: int = 10) -> FEVDResult:
        irf = self.irf(h, orthogonalized=True).irf
        k = self.k
        out = []
        mse = [[0.0] * k for _ in range(k)]              # cumulative diagonal MSE
        contrib_cum = [[0.0] * k for _ in range(k)]
        for step in range(h + 1):
            Theta = irf[step]
            for i in range(k):
                for j in range(k):
                    contrib_cum[i][j] += Theta[i][j] ** 2
            tab = [[0.0] * k for _ in range(k)]
            for i in range(k):
                tot = _kahan_sum(contrib_cum[i][j] for j in range(k))
                for j in range(k):
                    tab[i][j] = _safe_div(contrib_cum[i][j], tot, default=0.0)
            out.append(tab)
        return FEVDResult(out, h, self.names)

    def test_causality(self, caused, *, causing) -> GrangerCausalityResult:
        """Granger causality: an F-test that ``causing`` variables' lags do not
        help predict ``caused`` (fit restricted vs unrestricted equations)."""
        names = self.names
        ci = names.index(caused) if isinstance(caused, str) else caused
        cj = [names.index(c) if isinstance(c, str) else c
              for c in ([causing] if isinstance(causing, (str, int)) else causing)]
        data = self._model.data
        p, k = self.p, self.k
        n = len(data)
        # unrestricted equation for `caused`
        Y = [data[t][ci] for t in range(p, n)]
        Xu = []
        for t in range(p, n):
            row = [1.0]
            for lag in range(1, p + 1):
                row += [data[t - lag][j] for j in range(k)]
            Xu.append(row)
        cu, _ = qr_lstsq(Xu, Y, ridge=1e-10)
        ru = [Y[i] - _kahan_sum(Xu[i][j] * cu[j] for j in range(len(cu)))
              for i in range(len(Y))]
        rss_u = _kahan_sum(e * e for e in ru)
        # restricted: drop lags of `causing`
        Xr = []
        for t in range(p, n):
            row = [1.0]
            for lag in range(1, p + 1):
                row += [data[t - lag][j] for j in range(k) if j not in cj]
            Xr.append(row)
        cr, _ = qr_lstsq(Xr, Y, ridge=1e-10)
        rr = [Y[i] - _kahan_sum(Xr[i][j] * cr[j] for j in range(len(cr)))
              for i in range(len(Y))]
        rss_r = _kahan_sum(e * e for e in rr)
        num_df = p * len(cj)
        den_df = len(Y) - len(cu)
        F = _safe_div((rss_r - rss_u) / num_df, rss_u / max(den_df, 1),
                      default=0.0)
        pval = f_sf(F, dfn=num_df, dfd=max(den_df, 1))
        return GrangerCausalityResult(
            [caused] if isinstance(caused, str) else [names[ci]],
            [names[j] for j in cj], F, pval, (num_df, den_df), "F")

    def summary(self) -> str:
        return ("VAR(%d) k=%d nobs=%d  aic=%.3f bic=%.3f fpe=%.4g"
                % (self.p, self.k, self.nobs, self.aic, self.bic, self.fpe))


class VAR:
    """Vector autoregression: multivariate OLS estimation of ``k`` interdependent
    series.  ``data`` is a list of length-``k`` observation vectors (rows = time)."""

    def __init__(self, data: Sequence[Sequence[float]], *,
                 names: Optional[Sequence[str]] = None) -> None:
        self.data = [[float(v) for v in row] for row in data]
        self.k = len(self.data[0])
        self.names = list(names) if names else ["y%d" % i for i in range(self.k)]

    def fit(self, *, maxlags: int = 1, ic: Optional[str] = None,
            trend: str = "c") -> VARResults:
        if ic is not None:
            sel = var_select_order(self.data, maxlags=maxlags, trend=trend)
            p = {"aic": sel.selected_aic, "bic": sel.selected_bic}.get(ic,
                                                                       sel.selected_aic)
            p = max(1, p)
        else:
            p = max(1, maxlags)
        data = self.data
        k = self.k
        n = len(data)
        rows, targets = [], []
        for t in range(p, n):
            row = [1.0] if "c" in trend else []
            for lag in range(1, p + 1):
                row += [data[t - lag][j] for j in range(k)]
            rows.append(row)
            targets.append([data[t][i] for i in range(k)])
        ncoef = len(rows[0])
        coefs_by_eq = []
        for i in range(k):
            yi = [targets[t][i] for t in range(len(targets))]
            ci, _ = qr_lstsq(rows, yi, ridge=1e-10)
            coefs_by_eq.append(ci)
        intercept = [coefs_by_eq[i][0] if "c" in trend else 0.0 for i in range(k)]
        base = 1 if "c" in trend else 0
        A = []
        for lag in range(p):
            Alag = [[coefs_by_eq[i][base + lag * k + j] for j in range(k)]
                    for i in range(k)]
            A.append(Alag)
        fitted = []
        resid = []
        for t in range(len(rows)):
            fit_row = [_kahan_sum(rows[t][j] * coefs_by_eq[i][j]
                       for j in range(ncoef)) for i in range(k)]
            fitted.append(fit_row)
            resid.append([targets[t][i] - fit_row[i] for i in range(k)])
        neff = len(resid)
        sigma_u = [[_kahan_sum(resid[t][i] * resid[t][j] for t in range(neff))
                    / neff for j in range(k)] for i in range(k)]
        sigma_ml = sigma_u
        ld = logdet(sigma_ml) if is_pos_def(sigma_ml) else logdet_chol(
            cholesky(sigma_ml, jitter=1e-8) or eye(k))
        llf = -0.5 * neff * (k * _LOG2PI + ld + k)
        nparams = k * ncoef
        aic = -2.0 * llf + 2.0 * nparams
        bic = -2.0 * llf + nparams * math.log(neff)
        hqic = -2.0 * llf + 2.0 * nparams * math.log(math.log(neff)) \
            if neff > 2 else _INF
        fpe = ((neff + ncoef) / (neff - ncoef)) ** k * math.exp(ld) \
            if neff > ncoef else _INF
        return VARResults(A, intercept, sigma_u, resid, fitted, p, k, self.names,
                          neff, llf, aic, bic, hqic, fpe, self)


def var_select_order(data: Sequence[Sequence[float]], *, maxlags: int = 4,
                     trend: str = "c") -> OrderSelectionResult:
    """Select the VAR lag order by AIC/BIC/HQIC/FPE on the common sample."""
    data = [[float(v) for v in row] for row in data]
    k = len(data[0])
    n = len(data)
    aic, bic, hqic, fpe = {}, {}, {}, {}
    for p in range(1, maxlags + 1):
        if n - maxlags < k * p + 2:
            continue
        # fit on the COMMON sample (start at maxlags for comparability)
        rows, targets = [], []
        for t in range(maxlags, n):
            row = [1.0] if "c" in trend else []
            for lag in range(1, p + 1):
                row += [data[t - lag][j] for j in range(k)]
            rows.append(row)
            targets.append([data[t][i] for i in range(k)])
        ncoef = len(rows[0])
        resid = []
        for t in range(len(rows)):
            fit_row = []
            for i in range(k):
                yi = [targets[s][i] for s in range(len(targets))]
                ci, _ = qr_lstsq(rows, yi, ridge=1e-10)
                fit_row.append(_kahan_sum(rows[t][j] * ci[j] for j in range(ncoef)))
            resid.append([targets[t][i] - fit_row[i] for i in range(k)])
        neff = len(resid)
        su = [[_kahan_sum(resid[t][i] * resid[t][j] for t in range(neff)) / neff
               for j in range(k)] for i in range(k)]
        Ls = cholesky(su, jitter=1e-8)
        ld = logdet_chol(Ls) if Ls else 0.0
        nparams = k * ncoef
        llf = -0.5 * neff * (k * _LOG2PI + ld + k)
        aic[p] = -2.0 * llf + 2.0 * nparams
        bic[p] = -2.0 * llf + nparams * math.log(neff)
        hqic[p] = -2.0 * llf + 2.0 * nparams * math.log(math.log(neff)) \
            if neff > 2 else _INF
        fpe[p] = ((neff + ncoef) / (neff - ncoef)) ** k * math.exp(ld) \
            if neff > ncoef else _INF
    sa = min(aic, key=aic.get) if aic else 1
    sb = min(bic, key=bic.get) if bic else 1
    return OrderSelectionResult(aic, bic, hqic, fpe, sa, sb)


# ===========================================================================
# 21. VOLATILITY: GARCH / GJR-GARCH
# ===========================================================================
@dataclass
class GARCHResult:
    params: Dict[str, float]
    loglike: float
    aic: float
    bic: float
    conditional_variance: List[float]
    std_resid: List[float]
    persistence: float
    unconditional_variance: float
    model: str
    converged: bool = True

    def summary(self) -> str:
        ps = "  ".join("%s=%.5f" % (k, v) for k, v in sorted(self.params.items()))
        return ("%s  persistence=%.4f  uncond_var=%.4f\n  %s\n  loglike=%.3f aic=%.3f"
                % (self.model, self.persistence, self.unconditional_variance, ps,
                   self.loglike, self.aic))


class GARCH:
    """GARCH(p, q) / GJR-GARCH conditional-variance model estimated by Gaussian
    (quasi-)maximum likelihood.  The mean is a constant; residuals feed the
    variance recursion sigma2_t = omega + sum alpha e^2 + sum beta sigma2 (+ GJR
    leverage on negative shocks)."""

    def __init__(self, endog: Sequence[float], *, p: int = 1, q: int = 1,
                 mean: str = "constant", model: str = "garch") -> None:
        self.y = _clean(endog)
        self.p = p                                       # ARCH order
        self.q = q                                       # GARCH order
        self.mean = mean
        self.model = model                               # 'garch' | 'gjr'

    def _recursion(self, mu, omega, alpha, beta, gjr):
        y = self.y
        n = len(y)
        e = [v - mu for v in y]
        var0 = max(_var(e), _EPS)
        h = [var0] * n
        for t in range(1, n):
            val = omega
            for i in range(self.p):
                if t - 1 - i >= 0:
                    et = e[t - 1 - i]
                    val += alpha[i] * et * et
                    if self.model == "gjr" and et < 0.0:
                        val += gjr[i] * et * et
                else:
                    val += alpha[i] * var0
            for j in range(self.q):
                val += beta[j] * (h[t - 1 - j] if t - 1 - j >= 0 else var0)
            h[t] = max(val, _EPS)
        return e, h

    def fit(self, *, seed: int = 0, maxiter: int = 1500) -> GARCHResult:
        y = self.y
        p, q = self.p, self.q
        ng = p if self.model == "gjr" else 0
        mu0 = _mean(y)
        uvar = max(_var(y), _EPS)

        def unpack(vec):
            i = 0
            mu = vec[i]; i += 1
            omega = math.exp(vec[i]); i += 1
            # alpha + beta constrained to sum < 1 via softmax
            weights = softmax_simplex(vec[i:i + p + q])
            alpha = weights[:p]
            beta = weights[p:p + q]
            i += p + q
            gjr = [expit(vec[i + t]) * 0.5 for t in range(ng)] if ng else [0.0] * p
            return mu, omega, alpha, beta, gjr

        def nll(vec):
            mu, omega, alpha, beta, gjr = unpack(vec)
            e, h = self._recursion(mu, omega, alpha, beta, gjr)
            ll = -0.5 * _kahan_sum(_LOG2PI + math.log(h[t]) + e[t] * e[t] / h[t]
                                   for t in range(len(y)))
            return -ll
        x0 = [mu0, math.log(0.1 * uvar)] + [0.0] * (p + q) + [0.0] * ng
        res = minimize_nelder_mead(nll, x0, restarts=3, seed=seed, maxiter=maxiter)
        mu, omega, alpha, beta, gjr = unpack(res.x)
        e, h = self._recursion(mu, omega, alpha, beta, gjr)
        std = [e[t] / math.sqrt(h[t]) for t in range(len(y))]
        pers = _kahan_sum(alpha) + _kahan_sum(beta) + 0.5 * _kahan_sum(gjr)
        uncond = _safe_div(omega, 1.0 - pers, default=uvar)
        ll = -res.fun
        k = 1 + 1 + p + q + ng
        n = len(y)
        aic = -2.0 * ll + 2.0 * k
        bic = -2.0 * ll + k * math.log(n)
        params = {"mu": mu, "omega": omega}
        for i in range(p):
            params["alpha%d" % (i + 1)] = alpha[i]
        for j in range(q):
            params["beta%d" % (j + 1)] = beta[j]
        for i in range(ng):
            params["gamma%d" % (i + 1)] = gjr[i]
        self._last = (mu, omega, alpha, beta, gjr, h, e)
        return GARCHResult(params, ll, aic, bic, h, std, pers, uncond,
                           self.model.upper(), res.converged)

    def forecast(self, result: GARCHResult, h: int = 1) -> List[float]:
        """Forecast conditional variance ``h`` steps ahead.  Iterates the FULL
        GARCH(p, q)/GJR recursion: known past squared residuals feed the near horizon
        (with their observed sign for the leverage term), while unknown future shocks
        contribute their expectation E[e^2] = E[sigma^2] -- with 0.5*gamma expected
        leverage under a symmetric conditional distribution.  Exact for GARCH(1,1);
        the old code dropped every ARCH/GARCH lag past the first and the GJR term."""
        mu, omega, alpha, beta, gjr, hist, e = self._last
        p, q = self.p, self.q
        n = len(hist)
        is_gjr = (self.model == "gjr")
        hf = list(hist)                                  # sigma^2 path, extended forward
        e2 = [ev * ev for ev in e]                       # known squared residuals
        neg = [ev < 0.0 for ev in e]
        out = []
        for k in range(h):
            m = n + k                                    # absolute time being forecast
            val = omega
            for i in range(1, p + 1):                    # ARCH lags
                t = m - i
                if t < n:                                # observed shock
                    val += alpha[i - 1] * e2[t]
                    if is_gjr and neg[t]:
                        val += gjr[i - 1] * e2[t]
                else:                                    # future shock: E[e^2] = E[h]
                    val += (alpha[i - 1] + 0.5 * gjr[i - 1]) * hf[t]
            for j in range(1, q + 1):                    # GARCH lags
                val += beta[j - 1] * hf[m - j]
            val = max(val, _EPS)
            hf.append(val)
            out.append(val)
        return out


# ===========================================================================
# 22. AUTOMATIC ORDER SELECTION
# ===========================================================================
@dataclass
class AutoARIMAResult:
    order: Tuple[int, int, int]
    seasonal_order: Tuple[int, int, int, int]
    trend: str
    ic: float
    ic_name: str
    results: ARIMAResults
    n_models: int
    trace: List[Tuple[Tuple, Tuple, float]] = field(default_factory=list)

    def forecast(self, h: int = 1, **kw) -> ForecastResult:
        return self.results.forecast(h, **kw)

    def summary(self) -> str:
        return ("auto_arima -> ARIMA%s x %s trend=%s  %s=%.3f  (%d models)\n%s"
                % (self.order, self.seasonal_order, self.trend, self.ic_name,
                   self.ic, self.n_models, self.results.summary()))


def _fit_arima_safe(y, order, seasonal_order, trend, *, method, seed):
    try:
        res = ARIMA(y, order=order, seasonal_order=seasonal_order,
                    trend=trend).fit(method=method, seed=seed)
        ic = res.aicc
        if not math.isfinite(ic):
            return None, _INF
        return res, ic
    except (ValueError, ZeroDivisionError, OverflowError):
        return None, _INF


def auto_arima(y: Sequence[float], *, seasonal: bool = True, m: int = 1,
               d: Optional[int] = None, D: Optional[int] = None, max_p: int = 3,
               max_q: int = 3, max_P: int = 1, max_Q: int = 1, start_p: int = 2,
               start_q: int = 2, max_d: int = 2, max_D: int = 1,
               information_criterion: str = "aicc", stepwise: bool = True,
               trend: Optional[str] = None, method: str = "css",
               seed: int = 0, trace: bool = False) -> AutoARIMAResult:
    """Automatic ARIMA order selection (Hyndman-Khandakar).  Chooses the number of
    differences by unit-root tests, then searches (p, q, P, Q) -- stepwise around
    a start model by default, or exhaustively -- to minimise an information
    criterion."""
    y = _clean(y)
    if not seasonal or m < 2:
        m = 1
        max_P = max_Q = 0
    if d is None:
        d = ndiffs(y, max_d=max_d)
    if D is None:
        D = nsdiffs(y, m=m, max_D=max_D) if m >= 2 else 0
    tr = trend if trend is not None else ("c" if d + D <= 1 else "n")
    tried: Dict[Tuple, float] = {}
    trace_log: List[Tuple[Tuple, Tuple, float]] = []

    def evaluate(p, q, P, Q):
        key = (p, q, P, Q)
        if key in tried:
            return tried[key]
        res, ic = _fit_arima_safe(y, (p, d, q), (P, D, Q, m), tr,
                                  method=method, seed=seed)
        tried[key] = ic
        trace_log.append(((p, d, q), (P, D, Q, m), ic))
        return ic

    results_cache: Dict[Tuple, ARIMAResults] = {}

    def full_fit(p, q, P, Q):
        res, ic = _fit_arima_safe(y, (p, d, q), (P, D, Q, m), tr,
                                  method=method, seed=seed)
        return res

    if not stepwise:
        best_key = None
        best_ic = _INF
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                for P in range(max_P + 1):
                    for Q in range(max_Q + 1):
                        ic = evaluate(p, q, P, Q)
                        if ic < best_ic:
                            best_ic, best_key = ic, (p, q, P, Q)
    else:
        cur = (min(start_p, max_p), min(start_q, max_q),
               min(1, max_P), min(1, max_Q))
        best_ic = evaluate(*cur)
        best_key = cur
        # also try the null and a couple of seeds
        for seed_key in [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0)]:
            sk = (min(seed_key[0], max_p), min(seed_key[1], max_q),
                  min(seed_key[2], max_P), min(seed_key[3], max_Q))
            ic = evaluate(*sk)
            if ic < best_ic:
                best_ic, best_key = ic, sk
        improved = True
        while improved:
            improved = False
            p, q, P, Q = best_key
            neighbours = []
            for dp in (-1, 1):
                neighbours.append((p + dp, q, P, Q))
            for dq in (-1, 1):
                neighbours.append((p, q + dq, P, Q))
            if m >= 2:
                for dP in (-1, 1):
                    neighbours.append((p, q, P + dP, Q))
                for dQ in (-1, 1):
                    neighbours.append((p, q, P, Q + dQ))
                neighbours.append((p + 1, q + 1, P, Q))
                neighbours.append((p - 1, q - 1, P, Q))
            for (np_, nq, nP, nQ) in neighbours:
                if not (0 <= np_ <= max_p and 0 <= nq <= max_q
                        and 0 <= nP <= max_P and 0 <= nQ <= max_Q):
                    continue
                ic = evaluate(np_, nq, nP, nQ)
                if ic < best_ic - 1e-8:
                    best_ic, best_key = ic, (np_, nq, nP, nQ)
                    improved = True
                    break
    p, q, P, Q = best_key
    res = full_fit(p, q, P, Q)
    if res is None:                                      # fallback
        res = ARIMA(y, order=(0, d, 0), seasonal_order=(0, D, 0, m),
                    trend=tr).fit(method=method, seed=seed)
    return AutoARIMAResult((p, d, q), (P, D, Q, m), tr, best_ic,
                           information_criterion, res, len(tried),
                           trace_log if trace else [])


def grid_search_arima(y: Sequence[float], *, d: Optional[int] = None,
                      D: Optional[int] = None, m: int = 1, max_p: int = 3,
                      max_q: int = 3, max_P: int = 1, max_Q: int = 1,
                      trend: Optional[str] = None, method: str = "css",
                      seed: int = 0) -> AutoARIMAResult:
    """Exhaustive-grid ARIMA order selection (a thin wrapper on ``auto_arima``)."""
    return auto_arima(y, m=m, d=d, D=D, max_p=max_p, max_q=max_q, max_P=max_P,
                      max_Q=max_Q, stepwise=False, trend=trend, method=method,
                      seed=seed)


# ===========================================================================
# 23. BASELINE FORECASTERS + MODEL REGISTRY
# ===========================================================================
class BaseForecaster:
    """Uniform baseline-forecaster contract: ``fit(y)`` then
    ``forecast(h, level=...)``."""

    name = "base"

    def __init__(self, *, seasonality: int = 1) -> None:
        self.m = seasonality
        self.y_: List[float] = []
        self.fitted_: List[float] = []
        self.resid_: List[float] = []
        self.sigma_: float = 0.0

    def fit(self, y: Sequence[float]) -> "BaseForecaster":
        self.y_ = _clean(y)
        self._fit_impl()
        self.sigma_ = _std(self.resid_) if self.resid_ else 0.0
        return self

    def _fit_impl(self) -> None:
        raise NotImplementedError

    def _point(self, h: int) -> List[float]:
        raise NotImplementedError

    def forecast(self, h: int, *, level: Sequence[float] = (0.80, 0.95)
                 ) -> ForecastResult:
        mean = self._point(h)
        lower: Dict[float, List[float]] = {}
        upper: Dict[float, List[float]] = {}
        for lv in level:
            z = _level_z(lv)
            lower[lv] = [mean[k] - z * self.sigma_ * math.sqrt(k + 1)
                         for k in range(h)]
            upper[lv] = [mean[k] + z * self.sigma_ * math.sqrt(k + 1)
                         for k in range(h)]
        return ForecastResult(model=self.name, h=h, mean=mean, lower=lower,
                              upper=upper, level=list(level))

    @property
    def fittedvalues(self) -> List[float]:
        return self.fitted_

    @property
    def resid(self) -> List[float]:
        return self.resid_


class MeanForecaster(BaseForecaster):
    name = "mean"

    def _fit_impl(self) -> None:
        self._mu = _mean(self.y_)
        self.fitted_ = [self._mu] * len(self.y_)
        self.resid_ = [self.y_[t] - self._mu for t in range(len(self.y_))]

    def _point(self, h: int) -> List[float]:
        return [self._mu] * h


class NaiveForecaster(BaseForecaster):
    name = "naive"

    def _fit_impl(self) -> None:
        y = self.y_
        self.fitted_ = [y[t - 1] for t in range(1, len(y))]
        self.resid_ = [y[t] - y[t - 1] for t in range(1, len(y))]

    def _point(self, h: int) -> List[float]:
        return [self.y_[-1]] * h


class DriftForecaster(BaseForecaster):
    name = "drift"

    def _fit_impl(self) -> None:
        y = self.y_
        n = len(y)
        self._slope = (y[-1] - y[0]) / (n - 1) if n > 1 else 0.0
        self.fitted_ = [y[t - 1] + self._slope for t in range(1, n)]
        self.resid_ = [y[t] - self.fitted_[t - 1] for t in range(1, n)]

    def _point(self, h: int) -> List[float]:
        return [self.y_[-1] + self._slope * (k + 1) for k in range(h)]


class SeasonalNaiveForecaster(BaseForecaster):
    name = "seasonal_naive"

    def _fit_impl(self) -> None:
        y = self.y_
        m = max(self.m, 1)
        self.fitted_ = [y[t - m] for t in range(m, len(y))]
        self.resid_ = [y[t] - y[t - m] for t in range(m, len(y))]

    def _point(self, h: int) -> List[float]:
        y = self.y_
        m = max(self.m, 1)
        return [y[len(y) - m + (k % m)] for k in range(h)]


MODEL_REGISTRY: Dict[str, Callable[..., BaseForecaster]] = {}


def register_model(name: str) -> Callable:
    """Decorator registering a ``BaseForecaster`` factory under ``name``."""
    def deco(factory: Callable[..., BaseForecaster]) -> Callable:
        MODEL_REGISTRY[name] = factory
        return factory
    return deco


for _nm, _cls in [("mean", MeanForecaster), ("naive", NaiveForecaster),
                  ("drift", DriftForecaster),
                  ("seasonal_naive", SeasonalNaiveForecaster)]:
    MODEL_REGISTRY[_nm] = _cls


def make_forecaster(name: str, *, seasonality: int = 1, **params
                    ) -> BaseForecaster:
    """Instantiate a registered baseline forecaster (aliases: 'rw'->'naive',
    'snaive'->'seasonal_naive')."""
    alias = {"rw": "naive", "random_walk": "naive", "snaive": "seasonal_naive",
             "average": "mean"}
    key = alias.get(name, name)
    if key not in MODEL_REGISTRY:
        raise ValueError("unknown model %r; registered: %s"
                         % (name, sorted(MODEL_REGISTRY)))
    return MODEL_REGISTRY[key](seasonality=seasonality, **params)


# ===========================================================================
# 24. FORECAST-ACCURACY METRICS
# ===========================================================================
def _align(y_true, y_pred):
    n = min(len(y_true), len(y_pred))
    return [float(v) for v in y_true[:n]], [float(v) for v in y_pred[:n]]


def me(y_true, y_pred) -> float:
    a, p = _align(y_true, y_pred)
    return _mean([a[i] - p[i] for i in range(len(a))]) if a else float("nan")


def mae(y_true, y_pred) -> float:
    a, p = _align(y_true, y_pred)
    return _mean([abs(a[i] - p[i]) for i in range(len(a))]) if a else float("nan")


def mse(y_true, y_pred) -> float:
    a, p = _align(y_true, y_pred)
    return _mean([(a[i] - p[i]) ** 2 for i in range(len(a))]) if a else float("nan")


def rmse(y_true, y_pred) -> float:
    v = mse(y_true, y_pred)
    return math.sqrt(v) if v == v else float("nan")


def mape(y_true, y_pred, *, as_percent: bool = True) -> float:
    a, p = _align(y_true, y_pred)
    vals = [abs((a[i] - p[i]) / a[i]) for i in range(len(a)) if abs(a[i]) > _EPS]
    if not vals:
        return float("nan")
    return _mean(vals) * (100.0 if as_percent else 1.0)


def mdape(y_true, y_pred, *, as_percent: bool = True) -> float:
    a, p = _align(y_true, y_pred)
    vals = [abs((a[i] - p[i]) / a[i]) for i in range(len(a)) if abs(a[i]) > _EPS]
    if not vals:
        return float("nan")
    return _median(vals) * (100.0 if as_percent else 1.0)


def wmape(y_true, y_pred, *, as_percent: bool = True) -> float:
    a, p = _align(y_true, y_pred)
    num = _kahan_sum(abs(a[i] - p[i]) for i in range(len(a)))
    den = _kahan_sum(abs(v) for v in a)
    return _safe_div(num, den, default=float("nan")) * (100.0 if as_percent else 1.0)


def smape(y_true, y_pred, *, as_percent: bool = True) -> float:
    a, p = _align(y_true, y_pred)
    vals = []
    for i in range(len(a)):
        den = (abs(a[i]) + abs(p[i]))
        if den > _EPS:
            vals.append(2.0 * abs(a[i] - p[i]) / den)
    if not vals:
        # every pair had |actual| + |forecast| ~ 0, i.e. a perfect all-zero
        # forecast -- sMAPE is 0 (not nan, which would poison an aggregate).
        return 0.0
    return _mean(vals) * (100.0 if as_percent else 1.0)


def mase(y_true, y_pred, *, y_train: Sequence[float], seasonality: int = 1,
         denom: Optional[float] = None) -> float:
    """Mean absolute scaled error (scaled by the in-sample seasonal-naive MAE)."""
    a, p = _align(y_true, y_pred)
    if denom is None:
        yt = _clean(y_train)
        s = max(seasonality, 1)
        diffs = [abs(yt[t] - yt[t - s]) for t in range(s, len(yt))]
        denom = _mean(diffs) if diffs else _INF
    denom = max(denom, _EPS)
    return _mean([abs(a[i] - p[i]) for i in range(len(a))]) / denom if a \
        else float("nan")


def rmsse(y_true, y_pred, *, y_train: Sequence[float], seasonality: int = 1,
          denom: Optional[float] = None) -> float:
    """Root mean squared scaled error (the M5 metric)."""
    a, p = _align(y_true, y_pred)
    if denom is None:
        yt = _clean(y_train)
        s = max(seasonality, 1)
        diffs = [(yt[t] - yt[t - s]) ** 2 for t in range(s, len(yt))]
        denom = _mean(diffs) if diffs else _INF
    denom = max(denom, _EPS)
    return math.sqrt(_mean([(a[i] - p[i]) ** 2 for i in range(len(a))]) / denom) \
        if a else float("nan")


def pinball_loss(y_true, y_pred_q, *, quantile: float) -> float:
    """Pinball (quantile) loss for a single quantile forecast."""
    a, p = _align(y_true, y_pred_q)
    vals = []
    for i in range(len(a)):
        diff = a[i] - p[i]
        vals.append(quantile * diff if diff >= 0 else (quantile - 1.0) * diff)
    return _mean(vals) if vals else float("nan")


def interval_score(y_true, lower, upper, *, level: float = 0.9) -> float:
    """Winkler / interval score for prediction intervals (lower is better)."""
    a = [float(v) for v in y_true]
    lo = [float(v) for v in lower]
    hi = [float(v) for v in upper]
    n = min(len(a), len(lo), len(hi))
    alpha = 1.0 - level
    vals = []
    for i in range(n):
        li, ui = min(lo[i], hi[i]), max(lo[i], hi[i])
        s = ui - li
        if a[i] < li:
            s += 2.0 / alpha * (li - a[i])
        elif a[i] > ui:
            s += 2.0 / alpha * (a[i] - ui)
        vals.append(s)
    return _mean(vals) if vals else float("nan")


def msis(y_true, lower, upper, *, y_train: Sequence[float], seasonality: int = 1,
         level: float = 0.95) -> float:
    """Mean scaled interval score (the M4/M5 interval metric)."""
    yt = _clean(y_train)
    s = max(seasonality, 1)
    diffs = [abs(yt[t] - yt[t - s]) for t in range(s, len(yt))]
    denom = max(_mean(diffs) if diffs else _INF, _EPS)
    return interval_score(y_true, lower, upper, level=level) / denom


def empirical_coverage(y_true, lower, upper) -> float:
    """Fraction of actuals falling inside the interval."""
    a = [float(v) for v in y_true]
    lo = [float(v) for v in lower]
    hi = [float(v) for v in upper]
    n = min(len(a), len(lo), len(hi))
    if n == 0:
        return float("nan")
    inside = sum(1 for i in range(n)
                 if min(lo[i], hi[i]) <= a[i] <= max(lo[i], hi[i]))
    return inside / n


def theil_u(y_true, y_pred) -> float:
    """Theil's U2: RMSE of the forecast relative to the naive (no-change) forecast."""
    a, p = _align(y_true, y_pred)
    if len(a) < 2:
        return float("nan")
    num = _kahan_sum((a[i + 1] - p[i + 1]) ** 2 for i in range(len(a) - 1))
    den = _kahan_sum((a[i + 1] - a[i]) ** 2 for i in range(len(a) - 1))
    return math.sqrt(_safe_div(num, den, default=float("nan")))


@dataclass
class ForecastMetrics:
    n: int
    me: float
    mae: float
    rmse: float
    mape: float
    smape: float
    wmape: float
    mase: float
    rmsse: float
    coverage: Dict[float, float] = field(default_factory=dict)
    interval_score: Dict[float, float] = field(default_factory=dict)

    def summary(self) -> str:
        base = ("n=%d  MAE=%.4f  RMSE=%.4f  MAPE=%.2f%%  sMAPE=%.2f%%  MASE=%.4f"
                % (self.n, self.mae, self.rmse, self.mape, self.smape, self.mase))
        return base


def evaluate(y_true, y_pred, *, y_train: Optional[Sequence[float]] = None,
             seasonality: int = 1, lower: Optional[Dict[float, Sequence[float]]] = None,
             upper: Optional[Dict[float, Sequence[float]]] = None) -> ForecastMetrics:
    """Compute the standard forecast-accuracy metric bundle."""
    a, p = _align(y_true, y_pred)
    denom = None
    if y_train is not None:
        yt = _clean(y_train)
        s = max(seasonality, 1)
        diffs = [abs(yt[t] - yt[t - s]) for t in range(s, len(yt))]
        denom = _mean(diffs) if diffs else None
    cov: Dict[float, float] = {}
    isc: Dict[float, float] = {}
    if lower and upper:
        for lv in sorted(lower):
            if lv in upper:
                cov[lv] = empirical_coverage(a, lower[lv], upper[lv])
                isc[lv] = interval_score(a, lower[lv], upper[lv], level=lv)
    return ForecastMetrics(
        len(a), me(a, p), mae(a, p), rmse(a, p), mape(a, p), smape(a, p),
        wmape(a, p),
        mase(a, p, y_train=y_train or a, seasonality=seasonality, denom=denom)
        if y_train is not None else float("nan"),
        rmsse(a, p, y_train=y_train or a, seasonality=seasonality)
        if y_train is not None else float("nan"),
        cov, isc)


_METRIC_FUNCS = {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape,
                 "smape": smape, "wmape": wmape, "me": me, "mdape": mdape}


def score(y_true, y_pred, *, metric: str = "rmse", **kwargs) -> float:
    """Compute a single named metric."""
    if metric in ("mase", "rmsse"):
        fn = mase if metric == "mase" else rmsse
        return fn(y_true, y_pred, **kwargs)
    if metric not in _METRIC_FUNCS:
        raise ValueError("unknown metric %r" % metric)
    return _METRIC_FUNCS[metric](y_true, y_pred)


# ===========================================================================
# 25. BACKTESTING, DIEBOLD-MARIANO, ENSEMBLING
# ===========================================================================
class RollingOriginSplitter:
    """Yield (train_index, test_index) pairs for rolling-origin cross-validation.
    ``mode='expanding'`` grows the training window; ``'sliding'`` keeps it fixed."""

    def __init__(self, *, initial: int, horizon: int = 1, step: int = 1,
                 mode: str = "expanding", window: Optional[int] = None,
                 max_splits: Optional[int] = None) -> None:
        self.initial = initial
        self.horizon = horizon
        self.step = step
        self.mode = mode
        self.window = window
        self.max_splits = max_splits

    def split(self, n: int) -> Iterator[Tuple[range, range]]:
        origin = self.initial
        count = 0
        while origin + self.horizon <= n:
            if self.max_splits is not None and count >= self.max_splits:
                break
            if self.mode == "sliding" and self.window:
                tr_start = max(0, origin - self.window)
            else:
                tr_start = 0
            yield (range(tr_start, origin), range(origin, origin + self.horizon))
            origin += self.step
            count += 1

    def n_splits(self, n: int) -> int:
        return sum(1 for _ in self.split(n))


def train_test_split(y: Sequence[float], *, test_size: Optional[int] = None,
                     test_frac: float = 0.2) -> Tuple[List[float], List[float]]:
    """Split a series into (train, test) chronologically."""
    y = _clean(y)
    n = len(y)
    if test_size is None:
        test_size = max(1, int(round(n * test_frac)))
    test_size = min(test_size, n - 1)
    return y[:n - test_size], y[n - test_size:]


@dataclass
class BacktestResult:
    metrics: Dict[str, float]
    per_split: List[Dict[str, float]]
    n_splits: int
    horizon: int
    forecasts: List[List[float]] = field(default_factory=list)
    actuals: List[List[float]] = field(default_factory=list)

    def summary(self) -> str:
        ms = "  ".join("%s=%.4f" % (k, v) for k, v in sorted(self.metrics.items()))
        return "backtest  splits=%d  h=%d\n  %s" % (self.n_splits, self.horizon, ms)


def backtest(forecast_fn: Callable[[Sequence[float], int], Sequence[float]],
             y: Sequence[float], *, splitter: RollingOriginSplitter,
             seasonality: int = 1,
             metrics: Sequence[str] = ("mae", "rmse", "smape", "mase"),
             aggregate: str = "mean") -> BacktestResult:
    """Rolling-origin backtest.  ``forecast_fn(train, h)`` returns ``h`` point
    forecasts; accuracy is aggregated across all splits."""
    y = _clean(y)
    n = len(y)
    per_split: List[Dict[str, float]] = []
    all_fc: List[List[float]] = []
    all_act: List[List[float]] = []
    # a frozen scaling denominator from the first training window (M-competition style)
    frozen_denom = None
    for tr_idx, te_idx in splitter.split(n):
        train = [y[i] for i in tr_idx]
        actual = [y[i] for i in te_idx]
        if len(train) < max(seasonality + 1, 3):
            continue
        try:
            pred = list(forecast_fn(train, len(actual)))
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        if len(pred) < len(actual):
            continue
        pred = pred[:len(actual)]
        if frozen_denom is None:
            s = max(seasonality, 1)
            diffs = [abs(train[t] - train[t - s]) for t in range(s, len(train))]
            frozen_denom = _mean(diffs) if diffs else _EPS
        row = {}
        for mt in metrics:
            if mt in ("mase", "rmsse"):
                row[mt] = score(actual, pred, metric=mt, y_train=train,
                                seasonality=seasonality, denom=frozen_denom)
            else:
                row[mt] = score(actual, pred, metric=mt)
        per_split.append(row)
        all_fc.append(pred)
        all_act.append(actual)
    agg: Dict[str, float] = {}
    for mt in metrics:
        vals = [r[mt] for r in per_split if math.isfinite(r.get(mt, _INF))]
        if vals:
            agg[mt] = _mean(vals) if aggregate == "mean" else _median(vals)
        else:
            agg[mt] = float("nan")
    return BacktestResult(agg, per_split, len(per_split),
                          splitter.horizon, all_fc, all_act)


@dataclass
class DMResult:
    statistic: float
    pvalue: float
    horizon: int
    loss: str
    better: str


def diebold_mariano(e1: Sequence[float], e2: Sequence[float], *, horizon: int = 1,
                    loss: str = "squared", hln_correction: bool = True) -> DMResult:
    """Diebold-Mariano test comparing the forecast accuracy of two models via
    their error series (with the Harvey-Leybourne-Newbold small-sample correction)."""
    e1 = [float(v) for v in e1]
    e2 = [float(v) for v in e2]
    n = min(len(e1), len(e2))
    if loss == "squared":
        d = [e1[i] ** 2 - e2[i] ** 2 for i in range(n)]
    else:
        d = [abs(e1[i]) - abs(e2[i]) for i in range(n)]
    dbar = _mean(d)
    # long-run variance (Newey-West with horizon-1 lags)
    gamma0 = _var(d)
    lrv = gamma0
    for k in range(1, horizon):
        cov = _kahan_sum((d[t] - dbar) * (d[t - k] - dbar)
                         for t in range(k, n)) / n
        lrv += 2.0 * cov
    lrv = max(lrv, _EPS)
    stat = dbar / math.sqrt(lrv / n)
    if hln_correction and n > horizon:
        corr = math.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
        stat *= corr
    pval = 2.0 * (1.0 - t_cdf(abs(stat), df=max(n - 1, 1)))
    better = "model1" if dbar < 0 else ("model2" if dbar > 0 else "tie")
    return DMResult(stat, pval, horizon, loss, better)


def combine_forecasts(forecasts: Sequence[Sequence[float]], *, method: str = "mean",
                      weights: Optional[Sequence[float]] = None,
                      errors: Optional[Sequence[float]] = None) -> List[float]:
    """Combine several forecasts (mean / median / weighted / inverse-error)."""
    h = min(len(f) for f in forecasts)
    k = len(forecasts)
    if method == "median":
        return [_median([forecasts[j][i] for j in range(k)]) for i in range(h)]
    if method == "inverse_error" and errors is not None:
        inv = [1.0 / max(e, _EPS) for e in errors]
        tot = _kahan_sum(inv)
        w = [v / tot for v in inv]
    elif method == "weighted" and weights is not None:
        tot = _kahan_sum(weights)
        w = [v / tot for v in weights]
    else:
        w = [1.0 / k] * k
    return [_kahan_sum(w[j] * forecasts[j][i] for j in range(k)) for i in range(h)]


class EnsembleForecaster:
    """Combine several ``forecast_fn(train, h)`` callables by mean/median/weight."""

    def __init__(self, models: Dict[str, Callable[[Sequence[float], int], Sequence[float]]],
                 *, method: str = "mean",
                 weights: Optional[Dict[str, float]] = None) -> None:
        self.models = models
        self.method = method
        self.weights = weights

    def forecast_fn(self, train: Sequence[float], h: int) -> List[float]:
        names = sorted(self.models)
        fcs = [list(self.models[nm](train, h)) for nm in names]
        w = ([self.weights[nm] for nm in names] if self.weights else None)
        return combine_forecasts(fcs, method=self.method, weights=w)


def auto_select(y: Sequence[float], *,
                candidates: Dict[str, Callable[[Sequence[float], int], Sequence[float]]],
                splitter: RollingOriginSplitter, seasonality: int = 1,
                metric: str = "mase") -> Dict[str, Any]:
    """Backtest every candidate ``forecast_fn`` and return the ranked results plus
    the best model name."""
    scores: Dict[str, float] = {}
    for nm in sorted(candidates):
        bt = backtest(candidates[nm], y, splitter=splitter,
                      seasonality=seasonality, metrics=(metric,))
        scores[nm] = bt.metrics.get(metric, _INF)
    ranked = sorted(scores.items(), key=lambda kv: (kv[1]
                    if math.isfinite(kv[1]) else _INF, kv[0]))
    return {"scores": scores, "ranking": ranked,
            "best": ranked[0][0] if ranked else None, "metric": metric}


# ===========================================================================
# 26. SERIES SIMULATORS
# ===========================================================================
def simulate_sarima(n: int, *, order: Tuple[int, int, int] = (1, 0, 0),
                    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
                    phi: Sequence[float] = (), theta: Sequence[float] = (),
                    seasonal_phi: Sequence[float] = (),
                    seasonal_theta: Sequence[float] = (), sigma2: float = 1.0,
                    c: float = 0.0, m: int = 1, burnin: int = 200,
                    seed: int = 0) -> List[float]:
    """Simulate ``n`` observations from a (seasonal) ARIMA process."""
    rng = DeterministicRNG(seed)
    p, d, q = order
    P, D, Q, ms = seasonal_order
    if ms:
        m = ms
    ar = _expand_lag_polynomial(phi, m=m, seasonal=seasonal_phi, kind="ar")
    ma = _expand_lag_polynomial(theta, m=m, seasonal=seasonal_theta, kind="ma")
    total = n + burnin + len(ar) + len(ma) + 10
    sd = math.sqrt(max(sigma2, _EPS))
    e = [rng.gauss() * sd for _ in range(total)]
    x = [0.0] * total
    for t in range(total):
        val = c
        val -= _kahan_sum(ar[i] * x[t - i] for i in range(1, min(len(ar), t + 1)))
        val += e[t]
        val += _kahan_sum(ma[j] * e[t - j] for j in range(1, min(len(ma), t + 1)))
        x[t] = val
    w = x[burnin:burnin + n]
    if d == 0 and D == 0:
        return w
    # integrate up to the requested differencing order
    diffr = Differencer(d=d, D=D, m=m)
    base = [0.0] * (d + D * m)
    return diffr.forecast_integrate(base, w)


def simulate_arma(phi: Sequence[float], theta: Sequence[float], *, n: int = 200,
                  sigma2: float = 1.0, c: float = 0.0, seed: int = 0) -> List[float]:
    """Simulate an ARMA(p, q) process (thin wrapper on :func:`simulate_sarima`)."""
    return simulate_sarima(n, order=(len(phi), 0, len(theta)), phi=phi,
                           theta=theta, sigma2=sigma2, c=c, seed=seed)


def simulate_series(n: int, *, level: float = 100.0, trend: float = 0.0,
                    seasonality: int = 0, seasonal_amp: float = 0.0,
                    ar: Sequence[float] = (), ma: Sequence[float] = (),
                    noise_sd: float = 1.0, seed: int = 0) -> List[float]:
    """Build a synthetic series from level + linear trend + sinusoidal seasonality
    + an ARMA noise component (handy for demos and tests)."""
    if ar or ma:
        noise = simulate_arma(ar, ma, n=n, sigma2=noise_sd * noise_sd, seed=seed)
    else:
        # ONE rng drawn n times -- constructing DeterministicRNG(seed) inside the
        # comprehension would reseed every iteration and repeat a single value,
        # yielding a constant offset instead of white noise.
        _rng = DeterministicRNG(seed)
        noise = [_rng.gauss() * noise_sd for _ in range(n)]
    out = []
    for t in range(n):
        val = level + trend * t
        if seasonality >= 2:
            val += seasonal_amp * math.sin(2.0 * math.pi * t / seasonality)
        val += noise[t]
        out.append(val)
    return out


# ===========================================================================
# 27. PAYLOADS + DEMO + EXPORTS
# ===========================================================================
def build_result_payload(result: Any, *, name: str = "forecast",
                         forecast: Optional[ForecastResult] = None
                         ) -> Dict[str, Any]:
    """Strict-JSON snapshot of a fitted model / forecast / metric bundle."""
    payload: Dict[str, Any] = {"name": name, "kind": type(result).__name__}
    if isinstance(result, ForecastResult):
        payload.update({
            "model": result.model, "h": result.h, "mean": result.mean,
            "method": result.method,
            "lower": {str(lv): result.lower[lv] for lv in sorted(result.lower)},
            "upper": {str(lv): result.upper[lv] for lv in sorted(result.upper)},
            "level": result.level, "sigma": result.sigma,
        })
    elif isinstance(result, ARIMAResults):
        payload.update({
            "model": result.model.name, "phi": result.phi, "theta": result.theta,
            "seasonal_phi": result.seasonal_phi,
            "seasonal_theta": result.seasonal_theta, "mu": result.mu,
            "sigma2": result.sigma2, "loglik": result.llf, "aic": result.aic,
            "bic": result.bic, "aicc": result.aicc, "nobs": result.nobs,
            "k_params": result.k_params, "order": [len(result.phi), result.d,
                                                   len(result.theta)],
            "seasonal_order": [len(result.seasonal_phi), result.D,
                               len(result.seasonal_theta), result.m],
        })
    elif isinstance(result, ETSResult):
        payload.update({"model": result.model, "params": result.params,
                        "aic": result.aic, "bic": result.bic,
                        "sigma2": result.sigma2, "nobs": result.nobs})
    elif isinstance(result, VARResults):
        payload.update({"k": result.k, "p": result.p, "names": result.names,
                        "aic": result.aic, "bic": result.bic, "fpe": result.fpe,
                        "intercept": result.intercept})
    elif isinstance(result, GARCHResult):
        payload.update({"model": result.model, "params": result.params,
                        "loglike": result.loglike, "aic": result.aic,
                        "persistence": result.persistence})
    elif isinstance(result, ForecastMetrics):
        payload.update({"n": result.n, "mae": result.mae, "rmse": result.rmse,
                        "mape": result.mape, "smape": result.smape,
                        "mase": result.mase, "rmsse": result.rmsse,
                        "coverage": result.coverage})
    elif isinstance(result, BacktestResult):
        payload.update({"metrics": result.metrics, "n_splits": result.n_splits,
                        "horizon": result.horizon})
    elif isinstance(result, UnitRootTestResult):
        payload.update({"test": result.test, "statistic": result.statistic,
                        "pvalue": result.pvalue, "stationary": result.stationary})
    if forecast is not None:
        payload["forecast"] = build_result_payload(forecast, name="forecast")
    return _json_safe(payload)


def build_a2ui_payload(results: Dict[str, Any], *, ic: str = "aic"
                       ) -> Dict[str, Any]:
    """Leaderboard of fitted models ordered by an information criterion (lowest
    first; non-finite scores rank last)."""
    def score_of(r: Any) -> float:
        v = getattr(r, ic, None)
        if v is None:
            v = getattr(r, "aic", None)
        return v if (isinstance(v, (int, float)) and math.isfinite(v)) else _INF

    cards = []
    for nm in sorted(results):
        r = results[nm]
        cards.append({
            "model": nm,
            "ic": score_of(r),
            "aic": getattr(r, "aic", None),
            "bic": getattr(r, "bic", None),
            "loglik": getattr(r, "llf", getattr(r, "loglike", None)),
            "nobs": getattr(r, "nobs", None),
        })
    cards.sort(key=lambda c: (c["ic"] if c["ic"] is not None
               and math.isfinite(c["ic"]) else _INF, c["model"]))
    best = cards[0]["model"] if cards else None
    return _json_safe({"leaderboard": cards, "best_model": best, "criterion": ic})


def _demo() -> None:
    print("Autoregressive & ARIMA Forecasting demo")
    print("=" * 60)

    # --- a seasonal, trending, autoregressive series ----------------------
    y = simulate_series(180, level=100.0, trend=0.15, seasonality=12,
                        seasonal_amp=8.0, ar=[0.6], noise_sd=1.5, seed=0)
    print("series: n=%d  mean=%.2f  range=[%.1f, %.1f]"
          % (len(y), _mean(y), min(y), max(y)))

    # --- stationarity + diagnostics --------------------------------------
    adf = adf_test(y)
    kp = kpss_test(y)
    print("ADF stat=%.3f (stationary=%s) | KPSS stat=%.3f (stationary=%s)"
          % (adf.statistic, adf.stationary, kp.statistic, kp.stationary))
    print("ndiffs=%d  nsdiffs(m=12)=%d  detected period=%d"
          % (ndiffs(y), nsdiffs(y, m=12), detect_seasonal_period(y).period))

    # --- ARIMA / SARIMA ---------------------------------------------------
    train, test = train_test_split(y, test_size=12)
    m1 = ARIMA(train, order=(2, 1, 1), trend="c").fit(method="css")
    fc1 = m1.forecast(12, level=(0.80, 0.95))
    print("ARIMA(2,1,1): aic=%.2f  RMSE(test)=%.3f  coverage95=%.2f"
          % (m1.aic, rmse(test, fc1.mean),
             empirical_coverage(test, fc1.lower[0.95], fc1.upper[0.95])))

    # --- auto_arima -------------------------------------------------------
    aa = auto_arima(train, seasonal=True, m=12, max_p=2, max_q=2, max_P=1,
                    max_Q=1, method="css")
    fc_a = aa.forecast(12)
    print("auto_arima -> ARIMA%s x %s  aicc=%.2f  RMSE(test)=%.3f"
          % (aa.order, aa.seasonal_order, aa.ic, rmse(test, fc_a.mean)))

    # --- exponential smoothing + Theta ------------------------------------
    ets_m, ets_r = auto_ets(train, seasonal_periods=12)
    fc_e = ets_m.forecast(ets_r, 12)
    th = ThetaModel(train, seasonal_periods=12).fit()
    fc_t = th.forecast(12)
    print("auto_ets -> %s  RMSE=%.3f | Theta RMSE=%.3f"
          % (ets_r.model, rmse(test, fc_e.mean), rmse(test, fc_t.mean)))

    # --- baselines + a model leaderboard ----------------------------------
    board = build_a2ui_payload({"ARIMA(2,1,1)": m1, "auto_arima": aa.results,
                                "ETS(%s)" % ets_r.model: ets_r}, ic="aic")
    print("leaderboard best (by AIC): %s" % board["best_model"])

    # --- rolling-origin backtest ------------------------------------------
    def arima_fn(tr, h):
        return ARIMA(tr, order=(1, 1, 1), trend="c").fit(method="css").forecast(h).mean

    def snaive_fn(tr, h):
        return make_forecaster("seasonal_naive", seasonality=12).fit(tr).forecast(h).mean
    sp = RollingOriginSplitter(initial=120, horizon=12, step=12, max_splits=4)
    bt = backtest(arima_fn, y, splitter=sp, seasonality=12,
                  metrics=("mase", "rmse"))
    bt2 = backtest(snaive_fn, y, splitter=sp, seasonality=12,
                   metrics=("mase", "rmse"))
    print("backtest MASE: ARIMA=%.3f  seasonal-naive=%.3f  (%d splits)"
          % (bt.metrics["mase"], bt2.metrics["mase"], bt.n_splits))

    # --- VAR --------------------------------------------------------------
    a = [0.0]
    b = [0.0]
    rng = DeterministicRNG(1)
    for _ in range(200):
        na = 0.5 * a[-1] + 0.2 * b[-1] + rng.gauss()
        nb = 0.4 * b[-1] - 0.1 * a[-1] + rng.gauss()
        a.append(na)
        b.append(nb)
    vr = VAR([[a[t], b[t]] for t in range(len(a))], names=["a", "b"]).fit(
        maxlags=2, ic="aic")
    gc = vr.test_causality("a", causing="b")
    print("VAR(%d): aic=%.2f  Granger b->a p=%.4f" % (vr.p, vr.aic, gc.pvalue))

    # --- GARCH volatility -------------------------------------------------
    ret = [0.0]
    h = 1.0
    for _ in range(500):
        h = 0.05 + 0.10 * ret[-1] ** 2 + 0.85 * h
        ret.append(math.sqrt(h) * rng.gauss())
    gres = GARCH(ret[1:], p=1, q=1).fit()
    print("GARCH(1,1): persistence=%.3f  vol next=%.3f"
          % (gres.persistence, math.sqrt(gres.conditional_variance[-1])))

    # --- differencing round-trip (exactness gate) -------------------------
    dtest = Differencer(d=1, D=1, m=12)
    rec = dtest.inverse_transform(dtest.fit_transform(y))
    err = max(abs(y[i] - rec[i]) for i in range(len(y)))
    print("differencing round-trip max error: %.2e" % err)

    print("=" * 60)
    print("  demo OK")


__all__ = [
    "__version__",
    # special functions / RNG
    "norm_cdf", "norm_ppf", "norm_pdf", "t_cdf", "t_ppf", "chi2_cdf", "chi2_sf",
    "chi2_ppf", "f_sf", "gammainc", "betainc", "DeterministicRNG",
    # linear algebra
    "matmul", "matvec", "transpose", "outer", "eye", "zeros", "kron",
    "frobenius_norm", "solve", "inv", "logdet", "cholesky", "chol_solve",
    "is_pos_def", "qr_householder", "qr_lstsq", "levinson_durbin",
    "solve_discrete_lyapunov", "companion_matrix", "spectral_radius",
    "poly_mul", "durand_kerner_roots",
    # optimisers / maps
    "minimize_nelder_mead", "pacf_transform", "pacf_untransform",
    "softmax_simplex", "logit", "expit", "OptResult",
    # records
    "ForecastResult", "ICResult", "AcfResult", "LjungBoxResult",
    "NormalityTestResult", "HetTestResult", "UnitRootTestResult",
    "SeasonalPeriodResult", "DecompositionResult", "ForecastMetrics",
    "BacktestResult", "DMResult", "ModelComparison", "AutoARIMAResult",
    "ARIMAResults", "ARFitResult", "ARMAFitResult", "ETSResult",
    "StructuralResult", "StateSpaceModel", "KalmanFilterOutput", "GARCHResult",
    "IRFResult", "FEVDResult", "GrangerCausalityResult", "OrderSelectionResult",
    "VARResults",
    # information criteria
    "count_params", "gaussian_loglike", "information_criteria", "compare_models",
    # ts kernels
    "autocovariance", "acf", "pacf", "ccf", "ljung_box", "durbin_watson",
    "jarque_bera", "heteroskedasticity_test", "diagnose_residuals",
    # transforms / differencing
    "box_cox", "box_cox_inverse", "guerrero_lambda", "Standardizer", "detrend",
    "Differencer", "diff", "integrate", "fourier_terms",
    # decomposition
    "moving_average", "seasonal_decompose", "stl_decompose", "seasonal_strength",
    "trend_strength",
    # unit-root / order
    "adf_test", "kpss_test", "ndiffs", "nsdiffs", "detect_seasonal_period",
    # psi / intervals
    "psi_weights", "pi_weights", "forecast_error_variance",
    "analytic_forecast_intervals", "bootstrap_forecast_intervals",
    # state space
    "kalman_filter", "kalman_smoother", "simulate_ssm", "arima_ssm",
    # arima estimators + class
    "yule_walker", "burg", "hannan_rissanen", "css_estimate", "ARIMA",
    # ETS / theta
    "SimpleExpSmoothing", "Holt", "ExponentialSmoothing", "auto_ets",
    "ThetaModel",
    # structural
    "local_level", "local_linear_trend",
    # VAR
    "VAR", "var_select_order",
    # volatility
    "GARCH",
    # auto
    "auto_arima", "grid_search_arima",
    # baselines / registry
    "BaseForecaster", "MeanForecaster", "NaiveForecaster",
    "SeasonalNaiveForecaster", "DriftForecaster", "MODEL_REGISTRY",
    "register_model", "make_forecaster",
    # metrics
    "mae", "mse", "rmse", "me", "mape", "mdape", "wmape", "smape", "mase",
    "rmsse", "pinball_loss", "interval_score", "msis", "empirical_coverage",
    "theil_u", "evaluate", "score",
    # backtest / ensemble
    "RollingOriginSplitter", "train_test_split", "backtest", "diebold_mariano",
    "combine_forecasts", "EnsembleForecaster", "auto_select",
    # simulators
    "simulate_sarima", "simulate_series", "simulate_arma",
    # payloads
    "build_result_payload", "build_a2ui_payload",
]


if __name__ == "__main__":
    _demo()
