"""
State-Space Models & Dynamic Linear Models -- a self-contained, zero-dependency
toolkit for *general* linear (and nonlinear / non-Gaussian) state-space time-series
modelling, filtering, smoothing and learning, implemented from scratch on the
Python standard library.

Where the sibling *autoregressive* module uses a state-space form only as an
internal engine for exact-ML ARIMA, and the *tracking* modules use filters for
physical object tracking (motion models, data association, multi-target), THIS
module owns the general **Dynamic Linear Model / Structural Time Series /
Unobserved-Components** framework -- the Durbin-Koopman / West-Harrison / KFAS /
``statsmodels.tsa.statespace`` world -- as a first-class modelling toolbox:

  * **General linear-Gaussian state space** with fully *time-varying* system
    matrices ``y_t = d_t + Z_t a_t + e_t``, ``a_{t+1} = c_t + T_t a_t + R_t n_t``,
    ``e_t ~ N(0,H_t)``, ``n_t ~ N(0,Q_t)``.
  * **Kalman filter** -- multivariate, robust to *missing observations*, with the
    **exact diffuse initialisation** of Koopman-Durbin for non-stationary states,
    a concentrated scale, and the prediction-error-decomposition log-likelihood.
  * **Smoothers** -- RTS/fast state smoother, the **disturbance smoother**
    (de Jong) for smoothed state & observation disturbances, the **simulation
    smoother** (Durbin-Koopman 2002), and **auxiliary residuals** for outlier /
    structural-break detection.
  * **Composable unobserved components** -- local level, (damped) local linear
    trend, dummy & trigonometric seasonal, stochastic cycle, autoregressive,
    static & dynamic (time-varying-parameter) regression and irregular terms,
    stacked into one model and estimated by **maximum likelihood** or the
    **EM algorithm** (Shumway-Stoffer).
  * **Bayesian DLM** -- West-Harrison discount factors and **forward-filtering
    backward-sampling** (FFBS).
  * **Nonlinear / non-Gaussian filters** -- Extended (EKF) and Unscented (UKF)
    Kalman filters and a bootstrap **particle filter** (SIR) for user-supplied
    transition / observation functions.
  * **Forecasting & signal extraction** -- multi-step forecasts with prediction
    variances, smoothed trend / seasonal / cycle extraction with confidence bands,
    residual diagnostics and strict-JSON competition payloads.

Everything -- the dense linear algebra (LU, Cholesky, QR, least squares, matrix
inverse / log-determinant, discrete Lyapunov), the special functions (normal /
Student-t / chi-square via error, gamma and beta functions), the optimiser and the
pseudo-random generator (including correlated multivariate-normal sampling) -- is
implemented on ``math``, ``cmath``, ``random``, ``itertools`` and ``statistics``
only, so it drops straight into a competition / hackathon / online-judge / embedded
sandbox: just copy this one file.

Contents
--------
  0.  Scaffold: version, constants, type aliases
  1.  Special functions + deterministic RNG (normal / t / chi2, MVN sampling)
  2.  Compensated statistics helpers
  3.  Dense linear algebra (LU, Cholesky, QR, solve, inv, logdet, Lyapunov, nearest-PD)
  4.  Result records + strict-JSON helper
  5.  LinearGaussianSSM: general time-varying state-space model + validation
  6.  Kalman filter (multivariate, missing data, exact diffuse, concentrated)
  7.  Smoothers: state (RTS/fast), disturbance, simulation; auxiliary residuals
  8.  Log-likelihood, information criteria, MLE + EM parameter estimation
  9.  Unobserved-components: composable structural components + builder
 10.  Bayesian DLM: discount factors + forward-filtering backward-sampling
 11.  Nonlinear / non-Gaussian filters: EKF, UKF, particle filter
 12.  Forecasting, signal extraction, in-sample prediction
 13.  Diagnostics: standardized & auxiliary residuals, portmanteau, CUSUM
 14.  Model presets (local level / trend / BSM, ARMA & VAR to state space)
 15.  Simulators
 16.  Payloads + demo + __all__

Every model option is keyword-only.  The demo is byte-deterministic across
``PYTHONHASHSEED``.
"""

from __future__ import annotations

import cmath
import itertools
import math
import random
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple, Union)

# ===========================================================================
# 0. SCAFFOLD
# ===========================================================================
__version__ = "1.0.0"

_EPS = 1e-12
_INF = float("inf")
_NEG_INF = float("-inf")
_LOG2PI = math.log(2.0 * math.pi)
_BIG = 1e4                       # diffuse-prior variance for the "big-kappa" fallback
#   (relative to the data magnitude via _obs_scale).  It must be large enough that
#   the diffuse-prior approximation is uninformative, but NOT so large that mixing
#   the ~kappa predicted covariance with the ~1/kappa smoother information matrix
#   loses precision: kappa ~ 1e9 (the old 1e7 * a large obs-scale) makes the
#   smoothed COVARIANCE of a multi-diffuse-state model (BSM / seasonal) catastrophically
#   cancel in V = P - P N P and go NEGATIVE.  1e4 is the best fixed compromise
#   (validated vs the statsmodels EXACT-diffuse smoother): ~1e-6 smoothed states and
#   ~1e-6 covariances for few-diffuse-state models, and a bounded ~7% diffuse-phase
#   covariance error for a 13-state BSM (the exact-diffuse smoother would zero it out
#   -- the residual is an inherent limitation of the big-kappa smoother).

Vector = List[float]
Matrix = List[List[float]]
# a system matrix may be constant, one-per-time-step, or a callable of t
TimeVarying = Union[Matrix, Vector, float, Sequence[Any], Callable[[int], Any]]


# ===========================================================================
# 1. SPECIAL FUNCTIONS + DETERMINISTIC RNG
# ===========================================================================
def norm_pdf(x: float) -> float:
    """Standard-normal density."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    """Standard-normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Standard-normal quantile (inverse CDF): Acklam's rational approximation
    refined by one Halley step -- accurate to ~1e-15 across the tails that the
    95%/99% prediction and confidence bands rely on."""
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
    e = norm_cdf(x) - p                                  # Halley refinement
    u = e * math.sqrt(2.0 * math.pi) * math.exp(0.5 * x * x)
    x -= u / (1.0 + 0.5 * x * u)
    return x


def _gammaln(x: float) -> float:
    return math.lgamma(x)


def gammainc(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x) via a series / continued
    fraction -- the workhorse behind the chi-square CDF."""
    if x <= 0.0 or a <= 0.0:
        return 0.0
    if x < a + 1.0:                                       # series expansion
        term = 1.0 / a
        total = term
        n = a
        for _ in range(500):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return total * math.exp(-x + a * math.log(x) - _gammaln(a))
    # Lentz continued fraction for Q, then P = 1 - Q
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    dd = 1.0 / b
    h = dd
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        dd = an * dd + b
        if abs(dd) < tiny:
            dd = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        dd = 1.0 / dd
        delta = dd * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - _gammaln(a)) * h


def chi2_cdf(x: float, *, df: float) -> float:
    """Chi-square CDF."""
    return gammainc(df / 2.0, x / 2.0) if x > 0.0 else 0.0


def chi2_sf(x: float, *, df: float) -> float:
    """Chi-square survival function 1 - CDF."""
    return 1.0 - chi2_cdf(x, df=df)


def chi2_ppf(p: float, *, df: float) -> float:
    """Chi-square quantile via bisection on the CDF."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return _INF
    lo, hi = 0.0, max(df * 4.0, 10.0)
    while chi2_cdf(hi, df=df) < p:
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi2_cdf(mid, df=df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b) -- backs the Student-t CDF."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lead = math.exp(_gammaln(a + b) - _gammaln(a) - _gammaln(b)
                    + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return lead * _betacf(a, b, x) / a
    return 1.0 - lead * _betacf(b, a, 1.0 - x) / b


def t_cdf(x: float, *, df: float) -> float:
    """Student-t CDF."""
    if df <= 0.0:
        return float("nan")
    xt = df / (df + x * x)
    ib = 0.5 * betainc(df / 2.0, 0.5, xt)
    return 1.0 - ib if x > 0.0 else ib


def t_ppf(p: float, *, df: float) -> float:
    """Student-t quantile via bisection (Newton would need the density)."""
    if p <= 0.0:
        return _NEG_INF
    if p >= 1.0:
        return _INF
    lo, hi = -1e6, 1e6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df=df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class DeterministicRNG:
    """A small, explicitly-seeded PRNG wrapper so every simulation, particle
    filter and FFBS draw is byte-reproducible regardless of ``PYTHONHASHSEED``.
    Wraps ``random.Random`` and adds Gaussian and multivariate-normal helpers."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def random(self) -> float:
        return self._rng.random()

    def uniform(self, lo: float = 0.0, hi: float = 1.0) -> float:
        return lo + (hi - lo) * self._rng.random()

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        return self._rng.gauss(mu, sigma)

    def normal_vector(self, n: int) -> Vector:
        return [self._rng.gauss(0.0, 1.0) for _ in range(n)]

    def multivariate_normal(self, mean: Sequence[float], cov: Matrix,
                            *, chol: Optional[Matrix] = None) -> Vector:
        """Draw x ~ N(mean, cov) via a Cholesky factor of cov (supplied or
        computed with a small jitter for positive-semidefiniteness)."""
        n = len(mean)
        if n == 0:
            return []
        if chol is not None:
            L = chol
        else:
            # scale-relative PSD jitter: a fixed 1e-12 is scale-blind and inflates
            # a legitimately small covariance (small-unit data -> variances ~ 1e-12),
            # corrupting the sampled spread of FFBS / simulation-smoother draws.
            _cs = max((abs(cov[i][i]) for i in range(n)), default=0.0)
            L = cholesky(cov, jitter=(1e-12 * _cs) or 1e-12)
        z = self.normal_vector(n)
        return [mean[i] + _kahan_sum(L[i][j] * z[j] for j in range(i + 1))
                for i in range(n)]

    def choice_index(self, weights: Sequence[float]) -> int:
        """Sample an index in proportion to non-negative ``weights``."""
        total = _kahan_sum(weights)
        u = self._rng.random() * total
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if u <= acc:
                return i
        return len(weights) - 1


# ===========================================================================
# 2. COMPENSATED STATISTICS HELPERS
# ===========================================================================
def _kahan_sum(values: Any) -> float:
    """Neumaier-compensated summation -- keeps long recursions numerically tight."""
    total = 0.0
    comp = 0.0
    for v in values:
        t = total + v
        if abs(total) >= abs(v):
            comp += (total - t) + v
        else:
            comp += (v - t) + total
        total = t
    return total + comp


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
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


def _quantile(xs: Sequence[float], q: float) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    if n == 1:
        return s[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if abs(b) > _EPS else default


# ===========================================================================
# 3. DENSE LINEAR ALGEBRA
# ===========================================================================
def zeros(m: int, n: int) -> Matrix:
    return [[0.0] * n for _ in range(m)]


def eye(n: int, scale: float = 1.0) -> Matrix:
    return [[scale if i == j else 0.0 for j in range(n)] for i in range(n)]


def transpose(A: Matrix) -> Matrix:
    if not A:
        return []
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def matmul(A: Matrix, B: Matrix) -> Matrix:
    if not A or not B:
        return []
    BT = transpose(B)                                    # columns of B as rows
    return [[sum(a * b for a, b in zip(row, col)) for col in BT] for row in A]


def matvec(A: Matrix, x: Sequence[float]) -> Vector:
    return [sum(a * xi for a, xi in zip(row, x)) for row in A]


def vecmat(x: Sequence[float], A: Matrix) -> Vector:
    m = len(A[0]) if A else 0
    return [sum(x[i] * A[i][j] for i in range(len(x))) for j in range(m)]


def mat_add(A: Matrix, B: Matrix) -> Matrix:
    return [[A[i][j] + B[i][j] for j in range(len(A[i]))] for i in range(len(A))]


def mat_sub(A: Matrix, B: Matrix) -> Matrix:
    return [[A[i][j] - B[i][j] for j in range(len(A[i]))] for i in range(len(A))]


def mat_scale(A: Matrix, s: float) -> Matrix:
    return [[s * v for v in row] for row in A]


def outer(x: Sequence[float], y: Sequence[float]) -> Matrix:
    return [[xi * yj for yj in y] for xi in x]


def vec_add(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [a[i] + b[i] for i in range(len(a))]


def vec_sub(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [a[i] - b[i] for i in range(len(a))]


def vec_scale(a: Sequence[float], s: float) -> Vector:
    return [s * v for v in a]


def diag(vals: Sequence[float]) -> Matrix:
    n = len(vals)
    return [[vals[i] if i == j else 0.0 for j in range(n)] for i in range(n)]


def symmetrize(A: Matrix) -> Matrix:
    """Average with the transpose to kill accumulated asymmetry in covariances."""
    n = len(A)
    return [[0.5 * (A[i][j] + A[j][i]) for j in range(n)] for i in range(n)]


def kron(A: Matrix, B: Matrix) -> Matrix:
    ra, ca = len(A), len(A[0]) if A else 0
    rb, cb = len(B), len(B[0]) if B else 0
    out = zeros(ra * rb, ca * cb)
    for i in range(ra):
        for j in range(ca):
            for k in range(rb):
                for l in range(cb):
                    out[i * rb + k][j * cb + l] = A[i][j] * B[k][l]
    return out


def block_diag(*blocks: Matrix) -> Matrix:
    rows = sum(len(b) for b in blocks)
    cols = sum(len(b[0]) if b else 0 for b in blocks)
    out = zeros(rows, cols)
    r = c = 0
    for b in blocks:
        for i in range(len(b)):
            for j in range(len(b[0]) if b else 0):
                out[r + i][c + j] = b[i][j]
        r += len(b)
        c += len(b[0]) if b else 0
    return out


def lu_decompose(A: Matrix) -> Tuple[Matrix, List[int], int]:
    """LU with partial pivoting.  Returns (LU, piv, sign)."""
    n = len(A)
    lu = [row[:] for row in A]
    piv = list(range(n))
    sign = 1
    # Scale-relative singular-pivot floor.  An ABSOLUTE 1e-12 is scale-blind: a
    # legitimately small-magnitude matrix (e.g. an innovation covariance F ~ 1e-16
    # for data in tiny units) has every pivot below it, so each would be replaced
    # by 1e-12 -- inflating the inverse by orders of magnitude and silently
    # corrupting the result.  Scale by the matrix magnitude, capped at the historic
    # 1e-12 so normal / large-scale matrices decompose byte-identically.
    amax = 0.0
    for row in lu:
        for val in row:
            av = abs(val)
            if av > amax:
                amax = av
    piv_tol = _EPS * min(1.0, amax) if amax > 0.0 else _EPS
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(lu[i][k]))
        if abs(lu[p][k]) < piv_tol:
            lu[p][k] = piv_tol if lu[p][k] >= 0 else -piv_tol
        if p != k:
            lu[k], lu[p] = lu[p], lu[k]
            piv[k], piv[p] = piv[p], piv[k]
            sign = -sign
        for i in range(k + 1, n):
            lu[i][k] /= lu[k][k]
            f = lu[i][k]
            for j in range(k + 1, n):
                lu[i][j] -= f * lu[k][j]
    return lu, piv, sign


def solve(A: Matrix, b: Sequence[float], *, ridge: float = 0.0) -> Vector:
    """Solve A x = b (optionally ridge-regularised) via LU with partial pivoting."""
    n = len(A)
    if n == 0:
        return []
    M = [row[:] for row in A]
    if ridge:
        for i in range(n):
            M[i][i] += ridge
    lu, piv, _ = lu_decompose(M)
    x = [b[piv[i]] for i in range(n)]
    for i in range(n):                                   # forward substitution
        for j in range(i):
            x[i] -= lu[i][j] * x[j]
    for i in range(n - 1, -1, -1):                       # back substitution
        for j in range(i + 1, n):
            x[i] -= lu[i][j] * x[j]
        x[i] /= lu[i][i]
    return x


def solve_matrix(A: Matrix, B: Matrix, *, ridge: float = 0.0) -> Matrix:
    """Solve A X = B for a matrix right-hand side (column by column)."""
    BT = transpose(B)
    cols = [solve(A, col, ridge=ridge) for col in BT]
    return transpose(cols)


def inv(A: Matrix, *, ridge: float = 0.0) -> Matrix:
    n = len(A)
    if n == 0:
        return []
    return solve_matrix(A, eye(n), ridge=ridge)


def logdet(A: Matrix) -> float:
    """Natural log of |det(A)| for a (numerically) positive matrix via LU."""
    n = len(A)
    if n == 0:
        return 0.0
    lu, _, _ = lu_decompose(A)
    return _kahan_sum(math.log(abs(lu[i][i])) for i in range(n))


def cholesky(A: Matrix, *, jitter: float = 0.0) -> Matrix:
    """Lower-triangular Cholesky factor L with L L' = A.  A small ``jitter`` (or an
    automatic escalation) is added to the diagonal to survive semi-definiteness."""
    n = len(A)
    if n == 0:
        return []
    add = jitter
    for _ in range(40):
        L = zeros(n, n)
        ok = True
        for i in range(n):
            for j in range(i + 1):
                s = A[i][j] + (add if i == j else 0.0) \
                    - _kahan_sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
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
        add = max(add * 10.0, 1e-12)
    return eye(n, math.sqrt(max(add, _EPS)))              # last-resort diagonal


def chol_solve(L: Matrix, b: Sequence[float]) -> Vector:
    """Solve (L L') x = b given the lower Cholesky factor L."""
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - _kahan_sum(L[i][k] * y[k] for k in range(i))) / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - _kahan_sum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
    return x


def qr_householder(A: Matrix) -> Tuple[Matrix, Matrix]:
    """Householder QR of an m x n matrix (m >= n).  Returns (Q, R)."""
    m = len(A)
    n = len(A[0]) if A else 0
    R = [row[:] for row in A]
    Q = eye(m)
    for k in range(min(m, n)):
        x = [R[i][k] for i in range(k, m)]
        alpha = -math.copysign(math.sqrt(_kahan_sum(v * v for v in x)) or 0.0, x[0] or 1.0)
        if abs(alpha) < _EPS:
            continue
        v = x[:]
        v[0] -= alpha
        vnorm2 = _kahan_sum(vi * vi for vi in v)
        if vnorm2 < _EPS:
            continue
        for j in range(n):                               # R <- (I - 2vv'/v'v) R
            dot = _kahan_sum(v[i] * R[k + i][j] for i in range(m - k))
            f = 2.0 * dot / vnorm2
            for i in range(m - k):
                R[k + i][j] -= f * v[i]
        for j in range(m):                               # Q <- Q (I - 2vv'/v'v)
            dot = _kahan_sum(v[i] * Q[j][k + i] for i in range(m - k))
            f = 2.0 * dot / vnorm2
            for i in range(m - k):
                Q[j][k + i] -= f * v[i]
    return Q, R


def qr_lstsq(A: Matrix, b: Sequence[float], *, ridge: float = 0.0) -> Tuple[Vector, int]:
    """Least-squares solve of A x ~= b via Householder QR (ridge optional).
    Returns (x, numerical_rank)."""
    m = len(A)
    n = len(A[0]) if A else 0
    if ridge > 0.0:
        A = [row[:] for row in A] + [[math.sqrt(ridge) if i == j else 0.0
                                      for j in range(n)] for i in range(n)]
        b = list(b) + [0.0] * n
        m += n
    Q, R = qr_householder(A)
    qtb = matvec(transpose(Q), b)
    x = [0.0] * n
    rank = 0
    for i in range(n - 1, -1, -1):
        if abs(R[i][i]) < 1e-10:
            x[i] = 0.0
            continue
        rank += 1
        s = qtb[i] - _kahan_sum(R[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / R[i][i]
    return x, rank


def nearest_pd(A: Matrix, *, eps: float = 1e-10) -> Matrix:
    """Symmetrise and floor tiny/negative pivots so a covariance stays usable.

    The floor is scaled by the matrix magnitude (capped at ``eps`` so normal /
    large-scale covariances are floored identically to before): an ABSOLUTE floor
    is scale-blind and would inflate a legitimately small covariance (variances
    ~1e-12 for data in tiny units) up to ``eps``, corrupting any draw taken from
    it."""
    S = symmetrize(A)
    n = len(S)
    scale = 0.0
    for i in range(n):
        if S[i][i] > scale:
            scale = S[i][i]
    floor = eps * min(1.0, scale) if scale > 0.0 else eps
    for i in range(n):
        if S[i][i] < floor:
            S[i][i] = floor
    return S


def solve_discrete_lyapunov(T: Matrix, Q: Matrix, *, max_iter: int = 200,
                            tol: float = 1e-13) -> Matrix:
    """Solve P = T P T' + Q for symmetric P (stationary state covariance).  Uses
    the exact vectorised solve when T is a small stable matrix, else Smith's
    doubling iteration (O(n^3 log 1/tol))."""
    n = len(T)
    if n == 0:
        return []
    if n <= 8 and spectral_radius(T) < 1.0 - 1e-8:
        K = kron(T, T)
        n2 = n * n
        A = [[(1.0 if i == j else 0.0) - K[i][j] for j in range(n2)]
             for i in range(n2)]
        vecQ = [Q[i][j] for i in range(n) for j in range(n)]
        vecP = solve(A, vecQ, ridge=1e-14)
        P = [[vecP[i * n + j] for j in range(n)] for i in range(n)]
    else:
        P = [row[:] for row in Q]
        Tk = [row[:] for row in T]
        for _ in range(max_iter):
            Pn = mat_add(P, matmul(matmul(Tk, P), transpose(Tk)))
            Tk = matmul(Tk, Tk)
            if _frob(mat_sub(Pn, P)) < tol * (_frob(P) + 1.0):
                P = Pn
                break
            P = Pn
    return symmetrize(P)


def _frob(A: Matrix) -> float:
    return math.sqrt(_kahan_sum(v * v for row in A for v in row))


def spectral_radius(A: Matrix) -> float:
    """Largest |eigenvalue| via power iteration on A'A (returns the spectral norm
    upper bound's sqrt is avoided -- we iterate A directly for the dominant mode,
    then fall back to the characteristic polynomial roots for small A)."""
    n = len(A)
    if n == 0:
        return 0.0
    if n <= 4:
        roots = _companion_eigenvalues(A)
        return max((abs(r) for r in roots), default=0.0)
    v = [1.0] * n
    last = 0.0
    for _ in range(300):
        w = matvec(A, v)
        nrm = math.sqrt(_kahan_sum(x * x for x in w))
        if nrm < _EPS:
            return 0.0
        v = [x / nrm for x in w]
        val = _kahan_sum(v[i] * matvec(A, v)[i] for i in range(n))
        if abs(val - last) < 1e-12:
            break
        last = val
    return abs(last)


def _companion_eigenvalues(A: Matrix) -> List[complex]:
    """Eigenvalues of a small matrix via the characteristic polynomial
    (Faddeev-LeVerrier) + Durand-Kerner root finder."""
    n = len(A)
    coeffs = _charpoly(A)                                 # ascending powers
    return durand_kerner_roots(coeffs)


def _charpoly(A: Matrix) -> List[float]:
    """Characteristic polynomial coefficients (ascending powers) via
    Faddeev-LeVerrier.  Returns [c0, c1, ..., cn] for det(xI - A)."""
    n = len(A)
    M = zeros(n, n)
    c = [0.0] * (n + 1)
    c[n] = 1.0
    for k in range(1, n + 1):
        M = mat_add(matmul(A, M), eye(n, c[n - k + 1]))
        AM = matmul(A, M)
        trace = _kahan_sum(AM[i][i] for i in range(n))
        c[n - k] = -trace / k
    return c


def durand_kerner_roots(coefs: Sequence[float], *, max_iter: int = 500,
                        tol: float = 1e-13,
                        seed: complex = 0.4 + 0.9j) -> List[complex]:
    """All complex roots of a real polynomial (ascending powers) via the
    Weierstrass / Durand-Kerner simultaneous iteration (deterministic seed)."""
    c = list(coefs)
    while len(c) > 1 and abs(c[-1]) < _EPS:
        c.pop()
    if len(c) <= 1:
        return []
    lead = c[-1]
    c = [ci / lead for ci in c]
    deg = len(c) - 1
    roots = [seed ** k for k in range(deg)]

    def peval(z: complex) -> complex:
        acc = 0j
        for ci in reversed(c):
            acc = acc * z + ci
        return acc

    for _ in range(max_iter):
        delta = 0.0
        new = roots[:]
        for i in range(deg):
            zi = roots[i]
            denom = 1.0 + 0j
            for j in range(deg):
                if j != i:
                    denom *= (zi - roots[j])
            if abs(denom) < 1e-300:
                denom = 1e-300
            step = peval(zi) / denom
            new[i] = zi - step
            delta = max(delta, abs(step))
        roots = new
        if delta < tol:
            break
    return roots


# ===========================================================================
# 4. RESULT RECORDS + STRICT-JSON HELPER
# ===========================================================================
def _json_safe(obj: Any) -> Any:
    """Recursively coerce to strict JSON: round floats to 6 dp and map non-finite
    values to ``None`` so payloads never contain NaN / inf."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else round(obj, 6)
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# ===========================================================================
# 5. LINEAR GAUSSIAN STATE-SPACE MODEL
# ===========================================================================
def _is_matrix(x: Any) -> bool:
    return (isinstance(x, list) and len(x) > 0 and isinstance(x[0], list)
            and (len(x[0]) == 0 or not isinstance(x[0][0], list)))


def _is_vector(x: Any) -> bool:
    return isinstance(x, list) and (len(x) == 0 or isinstance(x[0], (int, float)))


def _mat_at(x: TimeVarying, t: int) -> Matrix:
    """Return a (possibly time-varying) system matrix at time ``t``."""
    if callable(x):
        return x(t)
    if _is_matrix(x):
        return x                                         # constant matrix
    return x[t]                                          # sequence of matrices


def _vec_at(x: Optional[TimeVarying], t: int) -> Optional[Vector]:
    if x is None:
        return None
    if callable(x):
        return x(t)
    if _is_vector(x):
        return x                                         # constant vector
    return x[t]                                          # sequence of vectors


@dataclass
class LinearGaussianSSM:
    """General linear-Gaussian state-space model in Durbin-Koopman form::

        y_t     = d_t + Z_t alpha_t + eps_t,     eps_t ~ N(0, H_t)
        alpha_1 = a1 (+ diffuse P_inf),          alpha_t ~ N(a1, P1)
        alpha_{t+1} = c_t + T_t alpha_t + R_t eta_t,  eta_t ~ N(0, Q_t)

    Every system matrix (``Z, H, T, R, Q`` and the optional intercepts ``d, c``)
    may be given as a constant matrix/vector, a per-time sequence, or a callable of
    the time index ``t`` -- so time-varying-parameter, dynamic-regression and
    seasonal-of-changing-length models all fit one interface.  ``P1_infinity``
    holds the diffuse part of the initial covariance for non-stationary states.

    Dimensions: ``k_states`` (m), ``k_obs`` (p), ``k_posdef`` (r = state shocks).
    """
    Z: TimeVarying
    T: TimeVarying
    R: TimeVarying
    Q: TimeVarying
    H: TimeVarying
    a1: Vector
    P1: Matrix
    c: Optional[TimeVarying] = None
    d: Optional[TimeVarying] = None
    P1_infinity: Optional[Matrix] = None
    name: str = "ssm"
    k_states: int = field(init=False, default=0)
    k_obs: int = field(init=False, default=0)
    k_posdef: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        z0 = _mat_at(self.Z, 0)
        self.k_obs = len(z0)
        self.k_states = len(z0[0]) if z0 else len(self.a1)
        q0 = _mat_at(self.Q, 0)
        self.k_posdef = len(q0)
        self._validate()

    def _validate(self) -> None:
        m, p, r = self.k_states, self.k_obs, self.k_posdef
        if len(self.a1) != m:
            raise ValueError("a1 length %d != k_states %d" % (len(self.a1), m))
        if len(self.P1) != m or any(len(row) != m for row in self.P1):
            raise ValueError("P1 must be k_states x k_states (%d)" % m)
        T0 = _mat_at(self.T, 0)
        if len(T0) != m or any(len(row) != m for row in T0):
            raise ValueError("T must be k_states x k_states (%d)" % m)
        R0 = _mat_at(self.R, 0)
        if len(R0) != m or (R0 and len(R0[0]) != r):
            raise ValueError("R must be k_states x k_posdef (%d x %d)" % (m, r))
        z0 = _mat_at(self.Z, 0)
        if len(z0) != p or any(len(row) != m for row in z0):
            raise ValueError("Z must be k_obs x k_states (%d x %d)" % (p, m))
        H0 = _mat_at(self.H, 0)
        if len(H0) != p or any(len(row) != p for row in H0):
            raise ValueError("H must be k_obs x k_obs (%d x %d)" % (p, p))
        Q0 = _mat_at(self.Q, 0)
        if len(Q0) != r or any(len(row) != r for row in Q0):
            raise ValueError("Q must be k_posdef x k_posdef (%d x %d)" % (r, r))
        # a variance (diagonal of a covariance) cannot be negative -- otherwise the
        # filter silently returns a finite but meaningless log-likelihood.  Zero is
        # allowed (a deterministic component / no observation noise).
        for label, cov in (("P1", self.P1), ("Q", Q0), ("H", H0)):
            for i in range(len(cov)):
                if cov[i][i] < -_EPS:
                    raise ValueError("%s has a negative variance on its diagonal "
                                     "(%s[%d][%d] = %r)" % (label, label, i, i,
                                                            cov[i][i]))
        if self.P1_infinity is not None:
            if len(self.P1_infinity) != m:
                raise ValueError("P1_infinity must be k_states x k_states")

    # -- time-indexed accessors ------------------------------------------------
    def Z_t(self, t: int) -> Matrix:
        return _mat_at(self.Z, t)

    def T_t(self, t: int) -> Matrix:
        return _mat_at(self.T, t)

    def R_t(self, t: int) -> Matrix:
        return _mat_at(self.R, t)

    def Q_t(self, t: int) -> Matrix:
        return _mat_at(self.Q, t)

    def H_t(self, t: int) -> Matrix:
        return _mat_at(self.H, t)

    def c_t(self, t: int) -> Vector:
        v = _vec_at(self.c, t)
        return v if v is not None else [0.0] * self.k_states

    def d_t(self, t: int) -> Vector:
        v = _vec_at(self.d, t)
        return v if v is not None else [0.0] * self.k_obs

    def RQR_t(self, t: int) -> Matrix:
        """State-disturbance covariance in state space: R_t Q_t R_t'."""
        R = self.R_t(t)
        Q = self.Q_t(t)
        return matmul(matmul(R, Q), transpose(R))

    def initial_diffuse_rank(self) -> int:
        if self.P1_infinity is None:
            return 0
        return sum(1 for i in range(self.k_states)
                   if self.P1_infinity[i][i] > _EPS)

    def copy_with(self, **kw: Any) -> "LinearGaussianSSM":
        base = dict(Z=self.Z, T=self.T, R=self.R, Q=self.Q, H=self.H,
                    a1=self.a1, P1=self.P1, c=self.c, d=self.d,
                    P1_infinity=self.P1_infinity, name=self.name)
        base.update(kw)
        return LinearGaussianSSM(**base)


@dataclass
class KalmanFilterResult:
    """Output of :func:`kalman_filter` (Durbin-Koopman quantities)."""
    predicted_state: List[Vector]          # a_{t|t-1}
    predicted_cov: List[Matrix]            # P_{t|t-1}
    filtered_state: List[Vector]           # a_{t|t}
    filtered_cov: List[Matrix]             # P_{t|t}
    forecast_error: List[Vector]           # v_t (per observed dim, full length p, 0 if missing)
    forecast_error_cov: List[Matrix]       # F_t
    loglike: float
    scale: float                           # concentrated sigma^2 (1.0 if not concentrated)
    nobs: int                              # scalar observations that entered the likelihood
    n_diffuse: int                         # number of diffuse scalar observations
    # per (t, i) scalar records for the univariate smoother
    _scalars: List[Tuple[int, int, float, float, float, Vector, bool, Vector]] = \
        field(default_factory=list, repr=False)
    _model: Optional[LinearGaussianSSM] = field(default=None, repr=False)

    def standardized_residuals(self) -> List[Vector]:
        """One-step prediction errors standardised by their sd, per obs dim."""
        out = []
        for t in range(len(self.forecast_error)):
            v = self.forecast_error[t]
            F = self.forecast_error_cov[t]
            out.append([_safe_div(v[i], math.sqrt(F[i][i]) if F[i][i] > 0 else 0.0)
                        for i in range(len(v))])
        return out


# ===========================================================================
# 6. KALMAN FILTER  (univariate treatment, exact diffuse, missing data)
# ===========================================================================
def _decorrelated_rows(Z: Matrix, d: Vector, y: Sequence[float], H: Matrix
                       ) -> List[Tuple[Vector, float, float]]:
    """Return a list of scalar observation triples ``(z_row, y_value, h_var)`` for
    the univariate treatment.  Missing (NaN/None) rows are dropped.  A non-diagonal
    H is decorrelated by its Cholesky factor so the transformed observations have
    a diagonal (unit) covariance.

    Note: the Cholesky decorrelation loses precision when H is NEAR-SINGULAR (an
    almost rank-deficient / redundant observation, i.e. two measurements whose noise
    correlation is ~= 1).  This is inherent to the efficient univariate treatment;
    the likelihood is exact for realistic observation-noise correlations."""
    p = len(Z)
    obs_idx = [i for i in range(p)
               if y[i] is not None and not (isinstance(y[i], float) and math.isnan(y[i]))]
    if not obs_idx:
        return [], 0.0
    # Scale-RELATIVE diagonality test: an absolute floor (1e-12) is scale-blind and
    # would classify a genuinely CORRELATED H as diagonal for small-unit data (its
    # off-diagonals ~ variance ~ 1e-16), silently dropping the correlation.  A truly
    # diagonal H has exact-zero off-diagonals, so this is byte-identical there.
    hmax = max((abs(H[i][j]) for i in range(p) for j in range(p)), default=0.0)
    htol = _EPS * hmax
    diagonal = all(abs(H[i][j]) <= htol
                   for i in range(p) for j in range(p) if i != j)
    if diagonal:
        return ([([Z[i][k] for k in range(len(Z[i]))], float(y[i]) - d[i], H[i][i])
                 for i in obs_idx], 0.0)
    # non-diagonal: decorrelate the observed sub-block with its Cholesky factor.
    # y* = L^-1 (y - d) has unit covariance; the loglik is then computed in the y*
    # space, so the density of the ORIGINAL y needs the change-of-variables Jacobian
    # log|det(dy*/dy)| = -log|det(L)| = -0.5*log|H| added back (else the reported
    # loglik is off by exactly 0.5*log|H| for correlated observation noise).
    Zs = [Z[i] for i in obs_idx]
    ds = [d[i] for i in obs_idx]
    ys = [float(y[i]) for i in obs_idx]
    Hs = [[H[i][j] for j in obs_idx] for i in obs_idx]
    hs_scale = max((abs(Hs[i][i]) for i in range(len(Hs))), default=1.0)
    L = cholesky(Hs, jitter=1e-12 * max(hs_scale, _EPS))  # scale-relative jitter
    log_jac = -_kahan_sum(math.log(L[k][k]) for k in range(len(obs_idx)))
    ytil = chol_solve_lower(L, [ys[k] - ds[k] for k in range(len(obs_idx))])
    Ztil = [chol_solve_lower(L, [Zs[k][j] for k in range(len(obs_idx))])
            for j in range(len(Zs[0]))]
    Ztil = transpose(Ztil)                               # back to rows
    return ([(Ztil[k], ytil[k], 1.0) for k in range(len(obs_idx))], log_jac)


def chol_solve_lower(L: Matrix, b: Sequence[float]) -> Vector:
    """Forward-substitution solve L x = b for a lower-triangular L."""
    n = len(L)
    x = [0.0] * n
    for i in range(n):
        x[i] = (b[i] - _kahan_sum(L[i][k] * x[k] for k in range(i))) / L[i][i]
    return x


def _time_invariant(model: LinearGaussianSSM) -> bool:
    """True if no system matrix is time-varying (enables per-step matrix caching)."""
    for x in (model.Z, model.T, model.R, model.Q, model.H, model.c, model.d):
        if x is None:
            continue
        if callable(x):
            return False
        if _is_matrix(x) or _is_vector(x):
            continue
        return False                                     # sequence of matrices
    return True


def _check_obs_finite(y: Sequence[Sequence[float]], k_obs: Optional[int] = None) -> None:
    """Reject +/-inf observations (NaN / None are allowed and mean 'missing') and,
    if ``k_obs`` is given, dimension mismatches -- otherwise extra observation
    columns are silently ignored (the user unknowingly fits the wrong model)."""
    for row in y:
        if k_obs is not None and len(row) != k_obs:
            raise ValueError("each observation must have k_obs=%d component(s); "
                             "got %d" % (k_obs, len(row)))
        for v in row:
            if v is None:
                continue
            if isinstance(v, float) and math.isnan(v):
                continue
            if not math.isfinite(v):
                raise ValueError("observation contains a non-finite (inf) value; "
                                 "use NaN or None to encode a missing value")


def _obs_scale(y: Sequence[Sequence[float]]) -> float:
    """Mean square of the observed values (>= 1) -- used to scale the diffuse
    prior so smoothing stays offset/scale-equivariant."""
    return max(1.0, _var_scale(y))


def _var_scale(y: Sequence[Sequence[float]]) -> float:
    """Mean square of the observed values (0 if all missing), WITHOUT the >=1 floor.
    Used to make degeneracy tolerances / regularisation ridges scale-RELATIVE: an
    absolute floor is scale-blind and silently corrupts legitimately small-scale
    problems (tiny units -> tiny variances)."""
    tot = 0.0
    cnt = 0
    for row in y:
        for v in row:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            tot += v * v
            cnt += 1
    return tot / cnt if cnt else 0.0


def kalman_filter(model: LinearGaussianSSM, y: Sequence[Sequence[float]], *,
                  concentrate: bool = False, diffuse_method: str = "exact",
                  store: bool = True) -> KalmanFilterResult:
    """Kalman filter for a :class:`LinearGaussianSSM` using the numerically robust
    *univariate treatment* of Durbin-Koopman: each observation dimension is brought
    in sequentially (scalar innovations, no matrix inversions), which makes missing
    data and the **exact diffuse initialisation** for non-stationary states fall out
    naturally.  With ``concentrate`` a single scale ``sigma^2`` common to ``H`` and
    ``Q`` is profiled out of the likelihood.

    ``y`` is a sequence of length-``k_obs`` observation vectors; ``None`` / ``NaN``
    entries are treated as missing.  Returns a :class:`KalmanFilterResult`."""
    n = len(y)
    m = model.k_states
    _check_obs_finite(y, model.k_obs)
    # Scale-relative tolerance for a *degenerate* (numerically zero-variance)
    # innovation.  An ABSOLUTE floor (1e-12) is scale-blind: for a legitimately
    # small-scale problem (tiny units -> tiny variances) EVERY F* would fall below
    # it, so every update would be silently skipped and the log-likelihood would be
    # a meaningless 0.  Scale by the data magnitude, but never ABOVE the historical
    # 1e-12 -- so normal / large-scale problems (and the demo) are byte-identical
    # and only sub-unit-scale problems get the (smaller) relaxed threshold.
    f_tol = _EPS * min(1.0, max(_var_scale(y), _EPS))
    a = list(model.a1)
    P = [row[:] for row in model.P1]
    Pinf = ([row[:] for row in model.P1_infinity]
            if model.P1_infinity is not None else zeros(m, m))
    diffuse = model.P1_infinity is not None and any(
        Pinf[i][i] > _EPS for i in range(m))
    if diffuse and diffuse_method != "exact":
        # "big-kappa": fold the diffuse part into a large proper prior so the
        # ordinary recursion (and the standard smoother built on it) applies.  The
        # likelihood is then kappa-dependent (use diffuse_method='exact' for that);
        # this mode exists for the smoother, whose smoothed states/disturbances
        # converge to the exact diffuse ones as kappa -> infinity.  Kappa is scaled
        # by the data magnitude so the smoother stays offset/scale-equivariant.
        kappa = _BIG * _obs_scale(y)
        for i in range(m):
            for j in range(m):
                P[i][j] += kappa * Pinf[i][j]
        Pinf = zeros(m, m)
        diffuse = False

    pred_a: List[Vector] = []
    pred_P: List[Matrix] = []
    filt_a: List[Vector] = []
    filt_P: List[Matrix] = []
    verr: List[Vector] = []
    vcov: List[Matrix] = []
    scalars: List[Tuple[int, int, float, float, float, Vector, bool, Vector]] = []
    keep_scalars = store or concentrate

    ti = _time_invariant(model)
    if ti:                                               # cache constant matrices
        Z_c = model.Z_t(0); d_c = model.d_t(0); H_c = model.H_t(0)
        T_c = model.T_t(0); Tt_c = transpose(T_c)
        cc_c = model.c_t(0); RQR_c = model.RQR_t(0)

    loglike = 0.0
    sum_v2F = 0.0
    n_used = 0
    n_diffuse = 0
    jac_total = 0.0                                       # non-diagonal-H decorrelation

    for t in range(n):
        if store:
            pred_a.append(list(a))
            pred_P.append([row[:] for row in P])
        Z = Z_c if ti else model.Z_t(t)
        d = d_c if ti else model.d_t(t)
        H = H_c if ti else model.H_t(t)
        rows, log_jac = _decorrelated_rows(Z, d, y[t], H)
        loglike += log_jac                               # correlated-noise Jacobian
        jac_total += log_jac
        for (zrow, yval, hvar) in rows:
            zA = _kahan_sum(zrow[k] * a[k] for k in range(m))
            v = yval - zA
            M = matvec(P, zrow)                          # P z'
            Fstar = _kahan_sum(zrow[k] * M[k] for k in range(m)) + hvar
            if diffuse:
                Minf = matvec(Pinf, zrow)
                Finf = _kahan_sum(zrow[k] * Minf[k] for k in range(m))
            else:
                Finf = 0.0
            if diffuse and Finf > _EPS:                  # exact-diffuse update
                inv_i = 1.0 / Finf
                for k in range(m):
                    a[k] += Minf[k] * v * inv_i
                fss = Fstar * inv_i * inv_i
                for i in range(m):                       # P_star (in place)
                    Mi, Mfi, Prow = M[i], Minf[i], P[i]
                    for j in range(m):
                        Prow[j] += (Mfi * Minf[j] * fss
                                    - (Mi * Minf[j] + Mfi * M[j]) * inv_i)
                for i in range(m):                       # P_inf (in place)
                    Mfi, Pir = Minf[i], Pinf[i]
                    for j in range(m):
                        Pir[j] -= Mfi * Minf[j] * inv_i
                loglike += -0.5 * math.log(Finf)         # NO log(2*pi) when diffuse
                # NOTE: this exact-diffuse term matches statsmodels to ~1e-7 for the
                # local level / linear trend (F_inf == 1 -> 0 contribution).  For the
                # DUMMY-seasonal representation the diffuse-resolution F_inf products
                # equal period^2, so the reported log-likelihood carries a constant
                # -log(period) offset vs statsmodels (a diffuse-prior normalisation
                # convention).  It is param-INDEPENDENT, so the MLE, fitted params,
                # smoothed states and forecasts are unaffected and match statsmodels;
                # only cross-period AIC comparisons of dummy-seasonal models see the
                # constant.  Compare seasonal AICs only WITHIN a fixed period.
                n_diffuse += 1
                if keep_scalars:
                    scalars.append((t, 0, v, Fstar, Finf, list(zrow), True, list(Minf)))
            else:                                        # ordinary update
                if Fstar <= f_tol:                        # scale-relative degeneracy
                    if keep_scalars:
                        scalars.append((t, 0, v, f_tol, 0.0, list(zrow), False, list(M)))
                    continue
                inv_f = 1.0 / Fstar
                for k in range(m):
                    a[k] += M[k] * v * inv_f
                for i in range(m):                       # P (in place)
                    Mi, Prow = M[i] * inv_f, P[i]
                    for j in range(m):
                        Prow[j] -= Mi * M[j]
                loglike += -0.5 * (_LOG2PI + math.log(Fstar) + v * v * inv_f)
                sum_v2F += v * v * inv_f
                n_used += 1
                if keep_scalars:
                    scalars.append((t, 0, v, Fstar, 0.0, list(zrow), False, list(M)))
        P = symmetrize(P)
        if store:
            v_full, F_full = _report_innovation(model, t, y[t], pred_a[t], pred_P[t])
            verr.append(v_full)
            vcov.append(F_full)
            filt_a.append(list(a))
            filt_P.append([row[:] for row in P])
        if diffuse:
            Pinf = symmetrize(Pinf)
            if all(abs(Pinf[i][j]) < 1e-10 for i in range(m) for j in range(m)):
                diffuse = False
        # time update (predict next state)
        T = T_c if ti else model.T_t(t)
        Tt = Tt_c if ti else transpose(T)
        cc = cc_c if ti else model.c_t(t)
        RQR = RQR_c if ti else model.RQR_t(t)
        a = [cc[k] + _kahan_sum(T[k][j] * a[j] for j in range(m)) for k in range(m)]
        P = symmetrize(mat_add(matmul(matmul(T, P), Tt), RQR))
        if diffuse:
            Pinf = symmetrize(matmul(matmul(T, Pinf), Tt))

    scale = 1.0
    if concentrate and n_used > 0:
        scale = max(sum_v2F / n_used, _EPS)
        # concentrated log-likelihood: replace the quadratic forms by n_used and add
        # the scale term (log|F| already accumulated with sigma^2 folded in the vars)
        loglike = 0.0
        # recompute with concentrated scale from the stored ordinary scalars
        for (t, _i, v, Fstar, Finf, zrow, is_dif, _M) in scalars:
            if is_dif:
                loglike += -0.5 * math.log(max(Finf, _EPS))
            elif Fstar > f_tol:
                loglike += -0.5 * (_LOG2PI + math.log(Fstar) + 1.0
                                   + math.log(scale))
        loglike += jac_total                             # correlated-noise Jacobian

    res = KalmanFilterResult(pred_a, pred_P, filt_a, filt_P, verr, vcov,
                             loglike, scale, n_used, n_diffuse)
    res._scalars = scalars
    res._model = model
    return res


def _report_innovation(model: LinearGaussianSSM, t: int, yt: Sequence[float],
                       a_pred: Vector, P_pred: Matrix) -> Tuple[Vector, Matrix]:
    """Compute the (multivariate) one-step prediction error v_t and its covariance
    F_t at the predicted state, for reporting / diagnostics.  Missing dims -> 0."""
    Z = model.Z_t(t)
    d = model.d_t(t)
    H = model.H_t(t)
    p = len(Z)
    m = model.k_states
    v = [0.0] * p
    F = [[H[i][j] for j in range(p)] for i in range(p)]
    ZP = [matvec(P_pred, Z[i]) for i in range(p)]        # (P Z')' rows
    for i in range(p):
        yi = yt[i]
        if yi is None or (isinstance(yi, float) and math.isnan(yi)):
            v[i] = 0.0
        else:
            v[i] = float(yi) - d[i] - _kahan_sum(Z[i][k] * a_pred[k] for k in range(m))
        for j in range(p):
            F[i][j] += _kahan_sum(Z[i][k] * ZP[j][k] for k in range(m))
    return v, F


# ===========================================================================
# 7. SMOOTHERS  (state / disturbance / simulation) + auxiliary residuals
# ===========================================================================
@dataclass
class SmootherResult:
    """Output of the smoother: E[.|y_1..y_n] quantities."""
    smoothed_state: List[Vector]               # alpha_hat_t
    smoothed_state_cov: List[Matrix]           # V_t
    smoothed_obs_disturbance: List[Vector]     # eps_hat_t
    smoothed_obs_disturbance_cov: List[Matrix]
    smoothed_state_disturbance: List[Vector]   # eta_hat_t
    smoothed_state_disturbance_cov: List[Matrix]
    smoothed_signal: List[Vector]              # Z_t alpha_hat_t (+ d_t)
    r0: Vector                                 # backward cumulant at t=0 (diagnostics)

    def component(self, idx: int) -> Vector:
        """Smoothed value of state element ``idx`` over time (e.g. a trend)."""
        return [a[idx] for a in self.smoothed_state]

    def component_ci(self, idx: int, *, level: float = 0.90) -> Tuple[Vector, Vector]:
        z = norm_ppf(0.5 + level / 2.0)
        lo, hi = [], []
        for t in range(len(self.smoothed_state)):
            sd = math.sqrt(max(self.smoothed_state_cov[t][idx][idx], 0.0))
            lo.append(self.smoothed_state[t][idx] - z * sd)
            hi.append(self.smoothed_state[t][idx] + z * sd)
        return lo, hi


def _observed_innovation(model: LinearGaussianSSM, t: int, yt: Sequence[float],
                         a_pred: Vector, P_pred: Matrix):
    """Observed-subset innovation: returns (obs_idx, v, F, Zobs, Hobs)."""
    Z = model.Z_t(t)
    d = model.d_t(t)
    H = model.H_t(t)
    p = len(Z)
    m = model.k_states
    obs = [i for i in range(p)
           if yt[i] is not None and not (isinstance(yt[i], float) and math.isnan(yt[i]))]
    Zobs = [Z[i] for i in obs]
    Hobs = [[H[i][j] for j in obs] for i in obs]
    v = []
    for i in obs:
        v.append(float(yt[i]) - d[i]
                 - _kahan_sum(Z[i][k] * a_pred[k] for k in range(m)))
    ZP = [matvec(P_pred, Zobs[r]) for r in range(len(obs))]
    F = [[Hobs[r][s] + _kahan_sum(Zobs[r][k] * ZP[s][k] for k in range(m))
          for s in range(len(obs))] for r in range(len(obs))]
    return obs, v, F, Zobs, Hobs


def disturbance_smoother(model: LinearGaussianSSM,
                         y: Sequence[Sequence[float]]) -> SmootherResult:
    """Durbin-Koopman / de Jong smoother: smoothed states, smoothed observation &
    state disturbances and their covariances, from one forward (big-kappa) filter
    pass and a backward ``r_t`` / ``N_t`` recursion.  Handles missing data."""
    n = len(y)
    m = model.k_states
    kf = kalman_filter(model, y, diffuse_method="bigkappa")
    # Scale-relative regularisation ridge for the innovation-covariance inverse:
    # an absolute 1e-14 is scale-blind and swamps F for small-scale problems
    # (F ~ variance -> tiny), corrupting the smoothed states.  Capped at the
    # historical 1e-14 so normal / large-scale problems are byte-identical.
    _ridge = 1e-14 * min(1.0, max(_var_scale(y), _EPS))
    r = [0.0] * m
    N = zeros(m, m)
    sm_state: List[Vector] = [None] * n         # type: ignore
    sm_cov: List[Matrix] = [None] * n           # type: ignore
    sm_eps: List[Vector] = [None] * n           # type: ignore
    sm_eps_cov: List[Matrix] = [None] * n       # type: ignore
    sm_eta: List[Vector] = [None] * n           # type: ignore
    sm_eta_cov: List[Matrix] = [None] * n       # type: ignore
    signal: List[Vector] = [None] * n           # type: ignore
    for t in range(n - 1, -1, -1):
        a_pred = kf.predicted_state[t]
        P_pred = kf.predicted_cov[t]
        T = model.T_t(t)
        R = model.R_t(t)
        Q = model.Q_t(t)
        H = model.H_t(t)
        obs, v, F, Zobs, Hobs = _observed_innovation(model, t, y[t], a_pred, P_pred)
        p = len(obs)
        # smoothed state disturbance eta_hat = Q R' r_t (uses incoming r = r_t)
        Rt_r = matvec(transpose(R), r)
        eta = matvec(Q, Rt_r)
        RN = matmul(matmul(transpose(R), N), R)
        eta_cov = mat_sub(Q, matmul(matmul(Q, RN), Q))
        if p > 0:
            Finv = inv(F, ridge=_ridge)
            Finv_v = matvec(Finv, v)
            # gain K_t = T P Z' Finv
            PZ = transpose([matvec(P_pred, Zobs[i]) for i in range(p)])  # m x p
            Kt = matmul(matmul(T, PZ), Finv)            # m x p
            Kt_r = matvec(transpose(Kt), r)             # p
            u = [Finv_v[i] - Kt_r[i] for i in range(p)]
            eps = matvec(Hobs, u)                       # smoothed obs disturbance
            # eps cov = H - H (Finv + K' N K) H  (observed block)
            KtNK = matmul(matmul(transpose(Kt), N), Kt)
            mid = mat_add(Finv, KtNK)
            eps_cov = mat_sub(Hobs, matmul(matmul(Hobs, mid), Hobs))
            # update r, N  (L = T - K Z)
            ZtFinv = matmul(transpose(Zobs), Finv)      # m x p
            L = mat_sub(T, matmul(Kt, Zobs))            # m x m
            r_prev = vec_add(matvec(ZtFinv, v), matvec(transpose(L), r))
            N_prev = mat_add(matmul(ZtFinv, Zobs),
                             matmul(matmul(transpose(L), N), L))
        else:
            eps = [0.0] * len(H)
            eps_cov = [row[:] for row in H]
            L = T
            r_prev = matvec(transpose(L), r)
            N_prev = matmul(matmul(transpose(L), N), L)
        # smoothed state and covariance
        alpha = vec_add(a_pred, matvec(P_pred, r_prev))
        PN = matmul(P_pred, N_prev)
        V = symmetrize(mat_sub(P_pred, matmul(PN, P_pred)))
        # Floor the variances at 0: during the diffuse phase of a multi-diffuse-state
        # model the big-kappa V = P - P N P can still cancel to a small NEGATIVE value;
        # a variance is non-negative by definition, and returning a negative would
        # crash a downstream sqrt for a confidence interval.
        for i in range(m):
            if V[i][i] < 0.0:
                V[i][i] = 0.0
        sm_state[t] = alpha
        sm_cov[t] = V
        sm_eps[t] = _scatter(eps, obs, len(H))
        sm_eps_cov[t] = _scatter_mat(eps_cov, obs, len(H))
        sm_eta[t] = eta
        sm_eta_cov[t] = symmetrize(eta_cov)
        d = model.d_t(t)
        Zt = model.Z_t(t)
        signal[t] = [d[i] + _kahan_sum(Zt[i][k] * alpha[k] for k in range(m))
                     for i in range(len(Zt))]
        r, N = r_prev, N_prev
    return SmootherResult(sm_state, sm_cov, sm_eps, sm_eps_cov, sm_eta, sm_eta_cov,
                          signal, list(r))


def _scatter(vals: Sequence[float], idx: Sequence[int], p: int) -> Vector:
    out = [0.0] * p
    for k, i in enumerate(idx):
        out[i] = vals[k]
    return out


def _scatter_mat(M: Matrix, idx: Sequence[int], p: int) -> Matrix:
    out = zeros(p, p)
    for a, i in enumerate(idx):
        for b, j in enumerate(idx):
            out[i][j] = M[a][b]
    return out


def kalman_smoother(model: LinearGaussianSSM,
                    y: Sequence[Sequence[float]]) -> SmootherResult:
    """Fixed-interval (RTS) state smoother -- an alias returning the full smoother
    output (smoothed states, covariances and signal)."""
    return disturbance_smoother(model, y)


def simulation_smoother(model: LinearGaussianSSM, y: Sequence[Sequence[float]], *,
                        n_sims: int = 1, seed: int = 0) -> List[List[Vector]]:
    """Durbin-Koopman (2002) simulation smoother: draw sample state paths from
    ``alpha | y`` by the mean-correction method -- simulate an unconditional
    (alpha+, y+) path, smooth the simulated observations, and add the smoothed
    residual to the real smoothed state.  Returns ``n_sims`` state paths."""
    rng = DeterministicRNG(seed)
    n = len(y)
    m = model.k_states
    real = disturbance_smoother(model, y).smoothed_state
    draws: List[List[Vector]] = []
    for s in range(n_sims):
        # simulate a path (states + observations) from the model
        alpha_plus, y_plus = _simulate_path(model, n, rng)
        sm_plus = disturbance_smoother(model, y_plus).smoothed_state
        path = [[real[t][k] + alpha_plus[t][k] - sm_plus[t][k] for k in range(m)]
                for t in range(n)]
        draws.append(path)
    return draws


def _simulate_path(model: LinearGaussianSSM, n: int, rng: DeterministicRNG
                   ) -> Tuple[List[Vector], List[Vector]]:
    """Draw a forward (alpha, y) path from the model's own dynamics (proper part of
    the initial covariance; diffuse states start at their mean)."""
    m = model.k_states
    a = list(model.a1)
    P1 = model.P1
    if any(P1[i][i] > _EPS for i in range(m)):
        a = vec_add(a, rng.multivariate_normal([0.0] * m, P1))
    alphas: List[Vector] = []
    ys: List[Vector] = []
    for t in range(n):
        alphas.append(list(a))
        Z = model.Z_t(t)
        d = model.d_t(t)
        H = model.H_t(t)
        p = len(Z)
        eps = rng.multivariate_normal([0.0] * p, H) if _any_pos_diag(H) else [0.0] * p
        ys.append([d[i] + _kahan_sum(Z[i][k] * a[k] for k in range(m)) + eps[i]
                   for i in range(p)])
        T = model.T_t(t)
        cc = model.c_t(t)
        R = model.R_t(t)
        Q = model.Q_t(t)
        r = len(Q)
        eta = rng.multivariate_normal([0.0] * r, Q) if _any_pos_diag(Q) else [0.0] * r
        Reta = matvec(R, eta)
        a = [cc[k] + _kahan_sum(T[k][j] * a[j] for j in range(m)) + Reta[k]
             for k in range(m)]
    return alphas, ys


def _any_pos_diag(A: Matrix) -> bool:
    return any(A[i][i] > _EPS for i in range(len(A)))


def auxiliary_residuals(model: LinearGaussianSSM,
                        y: Sequence[Sequence[float]]) -> Dict[str, List[Vector]]:
    """Standardised smoothed disturbances (Harvey-Koopman auxiliary residuals):
    an observation-disturbance residual flags *outliers*, a state-disturbance
    residual flags *structural breaks / level shifts*.  Values beyond ~+/-3 are
    suspect."""
    sm = disturbance_smoother(model, y)
    n = len(y)
    obs_res: List[Vector] = []
    state_res: List[Vector] = []
    for t in range(n):
        ec = sm.smoothed_obs_disturbance_cov[t]
        obs_res.append([_safe_div(sm.smoothed_obs_disturbance[t][i],
                                  math.sqrt(ec[i][i]) if ec[i][i] > _EPS else 0.0)
                        for i in range(len(ec))])
        qc = sm.smoothed_state_disturbance_cov[t]
        state_res.append([_safe_div(sm.smoothed_state_disturbance[t][i],
                                    math.sqrt(qc[i][i]) if qc[i][i] > _EPS else 0.0)
                          for i in range(len(qc))])
    return {"observation": obs_res, "state": state_res}


# ===========================================================================
# 8. LOG-LIKELIHOOD, INFORMATION CRITERIA, MLE + EM ESTIMATION
# ===========================================================================
def loglike(model: LinearGaussianSSM, y: Sequence[Sequence[float]], *,
            concentrate: bool = False) -> float:
    """Prediction-error-decomposition log-likelihood of ``y`` under ``model``."""
    return kalman_filter(model, y, concentrate=concentrate).loglike


@dataclass
class ICResult:
    aic: float
    bic: float
    aicc: float
    hqic: float


def information_criteria(*, loglik: float, n_params: int, nobs: int) -> ICResult:
    k, n = n_params, max(nobs, 1)
    aic = -2.0 * loglik + 2.0 * k
    bic = -2.0 * loglik + k * math.log(n)
    denom = n - k - 1
    aicc = aic + (2.0 * k * (k + 1) / denom if denom > 0 else _INF)
    hqic = -2.0 * loglik + 2.0 * k * math.log(max(math.log(n), 1.0 + _EPS))
    return ICResult(aic, bic, aicc, hqic)


def minimize_nelder_mead(f: Callable[[Sequence[float]], float],
                         x0: Sequence[float], *, maxiter: int = 2000,
                         tol: float = 1e-9, restarts: int = 2,
                         step: float = 0.5, seed: int = 0):
    """Deterministic Nelder-Mead simplex minimiser with a few fixed restarts --
    the parameter-estimation workhorse (from scratch, no SciPy)."""
    n = len(x0)
    best_x = list(x0)
    best_f = f(best_x)
    rng = DeterministicRNG(seed)
    for attempt in range(restarts + 1):
        if attempt == 0:
            simplex = [list(best_x)]
            for i in range(n):
                pt = list(best_x)
                pt[i] += step if pt[i] == 0 else step * (1.0 + abs(pt[i]))
                simplex.append(pt)
        else:
            simplex = [list(best_x)]
            for i in range(n):
                pt = list(best_x)
                pt[i] += (0.3 + 0.7 * rng.random()) * (1.0 + abs(pt[i]))
                simplex.append(pt)
        fvals = [f(p) for p in simplex]
        for _ in range(maxiter):
            order = sorted(range(n + 1), key=lambda i: fvals[i])
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            if abs(fvals[-1] - fvals[0]) <= tol * (abs(fvals[0]) + tol):
                break
            centroid = [_kahan_sum(simplex[i][j] for i in range(n)) / n
                        for j in range(n)]
            xr = [centroid[j] + (centroid[j] - simplex[-1][j]) for j in range(n)]
            fr = f(xr)
            if fr < fvals[0]:
                xe = [centroid[j] + 2.0 * (centroid[j] - simplex[-1][j])
                      for j in range(n)]
                fe = f(xe)
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
            elif fr < fvals[-2]:
                simplex[-1], fvals[-1] = xr, fr
            else:
                xc = [centroid[j] + 0.5 * (simplex[-1][j] - centroid[j])
                      for j in range(n)]
                fc = f(xc)
                if fc < fvals[-1]:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j]
                                      - simplex[0][j]) for j in range(n)]
                        fvals[i] = f(simplex[i])
        if fvals[0] < best_f:
            best_f, best_x = fvals[0], list(simplex[0])
    return best_x, best_f


@dataclass
class StateSpaceFit:
    """A fitted state-space model."""
    model: LinearGaussianSSM
    params: Vector
    param_names: List[str]
    loglike: float
    aic: float
    bic: float
    aicc: float
    hqic: float
    nobs: int
    n_params: int
    scale: float
    converged: bool
    method: str = "mle"

    def filter(self, y: Sequence[Sequence[float]]) -> KalmanFilterResult:
        return kalman_filter(self.model, y)

    def smooth(self, y: Sequence[Sequence[float]]) -> SmootherResult:
        return disturbance_smoother(self.model, y)

    def summary(self) -> str:
        lines = ["%s  method=%s  nobs=%d  logL=%.4f" % (
            self.model.name, self.method, self.nobs, self.loglike)]
        lines.append("  AIC=%.3f  BIC=%.3f  AICc=%.3f" % (self.aic, self.bic, self.aicc))
        for nm, v in zip(self.param_names, self.params):
            lines.append("    %-16s = %.6f" % (nm, v))
        return "\n".join(lines)


def fit_ssm_mle(build: Callable[[Sequence[float]], LinearGaussianSSM],
                start_params: Sequence[float],
                y: Sequence[Sequence[float]], *,
                param_names: Optional[Sequence[str]] = None,
                transform: Optional[Callable[[Sequence[float]], Sequence[float]]] = None,
                concentrate: bool = False, maxiter: int = 2000,
                restarts: int = 2, seed: int = 0) -> StateSpaceFit:
    """Maximum-likelihood estimation of a state-space model by numerical
    optimisation of the prediction-error log-likelihood.  ``build(params)`` returns
    a :class:`LinearGaussianSSM`; ``transform`` (optional) maps the *unconstrained*
    optimiser vector to the model parameters (e.g. ``exp`` for variances)."""
    n_obs_scalars = _count_obs(y)

    def to_params(u: Sequence[float]) -> Sequence[float]:
        return transform(u) if transform is not None else u

    def nll(u: Sequence[float]) -> float:
        try:
            model = build(to_params(u))
            ll = kalman_filter(model, y, concentrate=concentrate,
                               store=False).loglike
        except (ValueError, ZeroDivisionError, OverflowError):
            return 1e18
        return -ll if math.isfinite(ll) else 1e18

    u0 = list(start_params)
    best_u, best_nll = minimize_nelder_mead(nll, u0, maxiter=maxiter,
                                            restarts=restarts, seed=seed)
    params = list(to_params(best_u))
    model = build(params)
    kf = kalman_filter(model, y, concentrate=concentrate)
    k = len(params) + (1 if concentrate else 0)
    ic = information_criteria(loglik=kf.loglike, n_params=k, nobs=n_obs_scalars)
    names = list(param_names) if param_names else ["p%d" % i for i in range(len(params))]
    return StateSpaceFit(model, params, names, kf.loglike, ic.aic, ic.bic,
                         ic.aicc, ic.hqic, n_obs_scalars, k, kf.scale,
                         best_nll < 1e17, "mle")


def _count_obs(y: Sequence[Sequence[float]]) -> int:
    n = 0
    for row in y:
        for v in row:
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                n += 1
    return n


def em_algorithm(model: LinearGaussianSSM, y: Sequence[Sequence[float]], *,
                 max_iter: int = 100, tol: float = 1e-6,
                 update_obs: bool = True, update_state: bool = True,
                 update_init: bool = False) -> StateSpaceFit:
    """Expectation-Maximisation for the disturbance covariances ``H`` (observation)
    and ``Q`` (state) of a *time-invariant* state-space model, holding the
    structural matrices ``Z, T, R`` fixed (the usual unobserved-components setting).
    Each M-step is closed form in the smoothed disturbances and their covariances
    (Shumway-Stoffer / Watson-Engle).  Monotonically increases the likelihood."""
    n = len(y)
    m = model.k_states
    r = model.k_posdef
    p = model.k_obs
    H = [row[:] for row in model.H_t(0)]
    Q = [row[:] for row in model.Q_t(0)]
    a1 = list(model.a1)
    P1 = [row[:] for row in model.P1]
    cur = model
    prev_ll = _NEG_INF
    for _ in range(max_iter):
        sm = disturbance_smoother(cur, y)
        ll = kalman_filter(cur, y).loglike
        if abs(ll - prev_ll) < tol * (abs(prev_ll) + 1.0):
            prev_ll = ll
            break
        prev_ll = ll
        if update_obs:
            Hn = zeros(p, p)
            cnt = 0
            for t in range(n):
                if _row_observed(y[t]):
                    e = sm.smoothed_obs_disturbance[t]
                    ec = sm.smoothed_obs_disturbance_cov[t]
                    for i in range(p):
                        for j in range(p):
                            Hn[i][j] += e[i] * e[j] + ec[i][j]
                    cnt += 1
            H = [[Hn[i][j] / max(cnt, 1) for j in range(p)] for i in range(p)]
            H = nearest_pd(H)
        if update_state:
            Qn = zeros(r, r)
            for t in range(n):
                eta = sm.smoothed_state_disturbance[t]
                qc = sm.smoothed_state_disturbance_cov[t]
                for i in range(r):
                    for j in range(r):
                        Qn[i][j] += eta[i] * eta[j] + qc[i][j]
            Q = [[Qn[i][j] / max(n, 1) for j in range(r)] for i in range(r)]
            Q = nearest_pd(Q)
        if update_init:
            a1 = list(sm.smoothed_state[0])
            P1 = ([row[:] for row in model.P1] if model.P1_infinity is not None
                  else [row[:] for row in sm.smoothed_state_cov[0]])
        cur = cur.copy_with(H=H, Q=Q, a1=a1, P1=P1)
    n_obs = _count_obs(y)
    n_free = (p * (p + 1) // 2 if update_obs else 0) + \
             (r * (r + 1) // 2 if update_state else 0)
    ic = information_criteria(loglik=prev_ll, n_params=n_free, nobs=n_obs)
    params = [H[i][i] for i in range(p)] + [Q[i][i] for i in range(r)]
    names = (["H[%d]" % i for i in range(p)] + ["Q[%d]" % i for i in range(r)])
    return StateSpaceFit(cur, params, names, prev_ll, ic.aic, ic.bic, ic.aicc,
                         ic.hqic, n_obs, n_free, 1.0, True, "em")


def _row_observed(row: Sequence[float]) -> bool:
    return any(v is not None and not (isinstance(v, float) and math.isnan(v))
              for v in row)


# ===========================================================================
# 9. UNOBSERVED COMPONENTS: composable structural components + builder
# ===========================================================================
def _sigmoid(u: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(u, 40.0), -40.0)))


def _reflection_to_ar(refl: Sequence[float]) -> List[float]:
    """Levinson step-up: partial-autocorrelations (|.|<1) -> stationary AR coefs."""
    phi: List[float] = []
    for k in range(len(refl)):
        rk = refl[k]
        new = [phi[i] - rk * phi[k - 1 - i] for i in range(k)]
        new.append(rk)
        phi = new
    return phi


def _param_natural(u: float, kind: str) -> float:
    """Map an unconstrained optimiser value to the natural parameter space."""
    if kind == "var":
        return math.exp(max(min(u, 40.0), -40.0))
    if kind == "unit":                                   # (0, 1) damping
        return _sigmoid(u)
    if kind == "corr":                                   # (-1, 1) reflection / AR1
        return math.tanh(u)
    if kind == "freq":                                   # (0, pi) frequency
        return math.pi * _sigmoid(u)
    return u                                             # real


def _param_unconstrained(p: float, kind: str) -> float:
    """Inverse of :func:`_param_natural` (for start values)."""
    if kind == "var":
        return math.log(max(p, 1e-8))
    if kind == "unit":
        p = min(max(p, 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))
    if kind == "corr":
        p = min(max(p, -1 + 1e-6), 1 - 1e-6)
        return math.atanh(p)
    if kind == "freq":
        r = min(max(p / math.pi, 1e-6), 1 - 1e-6)
        return math.log(r / (1 - r))
    return p


class Component:
    """Base structural component contributing a block to the state space and a
    (loading) row to the single observed series.  Subclasses set the fixed
    structure and map their parameter slice to ``T`` / ``Q`` blocks."""
    name = "component"
    k_states = 0
    k_posdef = 0
    param_specs: List[Tuple[str, str]] = []              # (name, kind)
    diffuse_states: List[bool] = []
    time_varying_Z = False
    is_irregular = False

    @property
    def n_params(self) -> int:
        return len(self.param_specs)

    def start_params(self) -> List[float]:
        return [1.0 for _ in self.param_specs]

    def T(self, params: Sequence[float]) -> Matrix:
        return eye(self.k_states)

    def R(self) -> Matrix:
        return eye(self.k_states)[: self.k_states][: self.k_states] if False else \
            [[1.0 if i == j else 0.0 for j in range(self.k_posdef)]
             for i in range(self.k_states)]

    def Q(self, params: Sequence[float]) -> Matrix:
        return zeros(self.k_posdef, self.k_posdef)

    def Z_row(self, t: int) -> Vector:
        return [0.0] * self.k_states

    def a1(self) -> Vector:
        return [0.0] * self.k_states

    def P1_star(self, params: Sequence[float]) -> Matrix:
        return zeros(self.k_states, self.k_states)

    def obs_var(self, params: Sequence[float]) -> float:
        return 0.0


class Irregular(Component):
    """White-noise observation disturbance (the ``H`` term)."""
    name = "irregular"
    is_irregular = True
    param_specs = [("sigma2.irregular", "var")]

    def __init__(self, start: float = 1.0):
        self._start = start

    def start_params(self) -> List[float]:
        return [self._start]

    def obs_var(self, params: Sequence[float]) -> float:
        return params[0]


class LocalLevel(Component):
    """Random-walk level (integrated of order 1)."""
    name = "level"
    k_states = 1
    k_posdef = 1
    param_specs = [("sigma2.level", "var")]
    diffuse_states = [True]

    def __init__(self, start: float = 1.0):
        self._start = start

    def start_params(self):
        return [self._start]

    def T(self, params):
        return [[1.0]]

    def Q(self, params):
        return [[params[0]]]

    def Z_row(self, t):
        return [1.0]


class LocalLinearTrend(Component):
    """Level + slope, each a random walk (the classic local linear trend).  Set
    ``stochastic_level=False`` for an integrated random walk / smooth trend."""
    name = "trend"
    k_states = 2
    param_specs = [("sigma2.level", "var"), ("sigma2.trend", "var")]
    diffuse_states = [True, True]

    def __init__(self, *, stochastic_level: bool = True,
                 stochastic_trend: bool = True, damped: bool = False,
                 start: float = 1.0):
        self.stochastic_level = stochastic_level
        self.stochastic_trend = stochastic_trend
        self.damped = damped
        self._start = start
        specs = []
        if stochastic_level:
            specs.append(("sigma2.level", "var"))
        if stochastic_trend:
            specs.append(("sigma2.trend", "var"))
        if damped:
            specs.append(("damping.trend", "unit"))
        self.param_specs = specs
        self.k_posdef = int(stochastic_level) + int(stochastic_trend)
        if self.k_posdef == 0:
            self.k_posdef = 1                            # deterministic trend still needs R shape

    def start_params(self):
        out = []
        if self.stochastic_level:
            out.append(self._start)
        if self.stochastic_trend:
            out.append(self._start * 0.1)
        if self.damped:
            out.append(0.9)
        return out

    def _phi(self, params):
        return params[-1] if self.damped else 1.0

    def T(self, params):
        return [[1.0, 1.0], [0.0, self._phi(params)]]

    def R(self):
        cols = []
        if self.stochastic_level:
            cols.append([1.0, 0.0])
        if self.stochastic_trend:
            cols.append([0.0, 1.0])
        if not cols:
            cols = [[0.0, 0.0]]
        return transpose(cols)

    def Q(self, params):
        vs = []
        i = 0
        if self.stochastic_level:
            vs.append(params[i]); i += 1
        if self.stochastic_trend:
            vs.append(params[i]); i += 1
        if not vs:
            vs = [0.0]
        return diag(vs)

    def Z_row(self, t):
        return [1.0, 0.0]


class SeasonalDummy(Component):
    """Stochastic dummy-variable seasonal of period ``s`` (``s-1`` states summing
    to zero over a cycle, with a single disturbance)."""
    name = "seasonal"

    def __init__(self, period: int, *, start: float = 1.0):
        self.period = period
        self.k_states = period - 1
        self.k_posdef = 1
        self.param_specs = [("sigma2.seasonal", "var")]
        self.diffuse_states = [True] * self.k_states
        self._start = start

    def start_params(self):
        return [self._start]

    def T(self, params):
        s = self.k_states
        T = zeros(s, s)
        for j in range(s):
            T[0][j] = -1.0
        for i in range(1, s):
            T[i][i - 1] = 1.0
        return T

    def R(self):
        return [[1.0 if i == 0 else 0.0] for i in range(self.k_states)]

    def Q(self, params):
        return [[params[0]]]

    def Z_row(self, t):
        return [1.0] + [0.0] * (self.k_states - 1)


class TrigonometricSeasonal(Component):
    """Trigonometric (harmonic) seasonal of period ``s`` with ``h`` harmonics
    (default ``s//2``); a common disturbance variance across harmonics."""
    name = "freq_seasonal"

    def __init__(self, period: int, *, harmonics: Optional[int] = None,
                 start: float = 1.0):
        self.period = period
        self.harmonics = harmonics or (period // 2)
        self.k_states = 2 * self.harmonics
        self.k_posdef = 2 * self.harmonics
        self.param_specs = [("sigma2.freq_seasonal", "var")]
        self.diffuse_states = [True] * self.k_states
        self._start = start

    def start_params(self):
        return [self._start]

    def T(self, params):
        blocks = []
        for j in range(1, self.harmonics + 1):
            lam = 2.0 * math.pi * j / self.period
            c, s = math.cos(lam), math.sin(lam)
            blocks.append([[c, s], [-s, c]])
        return block_diag(*blocks)

    def R(self):
        return eye(self.k_states)

    def Q(self, params):
        return eye(self.k_states, params[0])

    def Z_row(self, t):
        return [1.0 if i % 2 == 0 else 0.0 for i in range(self.k_states)]


class Cycle(Component):
    """Stochastic cycle with damping ``rho`` in (0,1) and frequency ``lambda`` in
    (0, pi) (period ``2*pi/lambda``).

    When fit inside an :class:`UnobservedComponents` model the cycle frequency is
    automatically seeded from the data's dominant periodogram peak (unless you pass
    a non-default ``start_period``), which reliably recovers the period of a cycle
    fit on its own.  Note, however, that the cycle-frequency likelihood is strongly
    **multimodal**, and a *stochastic* level / trend in the same model can absorb
    low-frequency cyclical variation -- so a cycle combined with ``LocalLevel`` /
    ``LocalLinearTrend`` can still converge to the degenerate frequency-0 solution.
    If that happens, refit with more ``restarts``, pass an explicit ``start_period``
    close to the expected period, or hold the level fixed
    (``LocalLinearTrend(stochastic_level=False)`` / a deterministic level)."""
    name = "cycle"
    k_states = 2
    k_posdef = 2
    param_specs = [("sigma2.cycle", "var"), ("damping.cycle", "unit"),
                   ("freq.cycle", "freq")]

    def __init__(self, *, start_period: float = 10.0, start_var: float = 1.0):
        self._start_period = start_period
        self._start_var = start_var
        self.diffuse_states = [False, False]

    def start_params(self):
        return [self._start_var, 0.9, 2.0 * math.pi / self._start_period]

    def T(self, params):
        rho, lam = params[1], params[2]
        c, s = math.cos(lam), math.sin(lam)
        return [[rho * c, rho * s], [-rho * s, rho * c]]

    def R(self):
        return eye(2)

    def Q(self, params):
        return eye(2, params[0])

    def Z_row(self, t):
        return [1.0, 0.0]

    def P1_star(self, params):
        sig, rho = params[0], params[1]
        v = sig / max(1.0 - rho * rho, _EPS)
        return [[v, 0.0], [0.0, v]]


class AutoRegressive(Component):
    """Stationary AR(p) component (parameterised by reflection coefficients so the
    optimiser always stays in the stationary region)."""
    name = "ar"

    def __init__(self, order: int, *, start_var: float = 1.0):
        self.order = order
        self.k_states = order
        self.k_posdef = 1
        self.param_specs = ([("ar.refl%d" % (i + 1), "corr") for i in range(order)]
                            + [("sigma2.ar", "var")])
        self.diffuse_states = [False] * order
        self._start_var = start_var

    def start_params(self):
        return [0.1 * (-1) ** i for i in range(self.order)] + [self._start_var]

    def _phi(self, params):
        return _reflection_to_ar(params[: self.order])

    def T(self, params):
        phi = self._phi(params)
        p = self.order
        T = zeros(p, p)
        for j in range(p):
            T[0][j] = phi[j]
        for i in range(1, p):
            T[i][i - 1] = 1.0
        return T

    def R(self):
        return [[1.0 if i == 0 else 0.0] for i in range(self.order)]

    def Q(self, params):
        return [[params[-1]]]

    def Z_row(self, t):
        return [1.0] + [0.0] * (self.order - 1)

    def P1_star(self, params):
        T = self.T(params)
        Q = self.Q(params)
        RQR = matmul(matmul(self.R(), Q), transpose(self.R()))
        try:
            return solve_discrete_lyapunov(T, RQR)
        except (ValueError, ZeroDivisionError):
            return eye(self.order, params[-1])


class Regression(Component):
    """Static regression on exogenous ``X`` (n x k): the coefficients are constant
    (diffuse) states, giving generalized-least-squares estimates within the model.
    Set ``dynamic=True`` (with a variance) for time-varying coefficients."""
    name = "regression"
    time_varying_Z = True

    def __init__(self, X: Sequence[Sequence[float]], *, dynamic: bool = False,
                 start: float = 0.1):
        self.X = [list(row) for row in X]
        self.k = len(self.X[0]) if self.X else 0
        self.k_states = self.k
        self.dynamic = dynamic
        self.k_posdef = self.k if dynamic else 1
        self.param_specs = [("sigma2.reg", "var")] if dynamic else []
        self.diffuse_states = [True] * self.k
        self._start = start

    def start_params(self):
        return [self._start] if self.dynamic else []

    def T(self, params):
        return eye(self.k)

    def R(self):
        return eye(self.k) if self.dynamic else [[0.0] for _ in range(self.k)]

    def Q(self, params):
        return eye(self.k, params[0]) if self.dynamic else [[0.0]]

    def Z_row(self, t):
        return list(self.X[t]) if t < len(self.X) else [0.0] * self.k


def _dominant_cycle_freqs(y: Sequence[Optional[float]], k: int = 1) -> List[float]:
    """Angular frequencies (0, pi) of the ``k`` strongest periodogram peaks of the
    linearly-detrended series (missing -> 0 residual).  Used to seed a Cycle
    component's frequency from the data: the maximum-likelihood cycle frequency is
    badly multimodal and a fixed default start-period traps long-period cycles at
    the frequency-0 boundary (a degenerate non-cycle).  Peaks are returned strongest
    first and kept a little apart so several cycles get distinct seeds."""
    n = len(y)
    idx = [t for t in range(n) if y[t] is not None
           and not (isinstance(y[t], float) and math.isnan(y[t]))]
    if len(idx) < 4:
        return []
    xbar = sum(idx) / len(idx)
    ybar = sum(y[t] for t in idx) / len(idx)
    sxx = sum((t - xbar) ** 2 for t in idx)
    sxy = sum((t - xbar) * (y[t] - ybar) for t in idx)
    b = sxy / sxx if sxx > 0 else 0.0
    a = ybar - b * xbar
    r = [(y[t] - (a + b * t)) if t in set(idx) else 0.0 for t in range(n)]
    wmin = 2.0 * math.pi / n                              # lowest resolvable freq
    ngrid = min(4 * n, 1024)
    scored: List[Tuple[float, float]] = []
    for j in range(1, ngrid):
        w = math.pi * j / ngrid
        if w < wmin:
            continue
        cr = _kahan_sum(r[t] * math.cos(w * t) for t in range(n))
        ci = _kahan_sum(r[t] * math.sin(w * t) for t in range(n))
        scored.append((cr * cr + ci * ci, w))
    scored.sort(reverse=True)
    out: List[float] = []
    for _, w in scored:
        if all(abs(w - ww) > 0.05 for ww in out):
            out.append(w)
        if len(out) >= k:
            break
    return out


@dataclass
class UnobservedComponents:
    """Composable structural time-series model: add components (level, trend,
    seasonal, cycle, AR, regression, irregular) and fit the disturbance variances
    and hyper-parameters by maximum likelihood.  The classic *basic structural
    model* is ``LocalLinearTrend + SeasonalDummy + Irregular``."""
    endog: Sequence[float]
    components: List[Component]
    name: str = "ucm"

    def __post_init__(self):
        self.y = [[float(v)] if not (v is None or (isinstance(v, float)
                  and math.isnan(v))) else [None] for v in self.endog]
        self.n = len(self.endog)
        self._state_comps = [c for c in self.components if not c.is_irregular]
        self._irregular = [c for c in self.components if c.is_irregular]
        self._layout = self._compute_layout()
        self._time_varying_Z = any(c.time_varying_Z for c in self._state_comps)

    def _compute_layout(self):
        layout = []
        poff = 0
        soff = 0
        roff = 0
        for c in self.components:
            layout.append(dict(comp=c, p0=poff, p1=poff + c.n_params,
                               s0=soff, r0=roff))
            poff += c.n_params
            if not c.is_irregular:
                soff += c.k_states
                roff += c.k_posdef
        self.n_params_total = poff
        self.k_states = soff
        self.k_posdef = roff
        return layout

    def param_names(self) -> List[str]:
        names = []
        for c in self.components:
            names.extend(nm for nm, _ in c.param_specs)
        return names

    def _split(self, params):
        out = []
        for lay in self._layout:
            out.append(params[lay["p0"]:lay["p1"]])
        return out

    def build(self, params: Sequence[float]) -> LinearGaussianSSM:
        parts = self._split(params)
        Tb, Rb, Qb, a1, P1s, P1inf = [], [], [], [], [], []
        Zconst_row: Vector = []
        hvar = 0.0
        for lay, p in zip(self._layout, parts):
            c = lay["comp"]
            if c.is_irregular:
                hvar += c.obs_var(p)
                continue
            Tb.append(c.T(p))
            Rb.append(c.R())
            Qb.append(c.Q(p))
            a1.extend(c.a1())
            P1s.append(c.P1_star(p))
            di = c.diffuse_states or [False] * c.k_states
            P1inf.append(diag([1.0 if d else 0.0 for d in di]))
            if not self._time_varying_Z:
                Zconst_row.extend(c.Z_row(0))
        T = block_diag(*Tb) if Tb else [[0.0]]
        R = block_diag(*Rb) if Rb else [[0.0]]
        Q = block_diag(*Qb) if Qb else [[0.0]]
        P1 = block_diag(*P1s) if P1s else [[0.0]]
        P1_inf = block_diag(*P1inf) if P1inf else [[0.0]]
        if not a1:
            a1 = [0.0]
        H = [[max(hvar, 0.0)]]
        if self._time_varying_Z:
            comps = self._state_comps
            def Zfun(t: int, comps=comps) -> Matrix:
                row: Vector = []
                for c in comps:
                    row.extend(c.Z_row(t))
                return [row]
            Z: TimeVarying = Zfun
        else:
            Z = [Zconst_row]
        diffuse_present = any(any(row) for row in P1_inf)
        return LinearGaussianSSM(
            Z=Z, T=T, R=R, Q=Q, H=H, a1=a1, P1=P1,
            P1_infinity=(P1_inf if diffuse_present else None), name=self.name)

    def start_params(self) -> List[float]:
        out = []
        for c in self.components:
            out.extend(c.start_params())
        return out

    def _transform(self, u: Sequence[float]) -> List[float]:
        kinds = []
        for c in self.components:
            kinds.extend(kind for _, kind in c.param_specs)
        return [_param_natural(u[i], kinds[i]) for i in range(len(u))]

    def _untransform(self, p: Sequence[float]) -> List[float]:
        kinds = []
        for c in self.components:
            kinds.extend(kind for _, kind in c.param_specs)
        return [_param_unconstrained(p[i], kinds[i]) for i in range(len(p))]

    def fit(self, *, maxiter: int = 2000, restarts: int = 2,
            seed: int = 0) -> StateSpaceFit:
        # Seed each Cycle's frequency from the data's dominant spectral peak(s)
        # (unless the user set a non-default start_period).  Without this the
        # multimodal cycle likelihood traps long-period cycles at frequency 0.
        cyc = [c for c in self.components
               if isinstance(c, Cycle) and c._start_period == 10.0]
        if cyc:
            freqs = _dominant_cycle_freqs([row[0] for row in self.y], k=len(cyc))
            for i, c in enumerate(cyc):
                if i < len(freqs) and freqs[i] > _EPS:
                    c._start_period = 2.0 * math.pi / freqs[i]
        start_nat = self.start_params()
        start_u = self._untransform(start_nat)
        fit = fit_ssm_mle(self.build, start_u, self.y,
                          param_names=self.param_names(),
                          transform=self._transform, maxiter=maxiter,
                          restarts=restarts, seed=seed)
        fit.method = "mle"
        fit._ucm = self                                  # remember the layout
        self._fitted = fit
        return fit


# ===========================================================================
# 10. BAYESIAN DLM: discount factors + forward-filtering backward-sampling
# ===========================================================================
def ffbs(model: LinearGaussianSSM, y: Sequence[Sequence[float]], *,
         n_draws: int = 1, seed: int = 0) -> List[List[Vector]]:
    """Forward-Filtering Backward-Sampling: draw complete state paths from the
    joint posterior ``p(alpha_1..alpha_n | y)`` of a linear-Gaussian model -- the
    building block of Gibbs samplers for state-space models."""
    rng = DeterministicRNG(seed)
    kf = kalman_filter(model, y, diffuse_method="bigkappa")
    n = len(y)
    m = model.k_states
    # scale-relative ridge (absolute 1e-12 swamps Pp for small-scale problems)
    _ridge = 1e-12 * min(1.0, max(_var_scale(y), _EPS))
    draws: List[List[Vector]] = []
    for _ in range(n_draws):
        alpha: List[Vector] = [None] * n            # type: ignore
        alpha[n - 1] = rng.multivariate_normal(kf.filtered_state[n - 1],
                                               nearest_pd(kf.filtered_cov[n - 1]))
        for t in range(n - 2, -1, -1):
            T = model.T_t(t)
            ap = kf.predicted_state[t + 1]
            Pp = kf.predicted_cov[t + 1]
            Pf = kf.filtered_cov[t]
            af = kf.filtered_state[t]
            J = matmul(matmul(Pf, transpose(T)), inv(Pp, ridge=_ridge))
            mean = vec_add(af, matvec(J, vec_sub(alpha[t + 1], ap)))
            cov = mat_sub(Pf, matmul(J, matmul(T, Pf)))
            alpha[t] = rng.multivariate_normal(mean, nearest_pd(cov))
        draws.append(alpha)
    return draws


@dataclass
class DLMFilterResult:
    filtered_mean: List[Vector]
    filtered_cov: List[Matrix]
    forecast: List[float]
    forecast_var: List[float]
    obs_scale: List[float]                          # running estimate of obs variance
    dof: List[float]
    loglike: float


class DiscountDLM:
    """West-Harrison dynamic linear model with *discount factors*: instead of a
    state-disturbance covariance ``Q`` the prior variance is inflated by ``1/delta``
    each step (``delta`` in (0,1]).  The (scalar) observational variance is learned
    online with an inverse-gamma / variance-discount ``beta`` update, giving
    Student-t predictive densities.  Univariate observation.

        obs:    y_t = F' theta_t + v_t
        system: theta_t = G theta_{t-1} + w_t   (var set by the discount)
    """

    def __init__(self, F: Sequence[float], G: Matrix, *, delta: float = 0.95,
                 m0: Optional[Sequence[float]] = None,
                 C0: Optional[Matrix] = None, n0: float = 1.0, s0: float = 1.0,
                 beta: float = 1.0):
        self.F = list(F)
        self.G = [row[:] for row in G]
        self.m = len(F)
        self.delta = delta
        self.beta = beta
        self.m0 = list(m0) if m0 is not None else [0.0] * self.m
        self.C0 = [row[:] for row in C0] if C0 is not None else eye(self.m, 100.0)
        self.n0 = n0
        self.s0 = s0

    def filter(self, y: Sequence[float]) -> DLMFilterResult:
        m_t = list(self.m0)
        C_t = [row[:] for row in self.C0]
        n_t = self.n0
        s_t = self.s0
        fmean, fcov, fc, fv, scales, dofs = [], [], [], [], [], []
        loglike = 0.0
        for yt in y:
            a = matvec(self.G, m_t)                          # prior mean
            GC = matmul(self.G, C_t)
            R = mat_scale(mat_add(matmul(GC, transpose(self.G)), zeros(self.m, self.m)),
                          1.0 / self.delta)
            f = _kahan_sum(self.F[i] * a[i] for i in range(self.m))
            RF = matvec(R, self.F)
            q = _kahan_sum(self.F[i] * RF[i] for i in range(self.m)) + s_t
            fc.append(f)
            fv.append(q)
            if yt is None or (isinstance(yt, float) and math.isnan(yt)):
                m_t, C_t = a, R
                fmean.append(list(m_t)); fcov.append([r[:] for r in C_t])
                scales.append(s_t); dofs.append(n_t)
                continue
            e = float(yt) - f
            A = [RF[i] / q for i in range(self.m)]
            n_new = self.beta * n_t + 1.0
            s_new = s_t * (self.beta * n_t + e * e / q) / n_new
            m_t = vec_add(a, vec_scale(A, e))
            C_t = mat_scale(mat_sub(R, mat_scale(outer(A, A), q)), s_new / s_t)
            C_t = symmetrize(C_t)
            # Student-t predictive log density with n_t dof, scale q
            loglike += (_gammaln((n_t + 1) / 2.0) - _gammaln(n_t / 2.0)
                        - 0.5 * math.log(math.pi * n_t * q)
                        - 0.5 * (n_t + 1) * math.log(1.0 + e * e / (n_t * q)))
            n_t, s_t = n_new, s_new
            fmean.append(list(m_t)); fcov.append([r[:] for r in C_t])
            scales.append(s_t); dofs.append(n_t)
        return DLMFilterResult(fmean, fcov, fc, fv, scales, dofs, loglike)

    def forecast(self, y: Sequence[float], h: int) -> Tuple[List[float], List[float]]:
        """``h``-step forecasts (mean + variance) continuing from the filtered state."""
        res = self.filter(y)
        m_t = list(res.filtered_mean[-1])
        C_t = [row[:] for row in res.filtered_cov[-1]]
        s_t = res.obs_scale[-1]
        means, vars_ = [], []
        a = list(m_t)
        R = [row[:] for row in C_t]
        for k in range(h):
            a = matvec(self.G, a)
            GR = matmul(self.G, R)
            R = mat_scale(matmul(GR, transpose(self.G)), 1.0 / self.delta)
            f = _kahan_sum(self.F[i] * a[i] for i in range(self.m))
            RF = matvec(R, self.F)
            q = _kahan_sum(self.F[i] * RF[i] for i in range(self.m)) + s_t
            means.append(f)
            vars_.append(q)
        return means, vars_


# ===========================================================================
# 11. NONLINEAR / NON-GAUSSIAN FILTERS: EKF, UKF, particle filter
# ===========================================================================
@dataclass
class NonlinearFilterResult:
    filtered_state: List[Vector]
    filtered_cov: List[Matrix]
    predicted_state: List[Vector]
    predicted_cov: List[Matrix]
    loglike: float


def _num_jacobian(f: Callable[[Sequence[float]], Sequence[float]],
                  x: Sequence[float], *, eps: float = 1e-6) -> Matrix:
    """Central-difference Jacobian of a vector function."""
    n = len(x)
    fx = list(f(x))
    p = len(fx)
    J = zeros(p, n)
    for j in range(n):
        xp = list(x); xp[j] += eps
        xm = list(x); xm[j] -= eps
        fp = f(xp); fm = f(xm)
        for i in range(p):
            J[i][j] = (fp[i] - fm[i]) / (2.0 * eps)
    return J


def extended_kalman_filter(f: Callable[[Sequence[float]], Sequence[float]],
                           h: Callable[[Sequence[float]], Sequence[float]],
                           Q: Matrix, R: Matrix, x0: Sequence[float], P0: Matrix,
                           y: Sequence[Sequence[float]], *,
                           Fjac: Optional[Callable] = None,
                           Hjac: Optional[Callable] = None) -> NonlinearFilterResult:
    """Extended Kalman filter for ``x_{t+1} = f(x_t) + w``, ``y_t = h(x_t) + v`` with
    ``w ~ N(0,Q)``, ``v ~ N(0,R)``.  Jacobians of ``f`` / ``h`` are supplied or
    computed by central differences."""
    x = list(x0)
    P = [row[:] for row in P0]
    fs, fc, ps, pc = [], [], [], []
    loglike = 0.0
    for t, yt in enumerate(y):
        # predict (x0/P0 is the prior for the FIRST observation, matching the
        # linear-Gaussian Kalman convention -- so EKF reduces exactly to it)
        if t == 0:
            x_pred = list(x)
            P_pred = [r[:] for r in P]
        else:
            F = Fjac(x) if Fjac else _num_jacobian(f, x)
            x_pred = list(f(x))
            P_pred = symmetrize(mat_add(matmul(matmul(F, P), transpose(F)), Q))
        ps.append(list(x_pred)); pc.append([r[:] for r in P_pred])
        # update
        obs = [i for i in range(len(yt))
               if yt[i] is not None and not (isinstance(yt[i], float) and math.isnan(yt[i]))]
        if obs:
            H = Hjac(x_pred) if Hjac else _num_jacobian(h, x_pred)
            hx = list(h(x_pred))
            Hs = [H[i] for i in obs]
            Rs = [[R[i][j] for j in obs] for i in obs]
            v = [float(yt[i]) - hx[i] for i in obs]
            PHt = matmul(P_pred, transpose(Hs))
            S = mat_add(matmul(Hs, PHt), Rs)
            _sr = max((abs(S[i][i]) for i in range(len(S))), default=0.0)
            Sinv = inv(S, ridge=(1e-14 * _sr) or 1e-14)    # scale-relative ridge
            K = matmul(PHt, Sinv)
            x = vec_add(x_pred, matvec(K, v))
            KH = matmul(K, Hs)
            ImKH = mat_sub(eye(len(x)), KH)
            P = symmetrize(matmul(ImKH, P_pred))
            loglike += -0.5 * (len(obs) * _LOG2PI + logdet(S)
                               + _kahan_sum(v[i] * matvec(Sinv, v)[i]
                                            for i in range(len(obs))))
        else:
            x, P = x_pred, P_pred
        fs.append(list(x)); fc.append([r[:] for r in P])
    return NonlinearFilterResult(fs, fc, ps, pc, loglike)


def unscented_kalman_filter(f: Callable[[Sequence[float]], Sequence[float]],
                            h: Callable[[Sequence[float]], Sequence[float]],
                            Q: Matrix, R: Matrix, x0: Sequence[float], P0: Matrix,
                            y: Sequence[Sequence[float]], *, alpha: float = 1e-3,
                            beta: float = 2.0,
                            kappa: float = 0.0) -> NonlinearFilterResult:
    """Unscented Kalman filter (sigma-point / unscented transform) -- derivative
    free and more accurate than the EKF for strongly nonlinear ``f`` / ``h``."""
    n = len(x0)
    lam = alpha * alpha * (n + kappa) - n
    c = n + lam
    Wm = [lam / c] + [0.5 / c] * (2 * n)
    Wc = [lam / c + (1 - alpha * alpha + beta)] + [0.5 / c] * (2 * n)

    def sigma_points(x: Sequence[float], P: Matrix) -> List[Vector]:
        Ps = mat_scale(P, c)
        _pr = max((abs(Ps[i][i]) for i in range(len(Ps))), default=0.0)
        L = cholesky(Ps, jitter=(1e-12 * _pr) or 1e-12)    # scale-relative jitter
        pts = [list(x)]
        for i in range(n):
            col = [L[j][i] for j in range(n)]
            pts.append(vec_add(x, col))
            pts.append(vec_sub(x, col))
        return pts

    x = list(x0)
    P = [row[:] for row in P0]
    fs, fc, ps, pc = [], [], [], []
    loglike = 0.0
    for t, yt in enumerate(y):
        if t == 0:                                       # x0/P0 = prior for y_0
            x_pred = list(x)
            P_pred = [r[:] for r in P]
        else:
            pts = sigma_points(x, P)
            fp = [list(f(pt)) for pt in pts]
            x_pred = [_kahan_sum(Wm[k] * fp[k][i] for k in range(2 * n + 1))
                      for i in range(n)]
            P_pred = [row[:] for row in Q]
            for k in range(2 * n + 1):
                dx = vec_sub(fp[k], x_pred)
                for i in range(n):
                    for j in range(n):
                        P_pred[i][j] += Wc[k] * dx[i] * dx[j]
            P_pred = symmetrize(P_pred)
        ps.append(list(x_pred)); pc.append([r[:] for r in P_pred])
        obs = [i for i in range(len(yt))
               if yt[i] is not None and not (isinstance(yt[i], float) and math.isnan(yt[i]))]
        if obs:
            pts2 = sigma_points(x_pred, P_pred)
            hp = [[list(h(pt))[i] for i in obs] for pt in pts2]
            p = len(obs)
            y_mean = [_kahan_sum(Wm[k] * hp[k][i] for k in range(2 * n + 1))
                      for i in range(p)]
            S = [[R[obs[i]][obs[j]] for j in range(p)] for i in range(p)]
            Cxy = zeros(n, p)
            for k in range(2 * n + 1):
                dy = vec_sub(hp[k], y_mean)
                dx = vec_sub(pts2[k], x_pred)
                for i in range(p):
                    for j in range(p):
                        S[i][j] += Wc[k] * dy[i] * dy[j]
                for i in range(n):
                    for j in range(p):
                        Cxy[i][j] += Wc[k] * dx[i] * dy[j]
            _sr = max((abs(S[i][i]) for i in range(len(S))), default=0.0)
            Sinv = inv(S, ridge=(1e-14 * _sr) or 1e-14)    # scale-relative ridge
            K = matmul(Cxy, Sinv)
            v = [float(yt[obs[i]]) - y_mean[i] for i in range(p)]
            x = vec_add(x_pred, matvec(K, v))
            P = symmetrize(mat_sub(P_pred, matmul(matmul(K, S), transpose(K))))
            loglike += -0.5 * (p * _LOG2PI + logdet(S)
                               + _kahan_sum(v[i] * matvec(Sinv, v)[i] for i in range(p)))
        else:
            x, P = x_pred, P_pred
        fs.append(list(x)); fc.append([r[:] for r in P])
    return NonlinearFilterResult(fs, fc, ps, pc, loglike)


@dataclass
class ParticleFilterResult:
    filtered_mean: List[Vector]
    filtered_cov: List[Matrix]
    ess: List[float]                                # effective sample size per step
    loglike: float                                  # marginal likelihood estimate


def _systematic_resample(weights: Sequence[float], rng: DeterministicRNG) -> List[int]:
    n = len(weights)
    u0 = rng.random()                                    # ONE shared uniform
    positions = [(i + u0) / n for i in range(n)]
    cum = list(itertools.accumulate(weights))
    idx = [0] * n
    i = j = 0
    while i < n:
        if positions[i] < cum[j]:
            idx[i] = j
            i += 1
        else:
            j += 1
            if j >= n:
                j = n - 1
    return idx


def particle_filter(transition_sample: Callable[[Sequence[float], int, DeterministicRNG], Sequence[float]],
                    obs_loglik: Callable[[Sequence[float], Sequence[float], int], float],
                    initial_sample: Callable[[DeterministicRNG], Sequence[float]],
                    y: Sequence[Sequence[float]], *, n_particles: int = 500,
                    seed: int = 0, ess_threshold: float = 0.5
                    ) -> ParticleFilterResult:
    """Bootstrap particle filter (sequential importance resampling) for a general
    state-space model with a user-supplied transition sampler, observation
    log-likelihood and initial sampler.  Uses systematic resampling when the
    effective sample size falls below ``ess_threshold * n_particles`` and returns
    an unbiased marginal-likelihood estimate."""
    rng = DeterministicRNG(seed)
    N = n_particles
    parts = [list(initial_sample(rng)) for _ in range(N)]
    logw = [0.0] * N
    means, covs, esss = [], [], []
    loglike = 0.0
    for t, yt in enumerate(y):
        if t > 0:
            parts = [list(transition_sample(parts[i], t, rng)) for i in range(N)]
        ll = [obs_loglik(parts[i], yt, t) for i in range(N)]
        # marginal-likelihood increment is log sum exp(logw+ll) - log sum exp(logw)
        # (telescoping).  With adaptive resampling logw carries over accumulated
        # weights between steps, so subtracting log(N) -- valid only when logw is
        # uniform (t=0 or just after a resample) -- would badly bias the estimate.
        mx_old = max(logw)
        lse_old = mx_old + math.log(_kahan_sum(math.exp(lw - mx_old) for lw in logw))
        logw = [logw[i] + ll[i] for i in range(N)]
        mx = max(logw)
        w = [math.exp(lw - mx) for lw in logw]
        sw = _kahan_sum(w)
        loglike += (mx + math.log(sw)) - lse_old
        w = [wi / sw for wi in w]
        d = len(parts[0])
        mean = [_kahan_sum(w[i] * parts[i][k] for i in range(N)) for k in range(d)]
        cov = zeros(d, d)
        for i in range(N):
            dx = vec_sub(parts[i], mean)
            for a in range(d):
                for b in range(d):
                    cov[a][b] += w[i] * dx[a] * dx[b]
        means.append(mean); covs.append(symmetrize(cov))
        ess = 1.0 / _kahan_sum(wi * wi for wi in w)
        esss.append(ess)
        if ess < ess_threshold * N:
            idx = _systematic_resample(w, rng)
            parts = [list(parts[i]) for i in idx]
            logw = [0.0] * N
    return ParticleFilterResult(means, covs, esss, loglike)


# ===========================================================================
# 12. FORECASTING, SIGNAL EXTRACTION, IN-SAMPLE PREDICTION
# ===========================================================================
def _as_obs(y: Sequence[Any]) -> List[List[Any]]:
    """Normalise an input series to a list of observation vectors."""
    if not y:
        return []
    if isinstance(y[0], (list, tuple)):
        return [list(row) for row in y]
    return [[v] for v in y]


@dataclass
class ForecastResult:
    mean: List[Vector]                     # h forecast vectors (k_obs each)
    variance: List[Matrix]                 # forecast covariance per step
    lower: Dict[float, List[Vector]]
    upper: Dict[float, List[Vector]]
    level: List[float]
    state_mean: List[Vector]               # forecast state path
    state_cov: List[Matrix]

    def mean1d(self) -> List[float]:
        """First observation dimension of the forecast (univariate convenience)."""
        return [m[0] for m in self.mean]


def _check_level(level: float) -> float:
    """A prediction-interval confidence level must be a FRACTION in (0, 1); a
    percentage (95) or out-of-range value would silently give infinite- or
    negative-width bands via ``norm_ppf`` of an argument outside (0, 1)."""
    if not (isinstance(level, (int, float)) and 0.0 < level < 1.0):
        raise ValueError("confidence level must be a fraction in (0, 1), got %r"
                         % (level,))
    return level


def forecast(model_or_fit: Union[LinearGaussianSSM, StateSpaceFit],
             y: Sequence[Any], h: int, *,
             level: Sequence[float] = (0.80, 0.95)) -> ForecastResult:
    """Forecast ``h`` steps ahead: filter to the end of ``y`` then propagate the
    state (with no further observations), returning the predicted observation mean,
    covariance and Gaussian prediction bands per horizon."""
    model = model_or_fit.model if isinstance(model_or_fit, StateSpaceFit) \
        else model_or_fit
    if not isinstance(model, LinearGaussianSSM):
        raise TypeError("forecast expects a LinearGaussianSSM or a fitted "
                        "StateSpaceFit; got %s -- did you pass an unfitted model "
                        "(call .fit() first)?" % type(model_or_fit).__name__)
    if not _time_invariant(model):
        # a time-varying model (Regression / dynamic-regression / exog, or any
        # per-time Z/T/...) has UNKNOWN system matrices beyond the sample; silently
        # padding them (e.g. exog -> 0) would return a wrong forecast, so refuse.
        raise ValueError("forecast does not support time-varying models (e.g. a "
                         "Regression / exog component): the future system matrices "
                         "are unknown beyond the observed sample")
    for lv in level:
        _check_level(lv)
    obs = _as_obs(y)
    n = len(obs)
    if n == 0:
        raise ValueError("forecast requires at least one observation")
    m = model.k_states
    kf = kalman_filter(model, obs, diffuse_method="bigkappa")
    a = list(kf.filtered_state[-1])
    P = [row[:] for row in kf.filtered_cov[-1]]
    means, variances = [], []
    state_mean, state_cov = [], []
    lower: Dict[float, List[Vector]] = {lv: [] for lv in level}
    upper: Dict[float, List[Vector]] = {lv: [] for lv in level}
    for k in range(h):
        t = n + k
        T = model.T_t(t - 1)
        cc = model.c_t(t - 1)
        RQR = model.RQR_t(t - 1)
        a = vec_add(cc, matvec(T, a))
        P = symmetrize(mat_add(matmul(matmul(T, P), transpose(T)), RQR))
        state_mean.append(list(a)); state_cov.append([r[:] for r in P])
        Z = model.Z_t(t)
        d = model.d_t(t)
        H = model.H_t(t)
        p = len(Z)
        ymean = [d[i] + _kahan_sum(Z[i][j] * a[j] for j in range(m)) for i in range(p)]
        ZP = [matvec(P, Z[i]) for i in range(p)]
        F = [[H[i][j] + _kahan_sum(Z[i][kk] * ZP[j][kk] for kk in range(m))
              for j in range(p)] for i in range(p)]
        means.append(ymean); variances.append(symmetrize(F))
        for lv in level:
            z = norm_ppf(0.5 + lv / 2.0)
            sd = [math.sqrt(max(F[i][i], 0.0)) for i in range(p)]
            lower[lv].append([ymean[i] - z * sd[i] for i in range(p)])
            upper[lv].append([ymean[i] + z * sd[i] for i in range(p)])
    return ForecastResult(means, variances, lower, upper, list(level),
                          state_mean, state_cov)


def predict_in_sample(model: LinearGaussianSSM, y: Sequence[Any]) -> List[Vector]:
    """One-step-ahead in-sample predictions E[y_t | y_1..y_{t-1}]."""
    obs = _as_obs(y)
    if not obs:
        raise ValueError("predict_in_sample requires at least one observation")
    kf = kalman_filter(model, obs, diffuse_method="bigkappa")
    out = []
    for t in range(len(obs)):
        Z = model.Z_t(t)
        d = model.d_t(t)
        a = kf.predicted_state[t]
        out.append([d[i] + _kahan_sum(Z[i][j] * a[j] for j in range(model.k_states))
                    for i in range(len(Z))])
    return out


def decompose(fit_or_model: Union[LinearGaussianSSM, StateSpaceFit, "UnobservedComponents"],
              y: Optional[Sequence[Any]] = None) -> Dict[str, List[float]]:
    """Smoothed structural decomposition: the estimated contribution of each named
    component (trend / seasonal / cycle / ...) to the univariate series."""
    ucm = None
    if isinstance(fit_or_model, UnobservedComponents):
        ucm = fit_or_model
        model = (ucm._fitted.model if getattr(ucm, "_fitted", None) is not None
                 else ucm.build(ucm.start_params()))
        obs = ucm.y
    elif isinstance(fit_or_model, StateSpaceFit):
        ucm = getattr(fit_or_model, "_ucm", None)
        model = fit_or_model.model
        obs = ucm.y if ucm is not None else (_as_obs(y) if y is not None else None)
    else:
        model = fit_or_model
        obs = _as_obs(y) if y is not None else None
    if obs is None:
        raise ValueError("decompose needs the observed series y")
    sm = disturbance_smoother(model, obs)
    out: Dict[str, List[float]] = {}
    if ucm is not None:                                  # split by component loading
        off = 0
        for c in ucm._state_comps:
            idxs = [off + i for i in range(c.k_states)]
            series = []
            for t in range(len(obs)):
                zr_t = c.Z_row(t)
                series.append(_kahan_sum(zr_t[i] * sm.smoothed_state[t][idxs[i]]
                                         for i in range(c.k_states)))
            out[c.name] = series
            off += c.k_states
    else:
        for k in range(model.k_states):
            out["state%d" % k] = sm.component(k)
    return out


# ===========================================================================
# 13. DIAGNOSTICS
# ===========================================================================
def _acf(x: Sequence[float], nlags: int) -> List[float]:
    n = len(x)
    mu = _mean(x)
    dev = [v - mu for v in x]
    c0 = _kahan_sum(d * d for d in dev) / n
    out = [1.0]
    for k in range(1, nlags + 1):
        ck = _kahan_sum(dev[t] * dev[t - k] for t in range(k, n)) / n
        out.append(ck / c0 if c0 > _EPS else 0.0)
    return out


@dataclass
class LjungBoxResult:
    statistic: float
    pvalue: float
    lags: int
    dof: int


def ljung_box(resid: Sequence[float], *, lags: int = 10, model_df: int = 0
              ) -> LjungBoxResult:
    """Ljung-Box portmanteau test for residual autocorrelation (df corrected by the
    number of estimated parameters)."""
    resid = [r for r in resid if r is not None and not (isinstance(r, float) and math.isnan(r))]
    n = len(resid)
    if n == 0:
        raise ValueError("ljung_box requires a non-empty residual series")
    ac = _acf(resid, lags)
    stat = n * (n + 2.0) * _kahan_sum(ac[k] * ac[k] / (n - k) for k in range(1, lags + 1))
    dof = max(lags - model_df, 1)
    return LjungBoxResult(stat, chi2_sf(stat, df=dof), lags, dof)


@dataclass
class NormalityTestResult:
    statistic: float
    pvalue: float
    skew: float
    kurtosis: float


def jarque_bera(resid: Sequence[float]) -> NormalityTestResult:
    """Jarque-Bera test of residual normality (skewness + excess kurtosis)."""
    resid = [r for r in resid if r is not None and not (isinstance(r, float) and math.isnan(r))]
    n = len(resid)
    mu = _mean(resid)
    dev = [r - mu for r in resid]
    m2 = _kahan_sum(d * d for d in dev) / n
    m3 = _kahan_sum(d ** 3 for d in dev) / n
    m4 = _kahan_sum(d ** 4 for d in dev) / n
    skew = m3 / (m2 ** 1.5) if m2 > _EPS else 0.0
    kurt = m4 / (m2 * m2) if m2 > _EPS else 0.0
    jb = n / 6.0 * (skew * skew + 0.25 * (kurt - 3.0) ** 2)
    return NormalityTestResult(jb, chi2_sf(jb, df=2), skew, kurt)


def durbin_watson(resid: Sequence[float]) -> float:
    """Durbin-Watson statistic (~2 = no first-order autocorrelation)."""
    num = _kahan_sum((resid[t] - resid[t - 1]) ** 2 for t in range(1, len(resid)))
    den = _kahan_sum(r * r for r in resid)
    return _safe_div(num, den, default=2.0)


def diagnose_residuals(model: LinearGaussianSSM, y: Sequence[Any], *,
                       lags: int = 10, model_df: int = 0) -> Dict[str, Any]:
    """Standardised one-step prediction errors and their portmanteau / normality
    diagnostics -- the state-space analogue of ARIMA residual checks."""
    obs = _as_obs(y)
    kf = kalman_filter(model, obs)
    std = [row[0] for row in kf.standardized_residuals()
           if row and _row_observed_scalar(row[0])]
    lb = ljung_box(std, lags=lags, model_df=model_df)
    jb = jarque_bera(std)
    return {"standardized": std, "ljung_box": lb, "jarque_bera": jb,
            "durbin_watson": durbin_watson(std),
            "n": len(std)}


def _row_observed_scalar(v: float) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


# ===========================================================================
# 14. MODEL PRESETS
# ===========================================================================
def local_level(y: Sequence[float], **fit_kw: Any) -> StateSpaceFit:
    """Fit a local-level (random-walk-plus-noise) model by ML."""
    return UnobservedComponents(y, [LocalLevel(), Irregular()],
                                name="local_level").fit(**fit_kw)


def local_linear_trend(y: Sequence[float], *, damped: bool = False,
                       **fit_kw: Any) -> StateSpaceFit:
    """Fit a local-linear-trend model by ML."""
    return UnobservedComponents(
        y, [LocalLinearTrend(damped=damped), Irregular()],
        name="local_linear_trend").fit(**fit_kw)


def basic_structural_model(y: Sequence[float], *, period: int,
                           trigonometric: bool = False,
                           **fit_kw: Any) -> StateSpaceFit:
    """Fit the basic structural model (trend + seasonal + irregular) by ML."""
    seas = (TrigonometricSeasonal(period) if trigonometric
            else SeasonalDummy(period))
    return UnobservedComponents(
        y, [LocalLinearTrend(), seas, Irregular()], name="bsm").fit(**fit_kw)


def arma_to_ssm(*, phi: Sequence[float] = (), theta: Sequence[float] = (),
                sigma2: float = 1.0, mu: float = 0.0) -> LinearGaussianSSM:
    """State-space (Harvey) form of a stationary ARMA(p, q) process -- so ARMA can
    be embedded as a component of a larger structural model or filtered exactly."""
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
    Z = [[1.0] + [0.0] * (r - 1)]
    Q = [[sigma2]]
    H = [[0.0]]
    a1 = [0.0] * r
    try:
        P1 = solve_discrete_lyapunov(T, matmul(matmul(R, Q), transpose(R)))
    except (ValueError, ZeroDivisionError):
        P1 = eye(r, sigma2)
    d = [mu] if mu else None
    return LinearGaussianSSM(Z=Z, T=T, R=R, Q=Q, H=H, a1=a1, P1=P1, d=d,
                             name="arma_ssm")


def var_to_ssm(coefs: Sequence[Matrix], sigma_u: Matrix, *,
               intercept: Optional[Sequence[float]] = None) -> LinearGaussianSSM:
    """State-space form of a VAR(p): stacks the ``p`` lags of a ``k``-vector into the
    state so the multivariate series is filtered / forecast by the same engine."""
    p = len(coefs)
    k = len(coefs[0])
    m = k * p
    T = zeros(m, m)
    for j in range(p):
        for a in range(k):
            for b in range(k):
                T[a][j * k + b] = coefs[j][a][b]
    for i in range(k, m):
        T[i][i - k] = 1.0
    R = [[1.0 if (i < k and i == jj) else 0.0 for jj in range(k)] for i in range(m)]
    Q = [row[:] for row in sigma_u]
    Z = [[1.0 if j == i else 0.0 for j in range(m)] for i in range(k)]
    H = zeros(k, k)
    a1 = [0.0] * m
    try:
        P1 = solve_discrete_lyapunov(T, matmul(matmul(R, Q), transpose(R)))
    except (ValueError, ZeroDivisionError):
        P1 = eye(m, 1.0)
    d = list(intercept) if intercept else None
    return LinearGaussianSSM(Z=Z, T=T, R=R, Q=Q, H=H, a1=a1, P1=P1, d=d,
                             name="var_ssm")


# ===========================================================================
# 15. SIMULATORS
# ===========================================================================
def simulate_ssm(model: LinearGaussianSSM, n: int, *, seed: int = 0
                 ) -> Tuple[List[Vector], List[Vector]]:
    """Draw a state path and observation path of length ``n`` from ``model``."""
    return _simulate_path(model, n, DeterministicRNG(seed))


def sample_series(n: int, *, level: float = 10.0, trend: float = 0.0,
                  period: int = 0, seasonal_amp: float = 0.0,
                  noise: float = 1.0, level_noise: float = 0.1,
                  seed: int = 0) -> List[float]:
    """Synthetic univariate series (level + trend + seasonal + noise) for demos."""
    rng = DeterministicRNG(seed)
    lvl = level
    out = []
    for t in range(n):
        lvl += trend + level_noise * rng.gauss()
        s = seasonal_amp * math.sin(2.0 * math.pi * (t % period) / period) if period >= 2 else 0.0
        out.append(lvl + s + noise * rng.gauss())
    return out


# ===========================================================================
# 16. PAYLOADS + DEMO
# ===========================================================================
def build_result_payload(fit: StateSpaceFit) -> Dict[str, Any]:
    """Strict-JSON summary of a fitted state-space model (leaderboard-ready)."""
    return _json_safe({
        "model": fit.model.name,
        "method": fit.method,
        "loglike": fit.loglike,
        "aic": fit.aic, "bic": fit.bic, "aicc": fit.aicc, "hqic": fit.hqic,
        "nobs": fit.nobs, "n_params": fit.n_params,
        "k_states": fit.model.k_states,
        "params": {nm: v for nm, v in zip(fit.param_names, fit.params)},
        "converged": fit.converged,
    })


def build_a2ui_payload(fits: Dict[str, StateSpaceFit]) -> Dict[str, Any]:
    """Leaderboard payload ranking several fitted models by AIC."""
    rows = []
    for name, fit in sorted(fits.items()):
        p = build_result_payload(fit)
        p["name"] = name
        rows.append(p)
    rows.sort(key=lambda r: (r["aic"] if r["aic"] is not None else _INF))
    return _json_safe({"models": rows,
                       "best": rows[0]["name"] if rows else None})


def _demo() -> None:
    """Byte-deterministic demonstration across the whole toolkit."""
    print("=" * 60)
    print("State-Space Models & Dynamic Linear Models  v%s  -- demo" % __version__)
    print("=" * 60)
    y = sample_series(60, level=10.0, trend=0.15, period=12, seasonal_amp=3.0,
                      noise=0.6, level_noise=0.08, seed=7)
    yobs = [[v] for v in y]

    print("\n[1] Local level vs local linear trend (ML, exact diffuse):")
    ll = local_level(y, maxiter=300, restarts=0)
    llt = local_linear_trend(y, maxiter=300, restarts=0)
    print("    local level : logL=%.3f AIC=%.2f" % (ll.loglike, ll.aic))
    print("    local trend : logL=%.3f AIC=%.2f" % (llt.loglike, llt.aic))

    print("\n[2] Basic structural model (trend + seasonal(12) + irregular), fit once:")
    bsm_uc = UnobservedComponents(y, [LocalLinearTrend(), SeasonalDummy(12),
                                      Irregular()], name="bsm")
    bsm = bsm_uc.fit(maxiter=250, restarts=0)
    print("    k_states=%d  logL=%.3f AIC=%.2f" % (bsm.model.k_states, bsm.loglike,
                                                   bsm.aic))
    print("    variances: %s" % {k: round(v, 4)
                                 for k, v in zip(bsm.param_names, bsm.params)})

    print("\n[3] Smoothed structural decomposition (component tails):")
    dec = decompose(bsm)
    for nm in ("trend", "seasonal"):
        print("    %-9s tail = %s" % (nm, [round(v, 3) for v in dec[nm][-3:]]))

    print("\n[4] Forecast 8 steps ahead (90%% band):")
    fc = forecast(bsm, y, 8, level=(0.90,))
    print("    mean = %s" % [round(v, 2) for v in fc.mean1d()[:6]])
    print("    lo90 = %s" % [round(v[0], 2) for v in fc.lower[0.90][:6]])

    print("\n[5] Residual diagnostics + auxiliary residuals:")
    dg = diagnose_residuals(bsm.model, y)
    aux = auxiliary_residuals(bsm.model, yobs)
    n_out = sum(1 for row in aux["observation"] for x in row if abs(x) > 3.0)
    print("    Ljung-Box p=%.3f  Jarque-Bera p=%.3f  DW=%.3f  outliers>3sd=%d" % (
        dg["ljung_box"].pvalue, dg["jarque_bera"].pvalue, dg["durbin_watson"], n_out))

    print("\n[6] EM estimation of a local level:")
    m0 = UnobservedComponents(y, [LocalLevel(), Irregular()]).build([1.0, 1.0])
    em = em_algorithm(m0, yobs, max_iter=30)
    print("    EM logL=%.3f  H=%.4f  Q=%.4f" % (em.loglike, em.params[0], em.params[1]))

    print("\n[7] West-Harrison discount DLM + FFBS posterior draws:")
    dlm = DiscountDLM(F=[1.0, 0.0], G=[[1.0, 1.0], [0.0, 1.0]], delta=0.95,
                      m0=[10.0, 0.0], C0=eye(2, 10.0))
    dres = dlm.filter(y)
    fm, _ = dlm.forecast(y, 4)
    print("    DLM logL=%.3f  4-step forecast=%s" % (
        dres.loglike, [round(v, 2) for v in fm]))
    draws = ffbs(bsm.model, yobs, n_draws=15, seed=1)
    trend_draw = [d[-1][0] for d in draws]
    print("    FFBS trend[-1] posterior mean=%.3f sd=%.3f (15 draws)" % (
        _mean(trend_draw), _std(trend_draw)))

    print("\n[8] Nonlinear filters (EKF / UKF / particle):")
    f = lambda x: [x[0]]
    hh = lambda x: [x[0]]
    ekf = extended_kalman_filter(f, hh, [[0.3]], [[0.6]], [y[0]], [[10.0]], yobs)
    ukf = unscented_kalman_filter(f, hh, [[0.3]], [[0.6]], [y[0]], [[10.0]], yobs)
    print("    EKF logL=%.3f  UKF logL=%.3f  (agree on a linear model)" % (
        ekf.loglike, ukf.loglike))

    def tr(x, t, rng):
        return [x[0] + math.sqrt(0.3) * rng.gauss()]

    def oll(x, yt, t):
        return -0.5 * (math.log(2 * math.pi * 0.6) + (yt[0] - x[0]) ** 2 / 0.6)

    pf = particle_filter(tr, oll, lambda rng: [y[0] + 5.0 * rng.gauss()],
                         yobs, n_particles=300, seed=1)
    print("    particle filter logL=%.3f  ESS[-1]=%.0f / 300" % (
        pf.loglike, pf.ess[-1]))

    print("\n[9] Leaderboard payload (ranked by AIC):")
    payload = build_a2ui_payload({"local_level": ll, "local_trend": llt, "bsm": bsm})
    print("    best model = %s   JSON-safe = %s" % (
        payload["best"], _json_dumpable(payload)))

    print("\n" + "=" * 60)
    print("  demo OK")


def _json_dumpable(obj: Any) -> bool:
    import json
    try:
        json.dumps(obj, allow_nan=False)
        return True
    except (ValueError, TypeError):
        return False


__all__ = [
    "__version__",
    # special functions / RNG
    "norm_pdf", "norm_cdf", "norm_ppf", "gammainc", "betainc",
    "chi2_cdf", "chi2_sf", "chi2_ppf", "t_cdf", "t_ppf", "DeterministicRNG",
    # linear algebra
    "zeros", "eye", "transpose", "matmul", "matvec", "mat_add", "mat_sub",
    "mat_scale", "outer", "diag", "kron", "block_diag", "solve", "solve_matrix",
    "inv", "logdet", "cholesky", "chol_solve", "qr_householder", "qr_lstsq",
    "nearest_pd", "solve_discrete_lyapunov", "spectral_radius",
    "durand_kerner_roots",
    # model + filter + smoother
    "LinearGaussianSSM", "KalmanFilterResult", "kalman_filter",
    "SmootherResult", "disturbance_smoother", "kalman_smoother",
    "simulation_smoother", "auxiliary_residuals",
    # estimation
    "loglike", "ICResult", "information_criteria", "minimize_nelder_mead",
    "StateSpaceFit", "fit_ssm_mle", "em_algorithm",
    # unobserved components
    "Component", "Irregular", "LocalLevel", "LocalLinearTrend", "SeasonalDummy",
    "TrigonometricSeasonal", "Cycle", "AutoRegressive", "Regression",
    "UnobservedComponents",
    # bayesian
    "ffbs", "DiscountDLM", "DLMFilterResult",
    # nonlinear filters
    "extended_kalman_filter", "unscented_kalman_filter", "particle_filter",
    "NonlinearFilterResult", "ParticleFilterResult",
    # forecasting / decomposition / diagnostics
    "ForecastResult", "forecast", "predict_in_sample", "decompose",
    "ljung_box", "jarque_bera", "durbin_watson", "diagnose_residuals",
    "LjungBoxResult", "NormalityTestResult",
    # presets
    "local_level", "local_linear_trend", "basic_structural_model",
    "arma_to_ssm", "var_to_ssm",
    # simulators + payloads
    "simulate_ssm", "sample_series", "build_result_payload", "build_a2ui_payload",
]


if __name__ == "__main__":
    _demo()
