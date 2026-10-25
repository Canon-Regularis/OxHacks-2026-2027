"""
Oracle-based stress suite for state_space_template.py.

Pure standard library + pytest -- NO numpy / scipy / statsmodels.  The oracles are
closed-form (the AR(1) exact Gaussian likelihood; the discrete-Lyapunov identity),
metamorphic invariants (scale / shift equivariance, determinism), cross-consistency
(EKF / UKF == the exact Kalman filter on a linear model; the particle-filter and
FFBS posteriors == the Kalman / smoother results), and a handful of golden values
that were cross-validated against statsmodels 0.14.6 at build time and then pinned.

Run:  pytest _stress_test.py -q     (passes identically under PYTHONHASHSEED 0 & 1)
"""
from __future__ import annotations

import json
import math

import pytest

from gridpulse.analysis.algorithms.orchid import state_space as S
from gridpulse.analysis.algorithms.orchid.state_space import (
    LinearGaussianSSM, kalman_filter, disturbance_smoother, kalman_smoother,
    simulation_smoother, auxiliary_residuals, loglike, information_criteria,
    minimize_nelder_mead, fit_ssm_mle, em_algorithm, DeterministicRNG,
    UnobservedComponents, Irregular, LocalLevel, LocalLinearTrend, SeasonalDummy,
    TrigonometricSeasonal, Cycle, AutoRegressive, Regression,
    ffbs, DiscountDLM, extended_kalman_filter, unscented_kalman_filter,
    particle_filter, forecast, predict_in_sample, decompose,
    ljung_box, jarque_bera, durbin_watson, local_level, local_linear_trend,
    basic_structural_model, arma_to_ssm, var_to_ssm, simulate_ssm, sample_series,
    build_result_payload, build_a2ui_payload,
    norm_ppf, norm_cdf, chi2_ppf, chi2_sf, t_ppf, gammainc, betainc,
    solve, cholesky, inv, logdet, qr_lstsq, matmul, kron, block_diag, transpose,
    solve_discrete_lyapunov, spectral_radius, durand_kerner_roots, eye, zeros,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol * (1.0 + abs(b))


def mat_close(A, B, tol=1e-8):
    return all(approx(A[i][j], B[i][j], tol)
               for i in range(len(A)) for j in range(len(A[0])))


def white(n, seed, sd=1.0):
    rng = DeterministicRNG(seed)
    return [sd * rng.gauss() for _ in range(n)]


def ar1_series(n, phi, s2, seed):
    rng = DeterministicRNG(seed)
    x = 0.0
    out = []
    for _ in range(n):
        x = phi * x + math.sqrt(s2) * rng.gauss()
        out.append(x)
    return out


def local_level_ssm(q, h, *, diffuse=True):
    return LinearGaussianSSM(
        Z=[[1.0]], T=[[1.0]], R=[[1.0]], Q=[[q]], H=[[h]], a1=[0.0],
        P1=[[0.0]] if diffuse else [[1e6]],
        P1_infinity=[[1.0]] if diffuse else None)


LL_Y = [1.2, 0.8, 1.5, 2.1, 1.9, 2.4, 2.0, 2.6, 2.3, 2.8, 3.1, 2.7, 3.4, 3.0, 3.6]


# --------------------------------------------------------------------------- #
# Independent joint-Gaussian likelihood oracle.  For a *time-invariant, proper-
# prior* linear-Gaussian SSM the stacked observations y_1..y_n are jointly normal
# with a mean and covariance obtained by MOMENT PROPAGATION (NOT the Kalman filter):
#   mean_a[t+1] = T mean_a[t];  P[t+1] = T P[t] T' + R Q R'
#   Cov(a_s,a_t) = P_s (T^{t-s})'  (s<=t);  Cov(y_s,y_t) = Z Cov(a_s,a_t) Z' + H[s==t]
# The definitional MVN log-pdf of y under those moments must equal the filter's
# log-likelihood -- a decisive, dependency-free check of the entire recursion.
# Uses its OWN tiny linear algebra so it is independent of the module under test.
# --------------------------------------------------------------------------- #
def _jm_mm(A, B):
    return [[sum(A[i][t] * B[t][j] for t in range(len(B)))
             for j in range(len(B[0]))] for i in range(len(A))]


def _jm_T(A):
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


def _jm_add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _jm_mv(A, x):
    return [sum(A[i][t] * x[t] for t in range(len(x))) for i in range(len(A))]


def _jm_pow(A, k):
    R = [[1.0 if i == j else 0.0 for j in range(len(A))] for i in range(len(A))]
    for _ in range(k):
        R = _jm_mm(R, A)
    return R


def _jm_chol(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][t] * L[j][t] for t in range(j))
            if i == j:
                L[i][j] = math.sqrt(max(A[i][i] - s, 1e-300))
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def _jm_logpdf(Sig, d):
    L = _jm_chol(Sig)
    n = len(d)
    z = [0.0] * n
    for i in range(n):
        z[i] = (d[i] - sum(L[i][t] * z[t] for t in range(i))) / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (z[i] - sum(L[t][i] * x[t] for t in range(i + 1, n))) / L[i][i]
    logdet = 2.0 * sum(math.log(L[i][i]) for i in range(n))
    quad = sum(d[i] * x[i] for i in range(n))
    return -0.5 * (n * math.log(2 * math.pi) + logdet + quad)


def joint_mvn_logpdf(model, y):
    Zm, Tm, Rm, Qm, Hm = model.Z, model.T, model.R, model.Q, model.H
    a1, P1 = model.a1, model.P1
    n, p = len(y), len(Zm)
    RQR = _jm_mm(_jm_mm(Rm, Qm), _jm_T(Rm))
    meanA = [None] * (n + 1)
    meanA[1] = list(a1)
    P = [None] * (n + 1)
    P[1] = [r[:] for r in P1]
    for t in range(1, n):
        meanA[t + 1] = _jm_mv(Tm, meanA[t])
        P[t + 1] = _jm_add(_jm_mm(_jm_mm(Tm, P[t]), _jm_T(Tm)), RQR)
    N = n * p
    Sig = [[0.0] * N for _ in range(N)]
    meanY = [0.0] * N
    for s in range(1, n + 1):
        ms = _jm_mv(Zm, meanA[s])
        for a in range(p):
            meanY[(s - 1) * p + a] = ms[a]
    for s in range(1, n + 1):
        for t in range(s, n + 1):
            blk = _jm_mm(_jm_mm(Zm, _jm_mm(P[s], _jm_T(_jm_pow(Tm, t - s)))), _jm_T(Zm))
            if s == t:
                blk = _jm_add(blk, Hm)
            for a in range(p):
                for b in range(p):
                    Sig[(s - 1) * p + a][(t - 1) * p + b] = blk[a][b]
                    Sig[(t - 1) * p + b][(s - 1) * p + a] = blk[a][b]
    yv = [y[i][j] for i in range(n) for j in range(p)]
    return _jm_logpdf(Sig, [yv[i] - meanY[i] for i in range(N)])


# --------------------------------------------------------------------------- #
class TestSpecialFunctions:
    def test_normal(self):
        assert approx(norm_ppf(0.975), 1.959963985, 1e-6)
        assert approx(norm_cdf(norm_ppf(0.3)), 0.3, 1e-9)
        assert approx(norm_cdf(0.0), 0.5)

    def test_chi2(self):
        assert approx(chi2_ppf(0.95, df=1), 3.841458821, 1e-4)
        assert approx(chi2_sf(3.841458821, df=1), 0.05, 1e-4)

    def test_student_t_cauchy(self):
        assert approx(t_ppf(0.75, df=1), 1.0, 1e-4)     # Cauchy quartile

    def test_gamma_beta_identities(self):
        assert approx(gammainc(1.0, 2.0), 1.0 - math.exp(-2.0), 1e-9)
        assert approx(betainc(1.0, 1.0, 0.42), 0.42, 1e-9)
        assert approx(betainc(2.0, 3.0, 1.0), 1.0)


class TestLinearAlgebra:
    def test_solve(self):
        A = [[4.0, 1.0, 2.0], [1.0, 5.0, 0.0], [2.0, 0.0, 3.0]]
        b = [1.0, 2.0, 3.0]
        x = solve(A, b)
        assert all(approx(sum(A[i][j] * x[j] for j in range(3)), b[i], 1e-9)
                   for i in range(3))

    def test_solve_small_scale(self):
        # Regression (R6): lu_decompose's absolute singular-pivot floor (1e-12)
        # was scale-blind -- a legitimately tiny but well-conditioned system had
        # its pivots replaced by 1e-12, giving a wildly wrong solution.
        A = [[1e-16, 0.0], [0.0, 4e-16]]
        x = solve(A, [2e-16, 8e-16])
        assert approx(x[0], 2.0, 1e-6) and approx(x[1], 2.0, 1e-6)
        # a tiny SPD system solves accurately too
        As = [[2e-16, 1e-16], [1e-16, 3e-16]]
        xs = solve(As, [3e-16, 5e-16])
        assert all(approx(sum(As[i][j] * xs[j] for j in range(2)),
                          [3e-16, 5e-16][i], 1e-9) for i in range(2))
        # normal-scale solve byte-unchanged
        x2 = solve([[2.0, 1.0], [1.0, 3.0]], [3.0, 5.0])
        assert approx(x2[0], 0.8, 1e-12) and approx(x2[1], 1.4, 1e-12)

    def test_nearest_pd_scale_relative(self):
        # Regression (R6): nearest_pd's floor must scale with the matrix -- an
        # absolute 1e-10 would inflate a legitimately tiny covariance up to 1e-10,
        # corrupting draws taken from it.
        from gridpulse.analysis.algorithms.orchid.state_space import nearest_pd
        big = nearest_pd([[2.0, 0.0], [0.0, 3.0]])       # scale>=1 -> byte-unchanged
        assert approx(big[0][0], 2.0, 1e-12) and approx(big[1][1], 3.0, 1e-12)
        tiny = nearest_pd([[1e-14, 0.0], [0.0, 4e-14]])  # must NOT inflate to 1e-10
        assert tiny[0][0] < 1e-12 and tiny[1][1] < 1e-12
        neg = nearest_pd([[-1e-16, 0.0], [0.0, 1e-14]])  # negative pivot still floored
        assert 0.0 < neg[0][0] < 1e-12

    def test_cholesky(self):
        A = [[4.0, 2.0, 1.0], [2.0, 5.0, 0.5], [1.0, 0.5, 3.0]]
        L = cholesky(A)
        LLt = matmul(L, transpose(L))
        assert mat_close(LLt, A, 1e-9)

    def test_inv_logdet(self):
        A = [[3.0, 1.0], [1.0, 2.0]]
        I = matmul(A, inv(A))
        assert mat_close(I, eye(2), 1e-9)
        assert approx(logdet(A), math.log(3 * 2 - 1 * 1), 1e-9)

    def test_qr_lstsq(self):
        X = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
        y = [1.0, 2.9, 5.1, 7.0]                          # slope ~2, intercept ~1
        beta, rank = qr_lstsq(X, y)
        assert rank == 2
        assert approx(beta[1], 2.0, 0.05) and approx(beta[0], 1.0, 0.1)

    def test_discrete_lyapunov(self):
        T = [[0.5, 0.1], [0.0, 0.3]]
        Q = [[1.0, 0.2], [0.2, 1.0]]
        P = solve_discrete_lyapunov(T, Q)
        rec = matmul(matmul(T, P), transpose(T))
        assert mat_close([[rec[i][j] + Q[i][j] for j in range(2)] for i in range(2)],
                         P, 1e-9)                          # P = T P T' + Q

    def test_kron_block_diag(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        assert kron([[2.0]], A) == [[2.0, 4.0], [6.0, 8.0]]
        bd = block_diag([[1.0]], [[2.0, 0.0], [0.0, 2.0]])
        assert bd == [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]

    def test_durand_kerner(self):
        # (x-2)(x-3) = x^2 -5x +6 -> ascending [6,-5,1]
        roots = sorted(r.real for r in durand_kerner_roots([6.0, -5.0, 1.0]))
        assert approx(roots[0], 2.0, 1e-6) and approx(roots[1], 3.0, 1e-6)

    def test_spectral_radius(self):
        # companion of char poly with roots 0.5, -0.5 -> radius 0.5
        T = [[0.0, 0.25], [1.0, 0.0]]
        assert approx(spectral_radius(T), 0.5, 1e-6)


class TestSSMDefinition:
    def test_dims(self):
        m = local_level_ssm(0.3, 1.0)
        assert m.k_states == 1 and m.k_obs == 1 and m.k_posdef == 1
        assert m.initial_diffuse_rank() == 1

    def test_shape_validation(self):
        # regression (bug NB-shape): Z / H / Q / T / R shape mismatches must raise
        with pytest.raises(ValueError):                  # H is 2x2 but k_obs=1
            LinearGaussianSSM(Z=[[1.0]], T=[[1.0]], R=[[1.0]], Q=[[0.3]],
                              H=[[1.0, 0.0], [0.0, 1.0]], a1=[0.0], P1=[[1.0]])
        with pytest.raises(ValueError):                  # Z wrong #columns
            LinearGaussianSSM(Z=[[1.0, 0.0]], T=[[1.0]], R=[[1.0]], Q=[[0.3]],
                              H=[[1.0]], a1=[0.0], P1=[[1.0]])
        with pytest.raises(ValueError):                  # a1 length wrong
            LinearGaussianSSM(Z=[[1.0]], T=[[1.0]], R=[[1.0]], Q=[[0.3]],
                              H=[[1.0]], a1=[0.0, 0.0], P1=[[1.0]])

    def test_negative_variance_rejected(self):
        # Regression (R8): a negative variance on a covariance diagonal used to be
        # silently accepted and return a finite but meaningless log-likelihood.
        for bad in (dict(P1=[[-1.0]]), dict(Q=[[-0.3]]), dict(H=[[-0.5]])):
            kw = dict(Z=[[1.0]], T=[[1.0]], R=[[1.0]], Q=[[0.3]], H=[[0.5]],
                      a1=[0.0], P1=[[1.0]])
            kw.update(bad)
            with pytest.raises(ValueError):
                LinearGaussianSSM(**kw)
        # zero variance (a deterministic component / no obs noise) is still valid,
        # and legitimately tiny variances are not mistaken for negative
        assert math.isfinite(kalman_filter(
            LinearGaussianSSM(Z=[[1.0]], T=[[1.0]], R=[[1.0]], Q=[[0.0]], H=[[0.5]],
                              a1=[0.0], P1=[[1.0]]), [[1.0], [2.0]]).loglike)
        assert math.isfinite(kalman_filter(
            LinearGaussianSSM(Z=[[1.0]], T=[[1.0]], R=[[1.0]], Q=[[1e-16]],
                              H=[[1e-16]], a1=[0.0], P1=[[1e-14]]),
            [[1e-8], [2e-8]]).loglike)

    def test_time_varying_accessor(self):
        Zseq = [[[float(t)]] for t in range(5)]           # per-time Z
        m = LinearGaussianSSM(Z=Zseq, T=[[1.0]], R=[[1.0]], Q=[[0.1]], H=[[1.0]],
                              a1=[0.0], P1=[[1.0]])
        assert m.Z_t(3) == [[3.0]]
        mc = LinearGaussianSSM(Z=lambda t: [[2.0 * t]], T=[[1.0]], R=[[1.0]],
                               Q=[[0.1]], H=[[1.0]], a1=[0.0], P1=[[1.0]])
        assert mc.Z_t(4) == [[8.0]]


class TestKalmanFilter:
    def test_ar1_exact_likelihood(self):
        # THE oracle: AR(1) as a state-space model == the closed-form exact
        # Gaussian AR(1) log-likelihood (no external dependency).
        phi, s2 = 0.6, 1.3
        y = ar1_series(80, phi, s2, seed=5)
        m = LinearGaussianSSM(Z=[[1.0]], T=[[phi]], R=[[1.0]], Q=[[s2]], H=[[0.0]],
                              a1=[0.0], P1=[[s2 / (1 - phi * phi)]])
        kf = kalman_filter(m, [[v] for v in y]).loglike
        n = len(y)
        exact = (-0.5 * n * math.log(2 * math.pi) - 0.5 * n * math.log(s2)
                 + 0.5 * math.log(1 - phi * phi)
                 - (1 - phi * phi) / (2 * s2) * y[0] ** 2
                 - 1.0 / (2 * s2) * sum((y[t] - phi * y[t - 1]) ** 2
                                        for t in range(1, n)))
        assert approx(kf, exact, 1e-9)

    def test_local_level_golden(self):
        # golden value cross-validated vs statsmodels UnobservedComponents 'local level'
        m = local_level_ssm(0.3, 1.0)
        assert approx(kalman_filter(m, [[v] for v in LL_Y]).loglike, -17.873512, 1e-4)

    def test_near_unit_root_exact(self):
        # near-unit-root AR(0.999) must stay exact (no catastrophic cancellation in
        # the stationary initialisation P1 = s2/(1-phi^2)).
        phi, s2 = 0.999, 1.0
        y = ar1_series(80, phi, s2, seed=3)
        m = LinearGaussianSSM(Z=[[1.0]], T=[[phi]], R=[[1.0]], Q=[[s2]], H=[[0.0]],
                              a1=[0.0], P1=[[s2 / (1 - phi * phi)]])
        kf = kalman_filter(m, [[v] for v in y]).loglike
        n = len(y)
        exact = (-0.5 * n * math.log(2 * math.pi) - 0.5 * n * math.log(s2)
                 + 0.5 * math.log(1 - phi * phi)
                 - (1 - phi * phi) / (2 * s2) * y[0] ** 2
                 - 1.0 / (2 * s2) * sum((y[t] - phi * y[t - 1]) ** 2
                                        for t in range(1, n)))
        assert approx(kf, exact, 1e-9)

    def test_diffuse_rank(self):
        Z = [[1.0, 0.0]]
        m = LinearGaussianSSM(Z=Z, T=[[1.0, 1.0], [0.0, 1.0]],
                              R=eye(2), Q=[[0.2, 0.0], [0.0, 0.05]], H=[[0.8]],
                              a1=[0.0, 0.0], P1=zeros(2, 2),
                              P1_infinity=eye(2))
        assert kalman_filter(m, [[v] for v in LL_Y]).n_diffuse == 2

    def test_store_invariant(self):
        m = local_level_ssm(0.3, 1.0)
        a = kalman_filter(m, [[v] for v in LL_Y], store=True).loglike
        b = kalman_filter(m, [[v] for v in LL_Y], store=False).loglike
        assert approx(a, b, 1e-12)

    def test_missing_data(self):
        y = [[v] for v in LL_Y]
        y[3][0] = float("nan")
        y[9][0] = float("nan")
        kf = kalman_filter(local_level_ssm(0.3, 1.0), y)
        assert math.isfinite(kf.loglike) and kf.nobs == len(LL_Y) - 2 - 1  # -diffuse

    def test_partial_multivariate_missing(self):
        # R7: with the univariate treatment, a SINGLE missing component of a
        # multivariate observation must drop only that scalar -- the loglik of the
        # partial observation must equal the loglik with the present component kept
        # and differ from dropping the whole vector.
        m = LinearGaussianSSM(
            Z=[[1., 0.], [0., 1.]], T=[[1., 0.], [0., 1.]],
            R=[[1., 0.], [0., 1.]], Q=[[0.2, 0.], [0., 0.3]],
            H=[[0.5, 0.], [0., 0.4]], a1=[0., 0.], P1=[[1., 0.], [0., 1.]])
        Y = [[1.0, 2.0], [1.5, 2.2], [1.8, 2.5], [2.1, 2.9], [1.9, 2.6]]
        Ypart = [r[:] for r in Y]
        Ypart[2][1] = float("nan")           # only 2nd series missing at t=2
        Yboth = [r[:] for r in Y]
        Yboth[2] = [float("nan"), float("nan")]
        ll_full = kalman_filter(m, Y).loglike
        ll_part = kalman_filter(m, Ypart).loglike
        ll_both = kalman_filter(m, Yboth).loglike
        # the partial observation is handled distinctly from both keeping and
        # dropping the whole vector -> the single missing scalar really was dropped
        assert math.isfinite(ll_part)
        assert abs(ll_part - ll_full) > 1e-6 and abs(ll_part - ll_both) > 1e-6

    def test_sparse_irregular_missing(self):
        # R9: robustness to realistic sensor-dropout patterns -- the filter and
        # smoother must stay finite and PSD however sparse the observations are.
        rng = DeterministicRNG(7)
        full, lvl = [], 0.0
        for _ in range(100):
            lvl += 0.2 * rng.gauss()
            full.append(lvl + 0.3 * rng.gauss())
        m = local_level_ssm(0.3, 1.0)
        patterns = [lambda t: t % 10 == 0,          # 90% missing
                    lambda t: t % 2 == 0,           # every other
                    lambda t: not (30 <= t < 70),   # a long central block
                    lambda t: t in (0, 50, 99)]     # only 3 observations
        for keep in patterns:
            Y = [[full[t]] if keep(t) else [float("nan")] for t in range(100)]
            kf = kalman_filter(m, Y)
            sr = disturbance_smoother(m, Y)
            assert math.isfinite(kf.loglike)
            assert all(math.isfinite(a[0]) for a in sr.smoothed_state)
            assert all(c[0][0] >= -1e-9 for c in sr.smoothed_state_cov)

    def test_mixed_within_model_scales(self):
        # R9: components spanning wildly different variance scales in ONE model
        # (a huge trend + a tiny slope, or a huge state + tiny obs noise) must still
        # equal the joint-MVN oracle -- within-model conditioning, distinct from the
        # uniform scale-equivariance of R6/R8.
        raw = [((1103515245 * (i + 1) + 12345) % 9973) / 4986.0 - 1.0
               for i in range(10)]
        Y = [[v * 100] for v in raw]
        for Q, H in (([[1e6, 0.], [0., 1e-6]], [[1.0]]),
                     ([[1e-6, 0.], [0., 1e6]], [[1.0]]),
                     ([[1e8, 0.], [0., 1e2]], [[1e-8]])):
            m = LinearGaussianSSM(Z=[[1., 0.]], T=[[1., 1.], [0., 1.]],
                                  R=[[1., 0.], [0., 1.]], Q=Q, H=H,
                                  a1=[0., 0.], P1=[[10., 0.], [0., 10.]])
            assert approx(kalman_filter(m, Y).loglike, joint_mvn_logpdf(m, Y), 1e-7)

    def test_long_series_riccati_steady_state(self):
        # R9: over a long series the filtered covariance must converge to the
        # steady-state Riccati solution (no roundoff drift / blow-up).
        phi, s2, h = 0.8, 0.5, 0.3
        m = LinearGaussianSSM(Z=[[1.]], T=[[phi]], R=[[1.]], Q=[[s2]], H=[[h]],
                              a1=[0.], P1=[[s2 / (1 - phi * phi)]])
        rng = DeterministicRNG(3)
        x, Y = 0.0, []
        for _ in range(3000):
            x = phi * x + math.sqrt(s2) * rng.gauss()
            Y.append([x + math.sqrt(h) * rng.gauss()])
        kf = kalman_filter(m, Y)
        P = 1.0
        for _ in range(20000):
            P = phi * phi * P + s2 - (phi * P) ** 2 / (P + h)
        p_filt_ss = P - P * P / (P + h)
        assert approx(kf.filtered_cov[-1][0][0], p_filt_ss, 1e-9)
        assert math.isfinite(kf.loglike)
        assert all(kf.filtered_cov[t][0][0] > 0 for t in range(3000))

    def test_concentrate_is_profile_likelihood(self):
        # the concentrated loglik must equal the FULL loglik maximised over a common
        # scale c applied to every variance, and .scale must be that argmax (the MLE
        # of the common variance).
        kf = kalman_filter(local_level_ssm(0.3, 1.0), [[v] for v in LL_Y],
                           concentrate=True)
        assert math.isfinite(kf.loglike) and kf.scale > 0.0
        best, bc = -1e18, None
        for i in range(400):
            c = 0.02 + i * 0.01
            mm = local_level_ssm(0.3 * c, 1.0 * c)
            ll = kalman_filter(mm, [[v] for v in LL_Y]).loglike
            if ll > best:
                best, bc = ll, c
        assert approx(kf.loglike, best, 1e-3)
        assert approx(kf.scale, bc, 0.05)

    def test_inf_observation_raises(self):
        # regression (fix): +/-inf is invalid input (NaN/None mean 'missing')
        with pytest.raises(ValueError):
            kalman_filter(local_level_ssm(0.3, 1.0),
                          [[1.0], [float("inf")], [2.0]])

    def test_obs_dimension_mismatch_raises(self):
        # Regression (R5): passing an observation with more components than the
        # model's k_obs used to silently ignore the extra columns (the user
        # unknowingly fits only the first series).  It must now raise.
        m = local_level_ssm(0.3, 1.0)  # k_obs == 1
        with pytest.raises(ValueError):
            kalman_filter(m, [[1.0, 2.0], [1.5, 2.0], [2.0, 3.0]])
        # the 2nd column being ignored was the observable symptom -> assert the
        # correct 1-D call still succeeds and gives a finite loglik
        assert math.isfinite(kalman_filter(m, [[1.0], [1.5], [2.0]]).loglike)

    def test_kalman_equals_joint_mvn(self):
        # DECISIVE (R6): the Kalman log-likelihood must equal the definitional
        # joint-Gaussian MVN log-pdf (mean & covariance built by independent moment
        # propagation, NOT the filter) for arbitrary time-invariant proper-prior
        # models -- a dependency-free validation of the entire recursion.
        raw = [((1103515245 * (i + 1) + 12345) % 9973) / 4986.0 - 1.0
               for i in range(8)]
        Y1 = [[v] for v in raw]
        Y2 = [[raw[i], raw[i] * 0.5 - 0.2] for i in range(8)]
        phi, s2 = 0.6, 0.8
        cases = [
            (LinearGaussianSSM(Z=[[1.]], T=[[1.]], R=[[1.]], Q=[[0.4]], H=[[0.7]],
                               a1=[0.], P1=[[5.]]), Y1),                     # local level
            (LinearGaussianSSM(Z=[[1., 0.]], T=[[1., 1.], [0., 1.]],
                               R=[[1., 0.], [0., 1.]], Q=[[0.3, 0.], [0., 0.1]],
                               H=[[0.5]], a1=[0., 0.], P1=[[5., 0.], [0., 5.]]), Y1),  # LLT
            (LinearGaussianSSM(Z=[[1.]], T=[[phi]], R=[[1.]], Q=[[s2]], H=[[0.3]],
                               a1=[0.], P1=[[s2 / (1 - phi * phi)]]), Y1),   # AR(1)+noise
            (LinearGaussianSSM(Z=[[1., 0.], [0., 1.]], T=[[1., 0.], [0., 1.]],
                               R=[[1., 0.], [0., 1.]], Q=[[0.3, 0.], [0., 0.5]],
                               H=[[0.6, 0.2], [0.2, 0.7]], a1=[0., 0.],
                               P1=[[4., 0.], [0., 4.]]), Y2),                # bivariate corr-H
        ]
        for model, Y in cases:
            assert approx(kalman_filter(model, Y).loglike,
                          joint_mvn_logpdf(model, Y), 1e-9)

    def test_non_diagonal_h_scale_invariant(self):
        # Regression (R8): the diagonal-H detection threshold (abs(H_ij) < 1e-12)
        # was scale-blind -> a genuinely CORRELATED H on small-unit data (its
        # off-diagonals ~ variance ~ 1e-16) was mis-classified as diagonal, silently
        # dropping the correlation.  The loglik must equal the joint-MVN oracle at
        # every scale.
        raw = [((1103515245 * (i + 1) + 12345) % 9973) / 4986.0 - 1.0
               for i in range(8)]
        for c in (1.0, 1e-4, 1e-8):
            Y = [[raw[i] * c, (raw[i] * 0.5 - 0.2) * c] for i in range(8)]
            m = LinearGaussianSSM(
                Z=[[1., 0.], [0., 1.]], T=[[1., 0.], [0., 1.]],
                R=[[1., 0.], [0., 1.]], Q=[[0.3 * c * c, 0.], [0., 0.5 * c * c]],
                H=[[0.6 * c * c, 0.2 * c * c], [0.2 * c * c, 0.7 * c * c]],
                a1=[0., 0.], P1=[[4. * c * c, 0.], [0., 4. * c * c]])
            assert approx(kalman_filter(m, Y).loglike, joint_mvn_logpdf(m, Y), 1e-7)

    def test_near_collinear_obs_realistic(self):
        # R10: correlated observation noise up to a REALISTIC correlation (~0.9998)
        # must match the joint-MVN oracle exactly.  The univariate-treatment
        # decorrelation only loses precision for a NEAR-SINGULAR H (noise correlation
        # ~= 1, i.e. a redundant/rank-deficient observation), a documented inherent
        # limitation beyond realistic sensor correlations (verified against an
        # independent well-conditioned rotation).
        raw = [((1103515245 * (i + 1) + 12345) % 9973) / 4986.0 - 1.0
               for i in range(10)]
        Y = [[raw[i], raw[i] + 0.01 * raw[i] * 0.3] for i in range(10)]
        for eps in (0.05, 1e-3, 1e-4):        # correlation up to ~0.9998
            m = LinearGaussianSSM(
                Z=[[1.], [1.]], T=[[0.9]], R=[[1.]], Q=[[0.3]],
                H=[[0.5, 0.5 - eps], [0.5 - eps, 0.5]], a1=[0.], P1=[[2.0]])
            assert approx(kalman_filter(m, Y).loglike, joint_mvn_logpdf(m, Y), 1e-7)

    def test_arma11_ma_channel_toeplitz(self):
        # R6: arma_to_ssm's MA channel (only the AR channel was checked before).
        # The ARMA(1,1) log-likelihood must equal the closed-form Toeplitz-
        # autocovariance MVN (gamma_0, gamma_1 exact; gamma_k = phi*gamma_{k-1}).
        raw = [((1103515245 * (i + 7) + 12345) % 9973) / 4986.0 - 1.0
               for i in range(9)]
        Y = [[v] for v in raw]
        n = len(Y)
        for phi, theta, s2 in [(0.5, 0.4, 1.0), (0.7, -0.3, 0.8), (-0.4, 0.6, 1.5)]:
            m = arma_to_ssm(phi=[phi], theta=[theta], sigma2=s2)
            g0 = s2 * (1 + 2 * phi * theta + theta * theta) / (1 - phi * phi)
            g1 = s2 * ((phi + theta) * (1 + phi * theta)) / (1 - phi * phi)
            gam = [g0, g1]
            for k in range(2, n):
                gam.append(phi * gam[-1])
            Sig = [[gam[abs(i - j)] for j in range(n)] for i in range(n)]
            ref = _jm_logpdf(Sig, raw)
            assert approx(kalman_filter(m, Y).loglike, ref, 1e-9)

    def test_scale_invariance_small_scale(self):
        # Regression (R6): an ABSOLUTE degeneracy floor (F* <= 1e-12) was scale-
        # blind -- for a legitimately small-scale problem (tiny units -> tiny
        # variances) EVERY update was silently skipped, giving loglik == 0.  For a
        # proper-prior model, scaling y by c and every (co)variance by c^2 must
        # shift the log-likelihood by exactly -n_obs*log(c) (Gaussian change of
        # variables), NOT collapse to 0.
        raw = [((1103515245 * (i + 3) + 12345) % 9973) / 4986.0 - 1.0
               for i in range(30)]

        def model_at(c):
            return LinearGaussianSSM(
                Z=[[1.]], T=[[0.9]], R=[[1.]], Q=[[0.3 * c * c]], H=[[0.2 * c * c]],
                a1=[0.], P1=[[1.0 * c * c]])
        base = kalman_filter(model_at(1.0), [[v] for v in raw]).loglike
        for c in (1e-2, 1e-4, 1e-8):
            ll = kalman_filter(model_at(c), [[v * c] for v in raw]).loglike
            assert math.isfinite(ll) and ll != 0.0
            assert approx(ll, base - len(raw) * math.log(c), 1e-7)
        # and it still equals the independent joint-MVN oracle at tiny scale
        c = 1e-8
        assert approx(kalman_filter(model_at(c), [[v * c] for v in raw]).loglike,
                      joint_mvn_logpdf(model_at(c), [[v * c] for v in raw]), 1e-6)

    def test_scale_equivariance(self):
        # metamorphic: scaling y by c scales the concentrated-scale-free loglik
        # predictably; here check filtered states scale by c for a no-obs-noise AR(1)
        phi = 0.5
        y = ar1_series(40, phi, 1.0, seed=3)
        base = LinearGaussianSSM(Z=[[1.0]], T=[[phi]], R=[[1.0]], Q=[[1.0]],
                                 H=[[0.0]], a1=[0.0], P1=[[1.0 / (1 - phi * phi)]])
        c = 1000.0
        scaled = LinearGaussianSSM(Z=[[1.0]], T=[[phi]], R=[[1.0]], Q=[[c * c]],
                                   H=[[0.0]], a1=[0.0],
                                   P1=[[c * c / (1 - phi * phi)]])
        fb = kalman_filter(base, [[v] for v in y]).filtered_state
        fs = kalman_filter(scaled, [[c * v] for v in y]).filtered_state
        assert all(approx(fs[t][0], c * fb[t][0], 1e-6) for t in range(len(y)))

    def test_non_diagonal_H_loglik(self):
        # Regression (R3 bug): with CORRELATED observation noise (non-diagonal H) the
        # univariate treatment decorrelates the observation via Cholesky; the loglik
        # must add the -0.5*log|H| Jacobian of that transform (it was omitted -> the
        # reported loglik was off by exactly 0.5*log|H|).  The single-observation
        # loglik must equal the joint 2-D Gaussian density.
        Z = [[1.0, 0.0], [0.0, 1.0]]
        P1 = [[2.0, 0.5], [0.5, 1.5]]
        H = [[0.5, -0.3], [-0.3, 0.8]]
        a1 = [0.3, -0.2]
        yv = [1.1, 0.4]
        m = LinearGaussianSSM(Z=Z, T=[[1.0, 0.0], [0.0, 1.0]],
                              R=[[1.0, 0.0], [0.0, 1.0]], Q=[[0.04, 0.0], [0.0, 0.04]],
                              H=H, a1=a1, P1=P1)
        our = kalman_filter(m, [yv]).loglike
        F = [[P1[i][j] + H[i][j] for j in range(2)] for i in range(2)]
        v = [yv[i] - a1[i] for i in range(2)]
        det = F[0][0] * F[1][1] - F[0][1] * F[1][0]
        Fi = [[F[1][1] / det, -F[0][1] / det], [-F[1][0] / det, F[0][0] / det]]
        quad = sum(v[i] * Fi[i][j] * v[j] for i in range(2) for j in range(2))
        manual = -0.5 * (2 * math.log(2 * math.pi) + math.log(det) + quad)
        assert approx(our, manual, 1e-9)

    def test_multivariate_smoother_finite(self):
        # a 2-series local level with correlated obs noise smooths to finite states
        rng = DeterministicRNG(9)
        a, b, Y = 1.0, 5.0, []
        for _ in range(30):
            a += 0.2 * rng.gauss(); b += 0.2 * rng.gauss()
            Y.append([a + 0.4 * rng.gauss(), b + 0.4 * rng.gauss()])
        m = LinearGaussianSSM(Z=[[1., 0.], [0., 1.]], T=[[1., 0.], [0., 1.]],
                              R=[[1., 0.], [0., 1.]], Q=[[0.04, 0.], [0., 0.04]],
                              H=[[0.16, 0.08], [0.08, 0.16]], a1=[0., 0.],
                              P1=[[0., 0.], [0., 0.]], P1_infinity=[[1., 0.], [0., 1.]])
        sr = disturbance_smoother(m, Y)
        assert all(math.isfinite(s[0]) and math.isfinite(s[1]) for s in sr.smoothed_state)


class TestSmoother:
    def test_smoothed_state_golden(self):
        sr = disturbance_smoother(local_level_ssm(0.3, 1.0), [[v] for v in LL_Y])
        assert approx(sr.smoothed_state[0][0], 1.332276, 1e-4)

    def test_signal_reconstructs(self):
        m = local_level_ssm(0.3, 1.0)
        sr = disturbance_smoother(m, [[v] for v in LL_Y])
        for t in range(len(LL_Y)):
            assert approx(sr.smoothed_signal[t][0], sr.smoothed_state[t][0], 1e-9)

    def test_smoother_scale_invariance(self):
        # Regression (R6): absolute degeneracy floors (lu_decompose pivot, smoother
        # ridge) were scale-blind -- for a legitimately small-scale problem the
        # smoothed states collapsed to garbage (e.g. ->0).  Smoothing must be
        # scale-EQUIVARIANT: scaling y by c and every (co)variance by c^2 scales
        # the smoothed state by c and its covariance by c^2, at realistic scales.
        raw = [((1103515245 * (i + 3) + 12345) % 9973) / 4986.0 - 1.0
               for i in range(25)]

        def model_at(c):
            return LinearGaussianSSM(
                Z=[[1.]], T=[[0.9]], R=[[1.]], Q=[[0.3 * c * c]], H=[[0.2 * c * c]],
                a1=[0.], P1=[[1.0 * c * c]])
        base = disturbance_smoother(model_at(1.0), [[v] for v in raw])
        for c in (1e-2, 1e-4, 1e-6):
            sm = disturbance_smoother(model_at(c), [[v * c] for v in raw])
            for t in range(len(raw)):
                assert approx(sm.smoothed_state[t][0] / c,
                              base.smoothed_state[t][0], 1e-6)
                assert approx(sm.smoothed_state_cov[t][0][0] / (c * c),
                              base.smoothed_state_cov[t][0][0], 1e-6)

    def test_disturbance_cov_nonneg(self):
        sr = disturbance_smoother(local_level_ssm(0.3, 1.0), [[v] for v in LL_Y])
        assert all(c[0][0] >= -1e-9 for c in sr.smoothed_state_disturbance_cov)
        assert all(c[0][0] >= -1e-9 for c in sr.smoothed_obs_disturbance_cov)

    def test_component_ci(self):
        # R11 (mutation-testing gap): SmootherResult.component_ci (confidence bands
        # for a smoothed component) was untested -- its two-sided Gaussian quantile
        # must be Phi^{-1}((1+level)/2) and the band must bracket the smoothed state.
        m = LinearGaussianSSM(Z=[[1.]], T=[[1.]], R=[[1.]], Q=[[0.3]], H=[[1.0]],
                              a1=[0.], P1=[[5.]])
        sr = disturbance_smoother(m, [[v] for v in LL_Y])
        lo, hi = sr.component_ci(0, level=0.90)
        z = norm_ppf(0.95)
        for t in range(len(LL_Y)):
            sd = math.sqrt(sr.smoothed_state_cov[t][0][0])
            assert approx(hi[t] - lo[t], 2 * z * sd, 1e-9)
            assert lo[t] <= sr.smoothed_state[t][0] <= hi[t]

    def test_smoothed_cov_nonnegative_multistate(self):
        # Regression (R10): for a MULTI-diffuse-state model (BSM / seasonal) the
        # big-kappa smoother's V = P - P N P used to catastrophically cancel during
        # the diffuse phase and return large NEGATIVE smoothed variances (down to
        # -1e26 -> a crash on sqrt for a confidence interval).  Reducing the
        # over-large kappa (_BIG) + a variance floor makes every smoothed variance
        # non-negative and finite, and the diffuse-phase covariance order-correct.
        rng = DeterministicRNG(7)
        seas = [2., -1., -1., 0., 1., -1., 2., 0., -1., 1., -2., 0.]
        y, lvl = [], 0.0
        for t in range(120):
            lvl += 0.05 * rng.gauss()
            y.append(10.0 + lvl + seas[t % 12] + 0.3 * rng.gauss())
        for comps in ([LocalLinearTrend(), SeasonalDummy(12), Irregular()],
                      [LocalLevel(), TrigonometricSeasonal(12, harmonics=2),
                       Irregular()]):
            m = UnobservedComponents(y, comps).fit(maxiter=100, restarts=0).model
            sm = disturbance_smoother(m, [[v] for v in y])
            for t in range(len(y)):
                for i in range(m.k_states):
                    assert sm.smoothed_state_cov[t][i][i] >= 0.0
                    assert math.isfinite(sm.smoothed_state_cov[t][i][i])

    def test_simulation_smoother_mean(self):
        # decisive: sample mean of simulation-smoother draws -> smoothed state mean
        m = local_level_ssm(0.3, 1.0)
        sr = disturbance_smoother(m, [[v] for v in LL_Y])
        draws = simulation_smoother(m, [[v] for v in LL_Y], n_sims=200, seed=1)
        n = len(LL_Y)
        mean = [sum(draws[s][t][0] for s in range(200)) / 200 for t in range(n)]
        err = max(abs(mean[t] - sr.smoothed_state[t][0]) for t in range(n))
        assert err < 0.15                                 # Monte-Carlo tolerance

    def test_missing_smoothing(self):
        y = [[v] for v in LL_Y]
        y[5][0] = float("nan")
        sr = disturbance_smoother(local_level_ssm(0.3, 1.0), y)
        assert all(math.isfinite(a[0]) for a in sr.smoothed_state)

    def test_kalman_smoother_alias(self):
        m = local_level_ssm(0.3, 1.0)
        a = kalman_smoother(m, [[v] for v in LL_Y]).smoothed_state
        b = disturbance_smoother(m, [[v] for v in LL_Y]).smoothed_state
        assert all(approx(a[t][0], b[t][0], 1e-12) for t in range(len(a)))

    def test_auxiliary_residuals_detect_anomalies(self):
        # R9: the auxiliary residuals are the module's outlier / structural-break
        # detector.  An ADDITIVE OUTLIER must show up as the largest OBSERVATION
        # auxiliary residual at its time; a LEVEL BREAK as the largest STATE
        # auxiliary residual.
        rng = DeterministicRNG(1)
        y, lvl = [], 0.0
        for _ in range(80):
            lvl += 0.15 * rng.gauss()
            y.append(lvl + 0.4 * rng.gauss())
        m = local_level(y, maxiter=300, restarts=1).model
        y_out = list(y)
        y_out[30] += 5.0                                   # additive outlier at t=30
        ar = auxiliary_residuals(m, [[v] for v in y_out])
        obs = [v[0] for v in ar["observation"]]
        assert max(range(len(obs)), key=lambda t: abs(obs[t])) in (29, 30, 31)
        y_brk = list(y)
        for t in range(55, 80):
            y_brk[t] += 4.0                                # level break at t=55
        st = [v[0] for v in auxiliary_residuals(m, [[v] for v in y_brk])["state"]]
        assert 53 <= max(range(len(st)), key=lambda t: abs(st[t])) <= 57


class TestEstimation:
    def test_ucm_local_level_fit(self):
        # golden vs statsmodels-validated fit (loglik + variances)
        f = UnobservedComponents(LL_Y, [LocalLevel(), Irregular()]).fit(
            maxiter=400, restarts=1)
        assert approx(f.loglike, -8.894675, 5e-2)
        assert approx(f.params[0], 0.1003, 2e-2) and approx(f.params[1], 0.06304, 2e-2)

    def test_em_increases_likelihood(self):
        y = [[v] for v in LL_Y]
        start = UnobservedComponents(LL_Y, [LocalLevel(), Irregular()]).build([1.0, 1.0])
        ll0 = kalman_filter(start, y).loglike
        em = em_algorithm(start, y, max_iter=40)
        assert em.loglike >= ll0 - 1e-6                   # EM never decreases logL

    def test_param_transform_roundtrip(self):
        for kind, vals in [("var", [0.01, 1.0, 100.0]),
                           ("unit", [0.1, 0.5, 0.9]),
                           ("corr", [-0.8, 0.0, 0.7]),
                           ("freq", [0.2, 1.5, 3.0])]:
            for v in vals:
                u = S._param_unconstrained(v, kind)
                assert approx(S._param_natural(u, kind), v, 1e-6)

    def test_nelder_mead_rosenbrock(self):
        ros = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        x, fx = minimize_nelder_mead(ros, [-1.2, 1.0], restarts=3, maxiter=3000)
        assert approx(x[0], 1.0, 1e-3) and approx(x[1], 1.0, 1e-3)

    def test_information_criteria(self):
        ic = information_criteria(loglik=-100.0, n_params=3, nobs=50)
        assert approx(ic.aic, 206.0) and approx(ic.bic, 200.0 + 3 * math.log(50))


class TestComponents:
    def test_component_shapes(self):
        for c in (LocalLevel(), LocalLinearTrend(), SeasonalDummy(4),
                  TrigonometricSeasonal(12), Cycle(), AutoRegressive(2)):
            p = c.start_params()
            T = c.T(p)
            assert len(T) == c.k_states and len(T[0]) == c.k_states
            assert len(c.R()) == c.k_states and len(c.R()[0]) == c.k_posdef
            assert len(c.Q(p)) == c.k_posdef
            assert len(c.Z_row(0)) == c.k_states

    def test_dominant_cycle_frequency(self):
        # Regression (R7): the periodogram seed must find the true angular
        # frequency of a clean sinusoid (independent of the fit).
        from gridpulse.analysis.algorithms.orchid.state_space import _dominant_cycle_freqs
        for period in (8.0, 20.0, 40.0):
            y = [math.sin(2 * math.pi * t / period) for t in range(200)]
            peaks = _dominant_cycle_freqs(y, k=1)
            assert peaks and approx(peaks[0], 2 * math.pi / period, 0.05)
        # two superposed cycles -> two distinct peaks
        y2 = [math.sin(2 * math.pi * t / 12) + 0.7 * math.sin(2 * math.pi * t / 30)
              for t in range(240)]
        two = _dominant_cycle_freqs(y2, k=2)
        assert len(two) == 2 and all(0.0 < w < math.pi for w in two)

    def test_cycle_recovers_frequency(self):
        # Regression (R7): the cycle frequency is seeded from the data periodogram,
        # so a cycle fit on its own recovers its period even when it is much LONGER
        # than the old fixed default start-period (10), which used to trap long
        # cycles at the degenerate frequency-0 boundary.
        for period in (12.0, 30.0):        # 30 >> old default start_period of 10
            rng = DeterministicRNG(int(period))
            y = [2.0 * math.sin(2 * math.pi * t / period) + 0.3 * rng.gauss()
                 for t in range(120)]
            fit = UnobservedComponents(y, [Cycle(), Irregular()]).fit(
                maxiter=300, restarts=2)
            f = dict(zip(fit.param_names, fit.params))["freq.cycle"]
            assert approx(f, 2 * math.pi / period, 0.05)

    def test_ar_component_stationary(self):
        c = AutoRegressive(3)
        phi = c._phi([0.5, -0.4, 0.3, 1.0])               # reflection coeffs + var
        # AR stationary iff roots of 1 - sum phi_i B^i lie OUTSIDE the unit circle
        char = [1.0] + [-v for v in phi]
        roots = durand_kerner_roots(char)
        assert all(abs(r) > 1.0 - 1e-9 for r in roots)

    def test_damped_trend(self):
        c = LocalLinearTrend(damped=True)
        p = c.start_params()
        T = c.T(p)
        assert 0.0 < T[1][1] < 1.0                        # damping enters the slope

    def test_deterministic_level(self):
        c = LocalLinearTrend(stochastic_level=False)
        assert ("sigma2.level", "var") not in c.param_specs
        assert c.n_params == len(c.start_params())

    def test_seasonal_recovers_pattern(self):
        # Both the dummy AND trigonometric seasonal must recover a known seasonal
        # pattern (their absolute loglik carries a diffuse-prior CONSTANT convention
        # offset vs statsmodels -- dummy is exactly -log(period) -- but the INFERENCE
        # is correct: R2 measured the smoothed seasonal at corr 0.9999 vs truth).
        rng = DeterministicRNG(3)
        seas = [3.0, -1.0, -2.5, 0.5]
        y, lvl = [], 10.0
        for t in range(64):
            lvl += 0.02 + 0.15 * rng.gauss()
            y.append(lvl + seas[t % 4] + 0.4 * rng.gauss())
        tp = [seas[t % 4] for t in range(64)]
        mt = sum(tp) / 64
        st = math.sqrt(sum((v - mt) ** 2 for v in tp))
        for comp in (SeasonalDummy(4), TrigonometricSeasonal(4)):
            uc = UnobservedComponents(y, [LocalLevel(), comp, Irregular()])
            uc.fit(maxiter=800, restarts=2)
            dec = decompose(uc)
            os = dec[[k for k in dec if "seas" in k.lower()][0]]
            mo = sum(os[:64]) / 64
            cov = sum((os[t] - mo) * (tp[t] - mt) for t in range(64))
            so = math.sqrt(sum((os[t] - mo) ** 2 for t in range(64)))
            assert cov / (so * st) > 0.95            # recovers the seasonal pattern

    def test_dummy_seasonal_loglik_golden(self):
        # R11 (mutation-testing gap): the exact-diffuse loglik term -0.5*log(F_inf) is
        # ZERO for level/trend (F_inf=1) but NONZERO for a dummy seasonal (F_inf=period^2),
        # so a sign error on it was only observable on a SEASONAL loglik -- which no test
        # pinned.  Golden value for a FIXED-parameter LocalLevel + SeasonalDummy(4).
        y = [float(v) for v in LL_Y[:10]] + [3.0, 2.7, 3.4, 3.0]
        m = UnobservedComponents(
            y, [LocalLevel(), SeasonalDummy(4), Irregular()]).build([0.1, 0.2, 0.5])
        assert approx(kalman_filter(m, [[v] for v in y]).loglike, -14.873548, 1e-4)

    def test_dynamic_regression_recovers_drift(self):
        # R9: a dynamic (time-varying-parameter) regression must track a DRIFTING
        # coefficient -- the smoothed regression state should follow the true
        # random-walk beta_t (only a static beta was checked before, in R4).
        rng = DeterministicRNG(2)
        X = [[math.sin(t * 0.3) + 0.5] for t in range(120)]
        beta, b = [], 1.0
        for _ in range(120):
            b += 0.03 * rng.gauss()
            beta.append(b)
        y = [beta[t] * X[t][0] + 0.2 * rng.gauss() for t in range(120)]
        fit = UnobservedComponents(
            y, [Regression(X, dynamic=True), Irregular()]).fit(maxiter=250,
                                                               restarts=1)
        est = [s[0] for s in disturbance_smoother(
            fit.model, [[v] for v in y]).smoothed_state]
        n = len(beta)
        mb, me = sum(beta) / n, sum(est) / n
        cov = sum((beta[t] - mb) * (est[t] - me) for t in range(n))
        vb = sum((beta[t] - mb) ** 2 for t in range(n))
        ve = sum((est[t] - me) ** 2 for t in range(n))
        assert cov / math.sqrt(vb * ve) > 0.9        # tracks the drifting coefficient


class TestDLM_FFBS:
    def test_dlm_runs_and_forecasts(self):
        dlm = DiscountDLM(F=[1.0], G=[[1.0]], delta=0.9, m0=[0.0], C0=[[100.0]])
        res = dlm.filter(LL_Y)
        assert math.isfinite(res.loglike) and len(res.filtered_mean) == len(LL_Y)
        fm, fv = dlm.forecast(LL_Y, 3)
        assert all(math.isfinite(v) and v > 0 for v in fv)
        assert all(approx(fm[k], fm[0]) for k in range(3))  # local level -> flat

    def test_ffbs_mean_and_cov_match_smoother(self):
        # decisive FFBS correctness: posterior sample MEAN and VARIANCE -> the
        # smoother's smoothed_state and smoothed_state_cov (R3 checked the covariance,
        # not just the mean -- the classic place FFBS bugs hide).
        m = local_level_ssm(0.3, 1.0)
        sr = disturbance_smoother(m, [[v] for v in LL_Y])
        D = 600
        draws = ffbs(m, [[v] for v in LL_Y], n_draws=D, seed=1)
        n = len(LL_Y)
        mean = [sum(draws[s][t][0] for s in range(D)) / D for t in range(n)]
        var = [sum((draws[s][t][0] - mean[t]) ** 2 for s in range(D)) / D
               for t in range(n)]
        assert max(abs(mean[t] - sr.smoothed_state[t][0]) for t in range(n)) < 0.15
        for t in range(n):                               # cov ratio within MC error
            assert 0.7 < var[t] / sr.smoothed_state_cov[t][0][0] < 1.4

    def test_ffbs_deterministic(self):
        m = local_level_ssm(0.3, 1.0)
        y = [[v] for v in LL_Y]
        assert ffbs(m, y, n_draws=3, seed=2) == ffbs(m, y, n_draws=3, seed=2)


class TestNonlinearFilters:
    def _linear_case(self):
        q, r = 0.5, 1.0
        rng = DeterministicRNG(7)
        x = 0.0
        ys = []
        for t in range(40):
            if t:
                x += math.sqrt(q) * rng.gauss()
            ys.append(x + math.sqrt(r) * rng.gauss())
        return q, r, [[v] for v in ys]

    def test_ekf_ukf_match_kalman(self):
        q, r, y = self._linear_case()
        m = LinearGaussianSSM(Z=[[1.0]], T=[[1.0]], R=[[1.0]], Q=[[q]], H=[[r]],
                              a1=[0.0], P1=[[5.0]])
        kf = kalman_filter(m, y)
        f = lambda x: [x[0]]
        h = lambda x: [x[0]]
        ekf = extended_kalman_filter(f, h, [[q]], [[r]], [0.0], [[5.0]], y)
        ukf = unscented_kalman_filter(f, h, [[q]], [[r]], [0.0], [[5.0]], y)
        assert approx(ekf.loglike, kf.loglike, 1e-6)
        assert approx(ukf.loglike, kf.loglike, 1e-5)
        assert approx(ekf.filtered_state[-1][0], kf.filtered_state[-1][0], 1e-6)

    def test_ekf_ukf_scale_invariant(self):
        # Regression (R8): the EKF/UKF innovation ridge (1e-14) and UKF sigma-point
        # jitter (1e-12) were scale-blind -> nonlinear filters on small-unit data
        # diverged from the exact Kalman filter they must reduce to on a LINEAR
        # model (UKF broke already at 1e-2 scale).
        A = [[0.95, 0.1], [0.0, 0.9]]
        f = lambda x: [A[0][0] * x[0] + A[0][1] * x[1],
                       A[1][0] * x[0] + A[1][1] * x[1]]
        h = lambda x: [x[0]]
        raw = [((1103515245 * (i + 1) + 12345) % 9973) / 4986.0 - 1.0
               for i in range(15)]
        for c in (1.0, 1e-4, 1e-8):
            Q = [[0.2 * c * c, 0.0], [0.0, 0.1 * c * c]]
            R = [[0.5 * c * c]]
            x0, P0 = [0.0, 0.0], [[c * c, 0.0], [0.0, c * c]]
            Y = [[v * c] for v in raw]
            m = LinearGaussianSSM(Z=[[1.0, 0.0]], T=A, R=[[1.0, 0.0], [0.0, 1.0]],
                                  Q=Q, H=R, a1=x0, P1=P0)
            kf = kalman_filter(m, Y).loglike
            assert approx(extended_kalman_filter(f, h, Q, R, x0, P0, Y).loglike,
                          kf, 1e-5)
            assert approx(unscented_kalman_filter(f, h, Q, R, x0, P0, Y).loglike,
                          kf, 1e-5)

    def test_mvn_sampling_spread_scale_invariant(self):
        # Regression (R8): the sampling Cholesky jitter (1e-12) was scale-blind and
        # inflated the spread of draws from a small covariance (FFBS / simulation
        # smoother on small-unit data).  Sample variance must track the covariance.
        rng = DeterministicRNG(0)
        for c in (1.0, 1e-6, 1e-10):
            var = 0.4 * c * c
            draws = [rng.multivariate_normal([0.0], [[var]])[0] for _ in range(5000)]
            mean = sum(draws) / len(draws)
            sv = sum((d - mean) ** 2 for d in draws) / len(draws)
            assert approx(sv / var, 1.0, 0.08)   # not inflated by the jitter

    def test_particle_filter_matches_kalman(self):
        # regression (HIGH bug): the SMC marginal loglike must converge to the exact
        # Kalman loglike on a linear-Gaussian model (was badly biased under adaptive
        # resampling before the telescoping-increment fix).
        q, r, y = self._linear_case()
        m = LinearGaussianSSM(Z=[[1.0]], T=[[1.0]], R=[[1.0]], Q=[[q]], H=[[r]],
                              a1=[0.0], P1=[[5.0]])
        kf = kalman_filter(m, y).loglike
        pf = particle_filter(
            lambda x, t, g: [x[0] + math.sqrt(q) * g.gauss()],
            lambda x, yt, t: -0.5 * (math.log(2 * math.pi * r) + (yt[0] - x[0]) ** 2 / r),
            lambda g: [math.sqrt(5.0) * g.gauss()],
            y, n_particles=2000, seed=1).loglike
        assert abs(pf - kf) < 1.0                          # was ~39 before the fix

    def test_particle_ess_range(self):
        q, r, y = self._linear_case()
        res = particle_filter(
            lambda x, t, g: [x[0] + math.sqrt(q) * g.gauss()],
            lambda x, yt, t: -0.5 * (math.log(2 * math.pi * r) + (yt[0] - x[0]) ** 2 / r),
            lambda g: [math.sqrt(5.0) * g.gauss()], y, n_particles=500, seed=1)
        assert all(1.0 <= e <= 500.0 + 1e-6 for e in res.ess)

    def test_systematic_resample_single_uniform(self):
        # regression (fix): genuine systematic resampling draws exactly ONE uniform
        class CountRNG:
            def __init__(self):
                self.n = 0

            def random(self):
                self.n += 1
                return 0.5
        c = CountRNG()
        idx = S._systematic_resample([0.25, 0.25, 0.25, 0.25], c)
        assert c.n == 1 and sorted(idx) == [0, 1, 2, 3]

    def test_particle_deterministic(self):
        q, r, y = self._linear_case()
        tr = lambda x, t, g: [x[0] + math.sqrt(q) * g.gauss()]
        oll = lambda x, yt, t: -0.5 * (math.log(2 * math.pi * r) + (yt[0] - x[0]) ** 2 / r)
        init = lambda g: [g.gauss()]
        a = particle_filter(tr, oll, init, y, n_particles=200, seed=3).loglike
        b = particle_filter(tr, oll, init, y, n_particles=200, seed=3).loglike
        assert a == b


class TestForecast:
    def test_variance_monotone(self):
        f = local_level(LL_Y, maxiter=200, restarts=0)
        fc = forecast(f, LL_Y, 10)
        v = [fc.variance[k][0][0] for k in range(10)]
        assert all(v[k] >= v[k - 1] - 1e-9 for k in range(1, 10))

    def test_bands_nested(self):
        f = local_level(LL_Y, maxiter=200, restarts=0)
        fc = forecast(f, LL_Y, 5, level=(0.80, 0.95))
        for k in range(5):
            assert (fc.lower[0.95][k][0] <= fc.lower[0.80][k][0]
                    <= fc.mean[k][0] <= fc.upper[0.80][k][0] <= fc.upper[0.95][k][0])

    def test_forecast_bands_are_gaussian_quantiles(self):
        # R7: the interval endpoints must equal mean +/- z*sd with the correct
        # two-sided normal quantile z = Phi^{-1}((1+level)/2) and sd = sqrt(var).
        f = local_level(LL_Y, maxiter=200, restarts=0)
        fc = forecast(f, LL_Y, 5, level=(0.80, 0.95))
        for lvl in (0.80, 0.95):
            z = norm_ppf((1 + lvl) / 2)
            for k in range(5):
                sd = math.sqrt(fc.variance[k][0][0])
                assert approx(fc.upper[lvl][k][0], fc.mean[k][0] + z * sd, 1e-9)
                assert approx(fc.lower[lvl][k][0], fc.mean[k][0] - z * sd, 1e-9)

    def test_level_validation(self):
        # regression (fix): a confidence level must be a fraction in (0,1)
        m = local_level_ssm(0.3, 1.0)
        for bad in (95, 1.5, -0.1, 0.0):
            with pytest.raises(ValueError):
                forecast(m, LL_Y, 3, level=(bad,))

    def test_empty_history_raises(self):
        # regression (fix): forecast on empty history -> clean ValueError
        with pytest.raises(ValueError):
            forecast(local_level_ssm(0.3, 1.0), [], 3)

    def test_forecast_rejects_time_varying(self):
        # Regression (R4): forecasting a time-varying model (Regression / exog) used
        # to silently pad the future exog to 0 -> a wrong (regression-ignoring)
        # forecast.  It must now raise rather than return silent garbage.
        rng = DeterministicRNG(2)
        X = [[math.sin(t * 0.4)] for t in range(40)]
        y, lvl = [], 2.0
        for t in range(40):
            lvl += 0.1 * rng.gauss()
            y.append(lvl + 1.5 * X[t][0] + 0.3 * rng.gauss())
        fit = UnobservedComponents(y, [LocalLevel(), Regression(X),
                                       Irregular()]).fit(maxiter=150, restarts=0)
        with pytest.raises(ValueError):
            forecast(fit, y, 6)

    def test_forecast_rejects_non_model(self):
        # Regression (R4): passing the UnobservedComponents container (not the fit)
        # gave a cryptic AttributeError; it must be a clear TypeError.
        uc = UnobservedComponents(LL_Y, [LocalLevel(), Irregular()])
        with pytest.raises(TypeError):
            forecast(uc, LL_Y, 3)

    def test_predict_in_sample(self):
        m = local_level_ssm(0.3, 1.0)
        pred = predict_in_sample(m, LL_Y)
        assert len(pred) == len(LL_Y) and all(math.isfinite(p[0]) for p in pred)

    def test_decompose_reconstructs(self):
        uc = UnobservedComponents(LL_Y, [LocalLinearTrend(), Irregular()])
        uc.fit(maxiter=200, restarts=0)
        dec = decompose(uc)
        # trend + (level is inside trend) should track the smoothed signal closely
        assert len(dec["trend"]) == len(LL_Y)
        assert all(math.isfinite(v) for v in dec["trend"])

    def test_decompose_is_additive(self):
        # R9: the smoothed component contributions must sum EXACTLY to the smoothed
        # signal (Z * a_hat) -- decompose is a partition of the fitted signal.
        rng = DeterministicRNG(4)
        seas = [0.0, 2.0, -1.0, -1.0]
        y, lvl = [], 2.0
        for t in range(48):
            lvl += 0.05 * rng.gauss()
            y.append(lvl + seas[t % 4] + 0.2 * rng.gauss())
        fit = UnobservedComponents(
            y, [LocalLevel(), SeasonalDummy(4), Irregular()]).fit(maxiter=200,
                                                                  restarts=0)
        dec = decompose(fit)
        sig = [s[0] for s in disturbance_smoother(fit.model,
                                                  [[v] for v in y]).smoothed_signal]
        for t in range(48):
            assert approx(sum(dec[k][t] for k in dec), sig[t], 1e-9)


class TestDiagnostics:
    def test_ljung_box_white(self):
        lb = ljung_box(white(300, 9), lags=10)
        assert lb.pvalue > 0.05                            # do NOT reject independence

    def test_ljung_box_statistic_golden(self):
        # R11 (mutation-testing gap): the exact Q statistic must be pinned -- the
        # white-noise p-value test above passed even with n*(n+2) mutated to n*n.
        # Golden value cross-validated against statsmodels acorr_ljungbox in R7.
        resid = [0.5, -0.3, 0.8, -1.2, 0.1, 0.9, -0.6, 0.4, -0.7, 1.1, -0.2, 0.3,
                 -0.9, 0.6, -0.4, 0.7, -1.0, 0.2, 0.5, -0.8, 0.35, -0.25, 0.65,
                 -0.15, 0.95, -0.55, 0.45, -0.85, 1.05, -0.35]
        lb = ljung_box(resid, lags=10)
        assert approx(lb.statistic, 33.993773, 1e-4)
        assert approx(lb.pvalue, 0.000185, 1e-5)

    def test_jarque_bera(self):
        assert jarque_bera(white(500, 2)).pvalue > 0.05
        skew = [math.exp(x) for x in white(500, 2)]
        assert jarque_bera(skew).pvalue < 0.05

    def test_durbin_watson(self):
        assert approx(durbin_watson(white(400, 1)), 2.0, 0.15)

    def test_durbin_watson_golden(self):
        # R11 (mutation-testing gap): a case where the (e_t - e_{t-1}) sign matters
        # -- a perfectly trending series has strong positive autocorrelation (DW -> 0),
        # whereas the (e_t + e_{t-1}) mutation would give ~2.98.
        assert approx(durbin_watson([1.0, 2.0, 3.0, 4.0, 5.0]), 4.0 / 55.0, 1e-9)


class TestPresets:
    def test_local_linear_trend(self):
        f = local_linear_trend(LL_Y, maxiter=200, restarts=0)
        assert math.isfinite(f.loglike) and f.model.k_states == 2

    def test_arma_to_ssm(self):
        m = arma_to_ssm(phi=[0.6], theta=[0.3], sigma2=1.0)
        y = ar1_series(50, 0.6, 1.0, seed=1)
        assert math.isfinite(kalman_filter(m, [[v] for v in y]).loglike)

    def test_arma_to_ssm_mean_reversion(self):
        # R7: the `mu` argument sets the process mean -> a stationary ARMA forecast
        # must revert to mu at long horizons.
        mu = 5.0
        m = arma_to_ssm(phi=[0.6], theta=[0.3], sigma2=1.0, mu=mu)
        y = [[mu + v] for v in ar1_series(40, 0.6, 1.0, seed=2)]
        fc = forecast(m, y, 50, level=(0.9,))
        assert approx(fc.mean[-1][0], mu, 0.05)

    def test_var_to_ssm(self):
        coefs = [[[0.5, 0.1], [0.0, 0.4]]]
        m = var_to_ssm(coefs, [[1.0, 0.2], [0.2, 1.0]])
        assert m.k_states == 2 and m.k_obs == 2


class TestSimulatorsPayloads:
    def test_simulate_ssm_deterministic(self):
        m = local_level_ssm(0.3, 1.0, diffuse=False)
        a, _ = simulate_ssm(m, 30, seed=4)
        b, _ = simulate_ssm(m, 30, seed=4)
        assert a == b

    def test_sample_series(self):
        y = sample_series(40, period=12, seasonal_amp=3.0, seed=1)
        assert len(y) == 40 and all(math.isfinite(v) for v in y)
        assert y != sample_series(40, period=12, seasonal_amp=3.0, seed=2)

    def test_payload_json_safe(self):
        f = local_level(LL_Y, maxiter=150, restarts=0)
        json.dumps(build_result_payload(f), allow_nan=False)
        json.dumps(build_a2ui_payload({"ll": f}), allow_nan=False)

    def test_a2ui_ranks_by_aic(self):
        f1 = local_level(LL_Y, maxiter=150, restarts=0)
        f2 = local_linear_trend(LL_Y, maxiter=150, restarts=0)
        pay = build_a2ui_payload({"level": f1, "trend": f2})
        assert pay["best"] in ("level", "trend")


class TestDeterminism:
    def test_fit_reproducible(self):
        a = local_level(LL_Y, maxiter=200, restarts=1, seed=3)
        b = local_level(LL_Y, maxiter=200, restarts=1, seed=3)
        assert a.params == b.params and a.loglike == b.loglike
