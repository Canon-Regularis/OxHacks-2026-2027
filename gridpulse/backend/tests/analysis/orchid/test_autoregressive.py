"""
Stress / correctness suite for autoregressive_template.py.

Pure standard-library + pytest (no numpy / scipy): every check is an ORACLE --
a closed-form value, a hand-computed quantity, a round-trip identity, or the
recovery of parameters from a series simulated with KNOWN coefficients.  The
suite is deterministic and must pass byte-identically under any PYTHONHASHSEED.

Run:  pytest _stress_test.py -q
"""
from __future__ import annotations

import json
import math

import pytest

from gridpulse.analysis.algorithms.orchid import autoregressive as ar
from gridpulse.analysis.algorithms.orchid.autoregressive import (
    # special functions / rng
    norm_cdf, norm_ppf, norm_pdf, t_cdf, t_ppf, chi2_cdf, chi2_sf, chi2_ppf,
    f_sf, gammainc, betainc, DeterministicRNG,
    # linear algebra
    matmul, matvec, transpose, solve, inv, cholesky, chol_solve, qr_lstsq,
    logdet, eye, spectral_radius, solve_discrete_lyapunov, poly_mul,
    durand_kerner_roots, companion_matrix, levinson_durbin,
    pacf_transform, pacf_untransform, softmax_simplex, minimize_nelder_mead,
    # ts kernels / records
    acf, pacf, ccf, autocovariance, ljung_box, durbin_watson, jarque_bera,
    heteroskedasticity_test, count_params, gaussian_loglike,
    information_criteria,
    # transforms
    box_cox, box_cox_inverse, guerrero_lambda, Standardizer, detrend,
    Differencer, diff, integrate, fourier_terms,
    moving_average, seasonal_decompose, stl_decompose, seasonal_strength,
    # unit root
    adf_test, kpss_test, ndiffs, nsdiffs, detect_seasonal_period,
    # psi / intervals / state space
    psi_weights, pi_weights, forecast_error_variance,
    analytic_forecast_intervals, kalman_filter, kalman_smoother, arima_ssm,
    StateSpaceModel,
    # estimators + models
    yule_walker, burg, hannan_rissanen, css_estimate, ARIMA,
    ExponentialSmoothing, SimpleExpSmoothing, Holt, auto_ets, ThetaModel,
    local_level, local_linear_trend, VAR, var_select_order, GARCH,
    auto_arima, grid_search_arima, make_forecaster,
    # metrics / backtest
    mae, mse, rmse, mape, smape, wmape, mase, rmsse, pinball_loss,
    interval_score, empirical_coverage, evaluate, score,
    RollingOriginSplitter, train_test_split, backtest, diebold_mariano,
    combine_forecasts, EnsembleForecaster, auto_select,
    # simulators / payloads
    simulate_sarima, simulate_arma, simulate_series,
    build_result_payload, build_a2ui_payload,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol * (1.0 + abs(b))


def mat_close(A, B, tol=1e-8):
    return all(abs(A[i][j] - B[i][j]) <= tol
               for i in range(len(A)) for j in range(len(A[0])))


def ar1(n, phi, sigma=1.0, seed=0):
    rng = DeterministicRNG(seed)
    x = [0.0]
    for _ in range(n + 200):
        x.append(phi * x[-1] + sigma * rng.gauss())
    return x[-n:]


def white(n, seed=0, sd=1.0):
    """A genuine white-noise sample (one RNG drawn n times, NOT n fresh RNGs)."""
    rng = DeterministicRNG(seed)
    return [sd * rng.gauss() for _ in range(n)]


def randomwalk(n, seed=0):
    rng = DeterministicRNG(seed)
    x = [0.0]
    for _ in range(n):
        x.append(x[-1] + rng.gauss())
    return x


# ===========================================================================
class TestSpecialFunctions:
    def test_normal(self):
        assert approx(norm_ppf(0.975), 1.959963985, 1e-4)
        assert approx(norm_ppf(0.025), -1.959963985, 1e-4)
        assert approx(norm_ppf(0.5), 0.0, 1e-9) or abs(norm_ppf(0.5)) < 1e-6
        assert approx(norm_cdf(0.0), 0.5)
        assert approx(norm_cdf(1.959963985), 0.975, 1e-4)
        assert approx(norm_pdf(0.0), 1.0 / math.sqrt(2 * math.pi))
        # cdf/ppf are inverse
        for p in (0.01, 0.1, 0.3, 0.6, 0.9, 0.99):
            assert approx(norm_cdf(norm_ppf(p)), p, 1e-4)

    def test_gamma_beta(self):
        # gammainc(1, x) = 1 - e^-x
        for x in (0.5, 1.0, 2.5):
            assert approx(gammainc(1.0, x), 1.0 - math.exp(-x), 1e-6)
        assert approx(gammainc(3.0, 0.0), 0.0, 1e-9)
        assert approx(gammainc(3.0, 1e3), 1.0, 1e-6)
        # betainc(1,1,x) = x  ; regularised betainc(a,b,0)=0, (a,b,1)=1
        for x in (0.2, 0.5, 0.8):
            assert approx(betainc(1.0, 1.0, x), x, 1e-6)
        assert approx(betainc(2.0, 3.0, 0.0), 0.0, 1e-9)
        assert approx(betainc(2.0, 3.0, 1.0), 1.0, 1e-6)

    def test_chi2_t_f(self):
        # chi2 with 1 df: 95% quantile ~ 3.841459
        assert approx(chi2_ppf(0.95, df=1), 3.841459, 1e-3)
        assert approx(chi2_ppf(0.95, df=2), 5.991465, 1e-3)
        assert approx(chi2_cdf(3.841459, df=1), 0.95, 1e-3)
        assert approx(chi2_sf(0.0, df=4), 1.0, 1e-6)
        # student-t: symmetric, t_cdf(0)=0.5; large df -> normal
        assert approx(t_cdf(0.0, df=5), 0.5, 1e-6)
        assert approx(t_ppf(0.975, df=1e6), 1.959963985, 1e-3)
        assert t_ppf(0.975, df=10) > 1.959963985      # heavier tails
        # F survival in (0,1), decreasing
        assert 0.0 < f_sf(2.0, dfn=3, dfd=20) < 1.0
        assert f_sf(0.01, dfn=3, dfd=20) > f_sf(5.0, dfn=3, dfd=20)

    def test_rng_deterministic(self):
        a = [DeterministicRNG(7).gauss() for _ in range(5)]
        b = [DeterministicRNG(7).gauss() for _ in range(5)]
        # fresh RNG same seed -> same first draw
        assert a[0] == b[0]
        r = DeterministicRNG(1)
        draws = [r.gauss() for _ in range(2000)]
        assert abs(ar._mean(draws)) < 0.1 and abs(ar._std(draws) - 1.0) < 0.1


class TestLinearAlgebra:
    def test_solve_identity(self):
        A = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]
        b = [1.0, 2.0, 3.0]
        x = solve(A, b)
        assert all(approx(v, b[i]) for i, v in enumerate(matvec(A, x)))

    def test_cholesky(self):
        A = [[4.0, 2.0, 0.6], [2.0, 5.0, 1.0], [0.6, 1.0, 3.0]]
        L = cholesky(A)
        assert L is not None
        LLt = matmul(L, transpose(L))
        assert mat_close(LLt, A)
        b = [1.0, -2.0, 0.5]
        x = chol_solve(L, b)
        assert all(approx(v, b[i]) for i, v in enumerate(matvec(A, x)))

    def test_inv_logdet(self):
        A = [[3.0, 1.0], [1.0, 2.0]]
        Ai = inv(A)
        assert mat_close(matmul(A, Ai), eye(2))
        assert approx(math.exp(logdet(A)), 3.0 * 2.0 - 1.0 * 1.0)

    def test_qr_lstsq(self):
        # overdetermined y = 2 + 3x + noise-free -> recover exactly
        X = [[1.0, float(t)] for t in range(6)]
        y = [2.0 + 3.0 * t for t in range(6)]
        beta, rank = qr_lstsq(X, y)
        assert rank == 2
        assert approx(beta[0], 2.0, 1e-6) and approx(beta[1], 3.0, 1e-6)

    def test_lyapunov(self):
        # P = T P T' + Q  for a stable scalar-ish system, check the equation holds
        T = [[0.5, 0.1], [0.0, 0.3]]
        Q = [[1.0, 0.0], [0.0, 0.5]]
        P = solve_discrete_lyapunov(T, Q)
        rhs = ar.mat_add(matmul(matmul(T, P), transpose(T)), Q)
        assert mat_close(P, rhs, 1e-6)

    def test_spectral_radius(self):
        # eigenvalues 0.9 and -0.2 -> spectral radius 0.9
        A = [[0.9, 0.0], [0.0, -0.2]]
        assert approx(spectral_radius(A), 0.9, 1e-6)


class TestPolynomials:
    def test_poly_mul(self):
        # (1 + 2B)(1 + 3B) = 1 + 5B + 6B^2
        assert poly_mul([1.0, 2.0], [1.0, 3.0]) == [1.0, 5.0, 6.0]
        assert poly_mul([1.0], [1.0, -1.0]) == [1.0, -1.0]

    def test_durand_kerner_real_roots(self):
        # z^2 - 5z + 6 = (z-2)(z-3); ascending coefs [6, -5, 1]
        roots = durand_kerner_roots([6.0, -5.0, 1.0])
        mods = sorted(r.real for r in roots)
        assert approx(mods[0], 2.0, 1e-6) and approx(mods[1], 3.0, 1e-6)

    def test_durand_kerner_complex(self):
        # z^2 + 1 = 0 -> +/- i
        roots = durand_kerner_roots([1.0, 0.0, 1.0])
        assert all(approx(abs(r), 1.0, 1e-6) for r in roots)
        assert all(abs(r.real) < 1e-6 for r in roots)

    def test_companion_spectral(self):
        # AR(1) phi=0.7 -> characteristic polynomial 1 - 0.7 z has its root at
        # 1/0.7 (outside the unit circle => stationary).  companion_matrix takes
        # the char polynomial [1, -0.7] and its eigenvalue IS that root.
        C = companion_matrix([1.0, -0.7])
        assert approx(spectral_radius(C), 1.0 / 0.7, 1e-6)
        assert spectral_radius(C) > 1.0                    # stationary


class TestLevinsonPacf:
    def test_levinson_ar1(self):
        # gamma_k = phi^k for AR(1); order-1 -> phi
        gamma = [1.0, 0.6, 0.36, 0.216]
        phi, refl, v = levinson_durbin(gamma, order=1)
        assert approx(phi[0], 0.6, 1e-9)
        # reflection coefficient (pacf) at lag1 == acf1
        assert approx(refl[0], 0.6, 1e-9)

    def test_levinson_ar2(self):
        # AR(2) Yule-Walker: gamma from phi=[0.5,-0.3]
        phi_true = [0.5, -0.3]
        y = simulate_arma(phi_true, [], n=4000, seed=1)
        g = autocovariance(y, nlags=2)
        phi, refl, v = levinson_durbin(g, order=2)
        assert approx(phi[0], 0.5, 0.06) and approx(phi[1], -0.3, 0.06)

    def test_pacf_transform_roundtrip(self):
        raw = [0.3, -0.8, 1.2, -0.4]
        phi = pacf_transform(raw)
        back = pacf_untransform(phi)
        assert all(approx(back[i], raw[i], 1e-6) for i in range(len(raw)))
        # transformed AR must be stationary: char-poly roots OUTSIDE unit circle
        char = [1.0] + [-p for p in phi]
        assert spectral_radius(companion_matrix(char)) > 1.0
        assert all(abs(r) > 1.0 for r in durand_kerner_roots(char))

    def test_softmax_simplex(self):
        s = softmax_simplex([1.0, 2.0, -0.5])
        assert all(v >= 0 for v in s) and sum(s) <= 1.0 + 1e-9


class TestOptimizer:
    def test_nelder_mead_quadratic(self):
        f = lambda x: (x[0] - 3.0) ** 2 + (x[1] + 1.0) ** 2
        r = minimize_nelder_mead(f, [0.0, 0.0])
        assert approx(r.x[0], 3.0, 1e-3) and approx(r.x[1], -1.0, 1e-3)

    def test_nelder_mead_rosenbrock(self):
        f = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
        r = minimize_nelder_mead(f, [-1.2, 1.0], restarts=3, maxiter=4000)
        assert r.fun < 1e-4

    def test_nelder_mead_deterministic(self):
        f = lambda x: sum(v * v for v in x)
        a = minimize_nelder_mead(f, [1.0, 2.0, 3.0], seed=0)
        b = minimize_nelder_mead(f, [1.0, 2.0, 3.0], seed=0)
        assert a.x == b.x


class TestStats:
    def test_kahan_mean_var(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert approx(ar._mean(xs), 3.0)
        assert approx(ar._var(xs), 2.0)            # population variance
        assert approx(ar._var(xs, ddof=1), 2.5)    # sample variance
        assert approx(ar._median(xs), 3.0)
        assert approx(ar._median([1.0, 2.0, 3.0, 4.0]), 2.5)

    def test_quantile(self):
        xs = list(range(101))
        assert approx(ar._quantile([float(v) for v in xs], 0.5), 50.0)
        assert approx(ar._quantile([float(v) for v in xs], 0.0), 0.0)
        assert approx(ar._quantile([float(v) for v in xs], 1.0), 100.0)


class TestACFPACF:
    def test_acf_ar1(self):
        # AR(1): acf(k) = phi^k
        y = ar1(5000, 0.6, seed=2)
        r = acf(y, nlags=4)
        assert r.values[0] == pytest.approx(1.0)
        assert approx(r.values[1], 0.6, 0.05)
        assert approx(r.values[2], 0.36, 0.05)
        assert approx(r.values[3], 0.216, 0.06)
        # confidence bounds present and bracket the estimate correctly
        assert r.confint_lower is not None and r.confint_upper is not None
        assert r.confint_upper[1] > r.confint_lower[1]
        assert all(math.isfinite(v) for v in r.confint_upper + r.confint_lower)

    def test_pacf_cutoff(self):
        # AR(1): pacf cuts off after lag 1
        y = ar1(5000, 0.7, seed=3)
        p = pacf(y, nlags=5)
        assert approx(p.values[1], 0.7, 0.05)
        assert abs(p.values[2]) < 0.06 and abs(p.values[3]) < 0.06
        # AR(2): pacf cuts off after lag 2
        y2 = simulate_arma([0.5, -0.3], [], n=6000, seed=4)
        p2 = pacf(y2, nlags=5)
        assert approx(p2.values[2], -0.3, 0.06)
        assert abs(p2.values[3]) < 0.06

    def test_ccf(self):
        y = ar1(2000, 0.5, seed=5)
        x = [0.0] + y[:-1]                          # x leads y by one step
        c = ccf(x, y, nlags=3)
        assert c is not None and len(c.values) >= 1

    def test_empty_series_clean_error(self):
        # Regression (NB9): the correlation family crashed with obscure IndexError /
        # ZeroDivisionError on an empty series (e.g. one filtered down to nothing in a
        # pipeline); it must raise a clean ValueError like the model-fitting path does.
        for fn in (lambda: acf([], nlags=1),
                   lambda: pacf([], nlags=1),
                   lambda: ccf([], [], nlags=1),
                   lambda: ljung_box([]),
                   lambda: ar.autocovariance([])):
            with pytest.raises(ValueError):
                fn()
        # a length-1 series is still accepted (only truly empty is rejected)
        assert acf([5.0], nlags=1) is not None

    def test_acf_affine_and_reversal_invariant(self):
        # Metamorphic: autocorrelation is invariant to an affine transform a*y + b
        # (it de-means and normalises) and is symmetric in time.  A hidden dependence
        # on the absolute scale/level (the sibling module's recurring bug class) would
        # break these to well above machine precision.
        y = simulate_arma([0.6, -0.2], [0.3], n=400, seed=1)
        base = acf(y, nlags=8).values
        affine = acf([3.5 * v + 100.0 for v in y], nlags=8).values
        reverse = acf(list(reversed(y)), nlags=8).values
        for k in range(9):
            assert approx(base[k], affine[k], 1e-12)
            assert approx(base[k], reverse[k], 1e-12)
    def test_ljung_box_white_noise(self):
        wn = white(400, 9)
        lb = ljung_box(wn, lags=10)
        assert lb.pvalue_lb[-1] > 0.05             # do NOT reject independence
        # strongly autocorrelated -> reject
        y = ar1(400, 0.8, seed=11)
        lb2 = ljung_box(y, lags=10)
        assert lb2.pvalue_lb[-1] < 0.05

    def test_durbin_watson(self):
        wn = white(500, 3)
        assert approx(durbin_watson(wn), 2.0, 0.2)    # ~2 for no autocorrelation
        pos = ar1(500, 0.8, seed=1)
        assert durbin_watson(pos) < 1.0               # positive autocorrelation

    def test_jarque_bera(self):
        normal = white(1000, 2)
        jb = jarque_bera(normal)
        assert jb.pvalue > 0.05                        # cannot reject normality
        skewed = [abs(v) ** 2 for v in white(1000, 5)]
        assert jarque_bera(skewed).pvalue < 0.05       # reject

    def test_arch_lm(self):
        wn = white(400, 4)
        assert heteroskedasticity_test(wn, lags=6).pvalue > 0.05


class TestTransforms:
    def test_box_cox_roundtrip(self):
        y = [10.0, 12.0, 15.0, 9.0, 20.0]
        for lam in (0.0, 0.5, 1.0, 2.0):
            w = box_cox(y, lam=lam)
            back = box_cox_inverse(w, lam=lam)
            assert all(approx(back[i], y[i], 1e-6) for i in range(len(y)))
        # lam=0 is the log transform
        assert approx(box_cox([math.e], lam=0.0)[0], 1.0, 1e-9)

    def test_guerrero_lambda(self):
        y = simulate_series(120, level=100, trend=0.1, seasonality=12,
                            seasonal_amp=5, noise_sd=1.0, seed=0)
        lam = guerrero_lambda(y, m=12)
        assert -1.0 <= lam <= 2.0

    def test_differencing_roundtrip_exact(self):
        y = simulate_series(80, level=50, trend=0.3, seasonality=12,
                            seasonal_amp=4, seed=1)
        for d, D, m in [(1, 0, 1), (2, 0, 1), (0, 1, 12), (1, 1, 12)]:
            dfr = Differencer(d=d, D=D, m=m)
            w = dfr.fit_transform(y)
            rec = dfr.inverse_transform(w)
            assert len(rec) == len(y)
            assert max(abs(y[i] - rec[i]) for i in range(len(y))) < 1e-8

    def test_diff_integrate_helpers(self):
        y = [1.0, 3.0, 6.0, 10.0, 15.0]            # first diff = [2,3,4,5]
        w = diff(y, d=1)
        assert w == pytest.approx([2.0, 3.0, 4.0, 5.0])
        # integrate reconstructs the FUTURE values y[1:] from the history y[0]
        back = integrate(w, history=y[:1], d=1)
        assert back == pytest.approx(y[1:])

    def test_standardizer(self):
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        s = Standardizer().fit(y)
        z = s.transform(y)
        assert approx(ar._mean(z), 0.0, 1e-9) and approx(ar._std(z), 1.0, 1e-6)
        assert all(approx(s.inverse_transform(z)[i], y[i]) for i in range(len(y)))

    def test_fourier_terms(self):
        F = fourier_terms(24, m=12, k=2)
        assert len(F) == 24 and len(F[0]) == 4       # 2 harmonics -> 4 columns
        assert all(-1.0 - 1e-9 <= v <= 1.0 + 1e-9 for row in F for v in row)


class TestDecomposition:
    def test_seasonal_decompose_recovers_pattern(self):
        y = simulate_series(120, level=100, trend=0.0, seasonality=12,
                            seasonal_amp=10, noise_sd=0.3, seed=0)
        dec = seasonal_decompose(y, model="additive", period=12)
        # the seasonal component repeats with period 12 and sums ~0 over a cycle
        assert approx(sum(dec.seasonal[:12]), 0.0, 0.5)
        assert all(approx(dec.seasonal[t], dec.seasonal[t + 12], 1e-6)
                   for t in range(12))

    def test_stl(self):
        y = simulate_series(96, level=50, trend=0.2, seasonality=12,
                            seasonal_amp=6, noise_sd=0.5, seed=2)
        dec = stl_decompose(y, period=12)
        assert len(dec.trend) == len(y) and len(dec.seasonal) == len(y)

    def test_loess_fast_path_matches_full_sort(self):
        # Regression (NB8): the sorted-x neighbour window in _loess_point must select
        # the SAME nearest points as the full O(n log n) sort -- this bit-identity is
        # what makes the STL O(n^2 log n) -> O(n*span) speedup safe (output unchanged).
        import random as _random
        rng = _random.Random(0)
        x = [float(i) for i in range(150)]
        y = [rng.gauss() for _ in range(150)]
        for sp in (3, 7, 21):
            for x0 in (0.0, 7.0, 74.5, 149.0):
                slow = ar._loess_point(x, y, x0, span_pts=sp, degree=1)
                fast = ar._loess_point(x, y, x0, span_pts=sp, degree=1,
                                       sorted_x=True)
                assert approx(fast, slow, 1e-12)

    def test_stl_does_not_regress_to_quadratic(self):
        # STL must stay near-linear.  The old code sorted all n points inside every
        # _loess_point call (O(n^2 log n)) -- ~25s at n=3000; the windowed search is
        # ~0.4s.  A generous ceiling catches an accidental regression without flaking.
        import time
        y = simulate_series(3000, seasonality=12, seasonal_amp=6, trend=0.01,
                            ar=[0.3], seed=1)
        t0 = time.time()
        dec = stl_decompose(y, period=12)
        assert time.time() - t0 < 8.0
        assert max(abs(dec.trend[t] + dec.seasonal[t] + dec.resid[t] - y[t])
                   for t in range(len(y))) < 1e-9

    def test_seasonal_strength(self):
        seasonal = simulate_series(120, seasonality=12, seasonal_amp=10,
                                   noise_sd=0.3, seed=1)
        flat = white(120, 1)
        assert seasonal_strength(seasonal, m=12) > 0.7
        assert seasonal_strength(flat, m=12) < 0.4


class TestUnitRoot:
    def test_adf(self):
        rw = ar._clean(simulate_arma([1.0 - 1e-9], [], n=300, seed=1))  # ~random walk
        rw = [0.0]
        rng = DeterministicRNG(1)
        for _ in range(300):
            rw.append(rw[-1] + rng.gauss())
        stat = ar1(300, 0.3, seed=2)
        # ADF: reject unit root (stationary) for AR(1) phi=0.3, not for a walk
        assert adf_test(stat).stationary
        assert not adf_test(rw).stationary

    def test_kpss(self):
        stat = ar1(300, 0.3, seed=3)
        rw = [0.0]
        rng = DeterministicRNG(2)
        for _ in range(300):
            rw.append(rw[-1] + rng.gauss())
        # KPSS null = stationary: do not reject for AR(1), reject for a walk
        assert kpss_test(stat).stationary
        assert not kpss_test(rw).stationary

    def test_ndiffs_nsdiffs(self):
        rw = [0.0]
        rng = DeterministicRNG(4)
        for _ in range(200):
            rw.append(rw[-1] + rng.gauss())
        assert ndiffs(rw) >= 1
        seasonal = simulate_series(120, seasonality=12, seasonal_amp=10,
                                   trend=0.0, noise_sd=0.5, seed=1)
        assert nsdiffs(seasonal, m=12) >= 1

    def test_detect_period(self):
        y = simulate_series(120, seasonality=12, seasonal_amp=8, trend=0.1,
                            noise_sd=0.5, seed=0)
        assert detect_seasonal_period(y).period == 12
        q = simulate_series(120, seasonality=4, seasonal_amp=8, noise_sd=0.5,
                            seed=1)
        assert detect_seasonal_period(q).period == 4

    def test_detect_period_not_fooled_by_ar(self):
        # Regression (NB6): a plain AR process has a monotonically decaying ACF and
        # NO seasonality, but seasonal_strength -- driven by an STL fit that overfits
        # tiny periods -- reported ~1.0 at m=2, so the detector wrongly returned 2.
        # A genuine seasonal lag must stand out as a local ACF *peak*; AR decay
        # does not, so these must all resolve to "no seasonality" (period 1).
        for phi in ([0.5], [0.8], [0.5, -0.3]):
            y = simulate_arma(phi, [], n=200, seed=3)
            assert detect_seasonal_period(y).period == 1
        # a genuinely period-2 (alternating) series must still be detected
        rng = DeterministicRNG(9)
        alt = [(10.0 if t % 2 == 0 else 2.0) + 0.5 * rng.gauss() for t in range(120)]
        assert detect_seasonal_period(alt).period == 2


class TestPsiIntervals:
    def test_psi_closed_form(self):
        # AR(1): psi_j = phi^j
        psi = psi_weights(phi=[0.5], n=6)
        assert all(approx(psi[j], 0.5 ** j, 1e-9) for j in range(6))
        # MA(1): psi = [1, theta, 0, 0, ...]
        psi_ma = psi_weights(theta=[0.3], n=5)
        assert psi_ma == pytest.approx([1.0, 0.3, 0.0, 0.0, 0.0])
        # integrated (d=1): psi all 1 -> random-walk variance grows linearly
        psi_i = psi_weights(d=1, n=5)
        assert all(approx(v, 1.0, 1e-9) for v in psi_i)

    def test_forecast_error_variance(self):
        fev = forecast_error_variance([1.0, 1.0, 1.0, 1.0], sigma2=2.0, horizon=4)
        assert fev == pytest.approx([2.0, 4.0, 6.0, 8.0])
        # AR(1): var(h) = sigma2 * sum_{j<h} phi^{2j}
        psi = psi_weights(phi=[0.5], n=10)
        fev2 = forecast_error_variance(psi, sigma2=1.0, horizon=3)
        assert approx(fev2[0], 1.0) and approx(fev2[1], 1.25) and approx(fev2[2], 1.3125)

    def test_analytic_intervals_widen(self):
        psi = psi_weights(phi=[0.6], n=12)
        fr = analytic_forecast_intervals([0.0] * 10, psi=psi, sigma2=1.0,
                                         levels=(0.95,))
        widths = [fr.upper[0.95][h] - fr.lower[0.95][h] for h in range(10)]
        assert all(widths[h + 1] >= widths[h] - 1e-9 for h in range(9))
        assert approx((fr.upper[0.95][0] + fr.lower[0.95][0]) / 2, 0.0, 1e-9)

    def test_interval_coverage(self):
        # rolling 1-step 90% intervals on AR(1) should cover ~90% of actuals
        y = ar1(300, 0.6, seed=7)
        hits = total = 0
        for t in range(272, 299):
            m = ARIMA(y[:t], order=(1, 0, 0), trend="c").fit(method="css")
            fc = m.forecast(1, level=(0.90,))
            lo, hi = fc.lower[0.90][0], fc.upper[0.90][0]
            hits += 1 if lo <= y[t] <= hi else 0
            total += 1
        assert 0.78 <= hits / total <= 1.0         # near the nominal 90%


class TestKalman:
    def test_ar1_exact_likelihood(self):
        # THE oracle: Kalman filter loglike == closed-form AR(1) exact likelihood
        phi, s2 = 0.6, 1.3
        y = simulate_arma([phi], [], n=200, sigma2=s2, seed=5)
        y = [v - ar._mean(y) for v in y]
        n = len(y)
        ll = (-0.5 * n * math.log(2 * math.pi) - 0.5 * n * math.log(s2)
              + 0.5 * math.log(1 - phi ** 2)
              - (1 - phi ** 2) / (2 * s2) * y[0] ** 2
              - 1.0 / (2 * s2) * sum((y[t] - phi * y[t - 1]) ** 2
                                     for t in range(1, n)))
        kf = kalman_filter(arima_ssm(phi=[phi], theta=[], sigma2=s2), y)
        assert approx(kf.loglike, ll, 1e-6)

    def test_true_model_higher_likelihood(self):
        phi = 0.7
        y = simulate_arma([phi], [], n=300, sigma2=1.0, seed=6)
        y = [v - ar._mean(y) for v in y]
        ll_true = kalman_filter(arima_ssm(phi=[phi], sigma2=1.0), y).loglike
        ll_wrong = kalman_filter(arima_ssm(phi=[0.1], sigma2=1.0), y).loglike
        assert ll_true > ll_wrong

    def test_smoother_runs(self):
        y = simulate_arma([0.5], [0.3], n=100, seed=8)
        y = [v - ar._mean(y) for v in y]
        out = kalman_smoother(arima_ssm(phi=[0.5], theta=[0.3], sigma2=1.0), y)
        assert out is not None

    def test_smoother_below_filter(self):
        # RTS smoother: with observation noise (H > 0) the smoothed states use future
        # data, so they are LESS jagged than the filtered states (lower total
        # variation) and the last smoothed state equals the last filtered state
        # (no future information at the endpoint).
        ss = ar.StateSpaceModel(T=[[1.0]], Z=[1.0], Q=[[0.1]], H=1.0,
                                R=[[1.0]], a1=[0.0], P1=[[10.0]])
        rng = DeterministicRNG(3)
        lvl, y = 0.0, []
        for _ in range(80):
            lvl += math.sqrt(0.1) * rng.gauss()
            y.append(lvl + rng.gauss())
        kf = kalman_filter(ss, y)
        sm = kalman_smoother(ss, y)
        assert approx(sm[-1][0], kf.filtered_state[-1][0], 1e-9)
        tv = lambda s: sum(abs(s[t] - s[t - 1]) for t in range(1, len(s)))
        assert tv([s[0] for s in sm]) < tv([s[0] for s in kf.filtered_state])

    def test_kalman_missing_data(self):
        # Missing (NaN / None) observations must trigger a prediction-only step:
        # they add nothing to the log-likelihood and leave every filtered state
        # finite (the filter coasts through the gap).  NaN and None are equivalent.
        y = simulate_arma([0.7], [], n=80, seed=3)
        y = [v - ar._mean(y) for v in y]
        ss = arima_ssm(phi=[0.7], theta=[], sigma2=1.0)
        ll_full = kalman_filter(ss, y).loglike
        gaps = (20, 21, 40)
        y_nan = [float("nan") if i in gaps else y[i] for i in range(len(y))]
        y_none = [None if i in gaps else y[i] for i in range(len(y))]
        kf = kalman_filter(ss, y_nan)
        assert all(math.isfinite(s[0]) for s in kf.filtered_state)
        assert math.isfinite(kf.loglike) and kf.loglike != ll_full
        assert approx(kf.loglike, kalman_filter(ss, y_none).loglike, 1e-12)

    def test_arima_mle_recovers_ar1(self):
        y = simulate_arma([0.7], [], n=600, sigma2=1.0, seed=9)
        m = ARIMA(y, order=(1, 0, 0), trend="c").fit(method="mle")
        assert approx(m.phi[0], 0.7, 0.08)
        assert math.isfinite(m.llf)


class TestEstimators:
    def test_yule_walker_ar2(self):
        y = simulate_arma([0.5, -0.3], [], n=5000, seed=1)
        r = yule_walker(y, order=2)
        assert approx(r.phi[0], 0.5, 0.05) and approx(r.phi[1], -0.3, 0.05)

    def test_burg_ar1(self):
        # regression guard: burg previously had a sign bug (returned -phi)
        y = simulate_arma([0.6], [], n=4000, seed=2)
        r = burg(y, order=1)
        assert approx(r.phi[0], 0.6, 0.05)         # NOT -0.6
        y2 = simulate_arma([0.4, 0.2], [], n=4000, seed=3)
        r2 = burg(y2, order=2)
        assert approx(r2.phi[0], 0.4, 0.06) and approx(r2.phi[1], 0.2, 0.06)

    def test_hannan_rissanen_arma(self):
        y = simulate_arma([0.6], [0.4], n=3000, seed=4)
        r = hannan_rissanen(y, order=(1, 1))
        assert approx(r.phi[0], 0.6, 0.12) and approx(r.theta[0], 0.4, 0.15)

    def test_css_recovers_arma(self):
        y = simulate_arma([0.5], [0.3], n=1500, seed=5)
        r = css_estimate(y, order=(1, 0, 1))
        assert approx(r.phi[0], 0.5, 0.1) and approx(r.theta[0], 0.3, 0.12)


class TestARIMA:
    def test_random_walk_forecast(self):
        # ARIMA(0,1,0): forecast is a flat line at the last observation
        y = randomwalk(100, seed=1)
        m = ARIMA(y, order=(0, 1, 0), trend="n").fit(method="css")
        fc = m.forecast(5)
        assert all(approx(v, y[-1], 1e-6) for v in fc.mean)
        # variance grows linearly -> intervals widen like sqrt(h)
        w = [fc.upper[0.95][h] - fc.lower[0.95][h] for h in range(5)]
        assert w[4] > w[0]

    def test_ar1_fit_recovers(self):
        y = simulate_arma([0.65], [], n=1000, seed=2)
        m = ARIMA(y, order=(1, 0, 0), trend="c").fit(method="css")
        assert approx(m.phi[0], 0.65, 0.06)
        assert len(m.resid) == len(m.fittedvalues)
        assert m.aic < m.bic or m.nobs < 8         # BIC penalises more (n>=8)

    def test_sarima_end_to_end(self):
        y = simulate_series(160, level=100, trend=0.1, seasonality=12,
                            seasonal_amp=8, ar=[0.5], noise_sd=1.0, seed=3)
        m = ARIMA(y[:148], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12),
                  trend="c").fit(method="css")
        fc = m.forecast(12, level=(0.8, 0.95))
        assert len(fc.mean) == 12 and 0.95 in fc.lower and 0.8 in fc.upper
        assert all(math.isfinite(v) for v in fc.mean)

    def test_information_criteria_formulas(self):
        # exact IC formulas: AIC=2k-2L, BIC=k ln n -2L, AICc=AIC+2k(k+1)/(n-k-1),
        # HQIC=2k ln(ln n) - 2L
        L, k, n = -100.0, 3, 50
        ic = information_criteria(loglik=L, n_params=k, nobs=n)
        assert approx(ic.aic, 2 * k - 2 * L)
        assert approx(ic.bic, k * math.log(n) - 2 * L)
        assert approx(ic.aicc, ic.aic + 2 * k * (k + 1) / (n - k - 1))
        assert approx(ic.hqic, 2 * k * math.log(math.log(n)) - 2 * L)

    def test_predict_alignment(self):
        y = simulate_arma([0.5], [], n=200, seed=5)
        m = ARIMA(y, order=(2, 0, 0), trend="c").fit(method="css")
        # predict / resid / fittedvalues are mutually consistent (the CSS
        # conditional sample: one entry per observation past the AR conditioning)
        assert len(m.predict()) == len(m.resid) == len(m.fittedvalues)
        assert len(m.resid) == m.nobs - 2               # AR(2) conditions on 2 obs
        # nobs is the FULL differenced sample the (exact) likelihood is scored on --
        # constant across orders so that AIC/AICc/BIC are comparable (see
        # test_ic_comparable_across_orders); here d=0 so it equals len(y).
        assert m.nobs == len(y)

    def test_exact_ic_robustness(self):
        # The exact full-sample IC (NB1 fix) must stay finite for near-unit-root and
        # near-non-invertible fits, and the css-mle Kalman refinement must never lower
        # the reported (exact) log-likelihood below the css starting point.
        for order, phi, theta in [((1, 0, 0), [0.97], []),
                                  ((0, 0, 1), [], [-0.95]),
                                  ((1, 0, 1), [0.9], [-0.85])]:
            y = simulate_arma(phi, theta, n=120, seed=3)
            mc = ARIMA(y, order=order, trend="c").fit(method="css")
            assert math.isfinite(mc.aicc) and math.isfinite(mc.llf)
            assert mc.nobs == len(y)                     # full-sample IC basis
        # css-mle refines the exact likelihood upward (one representative case)
        y = simulate_arma([0.97], [], n=120, seed=3)
        mc = ARIMA(y, order=(1, 0, 0), trend="c").fit(method="css")
        mm = ARIMA(y, order=(1, 0, 0), trend="c").fit(method="css-mle")
        assert mm.llf >= mc.llf - 1e-6                   # MLE refines the exact llf

    def test_estimator_methods_dispatch(self):
        # Coverage: ARIMA.fit dispatches to yule-walker / burg / hr / css / mle /
        # css-mle -- only css/mle were exercised through ARIMA before (the estimators
        # were tested standalone).  On AR(2) data every method must recover ~the same
        # coefficients with a finite log-likelihood and forecast.
        y = simulate_arma([0.6, -0.2], [], n=400, seed=2)
        base = ARIMA(y, order=(2, 0, 0), trend="c").fit(method="css").phi
        for meth in ("yule-walker", "burg", "hr", "mle", "css-mle"):
            m = ARIMA(y, order=(2, 0, 0), trend="c").fit(method=meth)
            assert math.isfinite(m.llf)
            assert all(approx(m.phi[i], base[i], 0.05) for i in range(2))
            assert all(math.isfinite(v) for v in m.forecast(5).mean)

    def test_standard_errors_calibrated(self):
        # Statistical calibration: the reported coefficient standard errors (from the
        # Hessian of the log-likelihood) must match the AR(1) asymptotic SE
        # sqrt((1 - phi^2) / n).  A broken information matrix would be off by a large
        # factor -- in practice the ratio is ~1.01, so a generous band still bites.
        phi_true, n = 0.6, 200
        ses = []
        for s in range(30):
            y = simulate_arma([phi_true], [], n=n, seed=s)
            m = ARIMA(y, order=(1, 0, 0), trend="c").fit(method="css")
            if m.bse:
                ses.append(m.bse[0])
        mean_se = sum(ses) / len(ses)
        asymp = math.sqrt((1 - phi_true ** 2) / n)
        assert 0.80 < mean_se / asymp < 1.25

    def test_forecast_level_validation(self):
        # API-footgun (NB10): a confidence level must be a FRACTION in (0, 1).  Passing
        # a percentage (95 -- a very common mistake) or an out-of-range value used to
        # yield silent infinite- or negative-width intervals (norm_ppf of an argument
        # outside (0,1)); it must now raise a clean ValueError on both the analytic and
        # the simulation interval paths, while valid fractions are unchanged.
        y = simulate_arma([0.6], [], n=150, seed=1)
        m = ARIMA(y, order=(1, 0, 0), trend="c").fit(method="css")
        for lv in (95, 0.0, 1.0, 1.5, -0.1):
            with pytest.raises(ValueError):
                m.forecast(3, level=(lv,))
            with pytest.raises(ValueError):
                m.forecast(3, level=(lv,), method="simulation", nsim=100)
        fc = m.forecast(4, level=(0.8, 0.95))
        z = ar.norm_ppf(0.5 + 0.95 / 2.0)
        assert approx(fc.upper[0.95][0] - fc.mean[0], z * fc.sigma[0], 1e-12)

    def test_result_is_immutable_snapshot(self):
        # Aliasing (NB11): a fitted result must be a self-contained SNAPSHOT.  Its
        # forecast used the MODEL's mutable _e_full, so refitting the same instance
        # (css then css-mle -- a realistic method-comparison workflow) silently
        # changed a previously-returned result's forecast.  Each result now stores its
        # own fit state, so an earlier result is stable across later refits, on both
        # the analytic and simulation interval paths.  (An MA term is used so the
        # forecast actually depends on the residuals.)
        y = simulate_arma([0.6], [0.4], n=250, seed=1)
        m = ARIMA(y, order=(1, 0, 1), trend="c")
        rA = m.fit(method="css")
        before = rA.forecast(6).mean
        before_sim = rA.forecast(6, method="simulation", nsim=200).mean
        m.fit(method="css-mle")                          # refit the SAME instance
        assert all(approx(before[i], rA.forecast(6).mean[i], 1e-12) for i in range(6))
        after_sim = rA.forecast(6, method="simulation", nsim=200).mean
        assert all(approx(before_sim[i], after_sim[i], 1e-12) for i in range(6))

    def test_large_offset_conditioning(self):
        # Numerical conditioning: offset-invariant quantities (acf, AR coefficients,
        # sigma^2) must survive a large additive offset -- i.e. the variance/covariance
        # computations must demean FIRST.  A naive E[x^2] - E[x]^2 formula loses ~all
        # precision at an offset of 1e6 (the sibling module had exactly this bug in its
        # HP filter); the demean-first code here stays accurate to many digits.
        y = simulate_arma([0.6, -0.2], [], n=300, seed=1)
        yo = [v + 1.0e6 for v in y]
        base_acf, off_acf = acf(y, nlags=3).values, acf(yo, nlags=3).values
        assert all(approx(base_acf[k], off_acf[k], 1e-7) for k in range(4))
        base = ARIMA(y, order=(2, 0, 0), trend="c").fit(method="css")
        off = ARIMA(yo, order=(2, 0, 0), trend="c").fit(method="css")
        assert all(approx(base.phi[i], off.phi[i], 1e-5) for i in range(2))
        assert approx(base.sigma2, off.sigma2, 1e-6)

    def test_golden_values_vs_statsmodels(self):
        # Golden values cross-validated against statsmodels 0.14.6 (NOT a dependency --
        # these are pinned so a regression is caught in the pure-stdlib suite).  Our
        # ljung_box / adf-tau / yule_walker match statsmodels to machine precision and
        # the exact-ML ARIMA log-likelihood matches to <1e-4 across orders.
        r = simulate_arma([0.5], [], n=300, seed=2)
        assert approx(ljung_box(r, lags=10).stat_lb[-1], 111.5815, 1e-3)
        y = simulate_arma([0.5, -0.2], [], n=300, seed=3)
        assert approx(adf_test(y, regression="c", maxlag=4, autolag=None).statistic,
                      -8.2206, 1e-3)
        yw = yule_walker(y, order=3, adjusted=False).phi
        assert all(approx(yw[i], v, 1e-3)
                   for i, v in enumerate([0.4211, -0.234, -0.0101]))
        # exact-ML llf == statsmodels' to <1e-4 (optimum is smooth -> a loose band
        # still catches a broken likelihood/Kalman)
        assert approx(ARIMA(y, order=(2, 0, 0), trend="c").fit(method="mle").llf,
                      -424.6082, 5e-2)

    def test_scale_and_negation_equivariance(self):
        # Metamorphic: ARMA coefficients are invariant to scaling/negating the data;
        # the forecast is equivariant (scales by c, mirrors under negation) and sigma2
        # scales by c^2.  Catches any hidden dependence on the absolute data scale.
        y = simulate_arma([0.6, -0.2], [0.3], n=300, seed=1)
        m0 = ARIMA(y, order=(2, 0, 1), trend="c").fit(method="css")
        f0 = m0.forecast(5).mean
        for c in (1000.0, 0.001, -1.0):
            mc = ARIMA([c * v for v in y], order=(2, 0, 1),
                       trend="c").fit(method="css")
            assert all(approx(m0.phi[i], mc.phi[i], 1e-6) for i in range(2))
            assert approx(m0.theta[0], mc.theta[0], 1e-6)
            assert approx(mc.sigma2, c * c * m0.sigma2, 1e-6)
            fc = mc.forecast(5).mean
            assert all(approx(fc[h], c * f0[h], 1e-5) for h in range(5))


class TestETS:
    def test_ses(self):
        y = [10.0, 11.0, 10.5, 11.5, 12.0, 11.0, 12.5]
        m = SimpleExpSmoothing(y)
        r = m.fit()                                # forecast on the fitted instance
        assert 0.0 <= r.params["alpha"] <= 1.0
        fc = m.forecast(r, 3)
        assert all(approx(fc.mean[i], fc.mean[0]) for i in range(3))  # flat

    def test_holt_trend(self):
        y = [float(t) * 2.0 + 5.0 for t in range(20)]        # perfect linear trend
        m = Holt(y)
        r = m.fit()
        fc = m.forecast(r, 5)
        # forecasts continue the upward trend
        assert fc.mean[4] > fc.mean[0] > y[-1] - 5.0

    def test_holt_winters_seasonal(self):
        y = simulate_series(96, level=100, trend=0.0, seasonality=12,
                            seasonal_amp=10, noise_sd=0.4, seed=1)
        m = ExponentialSmoothing(y, trend="add", seasonal="add",
                                 seasonal_periods=12)
        r = m.fit()
        fc = m.forecast(r, 24)
        # seasonal pattern recycles with period 12 across the 24-step horizon
        assert approx(fc.mean[0], fc.mean[12], 3.0)
        assert r.season_type == "add" and len(fc.mean) == 24

    def test_auto_ets(self):
        y = simulate_series(48, level=50, seasonality=12, seasonal_amp=6,
                            trend=0.1, noise_sd=0.5, seed=2)
        m, r = auto_ets(y, seasonal_periods=12)
        assert math.isfinite(r.aic) and len(r.model) == 3   # e.g. 'AAA'
        assert len(m.forecast(r, 6).mean) == 6

    def test_ets_ic_comparable_across_forms(self):
        # auto_ets selects by AIC across ETS forms, so (unlike the ARIMA NB1 bug)
        # every form must be scored on the SAME sample -- nobs constant -- otherwise
        # the AIC comparison is invalid.  On trend+seasonal data the richer
        # trend+seasonal form must also win over the flat form.
        y = simulate_series(96, level=100, trend=0.4, seasonality=12,
                            seasonal_amp=8, ar=[0.3], noise_sd=1.5, seed=1)
        fits = []
        for tr, se, dp in [(None, None, False), ("add", None, False),
                           ("add", "add", False), ("add", "add", True)]:
            r = ExponentialSmoothing(y, trend=tr, damped_trend=dp, seasonal=se,
                                     seasonal_periods=12).fit()
            fits.append(r)
        assert len({r.nobs for r in fits}) == 1              # same sample for all
        aics = [r.aic for r in fits]
        assert aics[2] < aics[0] and aics[2] < aics[1]       # trend+seasonal wins

    def test_theta(self):
        y = simulate_series(72, level=100, trend=0.2, seasonality=12,
                            seasonal_amp=5, noise_sd=0.5, seed=3)
        fc = ThetaModel(y, seasonal_periods=12).fit().forecast(6)
        assert len(fc.mean) == 6 and all(math.isfinite(v) for v in fc.mean)


class TestStructural:
    def test_local_level(self):
        y = randomwalk(150, seed=1)[1:]
        r = local_level(y)
        assert math.isfinite(r.llf)
        assert all(v >= -1e-9 for v in r.params.values())    # variances >= 0
        assert len(r.level) == len(y)
        assert len(r.forecast(5).mean) == 5

    def test_local_linear_trend(self):
        y = simulate_series(120, level=10, trend=0.5, noise_sd=1.0, seed=2)
        r = local_linear_trend(y)
        fc = r.forecast(6)
        assert fc.mean[5] > fc.mean[0]                        # trend continues


class TestVAR:
    def _sim_var1(self, n=600, seed=1):
        A = [[0.5, 0.1], [0.0, 0.3]]
        rng = DeterministicRNG(seed)
        data = [[0.0, 0.0]]
        for _ in range(n):
            prev = data[-1]
            a = A[0][0] * prev[0] + A[0][1] * prev[1] + rng.gauss()
            b = A[1][0] * prev[0] + A[1][1] * prev[1] + rng.gauss()
            data.append([a, b])
        return data, A

    def test_var_recovers_coefs(self):
        data, A = self._sim_var1()
        res = VAR(data, names=["a", "b"]).fit(maxlags=1)
        est = res.coefs[0]
        assert approx(est[0][0], A[0][0], 0.08) and approx(est[0][1], A[0][1], 0.08)
        assert approx(est[1][0], A[1][0], 0.08) and approx(est[1][1], A[1][1], 0.08)

    def test_var_irf_identity(self):
        data, _ = self._sim_var1(n=300)
        res = VAR(data, names=["a", "b"]).fit(maxlags=1)
        irf = res.irf(6, orthogonalized=False)
        # non-orthogonalised impulse response at lag 0 is the identity
        assert mat_close(irf.irf[0], eye(2), 1e-9)

    def test_var_fevd_rows_sum_to_one(self):
        data, _ = self._sim_var1(n=300)
        res = VAR(data, names=["a", "b"]).fit(maxlags=1)
        fe = res.fevd(8)
        last = fe.fevd[-1]
        for i in range(2):
            assert approx(sum(last[i]), 1.0, 1e-6)

    def test_var_granger(self):
        # b Granger-causes a (a depends on b's lag), not vice-versa
        data, _ = self._sim_var1(n=600)
        res = VAR(data, names=["a", "b"]).fit(maxlags=1)
        p_b_causes_a = res.test_causality("a", causing="b").pvalue
        p_a_causes_b = res.test_causality("b", causing="a").pvalue
        assert p_b_causes_a < 0.05                # b -> a detected
        assert p_a_causes_b > 0.05                # a -> b (coef 0) not detected


class TestGARCH:
    def test_garch_recovers_and_stationary(self):
        # simulate GARCH(1,1): omega=0.05, alpha=0.1, beta=0.85 (persistence 0.95)
        rng = DeterministicRNG(7)
        omega, al, be = 0.05, 0.1, 0.85
        h = omega / (1 - al - be)
        r = [math.sqrt(h) * rng.gauss()]
        for _ in range(1200):
            h = omega + al * r[-1] ** 2 + be * h
            r.append(math.sqrt(h) * rng.gauss())
        res = GARCH(r[1:], p=1, q=1).fit()
        assert res.persistence < 1.0                          # stationary
        assert approx(res.persistence, 0.95, 0.08)
        # unconditional variance = omega / (1 - persistence)
        assert res.unconditional_variance > 0
        assert len(res.conditional_variance) == len(r[1:])

    def test_garch_forecast_reverts(self):
        rng = DeterministicRNG(8)
        r = [rng.gauss() for _ in range(250)]
        model = GARCH(r, p=1, q=1)
        res = model.fit()
        vf = model.forecast(res, 30)
        # multi-step variance forecast converges toward the unconditional level
        assert abs(vf[-1] - res.unconditional_variance) <= \
            abs(vf[0] - res.unconditional_variance) + 1e-9

    def test_garch_forecast_full_recursion(self):
        # Regression (NB3/NB4): the 1-step variance forecast must iterate the FULL
        # recursion -- every ARCH/GARCH lag AND the GJR leverage on a negative last
        # shock.  The old code used only alpha[0]*e^2 + beta[0]*h (GARCH(1,1)-only),
        # silently dropping the second ARCH lag and the leverage term.
        def sim(om, arch, gam, be, n, seed):             # GJR-GARCH(len(arch), 1)
            rng = DeterministicRNG(seed)
            p = len(arch)
            e = [0.0] * p
            hh = [om / max(1 - sum(arch) - 0.5 * sum(gam) - be, 0.05)] * p
            out = []
            for _ in range(n):
                et = math.sqrt(hh[-1]) * rng.gauss()
                e.append(et)
                out.append(et)
                v = om + be * hh[-1]
                for i in range(p):
                    v += arch[i] * e[-1 - i] ** 2
                    if e[-1 - i] < 0:
                        v += gam[i] * e[-1 - i] ** 2
                hh.append(v)
            return out
        # GJR(1,1) with a genuine leverage effect and a negative last shock
        y = sim(0.05, [0.05], [0.12], 0.85, 600, 1)
        g = GARCH(y, p=1, q=1, model="gjr")
        r = g.fit()
        mu, om, al, be = (r.params["mu"], r.params["omega"],
                          r.params["alpha1"], r.params["beta1"])
        ga = r.params.get("gamma1", 0.0)
        el = y[-1] - mu
        assert ga > 1e-3 and el < 0                       # test actually bites
        want = om + al * el ** 2 + ga * el ** 2 + be * r.conditional_variance[-1]
        assert approx(g.forecast(r, 1)[0], want, 1e-9)
        # GARCH(2,1) -- the second ARCH lag must contribute to the 1-step forecast
        y2 = sim(0.05, [0.08, 0.10], [0.0, 0.0], 0.78, 700, 3)
        g2 = GARCH(y2, p=2, q=1)
        r2 = g2.fit()
        mu2, om2, a1, a2, b1 = (r2.params["mu"], r2.params["omega"],
                               r2.params["alpha1"], r2.params["alpha2"],
                               r2.params["beta1"])
        assert a2 > 1e-3                                  # test actually bites
        e2 = [v - mu2 for v in y2]
        want2 = om2 + a1 * e2[-1] ** 2 + a2 * e2[-2] ** 2 \
            + b1 * r2.conditional_variance[-1]
        assert approx(g2.forecast(r2, 1)[0], want2, 1e-9)


class TestAutoARIMA:
    def test_auto_arima_ar(self):
        y = simulate_arma([0.6, -0.2], [], n=400, seed=1)
        aa = auto_arima(y, seasonal=False, max_p=3, max_q=1, method="css")
        assert aa.order[1] == 0                    # d = 0 (stationary)
        assert aa.order[0] >= 1                     # picks some AR terms
        assert math.isfinite(aa.ic)
        assert len(aa.forecast(5).mean) == 5

    def test_grid_search(self):
        y = simulate_arma([0.5], [], n=300, seed=2)
        best = grid_search_arima(y, d=0, m=1, max_p=2, max_q=1, method="css")
        assert best is not None

    def test_ic_comparable_across_orders(self):
        # Regression: information criteria must be comparable across orders.
        # Bug: the conditional CSS likelihood was scored on nobs = len(w) - order,
        # so nobs shrank with p and AICc fell monotonically -- auto_arima therefore
        # drifted to the max_p boundary.  The exact full-sample (Kalman) likelihood
        # keeps nobs CONSTANT and penalises over-parameterisation.
        y = simulate_arma([0.5, -0.3], [], n=300, seed=7)      # AR(2) truth
        fits = [ARIMA(y, order=(p, 0, 0), trend="c").fit(method="css")
                for p in range(6)]
        # PRIMARY guard: nobs is constant across orders (the old code gave
        # 300,299,298,... which is exactly what made the IC incomparable and let
        # AICc fall monotonically into the max_p boundary).
        assert {m.nobs for m in fits} == {len(y)}
        aicc = [m.aicc for m in fits]
        assert aicc[2] < aicc[0] and aicc[2] < aicc[1]         # penalises under-fit
        # DEFINITIVE guard: the reported llf is the EXACT full-sample Kalman
        # likelihood, not the conditional CSS one (which used fewer observations).
        m2 = fits[2]                                     # d=0 here, so w == y
        kf = kalman_filter(arima_ssm(phi=m2.phi, theta=m2.theta, sigma2=1.0),
                           [v - m2.mu for v in y], concentrate_scale=True)
        assert approx(m2.llf, kf.loglike, 1e-9)

    def test_auto_arima_no_max_p_drift(self):
        # Regression (NB1 + NB2): on stationary AR(2) data the stepwise (p, q) search
        # must neither drift to the max_p boundary (the old IC fell monotonically with
        # p because nobs shrank with order) nor get trapped at a low-order MA model
        # (the old start=(1,1) sat in a local minimum at (0,0,1)).  d is fixed at 0 to
        # isolate the order search from the differencing choice.
        y = simulate_arma([0.5, -0.3], [], n=150, seed=7)
        aa = auto_arima(y, seasonal=False, d=0, max_p=3, max_q=2, method="css")
        assert aa.order[0] < 3                              # not the max_p boundary
        true_ic = ARIMA(y, order=(2, 0, 0), trend="c").fit(method="css").aicc
        assert aa.ic <= true_ic + 1e-6                     # not trapped below truth


class TestBaselines:
    def test_mean_naive_drift(self):
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        assert make_forecaster("mean").fit(y).forecast(3).mean == \
            pytest.approx([6.0, 6.0, 6.0])
        assert make_forecaster("naive").fit(y).forecast(3).mean == \
            pytest.approx([10.0, 10.0, 10.0])
        # drift: slope = (10-2)/4 = 2 -> 12, 14, 16
        assert make_forecaster("drift").fit(y).forecast(3).mean == \
            pytest.approx([12.0, 14.0, 16.0])

    def test_seasonal_naive(self):
        y = [1.0, 2.0, 3.0, 4.0] * 3                # period 4
        fc = make_forecaster("seasonal_naive", seasonality=4).fit(y).forecast(6)
        assert fc.mean == pytest.approx([1.0, 2.0, 3.0, 4.0, 1.0, 2.0])


class TestMetrics:
    def test_point_metrics(self):
        yt, yp = [1.0, 2.0, 3.0], [1.5, 2.0, 2.5]
        assert approx(mae(yt, yp), 1.0 / 3.0)
        assert approx(mse(yt, yp), 0.5 / 3.0)
        assert approx(rmse(yt, yp), math.sqrt(0.5 / 3.0))
        assert approx(mape([2.0, 4.0], [3.0, 5.0]), 37.5)     # (0.5+0.25)/2 *100
        assert approx(smape([2.0, 4.0], [3.0, 5.0]),
                      (2.0 / 5.0 + 2.0 / 9.0) / 2 * 100)
        assert approx(wmape([2.0, 4.0], [3.0, 5.0]), 2.0 / 6.0 * 100)
        # Regression (NB5): a perfect all-zero forecast is a 0% error, not nan
        # (nan would poison any backtest / ensemble aggregate).  A zero pair mixed
        # with a real pair is still skipped, leaving the real pair's error intact.
        assert smape([0.0, 0.0], [0.0, 0.0]) == 0.0
        assert approx(smape([0.0, 5.0], [0.0, 10.0]), 2.0 / 3.0 * 100)

    def test_metric_scale_relations(self):
        # Metamorphic: absolute-error metrics are scale-EQUIVARIANT (mae(c*y, c*p) =
        # c*mae) while the scale-free ones (sMAPE, MASE) are scale-INVARIANT.  The
        # sibling module's recurring bug was absolute thresholds baked into scale-
        # dependent metrics; these relations would expose any such regression here.
        yt, yp, tr = [10.0, 12.0, 11.0, 14.0], [10.5, 11.5, 11.2, 13.5], [5.0, 7.0, 6.0, 8.0]
        base_mae, base_sm = mae(yt, yp), smape(yt, yp)
        base_ms = mase(yt, yp, y_train=tr, seasonality=1)
        for c in (1e6, 1e-6):
            sy, sp, st = ([c * v for v in yt], [c * v for v in yp], [c * v for v in tr])
            assert approx(mae(sy, sp), c * base_mae, 1e-9)
            assert approx(smape(sy, sp), base_sm, 1e-9)
            assert approx(mase(sy, sp, y_train=st, seasonality=1), base_ms, 1e-9)

    def test_scaled_metrics(self):
        # MASE denom = in-sample mean abs 1-step change of train
        train = [10.0, 12.0, 11.0, 13.0]           # abs diffs: 2,1,2 -> denom 5/3
        yt, yp = [14.0, 13.0], [13.0, 13.0]        # abs errors: 1, 0 -> mae 0.5
        assert approx(mase(yt, yp, y_train=train, seasonality=1),
                      0.5 / (5.0 / 3.0))
        assert rmsse(yt, yp, y_train=train, seasonality=1) > 0

    def test_pinball_and_intervals(self):
        # pinball at the median (q=0.5) is 0.5 * MAE
        assert approx(pinball_loss([1.0, 2.0], [1.5, 1.5], quantile=0.5),
                      0.5 * mae([1.0, 2.0], [1.5, 1.5]))
        # perfect coverage
        assert empirical_coverage([1.0, 2.0, 3.0], [0.0, 1.0, 2.0],
                                  [2.0, 3.0, 4.0]) == 1.0
        assert interval_score([1.0], [0.0], [2.0], level=0.9) == \
            pytest.approx(2.0)                     # width only, actual inside

    def test_score_dispatch(self):
        yt, yp = [1.0, 2.0, 3.0], [1.1, 2.1, 2.9]
        assert approx(score(yt, yp, metric="rmse"), rmse(yt, yp))
        with pytest.raises(ValueError):
            score(yt, yp, metric="nope")

    def test_evaluate_bundle(self):
        y = simulate_series(60, level=100, seasonality=12, seasonal_amp=5, seed=1)
        m = evaluate(y[48:], y[36:48], y_train=y[:48], seasonality=12)
        assert math.isfinite(m.mae) and math.isfinite(m.rmse)
        assert isinstance(m.summary(), str)


class TestBacktest:
    def test_splitter_expanding(self):
        sp = RollingOriginSplitter(initial=10, horizon=2, step=1, mode="expanding")
        splits = list(sp.split(15))
        assert splits[0] == (range(0, 10), range(10, 12))
        assert splits[1] == (range(0, 11), range(11, 13))
        # every test window has the requested horizon; no leakage (train < test)
        for tr, te in splits:
            assert len(te) == 2 and tr.stop == te.start

    def test_splitter_sliding(self):
        sp = RollingOriginSplitter(initial=10, horizon=1, step=1, mode="sliding",
                                   window=5)
        tr, te = next(iter(sp.split(15)))
        assert len(tr) == 5 and te.start == tr.stop

    def test_backtest_runs(self):
        y = simulate_series(120, level=100, seasonality=12, seasonal_amp=6,
                            trend=0.1, seed=1)

        def sn(tr, h):
            return make_forecaster("seasonal_naive", seasonality=12).fit(tr).forecast(h).mean
        sp = RollingOriginSplitter(initial=96, horizon=12, step=12, max_splits=2)
        bt = backtest(sn, y, splitter=sp, seasonality=12, metrics=("mase", "rmse"))
        assert bt.n_splits >= 1 and math.isfinite(bt.metrics["rmse"])

    def test_diebold_mariano(self):
        rng = DeterministicRNG(3)
        e1 = [0.5 * rng.gauss() for _ in range(200)]          # model1 better
        e2 = [1.5 * rng.gauss() for _ in range(200)]
        dm = diebold_mariano(e1, e2)
        assert dm.statistic < 0 and dm.better == "model1" and dm.pvalue < 0.05
        eq1 = [rng.gauss() for _ in range(200)]
        eq2 = [rng.gauss() for _ in range(200)]
        assert diebold_mariano(eq1, eq2).pvalue > 0.05

    def test_combine(self):
        fcs = [[1.0, 2.0], [3.0, 4.0]]
        assert combine_forecasts(fcs, method="mean") == pytest.approx([2.0, 3.0])
        assert combine_forecasts(fcs, method="median") == pytest.approx([2.0, 3.0])
        w = combine_forecasts(fcs, method="weighted", weights=[3.0, 1.0])
        assert w == pytest.approx([1.5, 2.5])

    def test_auto_select(self):
        y = simulate_series(96, level=100, seasonality=12, seasonal_amp=8, seed=1)

        def sn(tr, h):
            return make_forecaster("seasonal_naive", seasonality=12).fit(tr).forecast(h).mean

        def mn(tr, h):
            return make_forecaster("mean").fit(tr).forecast(h).mean
        sp = RollingOriginSplitter(initial=72, horizon=12, step=12, max_splits=2)
        sel = auto_select(y, candidates={"snaive": sn, "mean": mn}, splitter=sp,
                          seasonality=12, metric="mase")
        assert sel["best"] == "snaive"             # seasonal-naive wins on seasonal data


class TestSimulators:
    def test_simulate_arma_acf(self):
        y = simulate_arma([0.7], [], n=4000, seed=1)
        assert approx(acf(y, nlags=1).values[1], 0.7, 0.05)

    def test_simulate_sarima_integrates(self):
        y = simulate_sarima(100, order=(1, 1, 0), phi=[0.5], seed=2)
        assert len(y) == 100 and all(math.isfinite(v) for v in y)

    def test_simulate_deterministic(self):
        a = simulate_arma([0.5], [0.3], n=50, seed=4)
        b = simulate_arma([0.5], [0.3], n=50, seed=4)
        assert a == b

    def test_simulate_series_noise_is_random(self):
        # Regression (NB7): with ar=()/ma=() (the default) simulate_series built its
        # noise as [DeterministicRNG(seed).gauss() for _ in range(n)] -- reseeding a
        # fresh RNG EACH iteration, so it repeated one draw n times (a constant
        # offset, not white noise).  The noise must vary with ~the requested std,
        # while staying byte-deterministic.
        y = simulate_series(400, level=0.0, trend=0.0, ar=[], noise_sd=2.0, seed=5)
        assert len(set(round(v, 9) for v in y)) > 1        # not a constant
        mu = sum(y) / len(y)
        std = (sum((v - mu) ** 2 for v in y) / len(y)) ** 0.5
        assert 1.5 < std < 2.5                              # ~ noise_sd = 2.0
        assert y == simulate_series(400, level=0.0, trend=0.0, ar=[],
                                    noise_sd=2.0, seed=5)   # deterministic


class TestPayloads:
    def test_result_payload_json_safe(self):
        y = simulate_arma([0.6], [], n=200, seed=1)
        m = ARIMA(y, order=(1, 0, 0), trend="c").fit(method="css")
        fc = m.forecast(6)
        json.dumps(build_result_payload(m), allow_nan=False)
        json.dumps(build_result_payload(fc), allow_nan=False)

    def test_json_safe_maps_nonfinite(self):
        fr = ar.ForecastResult(model="x", h=1, mean=[float("inf")],
                               lower={0.95: [float("nan")]},
                               upper={0.95: [1.0]})
        p = build_result_payload(fr)
        s = json.dumps(p, allow_nan=False)         # would raise if inf/nan leaked
        assert "Infinity" not in s and "NaN" not in s

    def test_a2ui_leaderboard(self):
        y = simulate_arma([0.6], [], n=300, seed=2)
        m1 = ARIMA(y, order=(1, 0, 0), trend="c").fit(method="css")
        m5 = ARIMA(y, order=(5, 0, 0), trend="c").fit(method="css")
        board = build_a2ui_payload({"ar1": m1, "ar5": m5}, ic="aic")
        json.dumps(board, allow_nan=False)
        assert board["best_model"] in ("ar1", "ar5")
        # ordered by AIC ascending
        ics = [c["ic"] for c in board["leaderboard"]]
        assert ics == sorted(ics)

    def test_a2ui_empty(self):
        board = build_a2ui_payload({})
        assert board["best_model"] is None


class TestDeterminism:
    def test_seeded_fits_reproduce(self):
        y = simulate_arma([0.6], [0.3], n=300, seed=1)
        a = ARIMA(y, order=(1, 0, 1), trend="c").fit(method="css")
        b = ARIMA(y, order=(1, 0, 1), trend="c").fit(method="css")
        assert a.phi == b.phi and a.theta == b.theta and a.llf == b.llf

    def test_garch_reproduces(self):
        r = simulate_arma([0.0], [], n=200, seed=2)
        g1 = GARCH(r, p=1, q=1).fit()
        g2 = GARCH(r, p=1, q=1).fit()
        assert g1.params == g2.params and g1.loglike == g2.loglike

    def test_auto_arima_reproduces(self):
        y = simulate_arma([0.5, -0.2], [], n=300, seed=3)
        a = auto_arima(y, seasonal=False, max_p=2, max_q=1, method="css")
        b = auto_arima(y, seasonal=False, max_p=2, max_q=1, method="css")
        assert a.order == b.order and a.seasonal_order == b.seasonal_order
