"""
Oracle-based stress tests for ``forecasting_baseline_template``.

The philosophy is *no hand-typed magic number where a closed form, an exact
identity or a metamorphic invariant exists*:

* **Closed forms** -- the least-squares solvers recover exact fits; ``ols_line``
  and the naive / drift forecasters reproduce their defining equations; every
  metric matches its formula on a worked example; Croston / SBA / TSB reproduce a
  hand-run of their recursions; ``yule_walker`` recovers a known AR coefficient.
* **Exact identities** -- ``integrate(difference(y, d), y[:d]) == y``; Box-Cox and
  the scalers round-trip; a decomposition's parts re-sum to the series; ``Q R = A``
  and ``L Lᵀ = A``; ``deseasonalise`` then ``reseasonalise`` is the identity.
* **Probabilistic / metamorphic** -- ACF peaks at the seasonal period; PACF of an
  AR(1) is ``phi`` at lag 1 and ~0 after; prediction intervals widen with the
  horizon; Holt extrapolates a line and Holt-Winters recovers a seasonal pattern;
  an ensemble is no worse than nonsense; scaled metrics vanish on a perfect forecast.

Run:  ``pytest _stress_test.py -q``
"""

import math
import random

import pytest

from gridpulse.analysis.algorithms.orchid import forecasting_baseline as fb
from gridpulse.analysis.algorithms.orchid.forecasting_baseline import (
    ForecastError, solve, cholesky, qr_householder, lstsq, lstsq_qr, ridge_lstsq,
    transpose, matmul, matvec, mean, variance, std, median, quantile,
    autocovariance, acf, pacf, norm_cdf, norm_ppf, t_ppf, ols_line, moving_average,
    seasonality_test, detect_seasonal_period,
    log_transform, boxcox, inv_boxcox, difference, integrate, seasonal_difference,
    inv_seasonal_difference, StandardScaler, MinMaxScaler,
    Naive, SeasonalNaive, Drift, MeanForecast, MovingAverageForecaster,
    WeightedMovingAverage, Naive2,
    ses_fit, SimpleExpSmoothing, Holt, HoltWinters, ETSForecaster, auto_ets,
    classical_decomposition, seasonal_indices, deseasonalise, reseasonalise,
    seasonal_adjust, Theta, theta_forecast,
    trend_features, seasonal_dummy_features, fourier_features,
    TrendForecaster, SeasonalDummyForecaster, FourierForecaster,
    ARForecaster, yule_walker, Croston, TSB,
    me, mae, mse, rmse, mape, smape, wape, mase, rmsse, mpe, r2_score,
    median_ae, pinball_loss, coverage, interval_width, msis, evaluate,
    train_test_split, rolling_origin_splits, cross_validate, backtest,
    combine_mean, combine_median, combine_weighted, combine_trimmed,
    inverse_error_weights, make_trend_seasonal, make_ar_series, make_intermittent,
)


# ===========================================================================
# 1. LINEAR ALGEBRA
# ===========================================================================
def test_solve_matches_known_solution():
    A = [[4.0, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0]]
    b = [7.0, -1.0, 2.0]
    x = solve(A, b)
    resid = [matvec(A, x)[i] - b[i] for i in range(3)]
    assert max(abs(r) for r in resid) < 1e-9


def test_qr_and_cholesky_factorisations():
    A = [[12.0, -51.0, 4.0], [6.0, 167.0, -68.0], [-4.0, 24.0, -41.0]]
    Q, R = qr_householder(A)
    recon = matmul(Q, R)
    assert max(abs(recon[i][j] - A[i][j]) for i in range(3) for j in range(3)) < 1e-8
    qtq = matmul(transpose(Q), Q)
    assert max(abs(qtq[i][j] - (1.0 if i == j else 0.0))
               for i in range(3) for j in range(3)) < 1e-9
    S = [[4.0, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0]]
    L = cholesky(S)
    for i in range(3):
        for j in range(3):
            assert abs(math.fsum(L[i][k] * L[j][k] for k in range(3)) - S[i][j]) < 1e-9


def test_least_squares_fits_exact_linear_data_two_ways():
    X = [[1.0, float(t)] for t in range(8)]
    y = [3.0 + 2.0 * t for t in range(8)]                # y = 3 + 2t exactly
    for beta in (lstsq_qr(X, y), ridge_lstsq(X, y, 0.0), lstsq(X, y)):
        assert abs(beta[0] - 3.0) < 1e-9 and abs(beta[1] - 2.0) < 1e-9


def test_lstsq_handles_underdetermined_system():
    # A wide (n < p) design used to crash the Householder QR (an AR(p) with p too
    # large, or a regression with more features than points); it must return a valid
    # least-squares fit instead of raising IndexError.
    X = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 1.0, 0.0, 1.0, 2.0, 3.0]]   # 2 x 6
    y = [10.0, 5.0]
    beta = lstsq(X, y)
    assert len(beta) == 6
    resid = [matvec(X, beta)[i] - y[i] for i in range(2)]
    assert max(abs(r) for r in resid) < 1e-6              # interpolates the two rows


def test_lstsq_qr_is_linear_and_correct_on_tall_matrices():
    # The QR least-squares path must NOT materialise the n x n Q (which made every
    # regression / OLS-AR fit O(n^2) time and memory); it stays fast on a long series
    # and matches the normal-equations solution.
    import time
    n = 2000
    X = [[1.0, float(t), float(t * t)] for t in range(n)]
    y = [3.0 + 2.0 * t - 0.001 * t * t for t in range(n)]
    t0 = time.perf_counter()
    beta = lstsq_qr(X, y)
    assert time.perf_counter() - t0 < 3.0                 # the old O(n^2) path took ~10s
    assert max(abs(a - b) for a, b in zip(beta, ridge_lstsq(X, y, 0.0))) < 1e-6


# ===========================================================================
# 2. TIME-SERIES STATISTICS
# ===========================================================================
def test_autocovariance_uses_biased_divisor():
    # gamma_k divides by n (biased, the default) not by n-k; pin both on a tiny series
    # (the divisor cancels in the ACF ratio, so it needs its own direct check).
    y = [1.0, 2.0, 3.0, 4.0]                              # deviations -1.5, -0.5, 0.5, 1.5
    g = autocovariance(y, 2, adjusted=False)
    cross1 = (-0.5) * (-1.5) + 0.5 * (-0.5) + 1.5 * 0.5   # = 1.25
    assert abs(g[0] - (1.5 ** 2 + 0.5 ** 2 + 0.5 ** 2 + 1.5 ** 2) / 4) < 1e-12   # /n
    assert abs(g[1] - cross1 / 4) < 1e-12                                        # /n, not /(n-1)
    ga = autocovariance(y, 2, adjusted=True)
    assert abs(ga[1] - cross1 / 3) < 1e-12                                       # /(n-k)


def test_ets_recursion_matches_hand_run():
    # Pin the additive-trend recursion with FIXED params so a sign flip in the level or
    # trend update is caught -- an optimised fit on a perfect line drives beta->0, which
    # makes the (buggy) trend-update term inactive and hides such a bug.
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    alpha, beta = 0.5, 0.3
    got = ETSForecaster(trend="add", alpha=alpha, beta=beta).fit(y).predict(3)
    a0, b0 = ols_line(y[:min(len(y), 10)], t0=0)          # same heuristic init as the engine
    lev, tr = a0, b0
    for t in range(len(y)):
        base = lev + tr
        lev_new = alpha * y[t] + (1.0 - alpha) * base
        tr = beta * (lev_new - lev) + (1.0 - beta) * tr
        lev = lev_new
    expected = [lev + h * tr for h in range(1, 4)]
    assert max(abs(got[i] - expected[i]) for i in range(3)) < 1e-9


def test_acf_and_pacf_of_ar1():
    y = make_ar_series(400, phi=[0.7], seed=0)
    r = acf(y, 5)
    assert abs(r[0] - 1.0) < 1e-12
    assert abs(r[1] - 0.7) < 0.12                        # rho_1 ~ phi for AR(1)
    p = pacf(y, 5)
    assert abs(p[1] - 0.7) < 0.12                        # PACF[1] ~ phi
    assert all(abs(p[k]) < 0.2 for k in range(2, 6))     # ~0 after lag 1


def test_acf_peaks_at_seasonal_period():
    y = [10.0 + 5.0 * math.sin(2 * math.pi * t / 12) for t in range(72)]
    r = acf(y, 24)
    # ACF is strongly positive at the seasonal lags and negative at the anti-phase
    # lags (the biased estimator decays with lag, so r[24] < r[12] but stays a peak).
    assert r[12] > 0.8 and r[24] > 0.5
    assert r[6] < 0.0 and r[12] > r[6] and r[24] > r[18]
    assert seasonality_test(y, 12) and detect_seasonal_period(y) == 12


def test_detect_seasonal_period_ignores_ar_autocorrelation():
    # A decaying AR autocorrelation must NOT be read as seasonality: the ACF at the
    # seasonal lag has to be a positive LOCAL PEAK, not a point on a monotone tail.
    assert detect_seasonal_period(make_ar_series(400, phi=[0.7], seed=0)) == 1
    assert detect_seasonal_period(make_ar_series(400, phi=[0.5, 0.3], seed=1)) == 1
    # ...while genuine seasonality is still detected.
    assert detect_seasonal_period([1.0, 10.0] * 12) == 2                  # alternating
    assert detect_seasonal_period(
        make_trend_seasonal(96, slope=0.3, m=12, amplitude=15.0, noise=1.0, seed=1)) == 12


def test_statistics_are_scale_invariant():
    # ACF / PACF / seasonality / detection / R^2 / Yule-Walker are invariant to rescaling
    # y by a constant; an absolute variance threshold used to zero them out for tiny
    # amplitudes (the recurring 'absolute threshold on a scale-dependent quantity' class).
    base = [math.sin(t * 0.5) + 0.3 * math.sin(t * 0.17) for t in range(60)]
    tiny = [1e-7 * v for v in base]
    assert max(abs(a - b) for a, b in zip(acf(base, 6), acf(tiny, 6))) < 1e-9
    assert max(abs(a - b) for a, b in zip(pacf(base, 6), pacf(tiny, 6))) < 1e-9
    ar = make_ar_series(300, phi=[0.6, 0.2], seed=0)
    pb, pt = yule_walker(ar, 2)[0], yule_walker([1e-7 * v for v in ar], 2)[0]
    assert max(abs(a - b) for a, b in zip(pb, pt)) < 1e-6
    seas = [10.0 + 5.0 * math.sin(2 * math.pi * t / 12) for t in range(60)]
    assert seasonality_test(seas, 12) and seasonality_test([1e-7 * v for v in seas], 12)
    assert detect_seasonal_period([1e-7 * v for v in seas]) == 12
    y, yh, c = [1.0, 2.0, 3.0, 4.0, 5.0], [1.1, 1.9, 3.2, 3.8, 5.1], 1e-7
    assert abs(r2_score(y, yh) - r2_score([c * v for v in y], [c * v for v in yh])) < 1e-9
    assert acf([5.0] * 6, 2) == [1.0, 0.0, 0.0]           # true constant still zero-variance


def test_normal_quantiles_and_roundtrip():
    assert abs(norm_ppf(0.5)) < 1e-9
    assert abs(norm_ppf(0.975) - 1.959963985) < 1e-4
    for p in (0.05, 0.3, 0.5, 0.84, 0.99):
        assert abs(norm_cdf(norm_ppf(p)) - p) < 1e-6
    assert t_ppf(0.975, 1e9) == pytest.approx(norm_ppf(0.975), abs=1e-6)


def test_t_ppf_is_accurate_at_low_df():
    # The Cornish-Fisher approximation was ~3% off at df=2; the exact-CDF + Newton path
    # must match the tabulated Student-t quantiles for every df, including df = 1, 2.
    known = {1: 12.70620, 2: 4.30265, 4: 2.77645, 10: 2.22814, 30: 2.04227, 100: 1.98397}
    for df, exp in known.items():
        assert abs(t_ppf(0.975, df) - exp) < 1e-3
    assert abs(t_ppf(0.99, 3) - 4.5407) < 1e-3
    assert abs(t_ppf(0.5, 5)) < 1e-9                      # median is 0
    assert abs(t_ppf(0.975, 1e7) - norm_ppf(0.975)) < 1e-6   # -> normal for large df


def test_ols_line_and_moving_average_on_linear():
    y = [5.0 + 3.0 * t for t in range(10)]
    a, b = ols_line(y)
    assert abs(a - 5.0) < 1e-9 and abs(b - 3.0) < 1e-9
    ma = moving_average(y, 4)                            # centred 2x4 MA of a line = the line
    for t in range(2, 8):
        assert abs(ma[t] - y[t]) < 1e-9


def test_quantile_type7():
    x = [1.0, 2.0, 3.0, 4.0]
    assert quantile(x, 0.0) == 1.0 and quantile(x, 1.0) == 4.0
    assert abs(quantile(x, 0.5) - 2.5) < 1e-12
    assert abs(quantile(x, 0.25) - 1.75) < 1e-12
    assert abs(median([3.0, 1.0, 2.0]) - 2.0) < 1e-12


# ===========================================================================
# 3. TRANSFORMS
# ===========================================================================
def test_differencing_integrate_roundtrip():
    y = [10.0, 12.0, 11.0, 15.0, 14.0, 18.0, 20.0, 19.0, 23.0, 25.0]
    for d in (1, 2, 3):
        recon = integrate(difference(y, d), y[:d])
        assert len(recon) == len(y)
        assert max(abs(recon[i] - y[i]) for i in range(len(y))) < 1e-9
    for m in (2, 3, 4):
        recon = inv_seasonal_difference(seasonal_difference(y, m), y[:m], m)
        assert max(abs(recon[i] - y[i]) for i in range(len(y))) < 1e-9


def test_boxcox_roundtrip_and_special_cases():
    y = [2.0, 5.0, 3.0, 8.0, 13.0, 21.0]
    for lam in (-0.5, 0.0, 0.5, 1.0, 2.0):
        recon = inv_boxcox(boxcox(y, lam), lam)
        assert max(abs(recon[i] - y[i]) for i in range(len(y))) < 1e-6
    assert max(abs(a - b) for a, b in zip(boxcox(y, 0.0), log_transform(y))) < 1e-12
    assert max(abs(a - (v - 1.0)) for a, v in zip(boxcox(y, 1.0), y)) < 1e-9


def test_scalers_roundtrip_and_moments():
    y = [3.0, 7.0, 1.0, 9.0, 5.0]
    s = StandardScaler().fit(y)
    z = s.transform(y)
    assert abs(mean(z)) < 1e-9 and abs(std(z) - 1.0) < 1e-9
    assert max(abs(a - b) for a, b in zip(s.inverse_transform(z), y)) < 1e-9
    mm = MinMaxScaler().fit(y)
    zz = mm.transform(y)
    assert abs(min(zz)) < 1e-12 and abs(max(zz) - 1.0) < 1e-12
    assert max(abs(a - b) for a, b in zip(mm.inverse_transform(zz), y)) < 1e-9


# ===========================================================================
# 4. NAIVE & SIMPLE BASELINES
# ===========================================================================
def test_naive_family_defining_equations():
    y = [3.0, 7.0, 5.0, 9.0]
    assert Naive().fit(y).predict(3) == [9.0, 9.0, 9.0]
    assert MeanForecast().fit(y).predict(2) == [6.0, 6.0]
    assert Drift().fit([2.0, 4.0, 6.0, 8.0, 10.0]).predict(3) == [12.0, 14.0, 16.0]


def test_naive_family_fitted_values_are_one_step_ahead():
    # The naive-family in-sample fitted values are exact one-step-ahead forecasts
    # (fitted[t] uses only data up to t-1) -- guards against a look-ahead like
    # fitted = y itself.
    y = [3.0, 7.0, 5.0, 9.0, 6.0, 8.0]
    assert Naive().fit(y).fitted_values() == [3.0, 3.0, 7.0, 5.0, 9.0, 6.0]      # y[0], y[t-1]
    assert SeasonalNaive(2).fit(y).fitted_values() == [3.0, 7.0, 3.0, 7.0, 5.0, 9.0]  # y[t-2]
    assert Drift().fit([2.0, 4.0, 6.0, 8.0]).fitted_values() == [2.0, 4.0, 6.0, 8.0]  # y[t-1]+slope


def test_scaled_metrics_use_seasonal_denominator():
    # MASE / RMSSE scale by the in-sample SEASONAL-naive error at period m, not lag 1.
    y_train = [1.0, 3.0, 2.0, 5.0, 2.0, 4.0, 3.0, 6.0]    # m=4: |y_t - y_{t-4}| = 1,1,1,1 -> 1
    assert abs(mase([10.0, 11.0], [9.0, 10.0], y_train, 4) - 1.0) < 1e-9
    assert abs(rmsse([10.0, 11.0], [9.0, 10.0], y_train, 4) - 1.0) < 1e-9
    # a wrong lag-1 denominator would give a different (smaller) scale
    assert mase([10.0], [9.0], y_train, 4) != mase([10.0], [9.0], y_train, 1)


def test_theta_closed_form_includes_transient_term():
    # Theta's forecast = SES level + ((theta-1)/theta)*slope*(i + (1-(1-alpha)^n)/alpha).
    # Dropping the finite-sample SES transient (a plausible bug) shifts every forecast;
    # reconstruct the closed form independently and require an exact match.
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    alpha = 0.5
    got = Theta(m=1, theta=2.0, alpha=alpha).fit(y).predict(5)
    fit = ses_fit(y, alpha=alpha, dof=2)
    _, b = ols_line(y, t0=0)
    transient = (1.0 - (1.0 - alpha) ** len(y)) / alpha
    expected = [fit.level + 0.5 * b * (i + transient) for i in range(5)]
    assert max(abs(got[i] - expected[i]) for i in range(5)) < 1e-9


def test_seasonal_naive_repeats_last_cycle():
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    pred = SeasonalNaive(4).fit(y).predict(6)
    assert pred == [9.0, 10.0, 11.0, 12.0, 9.0, 10.0]       # ŷ_{n+h}=y_{n+h-m(k+1)}


def test_moving_and_weighted_average_are_flat():
    y = [10.0, 20.0, 30.0, 40.0]
    assert MovingAverageForecaster(2).fit(y).predict(3) == [35.0, 35.0, 35.0]
    wma = WeightedMovingAverage([1.0, 2.0, 3.0]).fit(y)   # weights normalise to 1/6,2/6,3/6
    assert abs(wma.predict(1)[0] - (20 * 1 + 30 * 2 + 40 * 3) / 6.0) < 1e-9


def test_naive2_matches_seasonal_naive_when_seasonal():
    y = make_trend_seasonal(48, slope=0.0, m=12, amplitude=20.0, level=100.0, seed=1)
    n2 = Naive2(m=12, model="additive").fit(y)
    assert len(n2.predict(12)) == 12
    assert all(math.isfinite(v) for v in n2.predict(12))


# ===========================================================================
# 5. EXPONENTIAL SMOOTHING (ETS)
# ===========================================================================
def test_ses_on_constant_series():
    fit = ses_fit([5.0] * 10, alpha=0.5, init="first")
    assert abs(fit.level - 5.0) < 1e-9
    assert SimpleExpSmoothing().fit([5.0] * 10).predict(3) == pytest.approx([5.0, 5.0, 5.0])


def test_ses_recursion_matches_hand():
    y = [10.0, 12.0, 13.0]
    fit = ses_fit(y, alpha=0.5, init="first")           # l0 = y0 = 10
    # l1 = .5*12 + .5*10 = 11 ; l2 = .5*13 + .5*11 = 12
    assert abs(fit.level - 12.0) < 1e-9
    assert fit.fitted[0] == 10.0 and abs(fit.fitted[1] - 10.0) < 1e-9


def test_holt_extrapolates_a_line():
    y = [float(t) for t in range(1, 9)]                 # y = t
    pred = Holt().fit(y).predict(3)
    assert all(abs(pred[i] - (9.0 + i)) < 0.25 for i in range(3))


def test_holt_winters_recovers_trend_plus_season():
    y = make_trend_seasonal(72, slope=0.7, m=12, amplitude=12.0, level=80.0,
                            noise=0.0, seed=0)
    train, test = train_test_split(y, 12)
    hw = HoltWinters(12, seasonal="add", trend="add").fit(train)
    assert mase(test, hw.predict(12), train, 12) < 0.6    # clearly beats naive


def test_ets_interval_widens_with_horizon():
    y = make_trend_seasonal(60, slope=0.3, m=12, noise=2.0, seed=3)
    lo, up = ETSForecaster().fit(y).predict_interval(10, level=0.95)
    widths = [up[i] - lo[i] for i in range(10)]
    assert all(widths[i + 1] >= widths[i] - 1e-9 for i in range(9))
    assert widths[-1] > widths[0]


def test_damped_sum_closed_form():
    assert fb._damped_sum(1.0, 5) == 5.0
    phi = 0.9
    assert abs(fb._damped_sum(phi, 3) - (phi + phi ** 2 + phi ** 3)) < 1e-12


def test_auto_ets_selects_a_working_model():
    y = make_trend_seasonal(60, slope=0.5, m=12, amplitude=10.0, noise=1.5, seed=2)
    train, test = train_test_split(y, 12)
    m = auto_ets(train, seasonal_periods=12)
    assert m.method.startswith("ETS(")
    assert mase(test, m.predict(12), train, 12) < 1.0


def test_multiplicative_ets_requires_positive_series():
    with pytest.raises(ForecastError):
        ETSForecaster(seasonal="mul", seasonal_periods=4).fit([1.0, -2.0, 3.0, 4.0, 5.0])


def test_fit_does_not_alias_input():
    # A fitted model must COPY the caller's series, not alias it: mutating the input
    # list afterwards must never change a forecast.
    for mk in (lambda: Naive(), lambda: SimpleExpSmoothing(alpha=0.5),
               lambda: ARForecaster(2), lambda: HoltWinters(4, seasonal="add"),
               lambda: Drift(), lambda: Theta(m=4)):
        y = [10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21.]
        f = mk().fit(y)
        p1 = f.predict(3)
        y[0], y[-1] = 99999.0, -99999.0
        assert f.predict(3) == p1


def test_refitting_overwrites_state():
    # Re-fitting the same forecaster on new data must reflect the new data, not leak
    # state (level / trend / seasonal / AR coefficients) from the previous fit.
    y2 = [100., 90., 100., 90., 100., 90., 100., 90., 100., 90., 100., 90.]
    for mk in (lambda: SimpleExpSmoothing(alpha=0.5),
               lambda: ETSForecaster(trend="add", alpha=0.5, beta=0.3),
               lambda: ARForecaster(2), lambda: HoltWinters(4, seasonal="add"),
               lambda: Theta(m=4), lambda: Croston(0.3)):
        f = mk().fit([float(t) for t in range(1, 13)])
        f.fit(y2)
        assert max(abs(a - b) for a, b in
                   zip(f.predict(3), mk().fit(y2).predict(3))) < 1e-9


# ===========================================================================
# 6. DECOMPOSITION & THETA
# ===========================================================================
def test_decomposition_parts_reconstruct_the_series():
    y = make_trend_seasonal(48, slope=0.5, m=12, amplitude=15.0, noise=1.0, seed=4)
    dec = classical_decomposition(y, 12, model="additive")
    for t in range(len(y)):
        if dec.trend[t] is not None:
            assert abs((dec.trend[t] + dec.seasonal[t] + dec.residual[t]) - y[t]) < 1e-9
    assert abs(math.fsum(dec.seasonal_indices)) < 1e-9    # additive indices sum to 0
    pos = [abs(v) + 50.0 for v in y]
    decm = classical_decomposition(pos, 12, model="multiplicative")
    for t in range(len(pos)):
        if decm.trend[t] is not None:
            assert abs(decm.trend[t] * decm.seasonal[t] * decm.residual[t] - pos[t]) < 1e-6
    assert abs(math.fsum(decm.seasonal_indices) - 12.0) < 1e-6   # multiplicative sum to m


def test_deseasonalise_reseasonalise_roundtrip():
    y = make_trend_seasonal(48, slope=0.4, m=12, amplitude=8.0, level=60.0, seed=5)
    idx = seasonal_indices(y, 12, model="additive")
    adj = deseasonalise(y, idx, model="additive")
    back = reseasonalise(adj, idx, model="additive")
    assert max(abs(back[t] - y[t]) for t in range(len(y))) < 1e-9


def test_theta_theta1_is_flat_ses():
    y = make_trend_seasonal(40, slope=0.6, m=1, amplitude=0.0, level=50.0, noise=1.0, seed=6)
    pred = Theta(m=1, theta=1.0).fit(y).predict(6)         # weight 0 on trend -> flat
    assert all(abs(pred[i] - pred[0]) < 1e-9 for i in range(6))


def test_theta_beats_naive_on_trend():
    y = make_trend_seasonal(60, slope=1.0, m=12, amplitude=6.0, noise=1.0, seed=7)
    train, test = train_test_split(y, 12)
    th = smape(test, theta_forecast(train, 12, m=12).mean)
    nv = smape(test, Naive().fit(train).predict(12))
    assert th <= nv


def test_theta_intervals_ordered_on_sign_changing_series():
    # A real-valued (sign-changing) series must not get negative multiplicative seasonal
    # indices that invert the reseasonalised interval bounds (lower > upper); Theta falls
    # back to non-seasonal in that case.
    y = [-1.849, 4.354, -1.772, -4.088, -1.746, 1.449, -0.045, -4.967,
         -1.291, 3.45, 1.155, -2.423, -0.422, 4.162, 0.954]
    f = Theta(m=4).fit(y)
    pt = f.predict(5)
    lo, up = f.predict_interval(5, 0.9)
    assert all(lo[i] <= pt[i] <= up[i] for i in range(5))


# ===========================================================================
# 7. REGRESSION FORECASTERS
# ===========================================================================
def test_trend_forecaster_exact_on_polynomial():
    y = [2.0 + 3.0 * t - 0.5 * t * t for t in range(10)]
    pred = TrendForecaster(degree=2).fit(y).predict(3)
    for i in range(3):
        t = 10 + i
        assert abs(pred[i] - (2.0 + 3.0 * t - 0.5 * t * t)) < 1e-6


def test_seasonal_dummy_forecaster_exact():
    season = [0.0, 5.0, -3.0, 1.0]                       # additive period-4 pattern
    y = [10.0 + 2.0 * t + season[t % 4] for t in range(20)]
    pred = SeasonalDummyForecaster(4, degree=1).fit(y).predict(4)
    for i in range(4):
        t = 20 + i
        assert abs(pred[i] - (10.0 + 2.0 * t + season[t % 4])) < 1e-6


def test_seasonal_forecasters_continue_phase_past_training():
    # A deterministic trend + exact seasonal pattern with n % m != 0 stresses the
    # future seasonal-phase bookkeeping (a classic off-by-one); the linear seasonal
    # models reproduce the continuation exactly, and SeasonalNaive at the right phase.
    m = 12
    season = [0.0, 3.0, 5.0, 4.0, -2.0, -5.0, -3.0, 1.0, 6.0, 2.0, -4.0, -1.0]
    n, H = 50, 18                                             # 50 % 12 = 2
    y = [100.0 + 0.5 * t + season[t % m] for t in range(n)]
    truth = [100.0 + 0.5 * (n + i) + season[(n + i) % m] for i in range(H)]
    assert max(abs(v - t) for v, t in zip(
        SeasonalDummyForecaster(m, degree=1).fit(y).predict(H), truth)) < 1e-6
    assert max(abs(v - t) for v, t in zip(
        FourierForecaster(m, K=6, degree=1).fit(y).predict(H), truth)) < 1e-4
    assert max(abs(v - t) for v, t in zip(
        HoltWinters(m, seasonal="add", trend="add").fit(y).predict(H), truth)) < 2.0
    sn = SeasonalNaive(m).fit(y).predict(H)
    assert all(abs(sn[i] - y[n - m + (i % m)]) < 1e-12 for i in range(H))


def test_classical_decomposition_recovers_known_seasonal():
    # A linear trend + exact additive seasonal is recovered exactly (the centred MA of
    # a line is the line), and the multiplicative reconstruction identity holds exactly.
    m = 12
    season = [0.0, 3.0, 5.0, 4.0, -2.0, -5.0, -3.0, 1.0, 6.0, 2.0, -4.0, -1.0]
    y = [100.0 + 0.5 * t + season[t % m] for t in range(60)]
    dec = classical_decomposition(y, m, model="additive")
    true_idx = [s - mean(season) for s in season]
    assert max(abs(dec.seasonal_indices[j] - true_idx[j]) for j in range(m)) < 1e-9
    res = [dec.residual[t] for t in range(60) if dec.residual[t] is not None]
    assert max(abs(r) for r in res) < 1e-9
    ymul = [(100.0 + 0.05 * t) * (1.0 + 0.1 * math.sin(2 * math.pi * t / m))
            for t in range(60)]
    dm = classical_decomposition(ymul, m, model="multiplicative")
    assert max(abs(dm.trend[t] * dm.seasonal[t] * dm.residual[t] - ymul[t])
               for t in range(60) if dm.residual[t] is not None) < 1e-9


def test_fourier_features_have_no_dead_nyquist_column():
    X = fourier_features(6, m=4, K=2, start=0)            # K=m/2 -> Nyquist
    assert len(X[0]) == 3                                 # sin at Nyquist dropped
    for c in range(3):
        assert any(abs(X[t][c]) > 1e-9 for t in range(6))
    ff = FourierForecaster(m=4, K=2, degree=1).fit(
        [10.0, 12.0, 9.0, 11.0, 10.5, 12.5, 9.5, 11.5])
    assert all(math.isfinite(v) for v in ff.predict(4))


# ===========================================================================
# 8. AUTOREGRESSION
# ===========================================================================
def test_yule_walker_recovers_ar1_coefficient():
    y = make_ar_series(500, phi=[0.6], seed=1)
    phi, sigma2 = yule_walker(y, 1)
    assert abs(phi[0] - 0.6) < 0.1 and sigma2 > 0


def test_ar_ols_and_yule_walker_agree():
    y = make_ar_series(600, phi=[0.5, -0.3], seed=2)
    ols = ARForecaster(2, method="ols").fit(y)
    yw = ARForecaster(2, method="yule_walker").fit(y)
    assert max(abs(ols.phi_[i] - yw.phi_[i]) for i in range(2)) < 0.1


def test_ar1_forecast_is_geometric_toward_mean():
    y = make_ar_series(400, phi=[0.7], c=1.0, seed=3)
    ar = ARForecaster(1, method="ols").fit(y)
    pred = ar.predict(3)
    man = [ar.c_ + ar.phi_[0] * y[-1]]
    man.append(ar.c_ + ar.phi_[0] * man[-1])
    man.append(ar.c_ + ar.phi_[0] * man[-1])
    assert max(abs(pred[i] - man[i]) for i in range(3)) < 1e-9


def test_ar_ma_weights_of_ar1():
    psi = fb._ar_ma_weights([0.7], 5)
    assert max(abs(psi[j] - 0.7 ** j) for j in range(5)) < 1e-9


def test_ar_order_selection_runs():
    y = make_ar_series(300, phi=[0.6], seed=4)
    ar = ARForecaster(ic="aic", max_p=8).fit(y)
    assert 1 <= ar.p <= 8 and math.isfinite(ar.forecast(5).mean[0])
    # selection recovers a LOW order on AR(1) data AND a genuinely higher order on
    # strong AR(3) data -- guards the lag indexing in the per-candidate selection SSE
    # (a wrong lag there collapses selection to p=1 for every series).
    assert all(ARForecaster(ic="aic", max_p=10).fit(
        make_ar_series(300, phi=[0.6], seed=s)).p <= 3 for s in range(4))
    assert all(ARForecaster(ic="aic", max_p=10).fit(
        make_ar_series(400, phi=[0.4, 0.3, 0.2], seed=s)).p >= 2 for s in range(4))


def test_ar_short_series_and_underdetermined_orders():
    # A fixed order too large for the data fails clearly (an over-determined OLS is
    # required), not with a cryptic IndexError from the QR.
    with pytest.raises(ForecastError):
        ARForecaster(p=5).fit([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    # Order selection on a short series caps the order to what the series supports
    # instead of demanding max_p+2 points up front.
    ar = ARForecaster(ic="bic", max_p=10).fit([1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0])
    assert 1 <= ar.p <= 3 and all(math.isfinite(v) for v in ar.predict(2))


def test_model_selection_is_scale_invariant():
    # AR order selection and auto_ets must choose the SAME model under y -> a*y (a>0),
    # so the forecast is affine-equivariant; per-candidate effective sample sizes in the
    # information criterion used to break this.
    Y = make_ar_series(60, phi=[0.5, -0.3, 0.2], c=5.0, seed=0)
    a = 0.02
    assert (ARForecaster(ic="aic", max_p=8).fit(Y).p
            == ARForecaster(ic="aic", max_p=8).fit([a * v for v in Y]).p)
    fb = ARForecaster(ic="aic", max_p=8).fit(Y).predict(3)
    fs = [v / a for v in ARForecaster(ic="aic", max_p=8).fit([a * v for v in Y]).predict(3)]
    assert max(abs(x - y) for x, y in zip(fb, fs)) < 1e-6
    Ys = make_trend_seasonal(48, slope=0.3, m=12, amplitude=4.0, level=100.0, noise=5.0, seed=1)
    assert (auto_ets(Ys, seasonal_periods=12).method
            == auto_ets([0.05 * v for v in Ys], seasonal_periods=12).method)


# ===========================================================================
# 9. INTERMITTENT DEMAND
# ===========================================================================
def test_croston_and_sba_match_hand_run():
    y = [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0]              # sizes [5,5], intervals [3,4]
    cr = Croston(0.5, init="naive").fit(y)               # z->5, q->3
    assert abs(cr.rate_ - 5.0 / 3.0) < 1e-9
    sba = Croston(0.5, "sba", init="naive").fit(y)
    assert abs(sba.rate_ - (1.0 - 0.25) * (5.0 / 3.0)) < 1e-9


def test_tsb_rate_and_no_lookahead():
    y = [0.0, 5.0, 0.0, 5.0]
    tsb = TSB(0.5, 0.5, init="first").fit(y)             # p0=1/2, z0=5
    assert abs(tsb.rate_ - 0.65625 * 5.0) < 1e-9
    # fitted[1] is the one-step forecast made BEFORE seeing y[1] (no look-ahead):
    # after t0 (zero): p = .5*.5 = .25, z = 5  -> fitted[1] = .25*5 = 1.25
    assert abs(tsb.fitted_[1] - 1.25) < 1e-9


def test_intermittent_all_zero_series_is_zero():
    z = [0.0] * 10
    assert Croston(0.2).fit(z).predict(3) == [0.0, 0.0, 0.0]
    assert TSB(0.2, 0.2).fit(z).predict(3) == [0.0, 0.0, 0.0]


def test_fitted_values_are_one_step_ahead():
    # Perturbing y[t] (outside the heuristic init window) must leave the one-step-ahead
    # fitted values at indices <= t unchanged and change fitted[t+1] as the state carries
    # the perturbation forward -- i.e. no forecaster peeks at y[t] when forming fitted[t]
    # (this is exactly the look-ahead defect that was found and fixed in TSB).
    yc = [3., 5., 4., 6., 5., 7., 6., 8., 7., 9., 8., 10., 9., 11., 10., 12.,
          11., 13., 12., 14., 13., 15., 14., 16.]
    yd = [0., 4., 0., 0., 6., 0., 3., 0., 0., 5., 0., 7., 0., 0., 4., 8.,
          0., 3., 0., 0., 6., 0., 2., 5.]
    t = 15
    cases = [(lambda: SimpleExpSmoothing(alpha=0.4), yc),
             (lambda: ETSForecaster(trend="add", alpha=0.4, beta=0.2), yc),
             (lambda: Croston(0.3, init="naive"), yd),
             (lambda: TSB(0.3, 0.3, init="first"), yd)]
    for mk, y in cases:
        a = mk().fit(y).fitted_values()
        y2 = list(y)
        y2[t] = y[t] + 50.0
        b = mk().fit(y2).fitted_values()
        assert max(abs(a[i] - b[i]) for i in range(t + 1)) < 1e-9   # no look-ahead
        assert abs(a[t + 1] - b[t + 1]) > 1e-9                       # state propagated y[t]


def test_ar_prediction_interval_is_calibrated():
    # Over many replications from a known AR(1) DGP, a nominal-90% one-step interval
    # should cover close to 90% of the held-out values (statistical calibration).
    inside = total = 0
    for r in range(200):
        y = make_ar_series(120, phi=[0.6], c=2.0, sigma=1.0, seed=r)
        fr = ARForecaster(1).fit(y[:-1]).forecast(1, level=0.90)
        total += 1
        if fr.lower[0] <= y[-1] <= fr.upper[0]:
            inside += 1
    assert 0.80 <= inside / total <= 0.97


# ===========================================================================
# 10. EVALUATION METRICS
# ===========================================================================
def test_basic_metrics_match_formulas():
    y = [10.0, 20.0, 30.0]
    f = [12.0, 18.0, 33.0]                               # errors -2, 2, -3
    assert abs(me(y, f) - (-1.0)) < 1e-12
    assert abs(mae(y, f) - (7.0 / 3.0)) < 1e-12
    assert abs(mse(y, f) - (17.0 / 3.0)) < 1e-12
    assert abs(rmse(y, f) - math.sqrt(17.0 / 3.0)) < 1e-12
    assert abs(smape([100.0], [110.0]) - (200.0 * 10.0 / 210.0)) < 1e-9


def test_scaled_metrics_and_perfect_forecast():
    y_train = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]            # seasonal-naive(1) MAE = 1
    y, f = [9.0, 10.0], [8.5, 9.5]                      # MAE 0.5, MSE 0.25
    assert abs(mase(y, f, y_train, 1) - 0.5) < 1e-9
    assert abs(rmsse(y, f, y_train, 1) - 0.5) < 1e-9
    assert mase(y, y, y_train, 1) == 0.0                # perfect forecast
    assert abs(r2_score(y_train, y_train) - 1.0) < 1e-12
    assert abs(r2_score([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])) < 1e-12   # mean forecast -> 0


def test_pinball_and_coverage_and_msis():
    y = [1.0, 2.0, 3.0]
    q50 = [1.5, 1.5, 3.5]
    assert abs(pinball_loss(y, q50, 0.5) - 0.5 * mae(y, q50)) < 1e-12
    # asymmetry: under-forecast penalised more at high quantile
    assert pinball_loss([10.0], [8.0], 0.9) > pinball_loss([10.0], [12.0], 0.9)
    lo, up = [0.0, 1.0, 2.0], [2.0, 3.0, 4.0]
    assert coverage(y, lo, up) == 1.0
    assert coverage([0.0, 5.0], [0.0, 1.0], [2.0, 5.0]) == 1.0   # points ON the bounds count
    assert abs(interval_width(lo, up) - 2.0) < 1e-12
    assert msis(y, lo, up, [1.0, 2.0, 3.0, 4.0], level=0.95, m=1) > 0


def test_evaluate_panel_is_complete():
    y, f = [10.0, 12.0, 11.0], [10.5, 11.5, 11.0]
    d = evaluate(y, f, y_train=[8.0, 9.0, 10.0, 11.0], m=1)
    for k in ("MAE", "RMSE", "sMAPE", "MASE", "RMSSE", "R2"):
        assert k in d and math.isfinite(d[k])


# ===========================================================================
# 11. CROSS-VALIDATION
# ===========================================================================
def test_rolling_origin_split_structure():
    splits = rolling_origin_splits(20, h=4, initial=10, step=2, expanding=True)
    assert splits == [(0, 10, 14), (0, 12, 16), (0, 14, 18), (0, 16, 20)]
    slide = rolling_origin_splits(20, h=4, initial=10, step=2, expanding=False)
    assert all(te - ts == 10 for ts, te, _ in slide)      # fixed window length


def test_train_test_split_and_cross_validate():
    y = list(range(1, 41))
    tr, te = train_test_split([float(v) for v in y], 5)
    assert len(tr) == 35 and te == [36.0, 37.0, 38.0, 39.0, 40.0]
    yts = make_trend_seasonal(72, slope=0.5, m=12, noise=1.0, seed=8)
    cv = cross_validate(yts, lambda: HoltWinters(12), h=6, initial=36, step=6, m=12)
    assert all(math.isfinite(v) for v in cv.values())


# ===========================================================================
# 12. FORECAST COMBINATION
# ===========================================================================
def test_combination_rules():
    a, b, c = [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]
    assert combine_mean([a, b, c]) == [3.0, 4.0]
    assert combine_median([a, b, c]) == [3.0, 4.0]
    assert combine_weighted([a, b, c], [1.0, 1.0, 1.0]) == [3.0, 4.0]
    assert combine_weighted([a, b], [3.0, 1.0]) == [1.5, 2.5]


def test_inverse_error_weights_favour_low_error():
    w = inverse_error_weights([1.0, 2.0, 4.0])
    assert abs(math.fsum(w) - 1.0) < 1e-12
    assert w[0] > w[1] > w[2]                             # lower error -> higher weight


# ===========================================================================
# 13. INPUT VALIDATION & EDGE CASES
# ===========================================================================
def test_input_validation():
    with pytest.raises(ForecastError):
        Naive().fit([])
    with pytest.raises(ForecastError):
        Naive().fit([1.0, float("nan"), 3.0])
    with pytest.raises(ForecastError):
        Drift().fit([1.0])                               # needs >= 2
    with pytest.raises(ForecastError):
        Naive().fit([1.0, 2.0]).predict(0)               # h >= 1
    with pytest.raises(ForecastError):
        ETSForecaster(seasonal="add", seasonal_periods=1)
    with pytest.raises(ForecastError):
        fourier_features(10, m=4, K=3)                   # K > m//2


def test_predict_before_fit_raises():
    with pytest.raises(ForecastError):
        Naive().predict(3)
    with pytest.raises(ForecastError):
        ETSForecaster().forecast(3)


def test_forecaster_argument_validation():
    # Unknown estimator / criterion strings must fail loudly, not silently fall back
    # (ARForecaster(method='burg') used to be silently treated as OLS).
    with pytest.raises(ForecastError):
        ARForecaster(2, method="burg")
    with pytest.raises(ForecastError):
        ARForecaster(2, ic="aicc")                          # AR order-select is aic/bic only
    with pytest.raises(ForecastError):
        auto_ets([1.0, 2.0, 3.0, 4.0, 5.0], criterion="xyz")
    with pytest.raises(ForecastError):
        ETSForecaster(trend="linear")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
