"""
Stress / correctness suite for smoothing_template.py
====================================================
Rigorous, mostly-oracle-based tests for the time-series smoothing & forecasting
module: known-value oracles, mathematical properties (difference/integrate and
Box-Cox round-trips, exact decomposition reconstruction, "a smoother of a
constant is that constant", Savitzky-Golay reproduces a polynomial of its
degree, LOWESS reproduces a line), parameter recovery on synthetic data (AR
coefficients, seasonal indices), cross-method consistency, and the full set of
failure modes (clear ValueErrors).

Heavy random batteries and long-horizon fits are marked ``@slow`` and skipped by
default, so the everyday run is a sub-second smoke test.  Set ``RUNSLOW`` to run
them too (self-contained here -- no conftest.py needed).

Run:  pytest smoothing/_stress_test.py -q             # fast smoke
      RUNSLOW=1 pytest smoothing/_stress_test.py -q   # full suite
"""

import json
import math
import os
import random
import sys

import pytest

from gridpulse.analysis.algorithms.orchid import smoothing as sm

_RUNSLOW = os.environ.get("RUNSLOW", "").strip().lower() not in ("", "0", "false", "no", "off")
slow = pytest.mark.skipif(not _RUNSLOW, reason="slow test -- set RUNSLOW=1 to run it")


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def _series_close(a, b, tol=1e-6):
    return len(a) == len(b) and all(abs(a[i] - b[i]) <= tol for i in range(len(a)))


# =========================================================================
# SERIES UTILITIES
# =========================================================================
class TestDifference:
    """Pins difference(): known values, length arithmetic (n - lag*order),
    order=0 copy semantics, and every failure mode / message."""

    def test_first_difference_known_values(self):
        assert _series_close(sm.difference([1.0, 3.0, 6.0, 10.0]), [2.0, 3.0, 4.0])

    def test_default_is_lag1_order1(self):
        y = [5.0, 2.0, 9.0, 1.0, 8.0]
        assert sm.difference(y) == sm.difference(y, 1, 1)

    def test_lag2_known_values(self):
        # y_t - y_{t-2}
        assert _series_close(sm.difference([1.0, 2.0, 4.0, 8.0, 16.0], lag=2, order=1),
                             [3.0, 6.0, 12.0])

    def test_second_order_known_values(self):
        # second difference of [1,4,9,16,25] (squares) is constant 2
        d2 = sm.difference([1.0, 4.0, 9.0, 16.0, 25.0], lag=1, order=2)
        assert _series_close(d2, [2.0, 2.0, 2.0])

    def test_second_order_equals_composition(self):
        y = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        once = sm.difference(y, lag=1, order=1)
        twice = sm.difference(once, lag=1, order=1)
        assert _series_close(sm.difference(y, lag=1, order=2), twice)

    def test_length_shrinks_by_lag_times_order(self):
        y = list(range(30))
        for lag in (1, 2, 3, 5):
            for order in (0, 1, 2, 3):
                if lag * order < len(y):
                    d = sm.difference(y, lag=lag, order=order)
                    assert len(d) == len(y) - lag * order

    def test_order_zero_returns_copy(self):
        y = [1.0, 2.0, 3.0]
        d = sm.difference(y, lag=1, order=0)
        assert d == y
        assert d is not y
        d[0] = 99.0
        assert y[0] == 1.0

    def test_order_zero_ignores_lag_bounds(self):
        # order=0 runs no differencing loop, so an oversized lag is harmless.
        assert sm.difference([1.0, 2.0], lag=99, order=0) == [1.0, 2.0]

    def test_returns_floats_from_ints(self):
        d = sm.difference([1, 3, 6])
        assert all(isinstance(v, float) for v in d)

    def test_of_constant_is_all_zero(self):
        assert _series_close(sm.difference([7.0] * 6), [0.0] * 5)

    def test_lag_zero_raises(self):
        with pytest.raises(ValueError):
            sm.difference([1.0, 2.0, 3.0], lag=0)

    def test_negative_lag_raises(self):
        with pytest.raises(ValueError):
            sm.difference([1.0, 2.0, 3.0], lag=-1)

    def test_negative_order_raises(self):
        with pytest.raises(ValueError):
            sm.difference([1.0, 2.0, 3.0], order=-1)

    def test_too_short_raises(self):
        # need len > lag to take one difference
        with pytest.raises(ValueError):
            sm.difference([1.0, 2.0], lag=2, order=1)

    def test_too_short_on_second_pass_raises(self):
        # len 3, lag 1: after first diff len is 2, after second len 1 -- but the
        # third pass has len 1 <= lag -> raises
        with pytest.raises(ValueError):
            sm.difference([1.0, 2.0, 3.0], lag=1, order=3)

    def test_nan_raises(self):
        with pytest.raises(ValueError):
            sm.difference([1.0, float("nan"), 3.0])


class TestIntegrate:
    """Pins integrate() as the exact inverse of a single lagged difference and
    its guard on insufficient initial values."""

    def test_known_reconstruction_lag1(self):
        out = sm.integrate([2.0, 3.0, 4.0], [1.0], lag=1)
        assert _series_close(out, [1.0, 3.0, 6.0, 10.0])

    def test_known_reconstruction_lag2(self):
        out = sm.integrate([3.0, 6.0, 12.0], [1.0, 2.0], lag=2)
        assert _series_close(out, [1.0, 2.0, 4.0, 8.0, 16.0])

    def test_length_grows_by_number_of_diffs(self):
        out = sm.integrate([1.0, 1.0, 1.0, 1.0], [0.0, 0.0], lag=2)
        assert len(out) == 2 + 4

    def test_empty_diffs_returns_initial_copy(self):
        out = sm.integrate([], [5.0, 6.0], lag=2)
        assert _series_close(out, [5.0, 6.0])

    def test_returns_floats(self):
        out = sm.integrate([1, 1], [0], lag=1)
        assert all(isinstance(v, float) for v in out)

    def test_too_few_initial_raises(self):
        with pytest.raises(ValueError):
            sm.integrate([1.0, 2.0], [0.0], lag=2)

    def test_zero_initial_with_lag1_raises(self):
        with pytest.raises(ValueError):
            sm.integrate([1.0], [], lag=1)


class TestDifferenceIntegrateRoundTrip:
    """difference and integrate are exact inverses across lags and, by
    composition, across differencing orders."""

    def test_round_trip_order1(self):
        y = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0]
        for lag in (1, 2, 3):
            d = sm.difference(y, lag=lag, order=1)
            rec = sm.integrate(d, y[:lag], lag=lag)
            assert _series_close(rec, y)

    def test_round_trip_order2_by_composition(self):
        y = [2.0, 5.0, 1.0, 8.0, 3.0, 7.0, 4.0, 9.0, 6.0, 0.0, 2.0]
        for lag in (1, 2):
            d1 = sm.difference(y, lag=lag, order=1)
            d2 = sm.difference(d1, lag=lag, order=1)
            rec1 = sm.integrate(d2, d1[:lag], lag=lag)
            assert _series_close(rec1, d1)
            rec0 = sm.integrate(rec1, y[:lag], lag=lag)
            assert _series_close(rec0, y)

    @slow
    def test_random_round_trip_battery(self):
        rng = random.Random(20260801)
        for _ in range(400):
            n = rng.randint(5, 40)
            y = [rng.uniform(-100.0, 100.0) for _ in range(n)]
            lag = rng.randint(1, max(1, n - 1))
            if n <= lag:
                continue
            d = sm.difference(y, lag=lag, order=1)
            rec = sm.integrate(d, y[:lag], lag=lag)
            assert _series_close(rec, y, tol=1e-6)


class TestSeasonalDifference:
    """seasonal_difference(y, m) is exactly difference(y, lag=m, order=1)."""

    def test_equivalence_to_difference(self):
        y = [10.0, 12.0, 14.0, 11.0, 13.0, 15.0, 12.0, 14.0, 16.0]
        for m in (2, 3, 4):
            assert _series_close(sm.seasonal_difference(y, m),
                                 sm.difference(y, lag=m, order=1))

    def test_removes_pure_seasonal_pattern(self):
        # a strictly period-3 repeating series seasonally differences to zeros
        y = [1.0, 2.0, 3.0] * 4
        assert _series_close(sm.seasonal_difference(y, 3), [0.0] * (len(y) - 3))

    def test_round_trip(self):
        y = [4.0, 9.0, 2.0, 7.0, 5.0, 1.0, 8.0, 3.0]
        m = 3
        d = sm.seasonal_difference(y, m)
        assert _series_close(sm.integrate(d, y[:m], lag=m), y)

    def test_period_below_one_raises(self):
        with pytest.raises(ValueError):
            sm.seasonal_difference([1.0, 2.0, 3.0], 0)

    def test_period_too_large_raises(self):
        with pytest.raises(ValueError):
            sm.seasonal_difference([1.0, 2.0, 3.0], 5)


class TestBoxCox:
    """box_cox / inv_box_cox: known values, exact round-trips across lambdas
    (incl. log and negative), the positivity guard, and the inverse's
    out-of-domain clamp to zero."""

    def test_log_transform_known_values(self):
        z = sm.box_cox([1.0, math.e, math.e ** 2], 0.0)
        assert _series_close(z, [0.0, 1.0, 2.0])

    def test_lambda_one_shift(self):
        # (v^1 - 1)/1 = v - 1
        assert _series_close(sm.box_cox([1.0, 2.0, 3.0], 1.0), [0.0, 1.0, 2.0])

    def test_lambda_two_known_values(self):
        # (v^2 - 1)/2
        assert _series_close(sm.box_cox([1.0, 2.0, 3.0], 2.0), [0.0, 1.5, 4.0])

    def test_round_trip_many_lambdas(self):
        y = [0.5, 1.0, 2.0, 3.5, 7.0, 12.25, 100.0]
        for lam in (-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
            z = sm.box_cox(y, lam)
            back = sm.inv_box_cox(z, lam)
            assert _series_close(back, y, tol=1e-6)

    def test_inv_of_log_is_exp(self):
        assert _series_close(sm.inv_box_cox([0.0, 1.0, 2.0], 0.0),
                             [1.0, math.e, math.e ** 2])

    def test_inv_out_of_domain_clamps_to_zero(self):
        # lam=1: lam*z + 1 = z + 1; for z = -2 that is -1 <= 0 -> 0.0
        assert sm.inv_box_cox([-2.0], 1.0)[0] == 0.0

    def test_positive_data_required_zero(self):
        with pytest.raises(ValueError):
            sm.box_cox([1.0, 0.0, 2.0], 0.5)

    def test_positive_data_required_negative(self):
        with pytest.raises(ValueError):
            sm.box_cox([1.0, -3.0, 2.0], 1.0)

    def test_empty_returns_empty(self):
        assert sm.box_cox([], 0.5) == []

    def test_nan_raises(self):
        with pytest.raises(ValueError):
            sm.box_cox([1.0, float("nan")], 0.5)


class TestTrainTestSplit:
    """train_test_split: exact partition, float coercion, and range guard on
    test_size in (0, n)."""

    def test_known_split(self):
        tr, te = sm.train_test_split([1.0, 2.0, 3.0, 4.0, 5.0], 2)
        assert _series_close(tr, [1.0, 2.0, 3.0])
        assert _series_close(te, [4.0, 5.0])

    def test_concatenation_reconstructs(self):
        y = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0]
        for k in range(1, len(y)):
            tr, te = sm.train_test_split(y, k)
            assert list(tr) + list(te) == y
            assert len(te) == k

    def test_returns_floats(self):
        tr, te = sm.train_test_split([1, 2, 3], 1)
        assert all(isinstance(v, float) for v in tr + te)

    def test_test_size_zero_raises(self):
        with pytest.raises(ValueError):
            sm.train_test_split([1.0, 2.0, 3.0], 0)

    def test_test_size_equals_len_raises(self):
        with pytest.raises(ValueError):
            sm.train_test_split([1.0, 2.0, 3.0], 3)

    def test_test_size_over_len_raises(self):
        with pytest.raises(ValueError):
            sm.train_test_split([1.0, 2.0, 3.0], 4)

    def test_negative_test_size_raises(self):
        with pytest.raises(ValueError):
            sm.train_test_split([1.0, 2.0, 3.0], -1)


class TestRollingWindows:
    """rolling_windows: count == n - window + 1, each window has length
    ``window`` and the correct contents; bounds on window in [1, n]."""

    def test_known_windows(self):
        w = sm.rolling_windows([1.0, 2.0, 3.0, 4.0], 2)
        assert w == [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]

    def test_count_and_length(self):
        y = list(range(20))
        for window in range(1, len(y) + 1):
            w = sm.rolling_windows(y, window)
            assert len(w) == len(y) - window + 1
            assert all(len(win) == window for win in w)

    def test_window_one_is_singletons(self):
        w = sm.rolling_windows([5.0, 6.0, 7.0], 1)
        assert w == [[5.0], [6.0], [7.0]]

    def test_window_equals_len_is_whole_series(self):
        y = [3.0, 1.0, 4.0, 1.0]
        w = sm.rolling_windows(y, 4)
        assert len(w) == 1
        assert _series_close(w[0], y)

    def test_returns_floats(self):
        w = sm.rolling_windows([1, 2, 3], 2)
        assert all(isinstance(v, float) for win in w for v in win)

    def test_windows_are_contiguous_slices(self):
        y = [10.0, 20.0, 30.0, 40.0, 50.0]
        w = sm.rolling_windows(y, 3)
        for i, win in enumerate(w):
            assert _series_close(win, y[i:i + 3])

    def test_window_zero_raises(self):
        with pytest.raises(ValueError):
            sm.rolling_windows([1.0, 2.0, 3.0], 0)

    def test_window_over_len_raises(self):
        with pytest.raises(ValueError):
            sm.rolling_windows([1.0, 2.0, 3.0], 4)

    def test_negative_window_raises(self):
        with pytest.raises(ValueError):
            sm.rolling_windows([1.0, 2.0, 3.0], -1)


# =========================================================================
# MOVING-AVERAGE & KERNEL SMOOTHERS
# =========================================================================
class TestConstantPreservation:
    """Every smoother applied to a constant series returns that constant.
    Pins the fundamental 'a smoother of a flat line is that flat line' law on
    the whole family (centered_moving_average only on its non-None interior)."""

    C = 7.25
    N = 9

    def _const(self):
        return [self.C] * self.N

    def test_simple_moving_average_trailing(self):
        out = sm.simple_moving_average(self._const(), 4)
        assert _series_close(out, self._const())

    def test_simple_moving_average_centered(self):
        out = sm.simple_moving_average(self._const(), 5, center=True)
        assert _series_close(out, self._const())

    def test_weighted_moving_average(self):
        out = sm.weighted_moving_average(self._const(), [1.0, 3.0, 2.0])
        assert _series_close(out, self._const())

    def test_cumulative_moving_average(self):
        out = sm.cumulative_moving_average(self._const())
        assert _series_close(out, self._const())

    def test_exponential_moving_average(self):
        out = sm.exponential_moving_average(self._const(), alpha=0.4)
        assert _series_close(out, self._const())

    def test_centered_moving_average_interior(self):
        out = sm.centered_moving_average(self._const(), 4)
        interior = [v for v in out if v is not None]
        assert len(interior) == self.N - 4
        assert all(_close(v, self.C) for v in interior)

    def test_median_filter(self):
        out = sm.median_filter(self._const(), 3)
        assert _series_close(out, self._const())

    def test_gaussian_filter(self):
        out = sm.gaussian_filter(self._const(), 1.5)
        assert _series_close(out, self._const())

    def test_savitzky_golay(self):
        out = sm.savitzky_golay(self._const(), 5, 2)
        assert _series_close(out, self._const())

    def test_lowess(self):
        out = sm.lowess(self._const(), frac=0.6)
        assert _series_close(out, self._const())

    def test_hodrick_prescott(self):
        out = sm.hodrick_prescott(self._const(), lam=1600.0)
        assert _series_close(out, self._const())


class TestSimpleMovingAverage:
    """Known-value oracles + length/centering invariants for simple_moving_average."""

    def test_trailing_known_values(self):
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        out = sm.simple_moving_average(y, 3)
        assert _series_close(out, [1.0, 1.5, 2.0, 3.0, 4.0])

    def test_centered_known_values(self):
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        out = sm.simple_moving_average(y, 3, center=True)
        assert _series_close(out, [1.5, 2.0, 3.0, 4.0, 4.5])

    def test_output_length_matches_input(self):
        y = [float(i) for i in range(13)]
        assert len(sm.simple_moving_average(y, 5)) == len(y)
        assert len(sm.simple_moving_average(y, 5, center=True)) == len(y)

    def test_window_one_is_identity(self):
        y = [3.0, 1.0, 4.0, 1.0, 5.0]
        assert _series_close(sm.simple_moving_average(y, 1), y)

    def test_window_full_last_equals_grand_mean(self):
        y = [1.0, 2.0, 3.0, 4.0]
        out = sm.simple_moving_average(y, 4)
        assert _close(out[-1], 2.5)

    def test_window_out_of_range_raises(self):
        with pytest.raises(ValueError):
            sm.simple_moving_average([1.0, 2.0, 3.0], 0)
        with pytest.raises(ValueError):
            sm.simple_moving_average([1.0, 2.0, 3.0], 4)


class TestWeightedMovingAverage:
    """Normalisation, known weighted value, and equivalence to trailing SMA
    under equal weights, for weighted_moving_average."""

    def test_known_weighted_values(self):
        y = [1.0, 2.0, 3.0, 4.0]
        out = sm.weighted_moving_average(y, [1.0, 2.0])
        assert _series_close(out, [1.0, 5.0 / 3.0, 8.0 / 3.0, 11.0 / 3.0])

    def test_weights_are_normalised(self):
        # scaling the weights must not change the result
        y = [2.0, 5.0, 1.0, 8.0, 3.0]
        a = sm.weighted_moving_average(y, [1.0, 2.0, 3.0])
        b = sm.weighted_moving_average(y, [10.0, 20.0, 30.0])
        assert _series_close(a, b)

    def test_equal_weights_equal_trailing_sma(self):
        y = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        w = sm.weighted_moving_average(y, [1.0, 1.0, 1.0])
        s = sm.simple_moving_average(y, 3)
        assert _series_close(w, s)

    def test_recency_weighting_tracks_last_value(self):
        # newest value gets the largest weight -> output close to last obs
        y = [0.0, 0.0, 0.0, 10.0]
        out = sm.weighted_moving_average(y, [0.0, 0.0, 1.0])
        assert _close(out[-1], 10.0)

    def test_too_many_weights_raises(self):
        with pytest.raises(ValueError):
            sm.weighted_moving_average([1.0, 2.0], [1.0, 1.0, 1.0])

    def test_zero_weights_raise(self):
        with pytest.raises(ValueError):
            sm.weighted_moving_average([1.0, 2.0, 3.0], [1.0, -1.0])


class TestCumulativeMovingAverage:
    """Expanding-window mean: known values and equivalence to running mean."""

    def test_known_values(self):
        y = [2.0, 4.0, 6.0, 8.0]
        assert _series_close(sm.cumulative_moving_average(y), [2.0, 3.0, 4.0, 5.0])

    def test_last_equals_grand_mean(self):
        y = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0]
        out = sm.cumulative_moving_average(y)
        assert _close(out[-1], sum(y) / len(y))

    def test_first_equals_first_obs(self):
        y = [42.0, 1.0, 2.0]
        assert _close(sm.cumulative_moving_average(y)[0], 42.0)


class TestExponentialMovingAverage:
    """Hand-computed recursion, span<->alpha map, and parameter validation."""

    def test_known_recursion(self):
        y = [1.0, 2.0, 3.0]
        out = sm.exponential_moving_average(y, alpha=0.5)
        assert _series_close(out, [1.0, 1.5, 2.25])

    def test_first_value_is_seed(self):
        y = [5.0, 9.0, 2.0, 7.0]
        assert _close(sm.exponential_moving_average(y, alpha=0.3)[0], 5.0)

    def test_span_maps_to_alpha(self):
        y = [1.0, 3.0, 2.0, 8.0, 5.0, 4.0]
        via_span = sm.exponential_moving_average(y, span=3)
        via_alpha = sm.exponential_moving_average(y, alpha=0.5)
        assert _series_close(via_span, via_alpha)

    def test_alpha_one_is_identity(self):
        y = [3.0, 1.0, 4.0, 1.0]
        assert _series_close(sm.exponential_moving_average(y, alpha=1.0), y)

    def test_alpha_out_of_range_raises(self):
        with pytest.raises(ValueError):
            sm.exponential_moving_average([1.0, 2.0], alpha=0.0)
        with pytest.raises(ValueError):
            sm.exponential_moving_average([1.0, 2.0], alpha=1.5)

    def test_missing_alpha_and_span_raises(self):
        with pytest.raises(ValueError):
            sm.exponential_moving_average([1.0, 2.0, 3.0])

    def test_bad_span_raises(self):
        with pytest.raises(ValueError):
            sm.exponential_moving_average([1.0, 2.0, 3.0], span=0)


class TestCenteredMovingAverage:
    """The classical-decomposition trend estimator: even m uses 2xm and is
    symmetric, odd m is a plain centered mean, endpoints are None."""

    def test_even_order_known_values(self):
        y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        out = sm.centered_moving_average(y, 4)
        assert out[0] is None and out[1] is None
        assert out[4] is None and out[5] is None
        assert _close(out[2], 3.0) and _close(out[3], 4.0)

    def test_odd_order_known_values(self):
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        out = sm.centered_moving_average(y, 3)
        assert out[0] is None and out[4] is None
        assert _close(out[1], 2.0) and _close(out[2], 3.0) and _close(out[3], 4.0)

    def test_even_non_none_count(self):
        y = [float(i) for i in range(12)]
        out = sm.centered_moving_average(y, 4)
        assert sum(1 for v in out if v is not None) == len(y) - 4

    def test_even_ma_is_symmetric_on_linear_trend(self):
        # a 2xm centered average of a straight line reproduces the line exactly
        y = [2.0 + 1.5 * i for i in range(10)]
        out = sm.centered_moving_average(y, 4)
        for t, v in enumerate(out):
            if v is not None:
                assert _close(v, 2.0 + 1.5 * t)

    def test_output_length_matches_input(self):
        y = [float(i) for i in range(8)]
        assert len(sm.centered_moving_average(y, 3)) == len(y)

    def test_order_out_of_range_raises(self):
        with pytest.raises(ValueError):
            sm.centered_moving_average([1.0, 2.0, 3.0], 0)
        with pytest.raises(ValueError):
            sm.centered_moving_average([1.0, 2.0, 3.0], 4)


class TestMedianFilter:
    """Robustness to impulse noise: a lone spike is removed exactly where a
    mean would smear it across the window."""

    def test_spike_removed_exactly(self):
        y = [1.0, 2.0, 3.0, 100.0, 5.0, 6.0, 7.0]
        out = sm.median_filter(y, 3)
        # the spike at index 3 is replaced by a neighbouring in-range value
        assert _close(out[3], 5.0)
        assert 4.0 < out[3] < 8.0

    def test_more_robust_than_mean_on_spike(self):
        y = [1.0, 2.0, 3.0, 100.0, 5.0, 6.0, 7.0]
        med = sm.median_filter(y, 3)
        mean = sm.simple_moving_average(y, 3, center=True)
        assert abs(med[3] - 4.0) < abs(mean[3] - 4.0)

    def test_window_one_is_identity(self):
        y = [3.0, 1.0, 4.0, 1.0, 5.0]
        assert _series_close(sm.median_filter(y, 1), y)

    def test_monotone_series_preserved(self):
        y = [float(i) for i in range(10)]
        out = sm.median_filter(y, 3)
        # interior points of a strictly increasing series are unchanged
        for t in range(1, len(y) - 1):
            assert _close(out[t], y[t])

    def test_output_length_matches_input(self):
        y = [float(i) for i in range(7)]
        assert len(sm.median_filter(y, 3)) == len(y)

    def test_window_out_of_range_raises(self):
        with pytest.raises(ValueError):
            sm.median_filter([1.0, 2.0, 3.0], 0)
        with pytest.raises(ValueError):
            sm.median_filter([1.0, 2.0, 3.0], 4)


class TestGaussianFilter:
    """Constant preservation, reversal symmetry, variance reduction, validation."""

    def test_reversal_symmetry(self):
        y = [1.0, 4.0, 9.0, 2.0, 7.0, 3.0]
        out = sm.gaussian_filter(y, 1.0)
        rev = sm.gaussian_filter(list(reversed(y)), 1.0)
        assert _series_close(out, list(reversed(rev)))

    def test_symmetric_input_gives_symmetric_output(self):
        y = [1.0, 2.0, 3.0, 2.0, 1.0]
        out = sm.gaussian_filter(y, 1.2)
        assert _series_close(out, list(reversed(out)))

    def test_reduces_noise_variance(self):
        rng = random.Random(20240501)
        y = [rng.gauss(0.0, 1.0) for _ in range(400)]
        out = sm.gaussian_filter(y, 2.0)

        def var(z):
            mu = sum(z) / len(z)
            return sum((v - mu) ** 2 for v in z) / len(z)

        assert var(out) < var(y)

    def test_preserves_mean_of_linear_trend_center(self):
        # a symmetric kernel reproduces a straight line in the interior
        y = [3.0 + 2.0 * i for i in range(30)]
        out = sm.gaussian_filter(y, 1.0)
        for t in range(5, 25):
            assert _close(out[t], 3.0 + 2.0 * t, 1e-6)

    def test_output_length_matches_input(self):
        y = [float(i) for i in range(20)]
        assert len(sm.gaussian_filter(y, 1.5)) == len(y)

    def test_nonpositive_sigma_raises(self):
        with pytest.raises(ValueError):
            sm.gaussian_filter([1.0, 2.0, 3.0], 0.0)
        with pytest.raises(ValueError):
            sm.gaussian_filter([1.0, 2.0, 3.0], -1.0)


class TestSavitzkyGolay:
    """The defining property: data on a degree-d polynomial is reproduced to
    ~1e-6 by savitzky_golay(window, poly>=d), interior AND shrunk edges."""

    def test_reproduces_quadratic_exactly(self):
        y = [2.0 + 3.0 * i - 0.5 * i * i for i in range(12)]
        out = sm.savitzky_golay(y, 5, 2)
        assert _series_close(out, y, 1e-6)

    def test_reproduces_cubic_exactly(self):
        y = [1.0 - 0.7 * i + 0.2 * i * i - 0.01 * i ** 3 for i in range(15)]
        out = sm.savitzky_golay(y, 7, 3)
        assert _series_close(out, y, 1e-6)

    def test_reproduces_line_with_low_degree(self):
        y = [4.0 + 2.5 * i for i in range(10)]
        out = sm.savitzky_golay(y, 5, 1)
        assert _series_close(out, y, 1e-6)

    def test_poly_zero_equals_centered_sma(self):
        # degree-0 fit is the window mean == centered simple moving average
        y = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0]
        sg = sm.savitzky_golay(y, 5, 0)
        sma = sm.simple_moving_average(y, 5, center=True)
        assert _series_close(sg, sma)

    def test_output_length_matches_input(self):
        y = [float(i) for i in range(11)]
        assert len(sm.savitzky_golay(y, 5, 2)) == len(y)

    def test_bad_window_raises(self):
        with pytest.raises(ValueError):
            sm.savitzky_golay([1.0, 2.0, 3.0], 0, 1)

    def test_bad_poly_raises(self):
        with pytest.raises(ValueError):
            sm.savitzky_golay([1.0, 2.0, 3.0, 4.0], 3, -1)


class TestLowess:
    """LOWESS reproduces a straight line to ~1e-6 and resists a gross outlier
    after robust re-weighting; frac is validated."""

    def test_reproduces_line(self):
        y = [3.0 + 2.0 * i for i in range(20)]
        out = sm.lowess(y, frac=0.5)
        assert _series_close(out, y, 1e-6)

    def test_reproduces_line_full_span(self):
        y = [-1.0 + 0.5 * i for i in range(15)]
        out = sm.lowess(y, frac=1.0)
        assert _series_close(out, y, 1e-6)

    def test_robust_iterations_beat_non_robust_on_outlier(self):
        rng = random.Random(3)
        n = 30
        true = [1.0 + 0.5 * i for i in range(n)]
        y = [true[i] + rng.gauss(0.0, 0.3) for i in range(n)]
        y[15] += 8.0                          # one contaminating outlier
        non_robust = sm.lowess(y, frac=0.4, iterations=1)
        robust = sm.lowess(y, frac=0.4, iterations=5)
        dev_nr = max(abs(non_robust[i] - true[i]) for i in range(n) if i != 15)
        dev_r = max(abs(robust[i] - true[i]) for i in range(n) if i != 15)
        # bisquare re-weighting must pull the fit back toward the true line
        assert dev_r < dev_nr
        assert dev_r < 0.5

    def test_outlier_barely_moves_far_point(self):
        n = 25
        true = [1.0 + 0.5 * i for i in range(n)]
        clean = sm.lowess(list(true), frac=0.5, iterations=4)
        y = list(true)
        y[12] += 500.0
        dirty = sm.lowess(y, frac=0.5, iterations=4)
        # a point far from the outlier is essentially unchanged
        assert abs(dirty[0] - clean[0]) < 1e-6

    def test_custom_x_line(self):
        x = [0.0, 1.0, 2.5, 4.0, 6.0, 9.0, 13.0]
        y = [2.0 + 3.0 * xi for xi in x]
        out = sm.lowess(y, frac=1.0, x=x)
        assert _series_close(out, y, 1e-6)

    def test_empty_returns_empty(self):
        assert sm.lowess([]) == []

    def test_output_length_matches_input(self):
        y = [float(i) for i in range(18)]
        assert len(sm.lowess(y, frac=0.4)) == len(y)

    def test_frac_out_of_range_raises(self):
        with pytest.raises(ValueError):
            sm.lowess([1.0, 2.0, 3.0, 4.0], frac=0.0)
        with pytest.raises(ValueError):
            sm.lowess([1.0, 2.0, 3.0, 4.0], frac=1.5)


class TestHodrickPrescott:
    """lam=0 returns the data; large lam -> OLS line; n<3 short-circuits;
    negative lam raises; solution satisfies its normal equations."""

    def test_lam_zero_returns_data(self):
        y = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        out = sm.hodrick_prescott(y, lam=0.0)
        assert _series_close(out, y, 1e-9)

    def test_large_lam_approaches_ols_line(self):
        rng = random.Random(7)
        n = 40
        line = [2.0 + 0.75 * i for i in range(n)]
        y = [line[i] + rng.gauss(0.0, 0.5) for i in range(n)]
        out = sm.hodrick_prescott(y, lam=1e10)
        # OLS trend of the noisy data
        xs = list(range(n))
        xm = sum(xs) / n
        ym = sum(y) / n
        b1 = sum((xs[i] - xm) * (y[i] - ym) for i in range(n)) / \
            sum((xs[i] - xm) ** 2 for i in range(n))
        b0 = ym - b1 * xm
        ols = [b0 + b1 * i for i in range(n)]
        assert _series_close(out, ols, 1e-3)

    def test_short_series_returned_unchanged(self):
        assert _series_close(sm.hodrick_prescott([1.0, 2.0]), [1.0, 2.0])
        assert sm.hodrick_prescott([5.0]) == [5.0]

    def test_negative_lam_raises(self):
        with pytest.raises(ValueError):
            sm.hodrick_prescott([1.0, 2.0, 3.0, 4.0], lam=-1.0)

    def test_satisfies_normal_equations(self):
        rng = random.Random(99)
        n = 20
        y = [math.sin(0.3 * i) + rng.gauss(0.0, 0.2) for i in range(n)]
        lam = 100.0
        tau = sm.hodrick_prescott(y, lam=lam)
        # reconstruct A = I + lam D'D and check A @ tau == y
        A = [[0.0] * n for _ in range(n)]
        for i in range(n):
            A[i][i] = 1.0
        for k in range(n - 2):
            idx = (k, k + 1, k + 2)
            coef = (1.0, -2.0, 1.0)
            for a in range(3):
                for b in range(3):
                    A[idx[a]][idx[b]] += lam * coef[a] * coef[b]
        for i in range(n):
            lhs = math.fsum(A[i][j] * tau[j] for j in range(n))
            assert _close(lhs, y[i], 1e-6)

    def test_output_length_matches_input(self):
        y = [float(i) for i in range(15)]
        assert len(sm.hodrick_prescott(y)) == len(y)

    @slow
    def test_trend_is_smoother_than_data(self):
        rng = random.Random(2024)
        n = 120
        y = [10.0 + 0.1 * i + 3.0 * math.sin(0.2 * i) + rng.gauss(0.0, 1.0)
             for i in range(n)]
        tau = sm.hodrick_prescott(y, lam=1600.0)

        def roughness(z):
            return math.fsum((z[t + 1] - 2 * z[t] + z[t - 1]) ** 2
                             for t in range(1, len(z) - 1))

        assert roughness(tau) < roughness(y)


# =========================================================================
# EXPONENTIAL SMOOTHING (ETS)
# =========================================================================
class TestForecastResultIntegrity:
    """Pins the structural contract of ForecastResult: field lengths, the
    residual identity residuals[i]==y[i]-fitted[i], forecast horizon length,
    and the exact params-dict key set implied by each ETS configuration."""

    def test_lengths_and_residual_identity_ses(self):
        y = [3.0, 7.0, 2.0, 9.0, 5.0, 6.0, 8.0, 4.0]
        r = sm.ses(y, h=4)
        assert isinstance(r, sm.ForecastResult)
        assert len(r.fitted) == len(y)
        assert len(r.residuals) == len(y)
        assert len(r.forecast) == 4
        for i in range(len(y)):
            assert _close(r.residuals[i], y[i] - r.fitted[i], 1e-12)

    def test_h_zero_gives_empty_forecast_and_no_intervals(self):
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        r = sm.exponential_smoothing(y, h=0)
        assert r.forecast == []
        assert r.lower is None and r.upper is None

    def test_params_keys_per_config(self):
        y = [10.0, 11.0, 9.0, 12.0, 11.0, 13.0, 12.0, 14.0,
             13.0, 15.0, 14.0, 16.0, 15.0, 17.0, 16.0, 18.0]
        assert set(sm.ses(y).params.keys()) == {"alpha"}
        assert set(sm.holt(y).params.keys()) == {"alpha", "beta"}
        assert set(sm.holt(y, damped=True).params.keys()) == {"alpha", "beta", "phi"}
        hw = sm.holt_winters(y, seasonal_periods=4, seasonal="add")
        assert set(hw.params.keys()) == {"alpha", "beta", "gamma"}
        hwd = sm.holt_winters(y, seasonal_periods=4, seasonal="add", damped=True)
        assert set(hwd.params.keys()) == {"alpha", "beta", "gamma", "phi"}

    def test_method_name_string(self):
        y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        assert sm.ses(y).method == "ETS(A,N,N)"
        assert sm.holt(y).method == "ETS(A,A,N)"
        assert sm.holt(y, damped=True).method == "ETS(A,Ad,N)"

    def test_state_dict_present(self):
        y = [1.0, 3.0, 5.0, 7.0, 9.0]
        r = sm.holt(y)
        assert set(["level", "trend", "seasonal"]).issubset(r.state.keys())


class TestSESOracle:
    """Pins SES against hand-computed recursions: alpha=1 collapses to the naive
    (shifted-y fitted, flat forecast at the last observation), and a fixed
    alpha=0.5 reproduces an exact level recursion.  Also checks that auto-alpha
    on a random walk is driven toward 1."""

    def test_alpha_one_is_naive(self):
        y = [3.0, 7.0, 2.0, 9.0]
        r = sm.ses(y, alpha=1.0, optimize=False, h=3)
        # fitted = [y0, y0, y1, y2]  (one-step naive after the seed)
        assert _series_close(r.fitted, [3.0, 3.0, 7.0, 2.0], 1e-12)
        # flat forecast at the last observation
        assert _series_close(r.forecast, [9.0, 9.0, 9.0], 1e-12)

    def test_fixed_alpha_recursion_oracle(self):
        y = [10.0, 12.0, 13.0, 12.0, 15.0]
        r = sm.ses(y, alpha=0.5, optimize=False, h=1)
        assert _series_close(r.fitted, [10.0, 10.0, 11.0, 12.0, 12.0], 1e-12)
        assert _series_close(r.forecast, [13.5], 1e-12)
        assert _close(r.params["alpha"], 0.5, 1e-15)
        # SSE over the full range (burn=0 for SES): 0+4+4+0+9
        assert _close(r.sse, 17.0, 1e-9)

    def test_multi_step_forecast_is_flat(self):
        y = [5.0, 6.0, 4.0, 7.0, 5.0, 8.0]
        r = sm.ses(y, h=5)
        assert all(_close(r.forecast[k], r.forecast[0], 1e-12) for k in range(5))

    def test_first_residual_is_zero(self):
        y = [2.0, 9.0, 4.0, 7.0]
        r = sm.ses(y, alpha=0.4, optimize=False)
        assert _close(r.residuals[0], 0.0, 1e-12)

    def test_auto_alpha_on_random_walk_is_large(self):
        rng = random.Random(20240501)
        y = [0.0]
        for _ in range(80):
            y.append(y[-1] + rng.gauss(0.0, 1.0))
        r = sm.ses(y, optimize=True)
        assert r.params["alpha"] >= 0.8
        assert 0.0 < r.params["alpha"] <= 0.9999


class TestHoltOracle:
    """Pins Holt's linear-trend method: a fully hand-traced fixed-parameter run,
    the exact arithmetic-progression structure of the undamped forecast, and the
    geometric flattening of the damped forecast (increment ratio == phi)."""

    def test_fixed_param_hand_trace(self):
        y = [1.0, 3.0, 5.0, 7.0]
        r = sm.holt(y, alpha=0.5, beta=0.5, optimize=False, h=1)
        assert _series_close(r.fitted, [3.0, 3.5, 4.625, 6.28125], 1e-12)
        assert _series_close(r.forecast, [8.2890625], 1e-12)

    def test_undamped_forecast_is_arithmetic_progression(self):
        y = [2.0, 5.0, 7.0, 10.0, 12.0, 15.0, 17.0, 20.0]
        r = sm.holt(y, h=6)
        diffs = [r.forecast[k + 1] - r.forecast[k] for k in range(5)]
        # constant increment == final trend slope
        assert all(_close(d, diffs[0], 1e-9) for d in diffs)

    def test_linear_series_forecast_continues_line(self):
        a, b = 4.0, 2.5
        y = [a + b * t for t in range(20)]
        r = sm.holt(y, h=5)
        step = r.forecast[1] - r.forecast[0]
        assert _close(step, b, 1e-3)
        # forecast collinear with the true line (loose)
        for k in range(5):
            assert _close(r.forecast[k], a + b * (20 + k), 1e-2)

    def test_damped_forecast_flattens_geometrically(self):
        y = [1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0]
        r = sm.holt(y, damped=True, h=6)
        phi = r.params["phi"]
        assert 0.8 <= phi <= 0.999
        incs = [r.forecast[k + 1] - r.forecast[k] for k in range(5)]
        # each increment == phi times the previous one, exactly
        for k in range(1, 5):
            assert _close(incs[k] / incs[k - 1], phi, 1e-7)
        # strictly shrinking magnitude (flattening)
        for k in range(1, 5):
            assert abs(incs[k]) < abs(incs[k - 1])

    def test_damped_forecast_is_bounded(self):
        y = [1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0]
        r = sm.holt(y, damped=True, h=200)
        phi = r.params["phi"]
        b = r.state["trend"]
        lvl = r.state["level"]
        bound = lvl + (phi / (1.0 - phi)) * b
        assert all(fc <= bound + 1e-6 for fc in r.forecast)


class TestHoltWintersSeasonal:
    """Pins the seasonal engine: on clean synthetic additive / multiplicative
    series the in-sample SSE is tiny and the forecast repeats with the true
    period; the seasonal state recovers the injected pattern shape."""

    def _add_series(self, reps=6):
        season = [5.0, -3.0, -6.0, 4.0]  # sums to zero
        return [50.0 + season[t % 4] for t in range(4 * reps)]

    def _mul_series(self, reps=6):
        season = [1.2, 0.8, 0.9, 1.1]
        return [100.0 * season[t % 4] for t in range(4 * reps)]

    def test_additive_no_trend_forecast_is_periodic(self):
        y = self._add_series()
        r = sm.exponential_smoothing(y, seasonal="add", seasonal_periods=4, h=8)
        # no trend -> base constant -> forecast exactly period-4
        for k in range(4):
            assert _close(r.forecast[k], r.forecast[k + 4], 1e-6)

    def test_multiplicative_no_trend_forecast_is_periodic(self):
        y = self._mul_series()
        r = sm.exponential_smoothing(y, seasonal="mul", seasonal_periods=4, h=8)
        for k in range(4):
            assert _close(r.forecast[k], r.forecast[k + 4], 1e-6)

    def test_additive_clean_series_small_sse(self):
        y = self._add_series()
        r = sm.holt_winters(y, seasonal_periods=4, seasonal="add", h=4)
        # deterministic clean series -> the fit should be essentially exact
        assert r.sse < 1e-2
        # one-period-ahead forecast recovers the seasonal shape
        season = [5.0, -3.0, -6.0, 4.0]
        for k in range(4):
            assert _close(r.forecast[k], 50.0 + season[k % 4], 0.5)

    def test_multiplicative_clean_series_small_sse(self):
        y = self._mul_series()
        r = sm.holt_winters(y, seasonal_periods=4, seasonal="mul", h=4)
        assert r.sse < 1e-1

    def test_seasonal_state_recovers_pattern(self):
        y = self._add_series()
        r = sm.exponential_smoothing(y, seasonal="add", seasonal_periods=4, h=0)
        s = r.state["seasonal"]
        assert len(s) == 4
        # differences between seasonal factors should match the injected pattern
        true = [5.0, -3.0, -6.0, 4.0]
        for i in range(4):
            for j in range(4):
                assert _close(s[i] - s[j], true[i] - true[j], 0.5)


class TestOptimizerAndInfoCriteria:
    """Pins that Nelder-Mead genuinely reduces the in-sample SSE relative to a
    mediocre fixed parameterisation, that optimised parameters land inside the
    admissible boxes, and that AIC/AICc are finite when data is plentiful."""

    def test_optimizer_reduces_sse_ses(self):
        rng = random.Random(7)
        y = [10.0 + 0.5 * t + rng.gauss(0, 2.0) for t in range(40)]
        opt = sm.ses(y, optimize=True)
        bad = sm.ses(y, alpha=0.01, optimize=False)
        assert opt.sse <= bad.sse + 1e-6

    def test_optimizer_reduces_sse_holt(self):
        rng = random.Random(11)
        y = [5.0 + 1.2 * t + rng.gauss(0, 1.0) for t in range(50)]
        opt = sm.holt(y, optimize=True)
        bad = sm.holt(y, alpha=0.02, beta=0.02, optimize=False)
        assert opt.sse <= bad.sse + 1e-6

    def test_optimized_params_in_range(self):
        rng = random.Random(3)
        y = [20.0 + 0.3 * t + 4.0 * math.sin(t) + rng.gauss(0, 0.5)
             for t in range(40)]
        r = sm.holt_winters(y, seasonal_periods=4, seasonal="add", damped=True)
        assert 0.0 < r.params["alpha"] <= 0.9999
        assert 0.0 < r.params["beta"] <= 0.9999
        assert 0.0 < r.params["gamma"] <= 0.9999
        assert 0.8 <= r.params["phi"] <= 0.999

    def test_aic_aicc_finite_with_enough_data(self):
        rng = random.Random(99)
        y = [3.0 + rng.gauss(0, 1.0) for _ in range(30)]
        r = sm.ses(y)
        assert r.aic is not None and math.isfinite(r.aic)
        assert r.aicc is not None and math.isfinite(r.aicc)
        assert r.aicc >= r.aic  # AICc adds a positive correction

    def test_sigma_matches_sse(self):
        y = [10.0, 12.0, 13.0, 12.0, 15.0, 14.0, 16.0]
        r = sm.ses(y, alpha=0.5, optimize=False)
        # burn=0 for SES, k_eff = n
        assert _close(r.sigma, math.sqrt(r.sse / len(y)), 1e-12)


class TestPredictionIntervals:
    """Pins interval sanity: lower < point < upper, the interval width grows
    with the horizon like sqrt(k+1), and z-scaling responds to the level."""

    def test_intervals_bracket_and_widen(self):
        rng = random.Random(42)
        y = [10.0 + 0.4 * t + rng.gauss(0, 1.5) for t in range(40)]
        r = sm.holt(y, h=6, level=0.95)
        assert r.lower is not None and r.upper is not None
        for k in range(6):
            assert r.lower[k] < r.forecast[k] < r.upper[k]
        widths = [r.upper[k] - r.lower[k] for k in range(6)]
        for k in range(1, 6):
            assert widths[k] > widths[k - 1]

    def test_width_scales_like_sqrt_horizon(self):
        rng = random.Random(1)
        y = [5.0 + rng.gauss(0, 1.0) for _ in range(50)]
        r = sm.ses(y, h=4)
        w = [r.upper[k] - r.lower[k] for k in range(4)]
        # w[k] proportional to sqrt(k+1)
        for k in range(4):
            assert _close(w[k] / w[0], math.sqrt(k + 1), 1e-9)

    def test_higher_level_gives_wider_interval(self):
        rng = random.Random(5)
        y = [7.0 + rng.gauss(0, 1.0) for _ in range(40)]
        r80 = sm.ses(y, h=3, level=0.80)
        r99 = sm.ses(y, h=3, level=0.99)
        for k in range(3):
            w80 = r80.upper[k] - r80.lower[k]
            w99 = r99.upper[k] - r99.lower[k]
            assert w99 > w80


class TestValidationErrors:
    """Pins the failure modes: too-short series, illegal trend/seasonal codes,
    damped-without-trend, seasonal misconfiguration, insufficient seasons, and
    NaN contamination all raise ValueError."""

    def test_too_short(self):
        with pytest.raises(ValueError):
            sm.exponential_smoothing([5.0])
        with pytest.raises(ValueError):
            sm.ses([1.0])

    def test_bad_trend_code(self):
        with pytest.raises(ValueError):
            sm.exponential_smoothing([1.0, 2.0, 3.0], trend="mul")

    def test_bad_seasonal_code(self):
        with pytest.raises(ValueError):
            sm.exponential_smoothing([1.0, 2.0, 3.0, 4.0], seasonal="foo",
                                     seasonal_periods=2)

    def test_damped_requires_trend(self):
        with pytest.raises(ValueError):
            sm.exponential_smoothing([1.0, 2.0, 3.0, 4.0], damped=True)

    def test_seasonal_needs_periods_ge_2(self):
        with pytest.raises(ValueError):
            sm.exponential_smoothing([1.0, 2.0, 3.0, 4.0], seasonal="add",
                                     seasonal_periods=1)
        with pytest.raises(ValueError):
            sm.exponential_smoothing([1.0, 2.0, 3.0, 4.0], seasonal="add")

    def test_seasonal_needs_two_full_seasons(self):
        # len 7 < 2*4
        with pytest.raises(ValueError):
            sm.holt_winters([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], seasonal_periods=4)

    def test_nan_rejected(self):
        with pytest.raises(ValueError):
            sm.ses([1.0, float("nan"), 3.0, 4.0])

    def test_bad_seasonal_code_via_holt_winters(self):
        y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        with pytest.raises(ValueError):
            sm.holt_winters(y, seasonal_periods=2, seasonal="bogus")


class TestHarshRandomBattery:
    """A seeded battery over many random configs asserting the invariants that
    must hold for ANY input: residual identity, forecast length, finite SSE,
    and non-crossing prediction intervals."""

    @slow
    def test_random_configs_invariants(self):
        rng = random.Random(123456)
        for _ in range(60):
            n = rng.randint(12, 40)
            base = rng.uniform(5, 50)
            slope = rng.uniform(-0.5, 0.5)
            y = [base + slope * t + rng.gauss(0, 1.5) for t in range(n)]
            trend = rng.choice([None, "add"])
            damped = trend == "add" and rng.random() < 0.5
            h = rng.randint(1, 8)
            r = sm.exponential_smoothing(y, trend=trend, damped=damped, h=h)
            assert len(r.fitted) == n
            assert len(r.forecast) == h
            for i in range(n):
                assert _close(r.residuals[i], y[i] - r.fitted[i], 1e-9)
            assert math.isfinite(r.sse) and r.sse >= 0.0
            if r.lower is not None:
                for k in range(h):
                    assert r.lower[k] <= r.forecast[k] <= r.upper[k]


# =========================================================================
# THETA, AUTO-ETS & DECOMPOSITION
# =========================================================================
class TestThetaMethod:
    """Pins the Theta method: shape, the constant-series oracle, drift-following
    on a clean trend, the deseasonalise/reseasonalise path, and n<3 errors."""

    def test_shape_and_type(self):
        rng = random.Random(1)
        y = [10.0 + 0.5 * t + rng.gauss(0, 1) for t in range(30)]
        r = sm.theta_method(y, h=7)
        assert isinstance(r, sm.ForecastResult)
        assert r.method == "Theta"
        assert len(r.forecast) == 7
        assert len(r.fitted) == len(y)
        assert len(r.residuals) == len(y)
        assert all(v == v for v in r.forecast)

    def test_constant_series_oracle(self):
        # On a constant series both the OLS drift line and the SES line equal c,
        # so every fitted value and every forecast must be exactly c, with zero
        # residuals (hence no prediction interval).
        c = 7.0
        n = 12
        y = [c] * n
        r = sm.theta_method(y, h=5)
        assert _series_close(r.forecast, [c] * 5, tol=1e-9)
        assert _series_close(r.fitted, [c] * n, tol=1e-9)
        assert all(abs(e) <= 1e-9 for e in r.residuals)
        assert r.sigma <= 1e-12
        assert r.lower is None and r.upper is None

    def test_drift_following_monotone(self):
        # For a strictly increasing linear series the SES multi-step forecast is
        # flat while the drift line rises, so the Theta forecast (their average)
        # must be strictly increasing across the horizon.
        n = 40
        y = [3.0 + 2.0 * t for t in range(n)]
        r = sm.theta_method(y, h=10)
        for k in range(1, len(r.forecast)):
            assert r.forecast[k] > r.forecast[k - 1] + 1e-9
        # and the first forecast should exceed the last observation (upward trend)
        assert r.forecast[0] > y[-1] - 1.0

    def test_drift_recovers_slope_params(self):
        # b1 is a plain OLS slope on the (optionally deseasonalised) work series;
        # on an exact line it must equal the true slope.
        n = 25
        y = [1.5 + 0.7 * t for t in range(n)]
        r = sm.theta_method(y, h=1)
        assert _close(r.params["b1"], 0.7, tol=1e-7)
        assert _close(r.params["b0"], 1.5, tol=1e-6)

    def test_seasonal_path_runs_and_finite(self):
        # Deseasonalise (multiplicative classical) then reseasonalise.
        m = 4
        seasonal = [1.3, 0.7, 1.1, 0.9]
        y = [(50.0 + 1.0 * t) * seasonal[t % m] for t in range(24)]
        r = sm.theta_method(y, h=8, seasonal_periods=m)
        assert len(r.forecast) == 8
        assert all(v == v and abs(v) < 1e9 for v in r.forecast)
        # the reseasonalised forecast should still be positive for this positive series
        assert all(v > 0 for v in r.forecast)

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            sm.theta_method([1.0, 2.0], h=3)

    def test_nan_rejected(self):
        with pytest.raises(ValueError):
            sm.theta_method([1.0, 2.0, float("nan"), 4.0], h=2)


class TestAutoEts:
    """Pins auto_ets: it returns a valid ForecastResult at the requested horizon,
    selects a SEASONAL model on strongly seasonal data, and stays non-seasonal /
    crash-free on a plain trend."""

    def test_plain_trend_no_crash_nonseasonal(self):
        rng = random.Random(7)
        y = [5.0 + 0.8 * t + rng.gauss(0, 0.5) for t in range(30)]
        r = sm.auto_ets(y, h=6)
        assert isinstance(r, sm.ForecastResult)
        assert len(r.forecast) == 6
        assert r.method.startswith("ETS(")
        # no seasonal_periods given -> seasonal component must be 'N'
        assert r.method.endswith(",N)")

    @slow
    def test_selects_seasonal_model(self):
        # Dominant seasonality with mild trend over 8 seasons -> a seasonal ETS
        # (last component != 'N') must win AICc.
        m = 4
        pattern = [20.0, -15.0, 10.0, -15.0]
        rng = random.Random(11)
        y = [100.0 + 0.3 * t + pattern[t % m] + rng.gauss(0, 0.3)
             for t in range(8 * m)]
        r = sm.auto_ets(y, seasonal_periods=m, h=8)
        assert isinstance(r, sm.ForecastResult)
        assert len(r.forecast) == 8
        assert not r.method.endswith(",N)"), r.method

    def test_requested_horizon_and_finite(self):
        rng = random.Random(3)
        y = [10.0 + rng.gauss(0, 1) for _ in range(20)]
        r = sm.auto_ets(y, h=4)
        assert len(r.forecast) == 4
        assert all(v == v for v in r.forecast)

    def test_short_series_falls_back(self):
        # Too few points for any AICc to be defined -> graceful SES fallback.
        y = [1.0, 3.0, 2.0]
        r = sm.auto_ets(y, h=2)
        assert isinstance(r, sm.ForecastResult)
        assert len(r.forecast) == 2


class TestClassicalDecomposition:
    """Pins classical decomposition: exact additive/multiplicative reconstruction
    on the interior, seasonal-index normalisation, known-seasonal recovery, and
    the full error surface."""

    def _additive_series(self, n=24, m=4):
        pattern = [5.0, -3.0, -1.0, -1.0]  # sums to 0
        return [100.0 + 2.0 * t + pattern[t % m] for t in range(n)], pattern

    def test_additive_reconstruction_interior(self):
        y, _ = self._additive_series()
        d = sm.classical_decomposition(y, 4, model="additive")
        for t in range(len(y)):
            if d.trend[t] is not None:
                assert _close(d.observed[t],
                              d.trend[t] + d.seasonal[t] + d.resid[t], tol=1e-9)

    def test_multiplicative_reconstruction_interior(self):
        m = 4
        seasonal = [1.2, 0.8, 1.1, 0.9]  # mean 1
        y = [(100.0 + 2.0 * t) * seasonal[t % m] for t in range(24)]
        d = sm.classical_decomposition(y, m, model="multiplicative")
        for t in range(len(y)):
            if d.trend[t] is not None:
                assert _close(d.observed[t],
                              d.trend[t] * d.seasonal[t] * d.resid[t], tol=1e-9)

    def test_additive_seasonal_sums_to_zero(self):
        y, _ = self._additive_series()
        d = sm.classical_decomposition(y, 4, model="additive")
        one_period = d.seasonal[:4]
        assert _close(math.fsum(one_period), 0.0, tol=1e-9)

    def test_multiplicative_seasonal_averages_to_one(self):
        m = 4
        seasonal = [1.2, 0.8, 1.1, 0.9]
        y = [(100.0 + 2.0 * t) * seasonal[t % m] for t in range(24)]
        d = sm.classical_decomposition(y, m, model="multiplicative")
        one_period = d.seasonal[:m]
        assert _close(math.fsum(one_period) / m, 1.0, tol=1e-9)

    def test_known_additive_seasonal_recovery(self):
        # Exact linear trend + exact repeating pattern, no noise: the centred MA
        # recovers the linear trend exactly, so the detrended series IS the
        # pattern and the recovered indices must match it.
        y, pattern = self._additive_series(n=24, m=4)
        d = sm.classical_decomposition(y, 4, model="additive")
        assert _series_close(d.seasonal[:4], pattern, tol=1e-6)

    def test_known_multiplicative_seasonal_recovery(self):
        m = 4
        seasonal = [1.2, 0.8, 1.1, 0.9]  # mean 1
        y = [(100.0 + 2.0 * t) * seasonal[t % m] for t in range(24)]
        d = sm.classical_decomposition(y, m, model="multiplicative")
        # the centred MA of a multiplicative series is not exactly the trend, so
        # recovery is only approximate -- but must land within a loose tolerance.
        assert _series_close(d.seasonal[:m], seasonal, tol=2e-3)

    def test_error_period_too_small(self):
        with pytest.raises(ValueError):
            sm.classical_decomposition([1.0] * 10, 1)

    def test_error_too_short(self):
        with pytest.raises(ValueError):
            sm.classical_decomposition([1.0, 2.0, 3.0], 2)  # n=3 < 2*2

    def test_error_multiplicative_nonpositive(self):
        y = [1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        with pytest.raises(ValueError):
            sm.classical_decomposition(y, 2, model="multiplicative")

    def test_error_bad_model(self):
        with pytest.raises(ValueError):
            sm.classical_decomposition([1.0] * 12, 4, model="banana")


class TestDecompositionSeasonalAdjust:
    """Pins Decomposition.seasonally_adjusted for both models."""

    def test_additive_adjustment(self):
        pattern = [5.0, -3.0, -1.0, -1.0]
        y = [100.0 + 2.0 * t + pattern[t % 4] for t in range(24)]
        d = sm.classical_decomposition(y, 4, model="additive")
        adj = d.seasonally_adjusted()
        expected = [d.observed[t] - d.seasonal[t] for t in range(len(y))]
        assert _series_close(adj, expected, tol=1e-12)

    def test_multiplicative_adjustment(self):
        seasonal = [1.2, 0.8, 1.1, 0.9]
        y = [(100.0 + 2.0 * t) * seasonal[t % 4] for t in range(24)]
        d = sm.classical_decomposition(y, 4, model="multiplicative")
        adj = d.seasonally_adjusted()
        expected = [d.observed[t] / d.seasonal[t] for t in range(len(y))]
        assert _series_close(adj, expected, tol=1e-12)


class TestStlDecomposition:
    """Pins simplified STL: exact additive reconstruction, centred seasonal, and
    matching error handling."""

    def test_additive_reconstruction_exact(self):
        rng = random.Random(5)
        m = 4
        pattern = [4.0, -2.0, 1.0, -3.0]
        y = [50.0 + 0.5 * t + pattern[t % m] + rng.gauss(0, 0.2)
             for t in range(24)]
        d = sm.stl_decomposition(y, m)
        for t in range(len(y)):
            assert _close(d.observed[t],
                          d.trend[t] + d.seasonal[t] + d.resid[t], tol=1e-9)

    def test_seasonal_is_centred(self):
        rng = random.Random(6)
        m = 4
        pattern = [4.0, -2.0, 1.0, -3.0]
        y = [50.0 + 0.5 * t + pattern[t % m] + rng.gauss(0, 0.2)
             for t in range(24)]
        d = sm.stl_decomposition(y, m)
        assert _close(math.fsum(d.seasonal) / len(d.seasonal), 0.0, tol=1e-9)

    def test_model_and_period_fields(self):
        y = [50.0 + t + [1.0, -1.0][t % 2] for t in range(12)]
        d = sm.stl_decomposition(y, 2)
        assert d.model == "additive"
        assert d.period == 2

    def test_error_period_too_small(self):
        with pytest.raises(ValueError):
            sm.stl_decomposition([1.0] * 10, 1)

    def test_error_too_short(self):
        with pytest.raises(ValueError):
            sm.stl_decomposition([1.0, 2.0, 3.0], 2)


# =========================================================================
# AUTOCORRELATION & AUTOREGRESSION
# =========================================================================
def _gen_ar(phi, n, seed, sd=1.0, mean=0.0, burn=500):
    """Generate a stationary AR(p) series with the given coefficients about a
    target unconditional mean, using a seeded Gaussian innovation stream."""
    rng = random.Random(seed)
    p = len(phi)
    c = mean * (1.0 - math.fsum(phi))
    hist = [mean] * p
    out = []
    for _ in range(n + burn):
        val = c + math.fsum(phi[k] * hist[-1 - k] for k in range(p)) + rng.gauss(0.0, sd)
        hist.append(val)
        out.append(val)
    return out[burn:]


class TestAutocovariance:
    """Pins the biased sample autocovariance: gamma_0 == population variance
    (divisor n), hand-computed lags on a known series, and input validation."""

    def test_known_values(self):
        g = sm.autocovariance([1.0, 2.0, 3.0, 4.0], 3)
        assert _close(g[0], 1.25)
        assert _close(g[1], 0.3125)
        assert _close(g[2], -0.375)
        assert _close(g[3], -0.5625)

    def test_gamma0_is_population_variance(self):
        rng = random.Random(11)
        y = [rng.gauss(3.0, 2.0) for _ in range(200)]
        mu = math.fsum(y) / len(y)
        var_n = math.fsum((v - mu) ** 2 for v in y) / len(y)
        g = sm.autocovariance(y, 5)
        assert _close(g[0], var_n, tol=1e-9)

    def test_nlags_zero_returns_single(self):
        g = sm.autocovariance([1.0, 5.0, 2.0, 8.0], 0)
        assert len(g) == 1
        assert _close(g[0], sm.autocovariance([1.0, 5.0, 2.0, 8.0], 3)[0])

    def test_length(self):
        assert len(sm.autocovariance(list(range(20)), 7)) == 8

    def test_negative_nlags_raises(self):
        with pytest.raises(ValueError):
            sm.autocovariance([1.0, 2.0, 3.0], -1)

    def test_nlags_too_large_raises(self):
        with pytest.raises(ValueError):
            sm.autocovariance([1.0, 2.0, 3.0], 3)
        with pytest.raises(ValueError):
            sm.autocovariance([1.0, 2.0, 3.0], 4)


class TestAutocorrelation:
    """Pins rho_0 == 1 exactly, |rho_k| <= 1, the covariance-ratio identity,
    hand-computed values, and the constant-series degenerate branch."""

    def test_rho0_is_one(self):
        rng = random.Random(3)
        y = [rng.gauss(0.0, 1.0) for _ in range(150)]
        r = sm.autocorrelation(y, 10)
        assert r[0] == 1.0

    def test_known_values(self):
        r = sm.autocorrelation([1.0, 2.0, 3.0, 4.0], 3)
        assert r[0] == 1.0
        assert _close(r[1], 0.25)
        assert _close(r[2], -0.3)
        assert _close(r[3], -0.45)

    def test_bounded_by_one(self):
        rng = random.Random(7)
        y = [rng.gauss(0.0, 5.0) + 0.5 * i for i in range(120)]
        r = sm.autocorrelation(y, 30)
        assert all(abs(rk) <= 1.0 + 1e-12 for rk in r)

    def test_is_covariance_ratio(self):
        rng = random.Random(21)
        y = [math.sin(0.3 * i) + rng.gauss(0, 0.2) for i in range(100)]
        g = sm.autocovariance(y, 12)
        r = sm.autocorrelation(y, 12)
        for k in range(13):
            assert _close(r[k], g[k] / g[0], tol=1e-12)

    def test_constant_series_degenerate(self):
        r = sm.autocorrelation([5.0] * 20, 4)
        assert r[0] == 1.0
        assert all(rk == 0.0 for rk in r[1:])

    def test_white_noise_acf_near_zero(self):
        rng = random.Random(2024)
        y = [rng.gauss(0.0, 1.0) for _ in range(4000)]
        r = sm.autocorrelation(y, 20)
        assert all(abs(r[k]) < 0.15 for k in range(1, 21))


class TestPartialAutocorrelation:
    """Pins pacf[0] == 1, pacf[1] == acf[1], hand-computed pacf[2], the AR(p)
    cut-off property (pacf ~ 0 beyond lag p), and validation."""

    def test_pacf0_is_one(self):
        p = sm.partial_autocorrelation([1.0, 3.0, 2.0, 5.0, 4.0], 3)
        assert p[0] == 1.0

    def test_nlags_zero(self):
        assert sm.partial_autocorrelation([1.0, 2.0, 3.0], 0) == [1.0]

    def test_pacf1_equals_acf1(self):
        y = [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0]
        p = sm.partial_autocorrelation(y, 4)
        a = sm.autocorrelation(y, 4)
        assert _close(p[1], a[1], tol=1e-12)

    def test_pacf2_known(self):
        # y=[1,2,3,4]: acf=[1,.25,-.3,-.45] -> pacf[2] = (acf2-acf1^2)/(1-acf1^2)
        p = sm.partial_autocorrelation([1.0, 2.0, 3.0, 4.0], 2)
        assert _close(p[1], 0.25)
        assert _close(p[2], (-0.3 - 0.25 ** 2) / (1.0 - 0.25 ** 2))

    def test_negative_nlags_raises(self):
        with pytest.raises(ValueError):
            sm.partial_autocorrelation([1.0, 2.0, 3.0], -2)

    @slow
    def test_ar1_cutoff(self):
        # AR(1): pacf[1] ~ phi, pacf[k] ~ 0 for k >= 2.
        phi = 0.7
        vals = []
        for seed in range(6):
            y = _gen_ar([phi], 3000, seed)
            vals.append(sm.partial_autocorrelation(y, 8))
        avg = [math.fsum(v[k] for v in vals) / len(vals) for k in range(9)]
        assert abs(avg[1] - phi) < 0.05
        assert all(abs(avg[k]) < 0.06 for k in range(2, 9))

    @slow
    def test_ar2_cutoff(self):
        # AR(2): pacf nonzero at lags 1,2 then cuts off.
        phi = [0.5, 0.3]
        vals = []
        for seed in range(6):
            y = _gen_ar(phi, 3000, 100 + seed)
            vals.append(sm.partial_autocorrelation(y, 8))
        avg = [math.fsum(v[k] for v in vals) / len(vals) for k in range(9)]
        assert abs(avg[2] - 0.3) < 0.06
        assert all(abs(avg[k]) < 0.06 for k in range(3, 9))


class TestYuleWalker:
    """Pins the order-1 closed form exactly, mean-correct intercept, positive
    sigma2, AR(1)/AR(2) coefficient recovery, and validation."""

    def test_order1_closed_form(self):
        y = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 3.0, 7.0]
        phi, c, s2 = sm.yule_walker(y, 1)
        g = sm.autocovariance(y, 1)
        rho1 = g[1] / g[0]
        mu = math.fsum(y) / len(y)
        assert _close(phi[0], rho1, tol=1e-12)
        assert _close(c, mu * (1.0 - phi[0]), tol=1e-12)
        assert _close(s2, g[0] * (1.0 - rho1 ** 2), tol=1e-12)

    def test_intercept_mean_correct(self):
        y = _gen_ar([0.5, 0.2], 800, 5, mean=4.0)
        phi, c, s2 = sm.yule_walker(y, 2)
        mu = math.fsum(y) / len(y)
        implied_mean = c / (1.0 - math.fsum(phi))
        assert _close(implied_mean, mu, tol=1e-6)

    def test_sigma2_positive(self):
        y = _gen_ar([0.6], 500, 9)
        _, _, s2 = sm.yule_walker(y, 3)
        assert s2 > 0.0

    def test_order_too_small_raises(self):
        with pytest.raises(ValueError):
            sm.yule_walker([1.0, 2.0, 3.0], 0)

    def test_order_too_large_raises(self):
        with pytest.raises(ValueError):
            sm.yule_walker([1.0, 2.0, 3.0], 3)

    @slow
    def test_ar1_recovery(self):
        est = []
        for seed in range(8):
            y = _gen_ar([0.7], 2500, seed)
            phi, _, _ = sm.yule_walker(y, 1)
            est.append(phi[0])
        assert abs(math.fsum(est) / len(est) - 0.7) < 0.03

    @slow
    def test_ar2_recovery(self):
        p1s, p2s = [], []
        for seed in range(8):
            y = _gen_ar([0.5, 0.3], 2500, 200 + seed)
            phi, _, _ = sm.yule_walker(y, 2)
            p1s.append(phi[0])
            p2s.append(phi[1])
        assert abs(math.fsum(p1s) / len(p1s) - 0.5) < 0.05
        assert abs(math.fsum(p2s) / len(p2s) - 0.3) < 0.05


class TestBurg:
    """Pins Burg AR fitting: mean-correct intercept, positive sigma2, AR(1)/AR(2)
    coefficient recovery, cross-agreement with Yule-Walker on long series, and
    validation."""

    def test_intercept_mean_correct(self):
        y = _gen_ar([0.6], 800, 13, mean=-2.0)
        phi, c, s2 = sm.burg(y, 1)
        mu = math.fsum(y) / len(y)
        assert _close(c / (1.0 - math.fsum(phi)), mu, tol=1e-6)

    def test_sigma2_positive(self):
        y = _gen_ar([0.5, 0.2], 600, 4)
        _, _, s2 = sm.burg(y, 2)
        assert s2 > 0.0

    def test_order_too_small_raises(self):
        with pytest.raises(ValueError):
            sm.burg([1.0, 2.0, 3.0], 0)

    def test_order_too_large_raises(self):
        with pytest.raises(ValueError):
            sm.burg([1.0, 2.0, 3.0], 3)

    @slow
    def test_ar1_recovery(self):
        est = []
        for seed in range(8):
            y = _gen_ar([0.7], 2000, 50 + seed)
            phi, _, _ = sm.burg(y, 1)
            est.append(phi[0])
        assert abs(math.fsum(est) / len(est) - 0.7) < 0.03

    @slow
    def test_ar2_recovery(self):
        p1s, p2s = [], []
        for seed in range(8):
            y = _gen_ar([0.5, 0.3], 2000, 300 + seed)
            phi, _, _ = sm.burg(y, 2)
            p1s.append(phi[0])
            p2s.append(phi[1])
        assert abs(math.fsum(p1s) / len(p1s) - 0.5) < 0.05
        assert abs(math.fsum(p2s) / len(p2s) - 0.3) < 0.05

    @slow
    def test_agrees_with_yule_walker(self):
        y = _gen_ar([0.6, -0.2], 4000, 77)
        pb, _, _ = sm.burg(y, 2)
        py, _, _ = sm.yule_walker(y, 2)
        assert _series_close(pb, py, tol=0.03)


class TestArForecast:
    """Pins AR-forecast shapes, both methods, decay of a stationary forecast
    toward the series mean, coefficient recording, and validation."""

    def test_shapes(self):
        y = _gen_ar([0.6], 300, 1)
        res = sm.ar_forecast(y, 2, 10)
        assert isinstance(res, sm.ForecastResult)
        assert len(res.fitted) == len(y)
        assert len(res.residuals) == len(y)
        assert len(res.forecast) == 10

    def test_both_methods(self):
        y = _gen_ar([0.5, 0.2], 400, 2)
        for method in ("yule_walker", "burg"):
            res = sm.ar_forecast(y, 2, 5, method=method)
            assert len(res.forecast) == 5

    def test_bad_method_raises(self):
        y = _gen_ar([0.5], 100, 3)
        with pytest.raises(ValueError):
            sm.ar_forecast(y, 1, 5, method="nope")

    def test_bad_order_raises(self):
        with pytest.raises(ValueError):
            sm.ar_forecast([1.0, 2.0, 3.0], 5, 2)

    def test_params_recorded(self):
        y = _gen_ar([0.6, -0.1], 300, 8)
        res = sm.ar_forecast(y, 2, 3)
        assert "intercept" in res.params
        assert "phi1" in res.params and "phi2" in res.params

    def test_zero_horizon(self):
        y = _gen_ar([0.5], 200, 9)
        res = sm.ar_forecast(y, 1, 0)
        assert res.forecast == []
        assert res.lower is None and res.upper is None

    @slow
    def test_forecast_decays_to_mean(self):
        y = _gen_ar([0.7], 2000, 4, mean=5.0)
        mu = math.fsum(y) / len(y)
        res = sm.ar_forecast(y, 1, 200)
        fc = res.forecast
        # a stationary AR(1) forecast converges monotonically toward the mean
        assert abs(fc[-1] - mu) < abs(fc[0] - mu)
        assert abs(fc[-1] - mu) < 1e-3

    def test_intervals_widen(self):
        y = _gen_ar([0.6], 400, 6)
        res = sm.ar_forecast(y, 1, 10)
        if res.lower is not None:
            widths = [res.upper[k] - res.lower[k] for k in range(10)]
            assert all(widths[k + 1] >= widths[k] - 1e-12 for k in range(9))


class TestLjungBox:
    """Pins Q >= 0, valid p-values, white-noise vs autocorrelated behaviour,
    the chi-squared tail oracle, and validation."""

    def test_returns_pair_in_range(self):
        rng = random.Random(5)
        y = [rng.gauss(0, 1) for _ in range(200)]
        q, p = sm.ljung_box(y, 10)
        assert q >= 0.0
        assert 0.0 <= p <= 1.0

    def test_bad_lags_raise(self):
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        with pytest.raises(ValueError):
            sm.ljung_box(y, 0)
        with pytest.raises(ValueError):
            sm.ljung_box(y, 5)

    def test_chi2_sf_df2_oracle(self):
        # chi-squared with 2 dof has closed-form survival exp(-x/2).
        for x in (0.5, 1.0, 2.0, 3.7):
            assert _close(sm._chi2_sf(x, 2), math.exp(-x / 2.0), tol=1e-9)

    def test_chi2_sf_median(self):
        # median of chi2_2 is 2*ln(2); survival there is exactly 0.5.
        assert _close(sm._chi2_sf(2.0 * math.log(2.0), 2), 0.5, tol=1e-9)

    def test_chi2_sf_nonpositive(self):
        assert sm._chi2_sf(0.0, 5) == 1.0
        assert sm._chi2_sf(-1.0, 5) == 1.0

    @slow
    def test_white_noise_large_pvalue(self):
        rng = random.Random(9090)
        y = [rng.gauss(0, 1) for _ in range(3000)]
        q, p = sm.ljung_box(y, 20)
        assert p > 0.05

    @slow
    def test_autocorrelated_small_pvalue(self):
        y = _gen_ar([0.8], 3000, 12)
        q, p = sm.ljung_box(y, 20)
        assert p < 0.01

    @slow
    def test_pvalue_decreases_with_autocorrelation(self):
        rng = random.Random(4242)
        noise = [rng.gauss(0, 1) for _ in range(2000)]
        weak = _gen_ar([0.3], 2000, 1)
        strong = _gen_ar([0.85], 2000, 1)
        _, p_noise = sm.ljung_box(noise, 15)
        _, p_weak = sm.ljung_box(weak, 15)
        _, p_strong = sm.ljung_box(strong, 15)
        assert p_noise > p_weak >= p_strong


# =========================================================================
# ACCURACY METRICS
# =========================================================================
class TestMetricsPerfectForecast:
    """Pins that a perfect forecast (predicted == actual) sends every error
    metric to exactly zero -- the single most important sanity anchor."""

    def test_all_error_metrics_zero(self):
        a = [3.0, -1.5, 42.0, 0.7, 100.0]
        assert _close(sm.me(a, a), 0.0)
        assert _close(sm.mae(a, a), 0.0)
        assert _close(sm.mse(a, a), 0.0)
        assert _close(sm.rmse(a, a), 0.0)
        assert _close(sm.mpe(a, a), 0.0)
        assert _close(sm.mape(a, a), 0.0)
        assert _close(sm.smape(a, a), 0.0)

    def test_scaled_metrics_zero_on_perfect(self):
        train = [1.0, 2.0, 4.0, 7.0, 11.0, 16.0]
        a = [22.0, 29.0, 37.0]
        assert _close(sm.mase(a, a, train), 0.0)
        assert _close(sm.rmsse(a, a, train), 0.0)

    def test_forecast_metrics_all_zero_perfect(self):
        train = [1.0, 3.0, 6.0, 10.0, 15.0]
        a = [21.0, 28.0]
        d = sm.forecast_metrics(a, a, train=train, m=1)
        for k, v in d.items():
            assert _close(v, 0.0), (k, v)


class TestMetricsKnownValues:
    """Pins each metric against hand-computed oracles on the canonical tiny
    example actual=[1,2,3], predicted=[1,2,4] (error a-p = [0,0,-1])."""

    A = [1.0, 2.0, 3.0]
    P = [1.0, 2.0, 4.0]

    def test_me(self):
        # mean(a-p) = (0+0-1)/3
        assert _close(sm.me(self.A, self.P), -1.0 / 3.0)

    def test_mae(self):
        assert _close(sm.mae(self.A, self.P), 1.0 / 3.0)

    def test_mse(self):
        assert _close(sm.mse(self.A, self.P), 1.0 / 3.0)

    def test_rmse(self):
        assert _close(sm.rmse(self.A, self.P), math.sqrt(1.0 / 3.0))

    def test_mpe(self):
        # 100 * mean[(a-p)/a] = 100 * (0 + 0 + (-1/3))/3 = -100/9
        assert _close(sm.mpe(self.A, self.P), -100.0 / 9.0)

    def test_mape(self):
        # 100 * mean[|a-p|/|a|] = 100 * (1/3)/3 = 100/9
        assert _close(sm.mape(self.A, self.P), 100.0 / 9.0)

    def test_smape(self):
        # only third term nonzero: 2*1/(3+4)=2/7 ; 100*(2/7)/3 = 200/21
        assert _close(sm.smape(self.A, self.P), 200.0 / 21.0)

    def test_me_sign_convention(self):
        # under-forecast (predicted < actual) => positive me
        assert sm.me([10.0, 10.0], [8.0, 9.0]) > 0
        # over-forecast => negative me
        assert sm.me([10.0, 10.0], [12.0, 11.0]) < 0

    def test_mase_known_value_m1(self):
        # train diffs all = 1 => scale = 1 ; mae = 1/3 => mase = 1/3
        train = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _close(sm.mase(self.A, self.P, train, m=1), 1.0 / 3.0)

    def test_rmsse_known_value_m1(self):
        # train squared diffs all = 1 => scale = 1 ; mse = 1/3 => rmsse = sqrt(1/3)
        train = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _close(sm.rmsse(self.A, self.P, train, m=1), math.sqrt(1.0 / 3.0))

    def test_mase_seasonal_scale(self):
        # m=2 train=[1,3,2,4,3,5]: |tr[i]-tr[i-2]| for i=2..5 = |2-1|,|4-3|,|3-2|,|5-4| = 1,1,1,1
        # scale = 4/4 = 1 ; mae(a,p)=1/3 -> mase = 1/3
        train = [1.0, 3.0, 2.0, 4.0, 3.0, 5.0]
        assert _close(sm.mase(self.A, self.P, train, m=2), 1.0 / 3.0)

    def test_mape_larger_oracle(self):
        # actual=[100,50], predicted=[110,45]: |0.1|,|0.1| -> 100*mean = 10.0
        assert _close(sm.mape([100.0, 50.0], [110.0, 45.0]), 10.0)

    def test_smape_extreme_opposite_sign_is_200(self):
        # a=1, p=-1 => 2*2/(1+1)=2 -> 200
        assert _close(sm.smape([1.0], [-1.0]), 200.0)


class TestMetricsProperties:
    """Pins the algebraic relationships every metric family must obey:
    rmse==sqrt(mse), mae<=rmse (Jensen), |mpe|<=mape, smape in [0,200]."""

    def _pairs(self, seed, n=40):
        rng = random.Random(seed)
        a = [rng.uniform(1.0, 100.0) for _ in range(n)]
        p = [ai + rng.uniform(-20.0, 20.0) for ai in a]
        return a, p

    def test_rmse_is_sqrt_mse(self):
        for seed in range(20):
            a, p = self._pairs(seed)
            assert _close(sm.rmse(a, p), math.sqrt(sm.mse(a, p)))

    def test_mae_le_rmse(self):
        for seed in range(20):
            a, p = self._pairs(seed)
            assert sm.mae(a, p) <= sm.rmse(a, p) + 1e-9

    def test_abs_mpe_le_mape(self):
        for seed in range(20):
            a, p = self._pairs(seed)
            assert abs(sm.mpe(a, p)) <= sm.mape(a, p) + 1e-9

    def test_smape_in_range(self):
        rng = random.Random(7)
        for _ in range(50):
            n = rng.randint(1, 30)
            a = [rng.uniform(-50, 50) for _ in range(n)]
            p = [rng.uniform(-50, 50) for _ in range(n)]
            s = sm.smape(a, p)
            assert -1e-9 <= s <= 200.0 + 1e-9

    def test_smape_symmetric(self):
        rng = random.Random(11)
        for _ in range(30):
            n = rng.randint(1, 20)
            a = [rng.uniform(0.1, 100) for _ in range(n)]
            p = [rng.uniform(0.1, 100) for _ in range(n)]
            assert _close(sm.smape(a, p), sm.smape(p, a), tol=1e-9)

    def test_mae_equals_abs_me_when_bias_one_sided(self):
        # if predicted is uniformly below actual, mean error magnitude == mae
        a = [5.0, 8.0, 12.0]
        p = [4.0, 7.0, 10.0]  # all under-forecast
        assert _close(sm.mae(a, p), sm.me(a, p))

    def test_mse_nonnegative_and_ge_me_squared(self):
        # MSE = variance(error) + mean(error)^2 >= me^2
        for seed in range(15):
            a, p = self._pairs(seed)
            assert sm.mse(a, p) >= sm.me(a, p) ** 2 - 1e-9


class TestMetricsZeroHandling:
    """Pins how percentage metrics treat zero-actual entries: mpe/mape skip
    them, and degenerate to +inf when ALL actuals are zero; smape uses the
    |a|+|p| denominator and returns 0.0 only when every denom is zero."""

    def test_mpe_skips_zero_actual(self):
        # zero-actual entry dropped; remaining actual=[2,4] pred=[1,2]
        # (2-1)/2 + (4-2)/4 = 0.5 + 0.5 = 1.0 ; /2 = 0.5 ; *100 = 50
        a = [0.0, 2.0, 4.0]
        p = [9.0, 1.0, 2.0]
        assert _close(sm.mpe(a, p), 50.0)

    def test_mape_skips_zero_actual(self):
        a = [0.0, 2.0, 4.0]
        p = [9.0, 1.0, 2.0]
        # |0.5| + |0.5| = 1 ; /2 *100 = 50
        assert _close(sm.mape(a, p), 50.0)

    def test_mpe_all_zero_actual_is_inf(self):
        assert sm.mpe([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == math.inf

    def test_mape_all_zero_actual_is_inf(self):
        assert sm.mape([0.0, 0.0], [1.0, 2.0]) == math.inf

    def test_smape_all_zero_pair_is_zero(self):
        # every |a|+|p| == 0 -> no terms -> convention returns 0.0
        assert _close(sm.smape([0.0, 0.0], [0.0, 0.0]), 0.0)

    def test_smape_skips_only_the_zero_denom_term(self):
        # first point both zero (skipped); second: a=2,p=4 -> 2*2/6=2/3
        a = [0.0, 2.0]
        p = [0.0, 4.0]
        assert _close(sm.smape(a, p), 100.0 * (2.0 / 3.0))


class TestScaledMetricScaleFree:
    """Pins that MASE and RMSSE are invariant when actual, predicted AND the
    training series are all multiplied by a common positive constant."""

    def _data(self, seed):
        rng = random.Random(seed)
        train = [rng.uniform(5.0, 50.0) for _ in range(24)]
        a = [rng.uniform(5.0, 50.0) for _ in range(6)]
        p = [ai + rng.uniform(-5.0, 5.0) for ai in a]
        return train, a, p

    def test_mase_scale_free(self):
        for seed in range(10):
            train, a, p = self._data(seed)
            for m in (1, 2, 4):
                base = sm.mase(a, p, train, m=m)
                for c in (0.001, 3.7, 1000.0):
                    scaled = sm.mase([c * v for v in a], [c * v for v in p],
                                     [c * v for v in train], m=m)
                    assert _close(base, scaled, tol=1e-6 * (1 + abs(base)))

    def test_rmsse_scale_free(self):
        for seed in range(10):
            train, a, p = self._data(seed)
            for m in (1, 2, 4):
                base = sm.rmsse(a, p, train, m=m)
                for c in (0.001, 3.7, 1000.0):
                    scaled = sm.rmsse([c * v for v in a], [c * v for v in p],
                                      [c * v for v in train], m=m)
                    assert _close(base, scaled, tol=1e-6 * (1 + abs(base)))

    def test_mase_uses_correct_seasonal_scale(self):
        # independent recomputation of the m-naive scale
        rng = random.Random(3)
        train = [rng.uniform(1, 100) for _ in range(30)]
        a = [rng.uniform(1, 100) for _ in range(5)]
        p = [rng.uniform(1, 100) for _ in range(5)]
        for m in (1, 2, 3, 7):
            scale = sum(abs(train[i] - train[i - m]) for i in range(m, len(train))) / (len(train) - m)
            assert _close(sm.mase(a, p, train, m=m), sm.mae(a, p) / scale, tol=1e-9)

    def test_rmsse_uses_correct_seasonal_scale(self):
        rng = random.Random(4)
        train = [rng.uniform(1, 100) for _ in range(30)]
        a = [rng.uniform(1, 100) for _ in range(5)]
        p = [rng.uniform(1, 100) for _ in range(5)]
        for m in (1, 2, 3, 7):
            scale = sum((train[i] - train[i - m]) ** 2 for i in range(m, len(train))) / (len(train) - m)
            assert _close(sm.rmsse(a, p, train, m=m), math.sqrt(sm.mse(a, p) / scale), tol=1e-9)

    def test_scaled_metrics_constant_train_is_inf(self):
        # constant train -> zero naive scale -> +inf
        train = [7.0] * 10
        a = [1.0, 2.0]
        p = [1.5, 2.5]
        assert sm.mase(a, p, train, m=1) == math.inf
        assert sm.rmsse(a, p, train, m=1) == math.inf


class TestForecastMetricsAggregator:
    """Pins forecast_metrics: key set, delegation to the individual metrics,
    and the presence/absence of scaled metrics keyed on the train argument."""

    def test_keys_without_train(self):
        a = [1.0, 2.0, 3.0]
        p = [1.1, 1.9, 3.2]
        d = sm.forecast_metrics(a, p)
        assert set(d) == {"me", "mae", "mse", "rmse", "mpe", "mape", "smape"}
        assert "mase" not in d and "rmsse" not in d

    def test_keys_with_train(self):
        a = [1.0, 2.0, 3.0]
        p = [1.1, 1.9, 3.2]
        train = [1.0, 2.0, 3.0, 4.0, 5.0]
        d = sm.forecast_metrics(a, p, train=train)
        assert set(d) == {"me", "mae", "mse", "rmse", "mpe", "mape",
                          "smape", "mase", "rmsse"}

    def test_values_match_individual_functions(self):
        rng = random.Random(99)
        a = [rng.uniform(1, 100) for _ in range(12)]
        p = [rng.uniform(1, 100) for _ in range(12)]
        train = [rng.uniform(1, 100) for _ in range(20)]
        d = sm.forecast_metrics(a, p, train=train, m=4)
        assert _close(d["me"], sm.me(a, p))
        assert _close(d["mae"], sm.mae(a, p))
        assert _close(d["mse"], sm.mse(a, p))
        assert _close(d["rmse"], sm.rmse(a, p))
        assert _close(d["mpe"], sm.mpe(a, p))
        assert _close(d["mape"], sm.mape(a, p))
        assert _close(d["smape"], sm.smape(a, p))
        assert _close(d["mase"], sm.mase(a, p, train, m=4))
        assert _close(d["rmsse"], sm.rmsse(a, p, train, m=4))

    def test_m_defaults_to_one(self):
        a = [1.0, 2.0]
        p = [1.5, 2.5]
        train = [1.0, 2.0, 3.0, 4.0, 5.0]
        d = sm.forecast_metrics(a, p, train=train)
        assert _close(d["mase"], sm.mase(a, p, train, m=1))


class TestMetricsErrorHandling:
    """Pins the failure modes: length mismatch, empty input, and scaled
    metrics whose training series is too short for the naive scale."""

    def test_length_mismatch_raises_all(self):
        for fn in (sm.me, sm.mae, sm.mse, sm.rmse, sm.mpe, sm.mape, sm.smape):
            with pytest.raises(ValueError):
                fn([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_length_mismatch_scaled_raises(self):
        train = [1.0, 2.0, 3.0, 4.0]
        with pytest.raises(ValueError):
            sm.mase([1.0, 2.0, 3.0], [1.0, 2.0], train)
        with pytest.raises(ValueError):
            sm.rmsse([1.0, 2.0, 3.0], [1.0, 2.0], train)

    def test_empty_raises(self):
        for fn in (sm.me, sm.mae, sm.mse, sm.rmse, sm.mpe, sm.mape, sm.smape):
            with pytest.raises(ValueError):
                fn([], [])

    def test_rmse_empty_raises(self):
        with pytest.raises(ValueError):
            sm.rmse([], [])

    def test_forecast_metrics_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            sm.forecast_metrics([1.0, 2.0], [1.0])

    def test_mase_train_too_short_raises(self):
        # len(train) == m is too short (need > m)
        with pytest.raises(ValueError):
            sm.mase([1.0], [1.0], [5.0], m=1)
        with pytest.raises(ValueError):
            sm.mase([1.0, 2.0], [1.0, 2.0], [5.0, 6.0], m=2)

    def test_rmsse_train_too_short_raises(self):
        with pytest.raises(ValueError):
            sm.rmsse([1.0], [1.0], [5.0], m=1)
        with pytest.raises(ValueError):
            sm.rmsse([1.0, 2.0], [1.0, 2.0], [5.0, 6.0, 7.0], m=3)

    def test_forecast_metrics_train_too_short_raises(self):
        with pytest.raises(ValueError):
            sm.forecast_metrics([1.0, 2.0], [1.0, 2.0], train=[5.0], m=1)

    def test_nan_input_raises(self):
        with pytest.raises(ValueError):
            sm.mae([1.0, float("nan")], [1.0, 2.0])


class TestMetricsSingleAndHarsh:
    """Extra harsh corners: single-element series, exact large-error oracle,
    and cross-checking me/mae/mse consistency on a mixed-sign example."""

    def test_single_element(self):
        assert _close(sm.me([5.0], [3.0]), 2.0)
        assert _close(sm.mae([5.0], [3.0]), 2.0)
        assert _close(sm.mse([5.0], [3.0]), 4.0)
        assert _close(sm.rmse([5.0], [3.0]), 2.0)
        assert _close(sm.mpe([5.0], [3.0]), 40.0)   # (5-3)/5*100
        assert _close(sm.mape([5.0], [3.0]), 40.0)
        # smape: 2*2/(5+3)=0.5 -> 50
        assert _close(sm.smape([5.0], [3.0]), 50.0)

    def test_mixed_sign_errors_oracle(self):
        a = [10.0, -10.0, 5.0, -5.0]
        p = [12.0, -6.0, 5.0, -9.0]
        # errors a-p = [-2, -4, 0, 4]
        assert _close(sm.me(a, p), (-2 - 4 + 0 + 4) / 4.0)     # -0.5
        assert _close(sm.mae(a, p), (2 + 4 + 0 + 4) / 4.0)     # 2.5
        assert _close(sm.mse(a, p), (4 + 16 + 0 + 16) / 4.0)   # 9.0

    def test_mase_default_m_is_one(self):
        train = [1.0, 2.0, 3.0, 4.0, 5.0]
        a = [1.0, 2.0]
        p = [1.5, 2.5]
        assert _close(sm.mase(a, p, train), sm.mase(a, p, train, m=1))

    @slow
    def test_metric_consistency_battery(self):
        rng = random.Random(2024)
        for _ in range(2000):
            n = rng.randint(1, 40)
            a = [rng.uniform(1.0, 200.0) for _ in range(n)]
            p = [ai * rng.uniform(0.5, 1.5) for ai in a]
            assert sm.mae(a, p) <= sm.rmse(a, p) + 1e-9
            assert _close(sm.rmse(a, p), math.sqrt(sm.mse(a, p)))
            assert abs(sm.mpe(a, p)) <= sm.mape(a, p) + 1e-9
            assert 0.0 <= sm.smape(a, p) <= 200.0 + 1e-9
            assert _close(sm.smape(a, p), sm.smape(p, a))


# =========================================================================
# FORECAST DISPATCH, BACK-TESTING, SAMPLES & PAYLOADS
# =========================================================================
class TestForecastDispatch:
    """Pins sm.forecast: it routes to every name in FORECASTERS, honours the h
    contract, flows m=/order= through, and rejects unknown methods / bad h -- with
    hand-computed oracles for the four trivial benchmarks."""

    def test_naive_oracle(self):
        y = [1.0, 2.0, 3.0, 5.0]
        assert _series_close(sm.forecast(y, "naive", 3), [5.0, 5.0, 5.0])

    def test_mean_oracle(self):
        y = [2.0, 4.0, 6.0]
        assert _series_close(sm.forecast(y, "mean", 2), [4.0, 4.0])

    def test_drift_oracle(self):
        # slope = (7-1)/(4-1) = 2 ; extrapolate from the last point
        y = [1.0, 3.0, 5.0, 7.0]
        assert _series_close(sm.forecast(y, "drift", 2), [9.0, 11.0])

    def test_drift_single_point_falls_back_to_last(self):
        assert _series_close(sm.forecast([4.0], "drift", 3), [4.0, 4.0, 4.0])

    def test_seasonal_naive_repeats_last_season(self):
        y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        got = sm.forecast(y, "seasonal_naive", 4, m=3)
        assert _series_close(got, [4.0, 5.0, 6.0, 4.0])

    def test_seasonal_naive_needs_m_kwarg(self):
        # holt_winters/seasonal_naive require m -> forecast must pass it through
        y = list(range(1, 13))
        got = sm.forecast([float(v) for v in y], "seasonal_naive", 5, m=4)
        # last season is [9,10,11,12]; then wraps
        assert _series_close(got, [9.0, 10.0, 11.0, 12.0, 9.0])

    def test_ar_order_flows_through(self):
        y = sm.get_series("seasonal")
        got = sm.forecast(y, "ar", 3, order=2)
        assert isinstance(got, list) and len(got) == 3
        assert all(math.isfinite(v) for v in got)

    def test_every_forecaster_returns_length_h(self):
        y = sm.get_series("seasonal")
        h = 4
        kw = {
            "seasonal_naive": {"m": 12},
            "holt_winters": {"m": 12},
            "theta": {"m": 12},
            "auto_ets": {"m": 12},
            "ar": {"order": 2},
        }
        for name in sm.FORECASTERS:
            got = sm.forecast(y, name, h, **kw.get(name, {}))
            assert isinstance(got, list), name
            assert len(got) == h, name
            assert all(math.isfinite(v) for v in got), name

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            sm.forecast([1.0, 2.0, 3.0], "does_not_exist", 2)

    def test_unknown_method_raises_even_with_bad_h(self):
        # method membership is checked before h
        with pytest.raises(ValueError):
            sm.forecast([1.0, 2.0, 3.0], "nope", 0)

    def test_h_below_one_raises(self):
        for bad in (0, -1, -5):
            with pytest.raises(ValueError):
                sm.forecast([1.0, 2.0, 3.0], "naive", bad)

    def test_nan_series_rejected(self):
        with pytest.raises(ValueError):
            sm.forecast([1.0, float("nan"), 3.0], "naive", 1)


class TestForecastersRegistry:
    """Pins the FORECASTERS registry itself: expected names present, all callable,
    and each produces exactly h points on a suitable series."""

    def test_expected_names_present(self):
        expected = {"naive", "seasonal_naive", "drift", "mean", "ses", "holt",
                    "holt_damped", "holt_winters", "theta", "auto_ets", "ar"}
        assert expected <= set(sm.FORECASTERS)

    def test_all_values_callable(self):
        assert all(callable(fn) for fn in sm.FORECASTERS.values())

    def test_holt_damped_is_damped(self):
        # holt_damped must differ from plain holt on a clear trend (damping bends
        # the extrapolation), and both must be length h.
        y = sm.get_series("trend")
        plain = sm.forecast(y, "holt", 8)
        damped = sm.forecast(y, "holt_damped", 8)
        assert len(plain) == len(damped) == 8
        assert not _series_close(plain, damped, tol=1e-6)


class TestRollingOriginBacktest:
    """Pins rolling_origin_backtest: origin count matches the range arithmetic,
    both modes run, the metrics dict has the full key-set, and it rejects
    impossible configs -- plus the headline Holt-Winters > naive/mean check."""

    def test_n_origins_matches_range_arithmetic(self):
        y = [float(i) for i in range(20)]
        bt = sm.rolling_origin_backtest(y, "naive", h=2, initial=10, step=1)
        # origins = range(10, 20-2+1, 1) = 10..18 -> 9 origins
        assert bt["n_origins"] == len(range(10, 20 - 2 + 1, 1)) == 9

    def test_n_origins_with_step(self):
        y = [float(i) for i in range(30)]
        bt = sm.rolling_origin_backtest(y, "naive", h=3, initial=6, step=4)
        assert bt["n_origins"] == len(range(6, 30 - 3 + 1, 4))

    def test_metrics_key_set(self):
        y = [float(i) for i in range(24)]
        bt = sm.rolling_origin_backtest(y, "naive", h=2, initial=10, m=1)
        expected = {"me", "mae", "mse", "rmse", "mpe", "mape", "smape",
                    "mase", "rmsse"}
        assert set(bt["metrics"]) == expected

    def test_top_level_keys(self):
        y = [float(i) for i in range(24)]
        bt = sm.rolling_origin_backtest(y, "naive", h=2, initial=10)
        assert set(bt) == {"method", "h", "mode", "n_origins", "metrics", "per_origin"}
        assert bt["method"] == "naive"
        assert bt["h"] == 2
        assert bt["mode"] == "expanding"
        assert len(bt["per_origin"]) == bt["n_origins"]

    def test_per_origin_rmse_matches_direct(self):
        y = [float(i) for i in range(15)]
        bt = sm.rolling_origin_backtest(y, "naive", h=2, initial=8, step=1)
        for rec in bt["per_origin"]:
            assert _close(rec["rmse"], sm.rmse(rec["actual"], rec["pred"]))

    def test_expanding_and_sliding_both_run(self):
        y = sm.get_series("trend")
        exp = sm.rolling_origin_backtest(y, "drift", h=2, initial=20, mode="expanding")
        sli = sm.rolling_origin_backtest(y, "drift", h=2, initial=20, mode="sliding",
                                         window=15)
        assert exp["n_origins"] == sli["n_origins"] > 0
        # a sliding window ignores older data -> generally different scores
        assert exp["mode"] == "expanding" and sli["mode"] == "sliding"

    def test_naive_expanding_score_is_exact_on_linear(self):
        # For a perfectly linear series, one-step naive always lags by the constant
        # slope, so every abs error == slope and RMSE == slope exactly.
        slope = 3.0
        y = [5.0 + slope * i for i in range(12)]
        bt = sm.rolling_origin_backtest(y, "naive", h=1, initial=6, step=1)
        assert _close(bt["metrics"]["rmse"], slope)
        assert _close(bt["metrics"]["mae"], slope)

    def test_bad_mode_raises(self):
        y = [float(i) for i in range(20)]
        with pytest.raises(ValueError):
            sm.rolling_origin_backtest(y, "naive", h=1, mode="bogus")

    def test_initial_plus_h_exceeds_n_raises(self):
        y = [float(i) for i in range(10)]
        with pytest.raises(ValueError):
            sm.rolling_origin_backtest(y, "naive", h=2, initial=9)

    def test_all_origins_error_raises(self):
        # holt_winters with m=1 -> seasonal_periods=1 invalid at every origin
        y = [float(i) for i in range(24)]
        with pytest.raises(ValueError):
            sm.rolling_origin_backtest(y, "holt_winters", h=2, initial=12, m=1)

    @slow
    def test_holt_winters_beats_naive_and_mean_on_airline(self):
        air = sm.get_series("airline")
        hw = sm.rolling_origin_backtest(air, "holt_winters", h=6, m=12,
                                        seasonal="mul")
        nv = sm.rolling_origin_backtest(air, "naive", h=6, m=12)
        mn = sm.rolling_origin_backtest(air, "mean", h=6, m=12)
        assert hw["metrics"]["rmse"] < nv["metrics"]["rmse"]
        assert hw["metrics"]["rmse"] < mn["metrics"]["rmse"]


class TestTimeSeriesCV:
    """Pins time_series_cv: returns {method: score} for the requested methods, and
    a method that errors at every origin scores +inf instead of crashing."""

    def test_returns_score_per_method(self):
        y = sm.get_series("trend")
        methods = ["naive", "drift", "mean"]
        scores = sm.time_series_cv(y, methods, h=2, initial=20, step=2)
        assert set(scores) == set(methods)
        assert all(math.isfinite(v) for v in scores.values())

    def test_drift_beats_naive_on_a_trend(self):
        y = sm.get_series("trend")
        scores = sm.time_series_cv(y, ["naive", "drift"], h=3, initial=20)
        assert scores["drift"] < scores["naive"]

    def test_failing_method_scores_inf_not_crash(self):
        # m=1 makes holt_winters invalid everywhere; the comparison must survive.
        y = sm.get_series("trend")
        scores = sm.time_series_cv(y, ["naive", "holt_winters"], h=2, initial=20, m=1)
        assert math.isfinite(scores["naive"])
        assert scores["holt_winters"] == float("inf")

    def test_metric_selection(self):
        y = sm.get_series("trend")
        s_rmse = sm.time_series_cv(y, ["naive"], h=2, initial=20, metric="rmse")
        s_mae = sm.time_series_cv(y, ["naive"], h=2, initial=20, metric="mae")
        # different metrics generally give different scores
        assert s_rmse["naive"] != s_mae["naive"]

    def test_bad_metric_gives_inf(self):
        # metrics dict has no such key -> .get(..., _INF) -> +inf
        y = sm.get_series("trend")
        scores = sm.time_series_cv(y, ["naive"], h=2, initial=20, metric="not_a_metric")
        assert scores["naive"] == float("inf")


class TestSampleSeries:
    """Pins the built-in demo data: documented lengths, deterministic generation,
    all-callable registry, and a clear error on unknown names."""

    def test_documented_lengths(self):
        assert len(sm.get_series("airline")) == 96
        assert len(sm.get_series("trend")) == 60
        assert len(sm.get_series("seasonal")) == 72
        assert len(sm.get_series("random_walk")) == 100
        assert len(sm.get_series("noisy_sine")) == 120

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            sm.get_series("no_such_series")

    def test_all_names_deterministic(self):
        for name in sm.SAMPLE_SERIES:
            a = sm.get_series(name)
            b = sm.get_series(name)
            assert _series_close(a, b, tol=0.0)

    def test_registry_all_callable_and_nonempty(self):
        assert set(sm.SAMPLE_SERIES) >= {"airline", "trend", "seasonal",
                                         "random_walk", "noisy_sine"}
        for name, fn in sm.SAMPLE_SERIES.items():
            assert callable(fn)
            out = fn()
            assert isinstance(out, list) and len(out) > 0

    def test_airline_is_positive(self):
        # multiplicative demo data must stay strictly positive
        assert all(v > 0 for v in sm.get_series("airline"))


class TestPayloads:
    """Pins the JSON payload builders: output is strict JSON (json.dumps succeeds)
    and every non-finite float is coerced to None."""

    def test_forecast_payload_is_json_serialisable(self):
        import json
        r = sm.ses(sm.get_series("random_walk"), h=3)
        payload = sm.build_forecast_payload(r)
        json.dumps(payload)  # must not raise
        assert payload["method"] == r.method
        assert len(payload["forecast"]) == 3

    def test_forecast_payload_maps_nonfinite_to_none(self):
        import json
        r = sm.ForecastResult(
            method="X", fitted=[1.0, 2.0], residuals=[0.0, 0.0],
            forecast=[float("inf"), float("nan"), 3.0],
            lower=None, upper=None, params={"a": float("inf")},
            sse=float("nan"), sigma=float("inf"), aic=None, aicc=None)
        payload = sm.build_forecast_payload(r)
        json.dumps(payload)  # strict JSON has no inf/nan
        assert payload["forecast"] == [None, None, 3.0]
        assert payload["sse"] is None
        assert payload["sigma"] is None
        assert payload["params"]["a"] is None

    def test_forecast_payload_intervals_preserved(self):
        import json
        r = sm.holt(sm.get_series("trend"), h=4)
        payload = sm.build_forecast_payload(r)
        json.dumps(payload)
        if r.lower is not None:
            assert len(payload["lower"]) == len(r.forecast)
            assert len(payload["upper"]) == len(r.forecast)

    def test_decomposition_payload_is_json_serialisable(self):
        import json
        dec = sm.classical_decomposition(sm.get_series("airline"), 12,
                                         model="multiplicative")
        payload = sm.build_decomposition_payload(dec)
        json.dumps(payload)  # trend/resid Nones must serialise fine
        assert payload["model"] == "multiplicative"
        assert payload["period"] == 12
        assert len(payload["observed"]) == len(dec.observed)

    def test_decomposition_payload_maps_nonfinite_to_none(self):
        import json
        dec = sm.Decomposition(
            observed=[1.0, 2.0], trend=[None, None],
            seasonal=[float("nan"), 1.0], resid=[None, float("inf")],
            model="additive", period=2)
        payload = sm.build_decomposition_payload(dec)
        json.dumps(payload)
        assert payload["seasonal"] == [None, 1.0]
        assert payload["resid"] == [None, None]
        # seasonally_adjusted = observed - seasonal = [nan, 1.0] -> [None, 1.0]
        assert payload["seasonally_adjusted"] == [None, 1.0]

    def test_decomposition_payload_none_trend_preserved(self):
        import json
        dec = sm.classical_decomposition(sm.get_series("seasonal"), 12)
        payload = sm.build_decomposition_payload(dec)
        json.dumps(payload)
        # classical decomposition leaves endpoints (trend None) -> serialise as null
        assert any(v is None for v in payload["trend"])


# ===========================================================================
# COVERAGE-GAP HARDENING  (untested code paths / weak assertions found by a
# post-authoring coverage critic -- exact oracles where possible)
# ===========================================================================
class TestCoverageGaps:
    """Pins code paths the per-area sections missed: the chi-squared
    continued-fraction branch, ljung_box's dof, AR residual identity, degenerate
    PACF, the multiplicative-seasonal guard, seasonally_adjusted's mul path,
    inv_box_cox's negative-lambda / empty cases, NaN rejection, and the
    hodrick_prescott lambda-validation order."""

    def test_chi2_sf_continued_fraction_branch(self):
        # for df=2 the chi-squared survival function is exactly exp(-x/2); x=20
        # forces the x >= a+1 continued-fraction branch of _gammaincc
        assert _close(sm._chi2_sf(20.0, 2), math.exp(-10.0), tol=1e-9)
        assert _close(sm._chi2_sf(8.0, 2), math.exp(-4.0), tol=1e-9)
        assert _close(sm._chi2_sf(1.0, 2), math.exp(-0.5), tol=1e-9)   # series branch
        assert _close(sm._chi2_sf(0.0, 5), 1.0)
        # survival is monotone decreasing in x and increasing in df (fixed x)
        assert sm._chi2_sf(5.0, 2) > sm._chi2_sf(9.0, 2)
        assert sm._chi2_sf(9.0, 2) < sm._chi2_sf(9.0, 8)

    def test_ljung_box_dof_reduces_pvalue(self):
        y = sm.get_series("noisy_sine")
        q0, p0 = sm.ljung_box(y, lags=10, dof=0)
        q3, p3 = sm.ljung_box(y, lags=10, dof=3)
        assert _close(q0, q3)                     # dof does not change Q
        assert p3 <= p0 + 1e-12                    # fewer df -> smaller tail p
        # dof >= lags floors the degrees of freedom at 1 (no crash / no zero df)
        _, pf = sm.ljung_box(y, lags=5, dof=9)
        assert 0.0 <= pf <= 1.0

    def test_ar_forecast_residual_identity_and_seed(self):
        r = _gen_ar([0.6], 200, 11)
        res = sm.ar_forecast(r, 2, 4, method="yule_walker")
        assert len(res.fitted) == len(r) and len(res.residuals) == len(r)
        assert all(_close(res.residuals[i], r[i] - res.fitted[i]) for i in range(len(r)))
        assert res.fitted[0] == r[0] and res.fitted[1] == r[0]   # first `order` seeded

    def test_partial_autocorrelation_constant_series(self):
        # degenerate (zero-variance) series exercises the |den| ~ 0 -> 0 branch
        assert _series_close(sm.partial_autocorrelation([5.0] * 6, 3), [1.0, 0.0, 0.0, 0.0])

    def test_multiplicative_seasonal_zero_level_guard(self):
        # first-season mean is exactly zero -> multiplicative init must raise
        y = [1.0, -1.0, 1.0, -1.0, 2.0, -2.0, 2.0, -2.0]   # mean of first 4 == 0
        with pytest.raises(ValueError):
            sm.exponential_smoothing(y, seasonal="mul", seasonal_periods=4)

    def test_seasonally_adjusted_multiplicative(self):
        dec = sm.classical_decomposition(sm.get_series("airline"), 12,
                                         model="multiplicative")
        adj = dec.seasonally_adjusted()
        assert all(_close(adj[t], dec.observed[t] / dec.seasonal[t])
                   for t in range(len(adj)) if dec.seasonal[t])

    def test_inv_box_cox_negative_lambda_and_empty(self):
        for lam in (-1.0, -0.5, -0.25):
            y = [0.5, 1.0, 2.0, 5.0]
            assert _series_close(sm.inv_box_cox(sm.box_cox(y, lam), lam), y)
        assert sm.inv_box_cox([], 1.0) == []
        assert sm.inv_box_cox([], 0.0) == []

    def test_nan_rejected_across_families(self):
        bad = [1.0, float("nan"), 3.0, 4.0]
        for fn in (lambda: sm.autocovariance(bad, 1),
                   lambda: sm.autocorrelation(bad, 1),
                   lambda: sm.lowess(bad, 0.5),
                   lambda: sm.simple_moving_average(bad, 2),
                   lambda: sm.exponential_moving_average(bad, 0.3),
                   lambda: sm.yule_walker(bad, 1)):
            with pytest.raises(ValueError):
                fn()

    def test_auto_ets_non_positive_series_no_crash(self):
        # a series with negatives must not attempt multiplicative seasonality
        y = [3.0 * math.sin(2 * math.pi * t / 12) for t in range(48)]
        res = sm.auto_ets(y, seasonal_periods=12, h=6)
        assert len(res.forecast) == 6
        assert not res.method.endswith("M)")     # no multiplicative seasonal chosen

    def test_hodrick_prescott_validates_lambda_before_length(self):
        # negative lambda is invalid input and must raise even for a short series
        with pytest.raises(ValueError):
            sm.hodrick_prescott([1.0, 2.0], lam=-1.0)
        # a valid short series still short-circuits to a copy
        assert sm.hodrick_prescott([1.0, 2.0], lam=100.0) == [1.0, 2.0]


# ===========================================================================
# BUG-HUNT ROUND 1  (adversarial find -> skeptical verify; fixes for genuine bugs)
# ===========================================================================
class TestBugHuntRound1:
    """R1 fixes.  (1) exponential_smoothing SILENTLY DROPPED a caller-fixed
    phi/beta/gamma when the matching damped/trend/seasonal flag was not set
    (contradicting the docstring's 'any of alpha/beta/gamma/phi may be fixed') --
    now it raises.  (2) Caller-fixed smoothing params BYPASSED the (0,1] box, so
    alpha=5 produced divergent nonsense instead of raising (unlike
    exponential_moving_average) -- now validated.  (3) yule_walker (the default AR
    method) CRASHED with 'singular linear system' on a constant / zero-variance
    series while burg handled it -- now the degenerate AR is flat at the mean."""

    Y = [10.0, 12.0, 15.0, 20.0, 24.0, 30.0]

    def test_fixed_phi_requires_damped(self):
        with pytest.raises(ValueError, match="phi"):
            sm.exponential_smoothing(self.Y, trend="add", phi=0.5, alpha=0.4,
                                     beta=0.2, optimize=False, h=2)
        # but a fixed phi IS honoured when damped=True (== the damped forecast)
        got = sm.exponential_smoothing(self.Y, trend="add", damped=True, phi=0.5,
                                       alpha=0.4, beta=0.2, optimize=False, h=3)
        assert "phi" in got.params and _close(got.params["phi"], 0.5)
        assert _series_close(got.forecast, [24.6136, 24.9419, 25.1061], tol=1e-3)

    def test_fixed_beta_requires_trend_and_gamma_requires_seasonal(self):
        with pytest.raises(ValueError, match="beta"):
            sm.exponential_smoothing(self.Y, beta=0.5, alpha=0.4, optimize=False)
        with pytest.raises(ValueError, match="gamma"):
            sm.exponential_smoothing(self.Y, gamma=0.5, alpha=0.4, optimize=False)
        # zero smoothing for a present component is legitimate (frozen init) -> no raise
        sm.holt(self.Y, alpha=0.4, beta=0.0, optimize=False, h=1)
        sm.holt_winters([1.0, 2.0, 3.0, 4.0] * 3, 4, alpha=0.3, beta=0.1,
                        gamma=0.0, optimize=False, h=1)

    def test_fixed_params_must_be_in_admissible_range(self):
        for kw in ({"alpha": 5.0}, {"alpha": 0.0}, {"alpha": -0.1}):
            with pytest.raises(ValueError):
                sm.ses(self.Y, optimize=False, h=2, **kw)
        with pytest.raises(ValueError):
            sm.holt(self.Y, alpha=0.4, beta=3.0, optimize=False)
        with pytest.raises(ValueError):
            sm.exponential_smoothing(self.Y, trend="add", damped=True, phi=1.5,
                                     alpha=0.4, beta=0.2, optimize=False)
        # boundary alpha=1 / phi=1 are valid
        sm.ses(self.Y, alpha=1.0, optimize=False, h=1)
        sm.exponential_smoothing(self.Y, trend="add", damped=True, phi=1.0,
                                 alpha=0.4, beta=0.2, optimize=False)

    def test_yule_walker_constant_series_is_flat_at_mean(self):
        c = [5.0] * 30
        # yule_walker (the default) used to crash 'singular linear system'
        phi, intercept, sigma2 = sm.yule_walker(c, 2)
        assert phi == [0.0, 0.0] and _close(intercept, 5.0) and _close(sigma2, 0.0)
        r_yw = sm.ar_forecast(c, 2, 3, method="yule_walker")
        r_bg = sm.ar_forecast(c, 2, 3, method="burg")
        assert _series_close(r_yw.forecast, [5.0, 5.0, 5.0])
        assert _series_close(r_yw.forecast, r_bg.forecast)   # methods agree now

    def test_yule_walker_on_real_ar_unaffected(self):
        # the zero-variance guard must not perturb a genuine AR fit
        import random as _r
        rng = _r.Random(0)
        x = [0.0]
        for _ in range(1200):
            x.append(0.6 * x[-1] + rng.gauss(0, 1))
        phi, _, sig2 = sm.yule_walker(x[300:], 1)
        assert 0.5 < phi[0] < 0.7 and sig2 > 0

    # ---- resume-phase finds (auto_ets / theta / scale-invariance / dispatch) ----

    def test_auto_ets_selects_seasonal_on_seasonal_data(self):
        # auto_ets used to rank on MIXED scales (seasonal models' aicc=None fell
        # back to raw SSE ~1e-31, always beaten by a non-seasonal negative AICc),
        # so it picked a runaway non-seasonal trend on a perfectly seasonal series
        y = [0.5 + 0.4 * math.sin(2 * math.pi * t / 12) for t in range(28)]
        res = sm.auto_ets(y, seasonal_periods=12, h=12)
        assert res.method.endswith("A)") or res.method.endswith("M)")   # seasonal
        assert all(0.0 < v < 1.1 for v in res.forecast)                 # bounded, not runaway
        # the fix: a seasonal fit now has a finite (comparable) AICc, not None
        seas = sm.exponential_smoothing(y, seasonal="add", seasonal_periods=12, h=1)
        assert seas.aicc is not None and math.isfinite(seas.aicc)

    def test_theta_method_non_positive_seasonal_no_crash(self):
        # hard-coded multiplicative decomposition crashed on any non-positive value
        y = [(-2.0 + 0.1 * t) + 3.0 * math.sin(2 * math.pi * t / 4) for t in range(24)]
        res = sm.theta_method(y, h=4, seasonal_periods=4)   # additive deseasonalise
        assert len(res.forecast) == 4 and all(math.isfinite(v) for v in res.forecast)

    def test_theta_method_period_one_is_non_seasonal(self):
        # seasonal_periods=1 must route to the plain Theta path, not crash
        y = [10.0 + 2.0 * t + math.sin(t) for t in range(20)]
        assert _series_close(sm.theta_method(y, h=3, seasonal_periods=1).forecast,
                             sm.theta_method(y, h=3, seasonal_periods=None).forecast)

    def test_autocorrelation_is_scale_invariant(self):
        import random as _r
        rng = _r.Random(5)
        base = [0.0]
        for _ in range(250):
            base.append(0.8 * base[-1] + rng.gauss(0, 1))
        base = base[50:]
        sensor = [5.0 + 3e-7 * v for v in base]      # tiny magnitude, real structure
        big = [5.0 + 1e-2 * v for v in base]
        assert _close(sm.autocorrelation(sensor, 5)[1],
                      sm.autocorrelation(big, 5)[1], tol=1e-6)
        assert sm.ljung_box(sensor, 10)[1] < 0.01    # not spuriously "white"
        # a genuinely constant series is still white noise
        assert _series_close(sm.autocorrelation([5.0] * 40, 3), [1.0, 0.0, 0.0, 0.0])

    def test_scaled_metrics_are_scale_free_at_small_magnitude(self):
        tr = [2.0e-6 + 1e-7 * i for i in range(20)]
        a, p = [2.6e-6, 2.7e-6], [2.55e-6, 2.68e-6]
        c = 1e6
        assert _close(sm.rmsse(a, p, tr),
                      sm.rmsse([x * c for x in a], [x * c for x in p], [x * c for x in tr]))
        assert _close(sm.mase(a, p, tr),
                      sm.mase([x * c for x in a], [x * c for x in p], [x * c for x in tr]))
        assert math.isfinite(sm.rmsse(a, p, tr)) and math.isfinite(sm.mase(a, p, tr))
        # a truly constant / zero training series still gives the inf scale
        assert sm.rmsse([1.0, 2.0], [1.0, 2.0], [5.0] * 10) == float("inf")
        assert sm.mase([1.0, 2.0], [1.0, 2.0], [0.0] * 10) == float("inf")

    def test_backtest_unknown_method_raises_catchable_valueerror(self):
        with pytest.raises(ValueError):
            sm.rolling_origin_backtest(sm.get_series("airline"), "nonesuch", h=3)
        # and time_series_cv survives a bad method (scores it +inf, ranks the rest)
        scores = sm.time_series_cv(sm.get_series("airline"), ["naive", "nonesuch"], h=3)
        assert scores["nonesuch"] == float("inf") and math.isfinite(scores["naive"])

    def test_theta_backtests_under_default_m(self):
        # rolling_origin_backtest defaults m=1; theta with m=1 must not crash it
        bt = sm.rolling_origin_backtest(sm.get_series("trend"), "theta", h=3)
        assert bt["n_origins"] > 0 and math.isfinite(bt["metrics"]["rmse"])


class TestBugHuntRound2:
    """R2 -- the recurring 'absolute threshold on a scale-dependent quantity' class
    from R1, hunted deeper.  Two fixes: yule_walker built the Yule-Walker system on
    the (variance-scaled) COVARIANCE matrix, so tiny-magnitude data (~1e-8) had
    entries below _solve's 1e-15 pivot floor and CRASHED 'singular linear system'
    (ar_forecast's default inherited it); burg's dk (a sum of squares) hit the
    absolute _EPS threshold and SILENTLY returned an all-zero fit.  Both now
    normalise (correlation matrix / unit-scale) so the AR fit is scale-invariant,
    as it must be.  The z-quantile and damped-phi=1 identities are pinned too."""

    @staticmethod
    def _ar_series(phi1, n, seed, scale=1.0, level=5.0):
        import random as _r
        rng = _r.Random(seed)
        x = [0.0]
        for _ in range(n + 200):
            x.append(phi1 * x[-1] + rng.gauss(0, 1))
        return [scale * (level + v) for v in x[200:]]

    def test_ar_fit_is_scale_invariant(self):
        base = self._ar_series(0.7, 500, 0)
        ref_yw = sm.yule_walker(base, 2)
        ref_bg = sm.burg(base, 2)
        for scale in (1e-10, 1e-6, 1.0, 1e6, 1e10):
            y = [scale * v for v in base]
            yw, bg = sm.yule_walker(y, 2), sm.burg(y, 2)
            assert _series_close(yw[0], ref_yw[0], tol=1e-7)   # phi scale-free
            assert _series_close(bg[0], ref_bg[0], tol=1e-7)
            # sigma2 scales as the square of the data scale
            if ref_yw[2] > 0:
                assert abs(yw[2] / (ref_yw[2] * scale * scale) - 1.0) < 1e-6

    def test_ar_fit_tiny_scale_does_not_crash_or_zero(self):
        y = self._ar_series(0.7, 400, 1, scale=1e-8)
        # yule_walker used to raise 'singular linear system' here
        phi_yw = sm.yule_walker(y, 2)[0]
        phi_bg = sm.burg(y, 2)[0]                 # burg used to return [0, 0]
        assert 0.5 < phi_yw[0] < 0.85 and abs(phi_bg[0] - phi_yw[0]) < 0.05
        fc = sm.ar_forecast(y, 2, 3).forecast     # default (yule_walker) must work
        assert len(fc) == 3 and all(math.isfinite(v) and v != 0.0 for v in fc)

    def test_ar_fit_constant_and_recovery(self):
        assert sm.yule_walker([5.0] * 30, 2) == ([0.0, 0.0], 5.0, 0.0)
        assert sm.burg([5.0] * 30, 2) == ([0.0, 0.0], 5.0, 0.0)
        base = self._ar_series(0.7, 800, 3)
        assert 0.6 < sm.yule_walker(base, 1)[0][0] < 0.75
        assert 0.6 < sm.burg(base, 1)[0][0] < 0.75

    def test_z_score_matches_normal_quantiles(self):
        for level, want in [(0.80, 1.28155), (0.90, 1.64485), (0.95, 1.95996),
                            (0.99, 2.57583), (0.999, 3.29053)]:
            assert _close(sm._z_score(level), want, tol=1e-2)
        with pytest.raises(ValueError):
            sm._z_score(1.0)
        with pytest.raises(ValueError):
            sm._z_score(0.0)

    def test_damped_phi_one_equals_undamped_holt(self):
        # phi=1 removes the damping, so a damped model with phi=1 is exactly Holt
        y = [10.0 + 2.0 * t + math.sin(t) for t in range(30)]
        undamped = sm.holt(y, alpha=0.4, beta=0.2, damped=False, optimize=False, h=6)
        damped1 = sm.exponential_smoothing(y, trend="add", damped=True, phi=1.0,
                                           alpha=0.4, beta=0.2, optimize=False, h=6)
        assert _series_close(undamped.forecast, damped1.forecast, tol=1e-9)


class TestBugHuntRound3:
    """R3 -- METAMORPHIC / invariance modality (a CLEAN round: no fix).  Linear
    smoothers must satisfy f(y+c)=f(y)+c and f(a*y)=a*f(y) EXACTLY; centred
    smoothers must be time-reversal symmetric; classical decomposition and the ETS
    engine (through the optimiser) must be shift/scale equivariant; ACF must be
    shift/scale/reversal invariant.  These strong properties are pinned as guards."""

    Y = [10.0 + math.sin(t / 3.0) + (0.4 if t % 5 else -0.6) for t in range(40)]
    LINEAR = {
        "sma": lambda z: sm.simple_moving_average(z, 5),
        "sma_center": lambda z: sm.simple_moving_average(z, 5, center=True),
        "wma": lambda z: sm.weighted_moving_average(z, [1.0, 2.0, 3.0]),
        "ema": lambda z: sm.exponential_moving_average(z, 0.3),
        "gaussian": lambda z: sm.gaussian_filter(z, 2.0),
        "savgol": lambda z: sm.savitzky_golay(z, 7, 2),
        "lowess": lambda z: sm.lowess(z, 0.3),
        "hp": lambda z: sm.hodrick_prescott(z, 100.0),
        "median": lambda z: sm.median_filter(z, 5),
    }

    def test_linear_smoothers_are_shift_and_scale_equivariant(self):
        for name, f in self.LINEAR.items():
            base = f(self.Y)
            assert _series_close(f([v + 100.0 for v in self.Y]),
                                 [b + 100.0 for b in base], tol=1e-6), name + " shift"
            assert _series_close(f([3.0 * v for v in self.Y]),
                                 [3.0 * b for b in base], tol=1e-6), name + " scale"

    def test_centered_smoothers_are_time_reversal_symmetric(self):
        for name, f in {"sma_center": lambda z: sm.simple_moving_average(z, 5, center=True),
                        "gaussian": lambda z: sm.gaussian_filter(z, 2.0),
                        "savgol": lambda z: sm.savitzky_golay(z, 7, 2),
                        "median": lambda z: sm.median_filter(z, 5),
                        "lowess": lambda z: sm.lowess(z, 0.3)}.items():
            assert _series_close(f(list(reversed(self.Y))),
                                 list(reversed(f(self.Y))), tol=1e-6), name

    def test_classical_decomposition_is_shift_and_scale_equivariant(self):
        ys = sm.get_series("seasonal")
        d0 = sm.classical_decomposition(ys, 12, model="additive")
        d1 = sm.classical_decomposition([v + 100.0 for v in ys], 12, model="additive")
        for t in range(len(ys)):
            if d0.trend[t] is not None:
                assert _close(d1.trend[t] - d0.trend[t], 100.0, tol=1e-6)
                assert _close(d1.resid[t], d0.resid[t], tol=1e-6)
        assert _series_close(d1.seasonal, d0.seasonal, tol=1e-6)
        assert _series_close(d1.seasonally_adjusted(),
                             [v + 100.0 for v in d0.seasonally_adjusted()], tol=1e-6)
        # multiplicative decomposition is scale-equivariant in the trend
        air = sm.get_series("airline")
        m0 = sm.classical_decomposition(air, 12, model="multiplicative")
        m1 = sm.classical_decomposition([7.0 * v for v in air], 12, model="multiplicative")
        assert _series_close(m1.seasonal, m0.seasonal, tol=1e-6)
        for t in range(len(air)):
            if m0.trend[t] is not None:
                assert abs(m1.trend[t] - 7.0 * m0.trend[t]) <= 1e-6 * abs(7.0 * m0.trend[t])

    def test_ets_is_shift_invariant_through_the_optimizer(self):
        # optimising on a shift-invariant SSE must yield the same params, so the
        # forecast shifts by exactly c -- a strong end-to-end optimiser check
        y = self.Y
        for f in (lambda z: sm.ses(z, h=5),
                  lambda z: sm.holt(z, h=5)):
            a = f(y).forecast
            b = f([v + 100.0 for v in y]).forecast
            assert _series_close(b, [x + 100.0 for x in a], tol=1e-5)
        ys = sm.get_series("seasonal")
        a = sm.holt_winters(ys, 12, seasonal="add", h=6).forecast
        b = sm.holt_winters([v + 100.0 for v in ys], 12, seasonal="add", h=6).forecast
        assert _series_close(b, [x + 100.0 for x in a], tol=1e-5)

    def test_multiplicative_hw_is_scale_equivariant(self):
        # forecast(c*y) == c*forecast(y) to optimiser precision.  Compare with a
        # RELATIVE tolerance: the values are ~1e3, and Nelder-Mead's converged
        # params differ only at ~1e-6 (the objective magnitude scales with c^2),
        # so an absolute tolerance would be the wrong test.
        air = sm.get_series("airline")
        a = sm.holt_winters(air, 12, seasonal="mul", h=12).forecast
        b = sm.holt_winters([7.0 * v for v in air], 12, seasonal="mul", h=12).forecast
        assert all(abs(b[i] - 7.0 * a[i]) <= 1e-5 * abs(7.0 * a[i]) for i in range(12))

    def test_acf_is_reversal_shift_and_scale_invariant(self):
        y = self.Y
        acf = sm.autocorrelation(y, 8)
        assert _series_close(sm.autocorrelation(list(reversed(y)), 8), acf, tol=1e-9)
        assert _series_close(sm.autocorrelation([v + 100.0 for v in y], 8), acf, tol=1e-9)
        assert _series_close(sm.autocorrelation([3.0 * v for v in y], 8), acf, tol=1e-9)

    def test_difference_order_two_round_trips(self):
        y = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0]
        for lag in (1, 2):
            d1 = sm.difference(y, lag, 1)
            d2 = sm.difference(d1, lag, 1)
            rec = sm.integrate(sm.integrate(d2, d1[:lag], lag), y[:lag], lag)
            assert _series_close(rec, y)

    def test_degenerate_windows_and_max_order(self):
        w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        assert sm.savitzky_golay(w, 1, 0) == w                 # window 1 == identity
        assert all(math.isfinite(v) for v in sm.savitzky_golay(w, 5, 9))   # poly>=window
        z = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0]
        for phi in (sm.yule_walker(z, len(z) - 1)[0], sm.burg(z, len(z) - 1)[0]):
            assert len(phi) == len(z) - 1 and all(math.isfinite(v) for v in phi)
        assert len(sm.autocorrelation(z, len(z) - 1)) == len(z)

    @slow
    def test_ses_prediction_interval_coverage_on_random_walk(self):
        # SES is optimal for a random walk, where the h-step error std is
        # sigma*sqrt(h) -- so the 95% residual-based interval should cover ~95%
        import random as _r
        rng = _r.Random(1)
        covered = total = 0
        for _ in range(400):
            walk = [0.0]
            for _ in range(60):
                walk.append(walk[-1] + rng.gauss(0, 1))
            r = sm.ses(walk[:-5], h=5, level=0.95)
            for k in range(5):
                total += 1
                if r.lower[k] <= walk[-5 + k] <= r.upper[k]:
                    covered += 1
        assert 0.90 <= covered / total <= 0.98


class TestMutationHardening:
    """Gaps found by a mutation-testing pass (R4).  One is a REAL BUG:
    stl_decomposition, started from a zero trend, let the trend LEAK into the
    seasonal component on trended data -- the trend collapsed to the global mean
    and the 'seasonal' absorbed the growth (garbage decomposition on the common
    trend+season case).  Fixed by seeding the trend with a centred moving average.
    The rest close genuine test gaps the survivors exposed: chi2_sf accuracy at
    df=4/6 (df=2 alone leaves lgamma(a)=0, hiding a normalisation mutant), burg's
    recursion vs Yule-Walker, the gaussian kernel shape, the AICc formula vs an
    independent oracle, _var/_std, the exact prediction-interval width, and the
    ETS one-step recursion (additive & multiplicative)."""

    def test_stl_recovers_trend_and_seasonal_on_trended_data(self):
        seas = [3.0, -1.0, -2.0, 0.0]
        y = [10.0 + 0.5 * t + seas[t % 4] for t in range(40)]
        d = sm.stl_decomposition(y, 4)
        # the seasonal must be the injected pattern, NOT trend leakage
        assert all(abs(d.seasonal[i] - seas[i % 4]) < 0.5 for i in range(4))
        # the trend must track the line 10 + 0.5t, not collapse to the mean
        assert abs(d.trend[10] - (10.0 + 0.5 * 10)) < 1.0
        assert d.trend[30] - d.trend[5] > 8.0        # genuinely increasing
        assert all(abs(y[t] - (d.trend[t] + d.seasonal[t] + d.resid[t])) < 1e-9
                   for t in range(40))
        # a flat series is still recovered exactly
        d0 = sm.stl_decomposition([20.0 + seas[t % 4] for t in range(40)], 4)
        assert _series_close(d0.seasonal[:4], seas, tol=1e-6)

    def test_chi2_sf_even_df_closed_forms(self):
        # df=4: (1+x/2)e^{-x/2} ; df=6: (1+x/2+x^2/8)e^{-x/2}.  df>=6 has a=3 with
        # lgamma(a)=log(2)!=0, exercising the incomplete-gamma normalisation term
        for x in (2.0, 5.0, 12.0, 25.0):
            assert _close(sm._chi2_sf(x, 4), (1 + x / 2) * math.exp(-x / 2), tol=1e-9)
            assert _close(sm._chi2_sf(x, 6),
                          (1 + x / 2 + x * x / 8) * math.exp(-x / 2), tol=1e-9)

    def test_burg_recursion_matches_yule_walker(self):
        import random as _r
        rng = _r.Random(0)
        x = [0.0, 0.0]
        for _ in range(1500):
            x.append(0.6 * x[-1] - 0.2 * x[-2] + rng.gauss(0, 1))
        x = x[300:]
        pyw, _, s2yw = sm.yule_walker(x, 2)
        pbg, _, s2bg = sm.burg(x, 2)
        assert all(abs(pyw[i] - pbg[i]) < 0.05 for i in range(2))   # coeffs agree
        assert abs(s2yw - s2bg) < 0.15 and s2bg > 0.5               # variance rescaled
        # a SHORT-series exact pin -- Burg's endpoint terms (dk init) matter here,
        # where they are negligible on the long series above; guards the recursion
        assert _series_close(sm.burg(x[:12], 2)[0], [0.330534, -0.711529], tol=1e-4)

    def test_gaussian_filter_kernel_shape(self):
        impulse = [0.0] * 11
        impulse[5] = 1.0
        g = sm.gaussian_filter(impulse, 1.5)               # gaussian kernel shape
        # exact kernel values (pin the exp(-0.5 (k/sigma)^2) exponent); the filter
        # renormalises per-point at the edges, so the response is NOT mass-conserving
        assert _close(g[5], 0.266012, tol=1e-5)
        assert _close(g[4], 0.213225, tol=1e-5) and _close(g[6], g[4])
        assert g[5] > g[4] > g[3]                          # monotone decay from centre

    def test_aicc_matches_independent_oracle(self):
        y = [float(v) for v in sm.get_series("random_walk")]
        r = sm.ses(y, alpha=0.5, optimize=False, h=1)
        n = len(y)
        k_eff, n_params = n, 2                              # SES: alpha + level state
        aic = k_eff * math.log(r.sse / k_eff) + 2 * n_params
        aicc = aic + 2 * n_params * (n_params + 1) / (k_eff - n_params - 1)
        assert _close(r.aic, aic, tol=1e-6) and _close(r.aicc, aicc, tol=1e-6)

    def test_var_and_std_known_values(self):
        assert _close(sm._var([1.0, 2.0, 3.0, 4.0, 5.0]), 2.0)        # population
        assert _close(sm._var([1.0, 2.0, 3.0, 4.0, 5.0], ddof=1), 2.5)
        assert _close(sm._std([1.0, 2.0, 3.0, 4.0, 5.0]), math.sqrt(2.0))

    def test_prediction_interval_width_is_exact(self):
        y = [float(v) for v in sm.get_series("random_walk")]
        r = sm.ses(y, h=5, level=0.95)
        z = sm._z_score(0.95)
        for k in range(5):
            w = z * r.sigma * math.sqrt(k + 1)
            assert _close(r.upper[k] - r.forecast[k], w, tol=1e-9)
            assert _close(r.forecast[k] - r.lower[k], w, tol=1e-9)

    def test_ets_one_step_recursion_additive_and_multiplicative(self):
        ym = [(10.0 + t) * (1.2 if t % 4 < 2 else 0.8) for t in range(24)]
        al, be, ga = 0.3, 0.1, 0.2
        lvl0 = sm._mean(ym[:4])
        b0 = (sm._mean(ym[4:8]) - sm._mean(ym[:4])) / 4
        mul = sm.holt_winters(ym, 4, seasonal="mul", alpha=al, beta=be, gamma=ga,
                              optimize=False, h=1)
        add = sm.holt_winters(ym, 4, seasonal="add", alpha=al, beta=be, gamma=ga,
                              optimize=False, h=1)
        # fitted[0] uses the initial state
        assert _close(mul.fitted[0], (lvl0 + b0) * (ym[0] / lvl0), tol=1e-9)
        assert _close(add.fitted[0], (lvl0 + b0) + (ym[0] - lvl0), tol=1e-9)
        # fitted[1] exercises the LEVEL UPDATE (the yt/season division for 'mul',
        # yt-season for 'add') -- hand-roll one step of the recursion
        sm0 = ym[0] / lvl0
        nl_m = al * (ym[0] / sm0) + (1 - al) * (lvl0 + b0)
        nb_m = be * (nl_m - lvl0) + (1 - be) * b0
        assert _close(mul.fitted[1], (nl_m + nb_m) * (ym[1] / lvl0), tol=1e-9)
        sa0 = ym[0] - lvl0
        nl_a = al * (ym[0] - sa0) + (1 - al) * (lvl0 + b0)
        nb_a = be * (nl_a - lvl0) + (1 - be) * b0
        assert _close(add.fitted[1], (nl_a + nb_a) + (ym[1] - lvl0), tol=1e-9)

    def test_theta_seasonal_reconstruction_preserves_phase(self):
        # a positive series deseasonalises MULTIPLICATIVELY; the forecast must keep
        # the input's seasonal phase (high phases 0,1 forecast above low phases 2,3)
        pos = [(100.0 + t) * (1.3 if t % 4 < 2 else 0.7) for t in range(24)]
        fp = sm.theta_method(pos, h=4, seasonal_periods=4).forecast
        assert min(fp[0], fp[1]) > max(fp[2], fp[3])       # phase preserved
        # pin the VALUES too -- a deseasonalise/reseasonalise inconsistency keeps the
        # phase but shifts the magnitudes (mult deseasonalise + mult reconstruct)
        assert _series_close(fp, [160.443, 161.5234, 87.3278, 87.44], tol=1e-2)
        # a sign-crossing series deseasonalises ADDITIVELY (no crash) and forecasts
        neg = [(-5.0 + t) + 3.0 * (1 if t % 4 < 2 else -1) for t in range(24)]
        fn = sm.theta_method(neg, h=4, seasonal_periods=4).forecast
        assert len(fn) == 4 and fn[0] > fn[2]
        assert len(sm.theta_method([10.0 + t + (1 if t % 2 else -1)
                                    for t in range(12)], h=3, seasonal_periods=2).forecast) == 3


class TestBugHuntRound5:
    """R5 -- doc-vs-behavior modality.  REAL BUG: stl_decomposition's docstring
    promised a ROBUST decomposition, but its inner LOWESS passes ran with
    ``iterations=1``, which disables LOWESS's bisquare re-weighting.  A single
    gross outlier therefore leaked into the trend and seasonal components instead
    of the residual -- i.e. STL was no more robust than classical decomposition,
    defeating the one property that distinguishes it.  Fixed by running the inner
    LOWESS with two iterations under a new ``robust=True`` default (bisquare on),
    with ``robust=False`` restoring the fast non-robust path."""

    def _clean(self, n=40):
        seas = [3.0, -1.0, -2.0, 0.0]
        return [10.0 + 0.5 * t + seas[t % 4] for t in range(n)]

    def test_stl_default_is_robust_to_a_gross_outlier(self):
        # a +40 spike must land almost entirely in the residual; trend & seasonal
        # (estimated from the CLEAN series) must barely move.
        clean = self._clean()
        dirty = list(clean); dirty[20] += 40.0
        dc = sm.stl_decomposition(clean, 4)          # robust=True default
        dd = sm.stl_decomposition(dirty, 4)
        seas_shift = max(abs(dd.seasonal[i] - dc.seasonal[i]) for i in range(40))
        trend_shift = max(abs(dd.trend[i] - dc.trend[i]) for i in range(40))
        assert dd.resid[20] > 30.0                   # outlier absorbed by resid
        assert seas_shift < 2.0                       # seasonal essentially intact
        assert trend_shift < 2.0                      # trend essentially intact

    def test_robust_flag_actually_changes_behavior(self):
        # verify-catch: robust=False reproduces the ORIGINAL non-robust behaviour
        # (the bug), so the two paths must differ measurably on outlier data.
        clean = self._clean()
        dirty = list(clean); dirty[20] += 40.0
        dc0 = sm.stl_decomposition(clean, 4, robust=False)
        dd0 = sm.stl_decomposition(dirty, 4, robust=False)
        seas_shift0 = max(abs(dd0.seasonal[i] - dc0.seasonal[i]) for i in range(40))
        # non-robust leaks the outlier into the seasonal far more than robust does
        dc1 = sm.stl_decomposition(clean, 4, robust=True)
        dd1 = sm.stl_decomposition(dirty, 4, robust=True)
        seas_shift1 = max(abs(dd1.seasonal[i] - dc1.seasonal[i]) for i in range(40))
        assert seas_shift0 > 3.0                       # bug behaviour reproduced
        assert seas_shift1 < seas_shift0 - 2.0         # robust is strictly better

    def test_robust_stl_handles_multiple_outliers(self):
        seas = [3.0, -1.0, -2.0, 0.0]
        clean = [20.0 + 0.3 * t + seas[t % 4] for t in range(80)]
        dirty = list(clean)
        for k in (17, 40, 63):
            dirty[k] += 35.0
        dc = sm.stl_decomposition(clean, 4)
        dd = sm.stl_decomposition(dirty, 4)
        assert max(abs(dd.seasonal[i] - dc.seasonal[i]) for i in range(80)) < 2.0
        assert max(abs(dd.trend[i] - dc.trend[i]) for i in range(80)) < 2.0
        for k in (17, 40, 63):
            assert dd.resid[k] > 28.0                  # each spike -> its residual

    def test_robust_default_barely_perturbs_a_clean_fit(self):
        # turning robustness on must NOT distort a clean (outlier-free) fit
        clean = self._clean()
        a = sm.stl_decomposition(clean, 4, robust=True)
        b = sm.stl_decomposition(clean, 4, robust=False)
        assert max(abs(a.trend[i] - b.trend[i]) for i in range(40)) < 0.05
        # reconstruction stays exact and the seasonal stays centred
        assert all(abs(a.observed[i] - (a.trend[i] + a.seasonal[i] + a.resid[i]))
                   < 1e-9 for i in range(40))
        assert abs(_mean_helper(a.seasonal)) < 1e-9


def _mean_helper(v):
    return math.fsum(v) / len(v)


class TestBugHuntRound6:
    """R6 -- cross-function / analytic-oracle modality + scalability.  The HP
    filter solved its ``(I + lam D'D) tau = y`` system with a DENSE O(n^3)
    Gaussian elimination -- correct, but ~18s at n=1000, an unusable hang for a
    hackathon-sized series.  The matrix is symmetric PENTADIAGONAL, so it is now
    factored in O(n) with a banded L D L^T solve (``_solve_symmetric_penta``);
    output is bit-identical (~1e-12) and n=1000 drops to ~3ms.  Also: the input
    guard rejected NaN but silently let +/-inf through, corrupting downstream
    smoothers -- now rejects any non-finite value.  The projection oracles
    (damped-trend geometric sum, HW seasonal phase over h>m, AR mean reversion)
    all verified CORRECT with no fix needed and are pinned here as guards."""

    def _dd(self, tau, n):
        # exact D'D tau (correct at the boundaries too): apply D (the (n-2)xn
        # second-difference operator) then D^T.  Interior rows reduce to the
        # (1,-4,6,-4,1) stencil, but the first/last two rows differ.
        dt = [tau[k] - 2.0 * tau[k + 1] + tau[k + 2] for k in range(n - 2)]  # D tau
        out = [0.0] * n
        for k in range(n - 2):                          # D^T (D tau)
            out[k] += dt[k]
            out[k + 1] += -2.0 * dt[k]
            out[k + 2] += dt[k]
        return out

    def test_penta_solver_matches_dense_gaussian(self):
        # the O(n) banded solver must agree with the dense _solve on random
        # symmetric-pentadiagonal SPD systems (independent of the HP assembly).
        rng = random.Random(202)
        for n in (3, 4, 5, 12, 40):
            d0 = [4.0 + 3.0 * rng.random() for _ in range(n)]       # diag-dominant -> SPD
            d1 = [rng.uniform(-1.0, 1.0) for _ in range(n - 1)]
            d2 = [rng.uniform(-0.5, 0.5) for _ in range(n - 2)]
            A = [[0.0] * n for _ in range(n)]
            for r in range(n):
                A[r][r] = d0[r]
                if r + 1 < n:
                    A[r][r + 1] = A[r + 1][r] = d1[r]
                if r + 2 < n:
                    A[r][r + 2] = A[r + 2][r] = d2[r]
            rhs = [rng.uniform(-5.0, 5.0) for _ in range(n)]
            xb = sm._solve_symmetric_penta(d0, d1, d2, rhs)
            xd = sm._solve(A, rhs)
            assert _series_close(xb, xd, tol=1e-9)

    def test_hp_satisfies_defining_normal_equations_at_scale(self):
        # independent oracle: the HP minimiser's gradient is zero, i.e.
        # (tau - y) + lam * D'D tau = 0 everywhere.  Holds regardless of solver,
        # so it validates the O(n) path on large n the old dense path never
        # feasibly reached in a test.
        rng = random.Random(77)
        for n, lam in ((60, 1600.0), (200, 100.0), (500, 1600.0)):
            y = [math.sin(0.05 * t) + 0.1 * rng.gauss(0, 1) + 0.01 * t for t in range(n)]
            tau = sm.hodrick_prescott(y, lam)
            dd = self._dd(tau, n)
            resid = max(abs((tau[i] - y[i]) + lam * dd[i]) for i in range(n))
            assert resid < 1e-6

    def test_hp_edge_and_invariants(self):
        assert sm.hodrick_prescott([3.0, 5.0], 1600.0) == [3.0, 5.0]     # n<3 passthrough
        ident = sm.hodrick_prescott([1.0, 2.0, 9.0, 3.0, 4.0], 0.0)      # lam=0 -> identity
        assert _series_close(ident, [1.0, 2.0, 9.0, 3.0, 4.0], tol=1e-12)
        line = [float(t) for t in range(25)]                             # linear data is a
        big = sm.hodrick_prescott(line, 1e10)                            # fixed point for any lam
        assert _series_close(big, line, tol=1e-3)
        with pytest.raises(ValueError):
            sm.hodrick_prescott([1.0, 2.0, 3.0, 4.0], -1.0)

    def test_non_finite_input_is_rejected(self):
        # the guard rejected NaN but let inf through (the bug); now BOTH raise,
        # across every entry point that ingests a series.
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError):
                sm.simple_moving_average([1.0, bad, 3.0], 2)
            with pytest.raises(ValueError):
                sm.ses([1.0, 2.0, bad, 4.0, 5.0])
            with pytest.raises(ValueError):
                sm.difference([1.0, bad, 3.0])
            with pytest.raises(ValueError):
                sm.autocorrelation([1.0, 2.0, bad, 4.0], 2)

    def test_damped_trend_forecast_is_a_geometric_series(self):
        # analytic oracle: damped Holt h-step increments shrink by exactly phi,
        # i.e. fc[h+1]-fc[h] = phi^(h+1)*trend -> successive-increment ratio == phi.
        y = [10.0 + 2.0 * t for t in range(30)]
        fc = sm.exponential_smoothing(y, trend="add", damped=True, alpha=0.5,
                                      beta=0.3, phi=0.9, optimize=False, h=15).forecast
        inc = [fc[i + 1] - fc[i] for i in range(len(fc) - 1)]
        ratios = [inc[i + 1] / inc[i] for i in range(len(inc) - 1) if abs(inc[i]) > 1e-12]
        assert ratios and all(abs(r - 0.9) < 1e-9 for r in ratios)

    def test_holt_winters_seasonal_phase_repeats_over_horizon(self):
        # analytic oracle: undamped additive HW forecast is periodic-plus-trend,
        # so fc[h+m]-fc[h] is CONSTANT (== m*trend) for every h -- a phase error
        # in the seasonal projection over h>m would break this.
        seas = [5.0, -3.0, -1.0, -1.0]
        y = [20.0 + 0.4 * t + seas[t % 4] for t in range(40)]
        fc = sm.exponential_smoothing(y, trend="add", seasonal="add",
                                      seasonal_periods=4, h=16).forecast
        diffs = [fc[h + 4] - fc[h] for h in range(len(fc) - 4)]
        assert max(diffs) - min(diffs) < 1e-6

    def test_stationary_ar1_forecast_reverts_to_mean_geometrically(self):
        # analytic oracle: AR(1) h-step forecast = mu + phi^h (y_n - mu); the
        # gap to the long-run mean must decay by exactly phi each step.
        rng = random.Random(7)
        y = [10.0]
        for _ in range(500):
            y.append(4.0 + 0.6 * y[-1] + rng.gauss(0, 0.5))
        phi, intercept, _ = sm.yule_walker(y, 1)
        mu = intercept / (1.0 - phi[0])
        fc = sm.ar_forecast(y, 1, 30).forecast
        ratios = [(fc[i] - mu) / (fc[i - 1] - mu) for i in range(1, 8)]
        assert all(abs(r - phi[0]) < 1e-9 for r in ratios)
        assert abs(fc[-1] - mu) < 1e-3                    # converged to the mean


class TestBugHuntRound7:
    """R7 -- cross-field internal-consistency modality (do the fields of a single
    returned object agree?).  REAL BUG: a ForecastResult exposed the FULL residual
    series but computed ``sse``/``sigma``/``aic`` over only the post-warm-up
    *effective sample* (skipping the first ``burn`` one-step errors, which reflect
    crude heuristic initial states).  So ``sse != sum(e*e for e in residuals)`` --
    ratio ~0.9 for seasonal models (burn=m) -- with NO exposed way to reconcile it,
    silently breaking the invariant a user checks when reconstructing RMSE from
    ``.residuals``.  The trim is statistically correct (it improves PI calibration
    and lets auto_ets compare models with different warm-ups fairly -- verified),
    so the fix keeps the numbers and instead makes the effective sample a
    first-class part of the contract: ``state['burn']`` / ``state['nobs_effective']``
    are now exposed and documented, so ``sse == sum(e*e for e in residuals[burn:])``
    is reconstructable.  The other invariants probed (no look-ahead leakage,
    AICc>=AIC, no backtest leakage, seasonal normalisation) all held and are pinned."""

    def _ets_cases(self):
        rng = random.Random(4)
        y_tr = [10.0 + 0.5 * t + rng.gauss(0, 1) for t in range(60)]
        y_se = [20.0 + 0.3 * t + [5, -3, -1, -1][t % 4] + rng.gauss(0, 0.5) for t in range(60)]
        y_pos = [abs(v) + 50.0 for v in y_se]
        return [
            (sm.ses(y_tr), y_tr),
            (sm.holt(y_tr), y_tr),
            (sm.holt(y_tr, damped=True), y_tr),
            (sm.holt_winters(y_se, 4, seasonal="add"), y_se),
            (sm.holt_winters(y_pos, 4, seasonal="mul"), y_pos),
            (sm.auto_ets(y_se, seasonal_periods=4), y_se),
            (sm.ar_forecast(y_tr, 2, 5), y_tr),
        ]

    def test_effective_sample_is_exposed_and_reconstructs_sse(self):
        for r, y in self._ets_cases():
            n = len(y)
            assert "burn" in r.state and "nobs_effective" in r.state   # the fix
            burn = r.state["burn"]
            assert r.state["nobs_effective"] == n - burn
            # residuals are the FULL series and equal actual - fitted
            assert len(r.residuals) == n and len(r.fitted) == n
            assert _series_close(r.residuals, [y[t] - r.fitted[t] for t in range(n)], tol=1e-9)
            # the documented invariant: sse is over the effective sample
            sse_eff = math.fsum(r.residuals[t] ** 2 for t in range(burn, n))
            assert abs(r.sse - sse_eff) <= 1e-6 * max(1.0, sse_eff)

    def test_ets_sigma_is_rmse_over_effective_sample(self):
        for r, y in self._ets_cases():
            if r.method.startswith("AR"):        # AR reports the innovation sd, not sqrt(sse/nobs)
                continue
            ke = r.state["nobs_effective"]
            if ke > 0 and r.sse > 0:
                assert abs(r.sigma - math.sqrt(r.sse / ke)) < 1e-9

    def test_fitted_has_no_look_ahead(self):
        # fitted[t] is a genuine one-step-ahead prediction: it must depend only on
        # y[0..t-1].  Perturbing y[t0] must leave fitted[0..t0] bit-identical (and
        # only shift the future).  Use FIXED params so the optimiser doesn't re-fit.
        rng = random.Random(8)
        y = [10.0 + 0.4 * t + [3, -1, -2, 0][t % 4] + rng.gauss(0, 0.3) for t in range(40)]
        t0 = 20
        y2 = list(y); y2[t0] += 100.0
        for kw in (dict(alpha=0.4),
                   dict(trend="add", alpha=0.4, beta=0.2),
                   dict(trend="add", seasonal="add", seasonal_periods=4,
                        alpha=0.4, beta=0.1, gamma=0.2),
                   dict(trend="add", seasonal="mul", seasonal_periods=4,
                        alpha=0.4, beta=0.1, gamma=0.2)):
            a = sm.exponential_smoothing(y, optimize=False, **kw)
            b = sm.exponential_smoothing(y2, optimize=False, **kw)
            assert all(a.fitted[t] == b.fitted[t] for t in range(t0 + 1))   # no leakage
            assert any(abs(a.fitted[t] - b.fitted[t]) > 1e-9
                       for t in range(t0 + 1, 40))                          # future does move

    def test_aicc_never_below_aic(self):
        rng = random.Random(15)
        for i in range(12):
            y = [rng.gauss(0, 1) + 0.1 * t for t in range(30 + 3 * i)]
            for r in (sm.ses(y), sm.holt(y), sm.holt(y, damped=True)):
                if r.aic is not None and r.aicc is not None \
                        and math.isfinite(r.aic) and math.isfinite(r.aicc):
                    assert r.aicc >= r.aic - 1e-9

    def test_backtest_matches_manual_truncated_forecast(self):
        # no leakage in the harness: each origin's forecast must equal a direct
        # forecast on the truncated prefix.
        y = [10.0 + 0.3 * t + [2, -1, -1, 0][t % 4] for t in range(40)]
        bt = sm.rolling_origin_backtest(y, "ses", h=1, initial=30, step=1, expanding=True)
        no = bt["n_origins"]
        man = [(y[o] - sm.ses(y[:o], h=1).forecast[0]) ** 2 for o in range(30, 30 + no)]
        man_rmse = math.sqrt(math.fsum(man) / len(man))
        assert abs(bt["metrics"]["rmse"] - man_rmse) < 1e-9

    def test_classical_decomposition_seasonal_is_normalised(self):
        rng = random.Random(21)
        y = [20.0 + 0.3 * t + [5, -3, -1, -1][t % 4] + rng.gauss(0, 0.4) for t in range(60)]
        da = sm.classical_decomposition(y, 4, model="additive")
        assert abs(math.fsum(da.seasonal[:4])) < 1e-9                 # additive: sums to 0
        pos = [abs(v) + 50.0 for v in y]
        dm = sm.classical_decomposition(pos, 4, model="multiplicative")
        assert abs(math.fsum(dm.seasonal[:4]) / 4 - 1.0) < 1e-9       # mult: averages to 1


def _ref_lowess(y, frac, iterations, x=None):
    """Independent, deliberately naive LOWESS: per point, full sort of all indices
    by distance and take the nearest r.  Used to pin that the optimised
    contiguous-window path in the module stays BIT-IDENTICAL."""
    yy = [float(v) for v in y]
    n = len(yy)
    if n == 0:
        return []
    xs = list(range(n)) if x is None else [float(v) for v in x]
    r = max(2, int(math.ceil(frac * n)))
    fitted = [0.0] * n
    rob = [1.0] * n
    for _ in range(max(1, iterations)):
        for i in range(n):
            order = sorted(range(n), key=lambda j: abs(xs[j] - xs[i]))[:r]
            h = max(abs(xs[j] - xs[i]) for j in order) or 1.0
            w = [((1 - (abs(xs[j] - xs[i]) / h) ** 3) ** 3
                  if abs(xs[j] - xs[i]) / h < 1 else 0.0) * rob[j] for j in order]
            sw = math.fsum(w)
            if sw <= 1e-12:
                fitted[i] = yy[i]
                continue
            swx = math.fsum(w[k] * xs[order[k]] for k in range(r))
            swy = math.fsum(w[k] * yy[order[k]] for k in range(r))
            swxx = math.fsum(w[k] * xs[order[k]] ** 2 for k in range(r))
            swxy = math.fsum(w[k] * xs[order[k]] * yy[order[k]] for k in range(r))
            denom = sw * swxx - swx * swx
            if abs(denom) < 1e-12:
                fitted[i] = swy / sw
            else:
                b1 = (sw * swxy - swx * swy) / denom
                fitted[i] = (swy - b1 * swx) / sw + b1 * xs[i]
        resid = [yy[i] - fitted[i] for i in range(n)]
        s = sorted(abs(e) for e in resid)
        med = s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
        if med <= 1e-12:
            break
        rob = [(1 - (e / (6 * med)) ** 2) ** 2 if abs(e) < 6 * med else 0.0 for e in resid]
    return fitted


class TestBugHuntRound8:
    """R8 -- perf/scalability + API-footgun modality.  (1) PERF: lowess re-sorted
    all n points by distance for EVERY point on EVERY bisquare iteration --
    O(n^2 log n).  The neighbour sets depend only on x, so they are precomputed
    once; for the default uniform grid the r nearest are the contiguous index
    window [i - r//2, i - r//2 + r) clamped to [0, n), reproducing the stable-sort
    tie-break exactly (smaller index wins a distance tie) and, since every
    regression sum uses math.fsum (order-independent), giving BIT-IDENTICAL output
    while dropping the sort factor (2-3.5x at small frac; STL ~1.6x).  (2) FOOTGUN:
    time_series_cv iterates ``methods``, so a bare string ``"ses"`` (easily
    confused with rolling_origin_backtest's singular ``method=``) silently scored
    the individual LETTERS ({'s': inf, 'e': inf}) -- now a clear TypeError.  (3)
    forecast(method='holt_winters') without ``m`` leaked a cryptic
    ``<lambda>() missing 1 required argument`` -- now a clear ValueError."""

    def test_lowess_optimised_path_is_bit_identical(self):
        rng = random.Random(31)
        for n in (5, 8, 13, 30, 60, 120):
            for frac in (0.05, 0.1, 0.25, 0.5, 0.9, 1.0):
                for it in (1, 2, 3):
                    y = [math.sin(0.1 * t) + rng.gauss(0, 0.4) + 0.02 * t for t in range(n)]
                    got = sm.lowess(y, frac, iterations=it)
                    ref = _ref_lowess(y, frac, it)
                    assert got == ref                      # exact, not approximate

    def test_lowess_even_and_odd_window_tie_break(self):
        # the window derivation must match the full-sort tie-break for BOTH even
        # and odd r (even r extends one further to the left).
        for n in (30, 31, 40, 41):
            for r_target in range(2, 12):
                y = [float(t * t % 7) + 0.1 * t for t in range(n)]
                assert sm.lowess(y, r_target / n, iterations=2) == _ref_lowess(y, r_target / n, 2)

    def test_lowess_custom_x_still_matches(self):
        rng = random.Random(32)
        for _ in range(15):
            n = rng.randint(10, 50)
            xv = sorted(rng.uniform(0, 100) for _ in range(n))     # non-uniform, sorted
            y = [math.sin(v * 0.1) + rng.gauss(0, 0.3) for v in xv]
            assert sm.lowess(y, 0.4, iterations=2, x=xv) == _ref_lowess(y, 0.4, 2, xv)
        # duplicate + unsorted x
        xd = [0, 1, 1, 2, 2, 2, 3, 5, 8, 8]
        yd = [1., 2., 1., 3., 2., 4., 3., 5., 4., 6.]
        assert sm.lowess(yd, 0.5, iterations=2, x=xd) == _ref_lowess(yd, 0.5, 2, xd)
        xu = [5, 1, 3, 2, 9, 7, 0, 8, 4, 6]
        assert sm.lowess(yd, 0.5, iterations=2, x=xu) == _ref_lowess(yd, 0.5, 2, xu)

    def test_time_series_cv_rejects_bare_string(self):
        y = [10.0 + 0.3 * t + [2, -1, -1, 0][t % 4] for t in range(40)]
        with pytest.raises(TypeError):
            sm.time_series_cv(y, "ses")                    # the footgun: was {'s':inf,'e':inf}
        scores = sm.time_series_cv(y, ["ses", "holt"])     # correct usage still works
        assert set(scores) == {"ses", "holt"} and all(math.isfinite(v) for v in scores.values())

    def test_forecast_holt_winters_without_m_is_clear_error(self):
        y = [10.0 + 0.3 * t + [2, -1, -1, 0][t % 4] for t in range(40)]
        with pytest.raises(ValueError):                    # was a cryptic lambda TypeError
            sm.forecast(y, "holt_winters", 4)
        fc = sm.forecast(y, "holt_winters", 4, m=4)        # with a period it works
        assert len(fc) == 4 and all(math.isfinite(v) for v in fc)
        # and the back-testing path (which passes m) is unaffected
        air = sm.get_series("airline")
        bt = sm.rolling_origin_backtest(air, "holt_winters", h=6, initial=48, m=12)
        assert bt["n_origins"] > 0 and math.isfinite(bt["metrics"]["rmse"])


class TestBugHuntRound9:
    """R9 -- property-based FUZZING modality.  ~12k randomized valid API calls (two
    broad passes + a targeted structural-invariant pass + a payload/metric pass)
    found ZERO genuine bugs: no non-ValueError crash, no non-finite output, no
    wrong-length result, no ACF-bound break, no negative variance, no reconstruction
    or PI-ordering failure -- across const/linear/seasonal/spike/tiny(1e-9)/huge
    (1e12)/near-zero/alt-sign series at boundary params.  (The ONLY fuzz 'suspects'
    were a FALSE harness invariant -- 'a smoother never raises sample variance' --
    which is not a finite-sample theorem for trailing / edge-renormalised filters
    that are not doubly stochastic; the CORRECT invariant below holds.)  These
    deterministic, bounded property tests lock the invariants in going forward.
    Marked @slow (randomised batteries) -- the fast suite already pins these
    invariants on specific cases via R1-R8."""

    pytestmark = slow

    def _rand_series(self, rng, n):
        kind = rng.choice(["noisy", "uniform", "altsign", "spike", "const", "linear"])
        if kind == "noisy":
            return [rng.gauss(0, 1) for _ in range(n)]
        if kind == "uniform":
            return [rng.uniform(-50, 50) for _ in range(n)]
        if kind == "altsign":
            return [((-1) ** t) * rng.uniform(1, 10) for t in range(n)]
        if kind == "spike":
            y = [1.0] * n; y[rng.randrange(n)] = rng.choice([1e3, -1e3]); return y
        if kind == "const":
            v = rng.uniform(-20, 20); return [v] * n
        return [rng.uniform(-5, 5) + rng.uniform(-3, 3) * t for t in range(n)]

    def test_convex_smoothers_stay_within_data_range(self):
        # sma/wma/ema/cma/median/gaussian are convex combinations (or order stats)
        # of the data, so every output must lie within [min(y), max(y)].  (savgol/
        # lowess/HP fit polynomials and may legitimately overshoot -- excluded.)
        rng = random.Random(4242)
        for _ in range(500):
            n = rng.choice([3, 4, 5, 6, 8, 10, 12, 16, 24])
            y = self._rand_series(rng, n)
            lo, hi = min(y), max(y)
            tol = 1e-9 * max(1.0, abs(lo), abs(hi))
            name, fn = rng.choice([
                ("sma_trail", lambda: sm.simple_moving_average(y, rng.randint(1, n))),
                ("sma_ctr", lambda: sm.simple_moving_average(y, rng.randint(1, n), center=True)),
                ("wma", lambda: sm.weighted_moving_average(
                    y, weights=[rng.uniform(0.1, 3) for _ in range(rng.randint(1, n))])),
                ("ema", lambda: sm.exponential_moving_average(y, alpha=rng.uniform(1e-3, 1.0))),
                ("cma", lambda: sm.centered_moving_average(y, rng.randint(2, n))),
                ("median", lambda: sm.median_filter(y, rng.randint(1, n))),
                ("gauss", lambda: sm.gaussian_filter(y, sigma=rng.uniform(0.3, 5))),
            ])
            for v in fn():
                if v is None:
                    continue
                assert lo - tol <= v <= hi + tol, (name, v, lo, hi, n)

    def test_smoothers_finite_and_right_length_at_boundaries(self):
        rng = random.Random(77)
        for _ in range(500):
            n = rng.choice([2, 3, 4, 5, 8, 12, 20])
            y = self._rand_series(rng, n)
            w = rng.randint(1, n)
            outs = [
                sm.simple_moving_average(y, w),
                sm.exponential_moving_average(y, alpha=rng.uniform(1e-4, 1.0)),
                sm.median_filter(y, w),
                sm.gaussian_filter(y, sigma=rng.uniform(0.3, 5)),
                sm.savitzky_golay(y, w, rng.randint(0, max(0, w - 1))),
                sm.lowess(y, rng.uniform(0.05, 1.0), iterations=rng.randint(1, 3)),
                sm.hodrick_prescott(y, rng.choice([0.0, 1.0, 1600.0, 1e6])),
            ]
            for o in outs:
                assert len(o) == n
                assert all(math.isfinite(v) for v in o if v is not None)

    def test_decomposition_reconstructs_for_random_inputs(self):
        rng = random.Random(303)
        for _ in range(300):
            p = rng.choice([2, 3, 4])
            n = rng.choice([2 * p, 3 * p, 4 * p, 5 * p])
            seas = [rng.uniform(-4, 4) for _ in range(p)]
            y = [10.0 + 0.2 * t + seas[t % p] + rng.gauss(0, 0.3) for t in range(n)]
            scale = max(1.0, max(abs(v) for v in y))
            da = sm.classical_decomposition(y, p, model="additive")
            assert max(abs(y[i] - (da.trend[i] + da.seasonal[i] + da.resid[i]))
                       for i in range(n) if da.trend[i] is not None) < 1e-6 * scale
            st = sm.stl_decomposition(y, p, robust=rng.random() < 0.5)
            assert max(abs(y[i] - (st.trend[i] + st.seasonal[i] + st.resid[i]))
                       for i in range(n)) < 1e-6 * scale
            sp = [rng.uniform(0.6, 1.5) for _ in range(p)]
            yp = [(20.0 + 0.3 * t) * sp[t % p] for t in range(n)]
            dm = sm.classical_decomposition(yp, p, model="multiplicative")
            assert max(abs(yp[i] - (dm.trend[i] * dm.seasonal[i] * dm.resid[i]))
                       for i in range(n) if dm.trend[i] is not None) < 1e-5 * max(yp)

    def test_prediction_intervals_ordered_and_symmetric(self):
        rng = random.Random(55)
        for _ in range(300):
            n = rng.choice([6, 8, 10, 16, 24])
            y = self._rand_series(rng, n)
            h = rng.randint(1, 6)
            kw = rng.choice([dict(alpha=1.0), dict(alpha=1e-4),
                             dict(trend="add", alpha=0.5, beta=0.0),
                             dict(trend="add", damped=True, alpha=0.5, beta=0.5, phi=0.8)])
            r = sm.exponential_smoothing(y, optimize=False, h=h,
                                         level=rng.choice([0.5, 0.8, 0.95, 0.99]), **kw)
            if r.lower is None:
                continue
            for k in range(h):
                assert r.lower[k] <= r.forecast[k] <= r.upper[k] + 1e-9
                assert abs((r.upper[k] - r.forecast[k]) - (r.forecast[k] - r.lower[k])) \
                    <= 1e-6 * max(1.0, abs(r.forecast[k]))

    def test_acf_bounded_and_variances_nonnegative(self):
        rng = random.Random(88)
        for _ in range(300):
            n = rng.choice([5, 8, 12, 20, 30])
            y = self._rand_series(rng, n)
            ac = sm.autocorrelation(y, n - 1)
            assert len(ac) == n and all(abs(v) <= 1.0 + 1e-9 for v in ac)
            for o in {1, (n - 1) // 2 or 1, n - 1}:            # low / mid / max order
                for est in (sm.yule_walker, sm.burg):
                    phi, _c, s2 = est(y, o)
                    assert all(math.isfinite(v) for v in phi) and s2 >= -1e-9

    def test_box_cox_round_trip(self):
        rng = random.Random(66)
        for _ in range(300):
            n = rng.choice([4, 6, 8, 12, 20])
            y = [rng.uniform(0.5, 200.0) for _ in range(n)]
            lam = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
            inv = sm.inv_box_cox(sm.box_cox(y, lam), lam)
            assert max(abs(inv[i] - y[i]) for i in range(n)) < 1e-6 * max(y)

    def test_payloads_are_strict_json(self):
        # build_forecast_payload / build_decomposition_payload must never leak
        # NaN/inf: json.dumps(..., allow_nan=False) must succeed for every result.
        rng = random.Random(4242)
        for _ in range(200):
            n = rng.choice([8, 10, 12, 16, 24]); h = rng.randint(1, 6); p = rng.choice([2, 3, 4])
            seas = [rng.uniform(-4, 4) for _ in range(p)]
            y = [10.0 + 0.2 * t + seas[t % p] for t in range(n)]
            c = rng.randrange(4)
            if c == 0:
                r = sm.exponential_smoothing(y, optimize=False, h=h, alpha=0.4)
            elif c == 1:
                r = sm.exponential_smoothing(y, trend="add", optimize=False, h=h,
                                             alpha=0.4, beta=0.2)
            elif c == 2:
                r = sm.theta_method(y, h=h)
            else:
                r = sm.ar_forecast(y, rng.randint(1, 3), h)
            json.dumps(sm.build_forecast_payload(r), allow_nan=False)
            if n >= 2 * p:
                d = sm.classical_decomposition(y, p, model="additive")
                json.dumps(sm.build_decomposition_payload(d), allow_nan=False)


class TestBugHuntRound10:
    """R10 -- numerical-conditioning modality (accuracy, not just finiteness, on a
    large DC offset -- prices/Kelvin/counts).  The core stats (_var, _std,
    autocovariance) already demean-first with math.fsum, so the whole pipeline is
    shift-stable to the float64 floor.  The one exception was the HP filter: its
    matrix (I + lam*D'D) is ill-conditioned (~1 + 16*lam), so a large offset in y
    lost precision.  Since HP preserves constants exactly, it now demeans before
    the solve and adds the mean back -- ~3000x more accurate at a 1e6 offset,
    normal-scale output unchanged to rounding.  These tests pin the offset
    robustness of HP and of the demean-first statistical primitives."""

    def _base(self):
        rng = random.Random(7)
        return [10.0 + 0.3 * t + [3, -1, -2, 0][t % 4] + rng.gauss(0, 0.5) for t in range(48)]

    def test_hp_robust_to_large_dc_offset(self):
        # HP preserves constants, so HP(y + off) - off == HP(y).  Post-demean the
        # error is ~float-floor; the pre-fix (solve on the raw offset) drifted to
        # ~1e-8 (1e6) / ~1e-5 (1e9) relative -- caught by these tight tolerances.
        base = self._base()
        b0 = sm.hodrick_prescott(base, 1600.0)
        scale = max(abs(v) for v in b0)
        for off, tol in ((1e6, 1e-10), (1e9, 1e-7)):
            b1 = sm.hodrick_prescott([v + off for v in base], 1600.0)
            err = max(abs((b1[i] - off) - b0[i]) for i in range(len(base))) / scale
            assert err < tol, (off, err)

    def test_hp_still_solves_normal_equations_with_offset(self):
        # demean-first must not change WHAT is solved: the minimiser's gradient
        # (tau - y) + lam*D'D tau = 0 must still hold on offset data.
        base = self._base()
        n = len(base)
        y = [v + 1e6 for v in base]
        tau = sm.hodrick_prescott(y, 1600.0)
        dt = [tau[k] - 2.0 * tau[k + 1] + tau[k + 2] for k in range(n - 2)]
        dd = [0.0] * n
        for k in range(n - 2):
            dd[k] += dt[k]; dd[k + 1] += -2.0 * dt[k]; dd[k + 2] += dt[k]
        resid = max(abs((tau[i] - y[i]) + 1600.0 * dd[i]) for i in range(n))
        assert resid < 1e-3                       # ~ float floor at offset 1e6

    def test_hp_normal_scale_unchanged_by_demean(self):
        # demeaning is a conditioning trick, not a behaviour change: at normal
        # scale the result must match a direct-solve reference to rounding.
        base = self._base()
        n = len(base)
        mu = math.fsum(base) / n
        d0 = [1.0] * n; d1 = [0.0] * (n - 1); d2 = [0.0] * (n - 2)
        coef = (1.0, -2.0, 1.0)
        for k in range(n - 2):
            idx = (k, k + 1, k + 2)
            for a in range(3):
                for b in range(3):
                    i, j = idx[a], idx[b]
                    if j < i:
                        continue
                    v = 1600.0 * coef[a] * coef[b]
                    if j == i:
                        d0[i] += v
                    elif j == i + 1:
                        d1[i] += v
                    else:
                        d2[i] += v
        ref = [t + mu for t in sm._solve_symmetric_penta(d0, d1, d2, [v - mu for v in base])]
        got = sm.hodrick_prescott(base, 1600.0)
        assert _series_close(got, ref, tol=1e-9)

    def test_statistical_primitives_stable_under_large_offset(self):
        # _var / autocorrelation / yule_walker / burg are shift-invariant; the
        # demean-first (not the naive E[x^2]-E[x]^2) formulation keeps them so even
        # with a 1e9 baseline.  A naive one-pass variance would blow these up.
        base = self._base()
        for off in (1e6, 1e9):
            y = [v + off for v in base]
            assert abs(sm._var(y) - sm._var(base)) < 1e-6 * sm._var(base)
            a0, a1 = sm.autocorrelation(base, 10), sm.autocorrelation(y, 10)
            assert max(abs(a1[i] - a0[i]) for i in range(11)) < 1e-6
            for est in (sm.yule_walker, sm.burg):
                p0, p1 = est(base, 2)[0], est(y, 2)[0]
                assert max(abs(p1[i] - p0[i]) for i in range(2)) < 1e-5


@slow
class TestDifferentialVsReference:
    """R11 -- DIFFERENTIAL testing against numpy/scipy (the reference libraries).
    Every EXACT-algorithm component matched to machine precision: _z_score 3e-9
    (Acklam's inherent accuracy), _chi2_sf 4e-15, autocorrelation 3e-16,
    yule_walker 1e-15, savitzky_golay(interior) 1e-13, metrics 3e-14, _solve
    1e-11, box_cox 5e-12, hodrick_prescott(vs dense) 2e-11, PACF 4e-15.  The only
    'discrepancy' -- _polyfit coeffs differing from numpy by ~7e-3 on a degree-4
    fit of 5 points -- is ill-conditioning non-uniqueness (both interpolate to
    SSE 0; VtV condition ~2.5e11), NOT a bug: _polyfit's least-squares SSE matches
    numpy to 9e-12 everywhere.  These tests are skipped when numpy/scipy are
    absent (the module itself stays pure-stdlib); where present they run as
    continuous external cross-validation."""

    def test_z_score_matches_scipy_norm_ppf(self):
        stats = pytest.importorskip("scipy.stats")
        for lvl in (0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999):
            assert abs(sm._z_score(lvl) - stats.norm.ppf(0.5 + lvl / 2)) < 1e-7

    def test_chi2_sf_matches_scipy(self):
        stats = pytest.importorskip("scipy.stats")
        for k in (1, 2, 3, 4, 6, 10, 20):
            for x in (0.1, 1.0, float(k), 2.0 * k, 50.0):
                assert abs(sm._chi2_sf(x, k) - stats.chi2.sf(x, k)) < 1e-10

    def test_acf_and_yule_walker_match_numpy(self):
        np = pytest.importorskip("numpy")
        rng = random.Random(1)
        for _ in range(60):
            n = rng.randint(20, 70)
            y = [0.0]
            for _ in range(n):
                y.append(0.6 * y[-1] + rng.gauss(0, 1))
            y = y[1:]
            arr = np.array(y); yc = arr - arr.mean(); d = float(np.sum(yc * yc))
            nl = rng.randint(1, min(8, n - 1))
            mine = sm.autocorrelation(y, nl)
            ref = [float(np.sum(yc[:n - k] * yc[k:]) / d) for k in range(nl + 1)]
            assert max(abs(mine[i] - ref[i]) for i in range(nl + 1)) < 1e-12
            o = rng.randint(1, 4)
            g = sm.autocovariance(y, o)
            R = np.array([[g[abs(i - j)] for j in range(o)] for i in range(o)])
            phi = sm.yule_walker(y, o)[0]
            refphi = np.linalg.solve(R, np.array(g[1:o + 1]))
            assert max(abs(phi[i] - refphi[i]) for i in range(o)) < 1e-9

    def test_savgol_interior_matches_scipy(self):
        signal = pytest.importorskip("scipy.signal")
        np = pytest.importorskip("numpy")
        rng = random.Random(3)
        for _ in range(50):
            n = rng.randint(15, 40); w = rng.choice([5, 7, 9, 11]); p = rng.randint(1, min(4, w - 1))
            if w > n:
                continue
            y = [math.sin(0.2 * t) + rng.gauss(0, 0.3) for t in range(n)]
            mine = sm.savitzky_golay(y, w, p)
            ref = signal.savgol_filter(np.array(y), w, p, mode="interp")
            half = w // 2
            for i in range(half, n - half):
                assert abs(mine[i] - ref[i]) < 1e-9

    def test_hp_and_solve_and_metrics_match_numpy(self):
        np = pytest.importorskip("numpy")
        rng = random.Random(5)
        for _ in range(40):
            n = rng.randint(5, 50); lam = rng.choice([1.0, 100.0, 1600.0])
            y = np.array([math.sin(0.1 * t) + rng.gauss(0, 0.4) + 0.02 * t for t in range(n)])
            D = np.zeros((n - 2, n))
            for k in range(n - 2):
                D[k, k], D[k, k + 1], D[k, k + 2] = 1, -2, 1
            ref = np.linalg.solve(np.eye(n) + lam * D.T @ D, y)
            mine = sm.hodrick_prescott(list(y), lam)
            assert max(abs(mine[i] - ref[i]) for i in range(n)) < 1e-8
        for _ in range(40):
            k = rng.randint(2, 7)
            A = np.array([[rng.uniform(-5, 5) for _ in range(k)] for _ in range(k)]) + k * np.eye(k)
            b = np.array([rng.uniform(-5, 5) for _ in range(k)])
            ms = sm._solve([list(r) for r in A], list(b)); rs = np.linalg.solve(A, b)
            assert max(abs(ms[i] - rs[i]) for i in range(k)) < 1e-7
            m = rng.randint(2, 15)
            a = np.array([rng.gauss(0, 5) for _ in range(m)]); c = np.array([rng.gauss(0, 5) for _ in range(m)])
            assert abs(sm.mae(list(a), list(c)) - float(np.mean(np.abs(a - c)))) < 1e-10
            assert abs(sm.rmse(list(a), list(c)) - math.sqrt(float(np.mean((a - c) ** 2)))) < 1e-10

    def test_polyfit_matches_numpy_in_fit_quality(self):
        # coefficients may differ under ill-conditioning, but the least-squares
        # objective (SSE) must match numpy -- that is what _polyfit actually solves.
        np = pytest.importorskip("numpy")
        rng = random.Random(9)
        for _ in range(200):
            n = rng.randint(5, 30); deg = rng.randint(0, min(4, n - 1))
            x = [rng.uniform(-3, 3) for _ in range(n)]
            y = [rng.uniform(-10, 10) for _ in range(n)]
            mine = sm._polyfit(x, y, deg)
            ref = np.polyfit(x, y, deg)[::-1]

            def sse(c):
                return sum((y[i] - sum(c[p] * x[i] ** p for p in range(deg + 1))) ** 2
                           for i in range(n))
            assert sse(mine) <= sse(ref) + 1e-6 * (1.0 + abs(sse(ref)))


class TestApiSurfaceIntegrity:
    """R11 addendum -- the public API surface (``__all__`` / ``import *``) was never
    pinned.  A template people do ``from smoothing_template import *`` on must keep
    __all__ consistent: every listed name must resolve (a typo crashes ``import *``),
    no duplicates, and no public function/class silently omitted."""

    def test_all_names_resolve(self):
        assert all(hasattr(sm, n) for n in sm.__all__), \
            [n for n in sm.__all__ if not hasattr(sm, n)]

    def test_all_has_no_duplicates(self):
        assert len(sm.__all__) == len(set(sm.__all__)), \
            [n for n in set(sm.__all__) if sm.__all__.count(n) > 1]

    def test_public_symbols_are_exported(self):
        import inspect
        public = [n for n, o in vars(sm).items()
                  if (inspect.isfunction(o) or isinstance(o, type))
                  and not n.startswith("_")
                  and getattr(o, "__module__", None) == sm.__name__]
        missing = [n for n in public if n not in sm.__all__]
        assert not missing, "public but not in __all__: %s" % missing


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
