import math
import sys
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from gridpulse.analysis.algorithms.errors import CalculationError, InvalidInputError
from gridpulse.analysis.algorithms.orchid import forecasting_baseline as fb
from gridpulse.analysis.algorithms.orchid import smoothing as sm
from gridpulse.analysis.algorithms.orchid import spectral as sp
from gridpulse.analysis.algorithms.series import HalfHourlySeries
from gridpulse.analysis.algorithms.signals import (
    difference,
    exponential_moving_average,
    rolling_median,
    rolling_quantile,
    rolling_rms,
    rolling_std,
    rolling_variance,
    trailing_mean,
    trailing_savitzky_golay,
    trailing_weighted_mean,
)

START = datetime(2026, 9, 1, tzinfo=UTC)
WEIGHTS = (1.0, 2.0, 4.0)

Feature = Callable[[HalfHourlySeries], HalfHourlySeries]
Oracle = Callable[[list[float]], list[float]]

# Each feature, with Orchid's output on complete data after the feature's warm-up.
CASES: dict[str, tuple[Feature, Oracle]] = {
    "trailing_mean": (
        lambda s: trailing_mean(s, 5),
        lambda y: sm.simple_moving_average(y, 5, center=False)[4:],
    ),
    "trailing_weighted_mean": (
        lambda s: trailing_weighted_mean(s, WEIGHTS),
        lambda y: sm.weighted_moving_average(y, WEIGHTS)[2:],
    ),
    # Degree 0 has taps of exactly 1/4; higher degrees are checked against least squares below.
    "trailing_savitzky_golay": (
        lambda s: trailing_savitzky_golay(s, 4, 0),
        lambda y: sp.lfilter([0.25] * 4, [1.0], y)[3:],
    ),
    "exponential_moving_average": (
        lambda s: exponential_moving_average(s, span=4),
        lambda y: sm.exponential_moving_average(y, span=4)[3:],
    ),
    "rolling_variance": (
        lambda s: rolling_variance(s, 5),
        lambda y: [fb.variance(w, 1) for w in sm.rolling_windows(y, 5)],
    ),
    "rolling_variance_ddof0": (
        lambda s: rolling_variance(s, 5, ddof=0),
        lambda y: [fb.variance(w, 0) for w in sm.rolling_windows(y, 5)],
    ),
    "rolling_std": (
        lambda s: rolling_std(s, 5),
        lambda y: [fb.std(w, 1) for w in sm.rolling_windows(y, 5)],
    ),
    "rolling_median": (
        lambda s: rolling_median(s, 4),
        lambda y: [fb.median(w) for w in sm.rolling_windows(y, 4)],
    ),
    "rolling_quantile": (
        lambda s: rolling_quantile(s, 6, 0.9),
        lambda y: [fb.quantile(w, 0.9) for w in sm.rolling_windows(y, 6)],
    ),
    "rolling_rms": (
        lambda s: rolling_rms(s, 5),
        lambda y: [sp.rms(w) for w in sm.rolling_windows(y, 5)],
    ),
    "difference": (
        lambda s: difference(s, lag=3, order=2),
        lambda y: sm.difference(y, lag=3, order=2),
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_matches_orchid_after_the_warm_up(
    make_series: Callable[..., HalfHourlySeries], name: str
) -> None:
    feature, orchid = CASES[name]
    series = make_series(50)
    expected = orchid(series.complete_values())
    result = feature(series)
    assert result.start == series.start
    assert result.values == (None,) * (len(series) - len(expected)) + tuple(expected)


@pytest.mark.parametrize("name", list(CASES))
def test_outputs_depend_only_on_earlier_periods(
    synthetic_values: Callable[..., list[float]], name: str
) -> None:
    feature = CASES[name][0]
    values, other = synthetic_values(30), synthetic_values(30, seed=1)
    baseline = feature(HalfHourlySeries(START, values)).values
    for t in range(len(values) - 1):
        futures: list[list[float | None]] = [[*other[t + 1 :]], [None, *other[t + 2 :]]]
        for future in futures:
            perturbed = feature(HalfHourlySeries(START, [*values[: t + 1], *future]))
            assert perturbed.values[: t + 1] == baseline[: t + 1]


@pytest.mark.parametrize("feature", [trailing_mean, rolling_median])
def test_windows_touching_a_gap_are_missing(
    feature: Callable[[HalfHourlySeries, int], HalfHourlySeries],
) -> None:
    series = HalfHourlySeries(START, [1.0, 2.0, 3.0, 4.0, None, 6.0, 7.0, None, 9.0, 10.0, 11.0])
    expected = (None, None, 2.0, 3.0, None, None, None, None, None, None, 10.0)
    assert feature(series, 3).values == expected


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        (1, (None, None, 3.0, 5.0, None, 9.0, None, 13.0, 15.0, 17.0)),
        (2, (None, None, None, None, None, 4.0, None, 4.0, None, 4.0)),
    ],
)
def test_difference_misses_only_rows_that_use_the_gap(
    order: int, expected: tuple[float | None, ...]
) -> None:
    triangular = HalfHourlySeries(START, [0.0, 1.0, 3.0, 6.0, None, 15.0, 21.0, 28.0, 36.0, 45.0])
    assert difference(triangular, lag=2, order=order).values == expected


@pytest.mark.parametrize(
    ("span", "alpha", "min_periods", "expected"),
    [
        (None, 0.5, 1, (1.0, 1.5, 2.25, None, 5.0, 5.5)),
        (None, 0.5, None, (None, None, 2.25, None, None, None)),
        (3, None, None, (None, None, 2.25, None, None, None)),
        (3, None, 2, (None, 1.5, 2.25, None, None, 5.5)),
    ],
)
def test_ewma_restarts_after_a_gap(
    span: int | None,
    alpha: float | None,
    min_periods: int | None,
    expected: tuple[float | None, ...],
) -> None:
    series = HalfHourlySeries(START, [1.0, 2.0, 3.0, None, 5.0, 6.0])
    result = exponential_moving_average(series, span=span, alpha=alpha, min_periods=min_periods)
    assert result.values == expected


def test_rolling_std_of_negative_and_zero_prices_ignores_the_level(
    synthetic_values: Callable[..., list[float]],
) -> None:
    prices = synthetic_values(48, level=0.0)
    prices[20:26] = [0.0] * 6
    assert min(prices) < 0
    result = rolling_std(HalfHourlySeries(START, prices), 6).values
    shifted = rolling_std(HalfHourlySeries(START, [price + 150.0 for price in prices]), 6).values
    assert result == pytest.approx(shifted, abs=1e-9)
    assert result[25] == 0.0


@pytest.mark.parametrize(("window", "degree"), [(3, 1), (5, 2), (8, 3), (49, 3)])
def test_trailing_savitzky_golay_is_the_newest_point_of_a_least_squares_fit(
    synthetic_values: Callable[..., list[float]], window: int, degree: int
) -> None:
    values = synthetic_values(120, level=0.0)
    design = fb.trend_features(window, degree, 1 - window)  # the newest period is at time 0
    expected = [fb.lstsq(design, w)[0] for w in sm.rolling_windows(values, window)]
    result = trailing_savitzky_golay(HalfHourlySeries(START, values), window, degree).values
    assert result[: window - 1] == (None,) * (window - 1)
    assert result[window - 1 :] == pytest.approx(expected, abs=1e-11)


def test_trailing_savitzky_golay_weights_are_the_textbook_endpoint_weights() -> None:
    # A five-point quadratic's newest point weights the window (3, -5, -3, 9, 31) / 35.
    for j, weight in enumerate((3, -5, -3, 9, 31)):
        impulse = HalfHourlySeries(START, [float(i == j) for i in range(5)])
        newest = trailing_savitzky_golay(impulse, 5, 2).values[-1]
        assert newest == pytest.approx(weight / 35, abs=1e-14)


def test_trailing_savitzky_golay_follows_a_ramp_without_lag_between_gaps() -> None:
    ramp = HalfHourlySeries(START, [0.0, 1.0, 2.0, 3.0, None, 5.0, 6.0, 7.0, 8.0])
    expected = [None, None, 2.0, 3.0, None, None, None, 7.0, 8.0]
    assert trailing_savitzky_golay(ramp, 3, 1).values == pytest.approx(expected)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda s: trailing_mean(s, 0), id="window_0"),
        pytest.param(lambda s: rolling_median(s, True), id="bool_window"),
        pytest.param(lambda s: rolling_variance(s, 1), id="window_below_ddof_plus_1"),
        pytest.param(
            lambda s: exponential_moving_average(s, span=3, alpha=0.5), id="span_and_alpha"
        ),
        pytest.param(lambda s: exponential_moving_average(s), id="neither_span_nor_alpha"),
        pytest.param(lambda s: exponential_moving_average(s, alpha=0.0), id="alpha_0"),
        pytest.param(lambda s: exponential_moving_average(s, alpha=1.5), id="alpha_1.5"),
        pytest.param(lambda s: trailing_weighted_mean(s, []), id="no_weights"),
        pytest.param(lambda s: trailing_weighted_mean(s, [1.0, 0.0]), id="zero_weight"),
        pytest.param(lambda s: trailing_weighted_mean(s, [1.0, -1.0]), id="negative_weight"),
        pytest.param(lambda s: trailing_weighted_mean(s, [1.0, math.inf]), id="infinite_weight"),
        pytest.param(lambda s: trailing_weighted_mean(s, [1e308, 1e308]), id="infinite_sum"),
        pytest.param(  # a plain sum stays finite on Python 3.11; Orchid's fsum overflows
            lambda s: trailing_weighted_mean(s, [sys.float_info.max, 0.6 * 2**970, 0.6 * 2**970]),
            id="sum_overflowing_only_when_exact",
        ),
        pytest.param(lambda s: exponential_moving_average(s, span=0), id="span_0"),
        pytest.param(lambda s: exponential_moving_average(s, span=True), id="bool_span"),
        pytest.param(
            lambda s: exponential_moving_average(s, span=3, min_periods=0), id="min_periods_0"
        ),
        pytest.param(lambda s: exponential_moving_average(s, alpha=10**400), id="huge_alpha"),
        pytest.param(lambda s: rolling_std(s, 5, ddof=2), id="ddof_2"),
        pytest.param(lambda s: rolling_quantile(s, 0, 0.5), id="quantile_window_0"),
        pytest.param(lambda s: rolling_quantile(s, 5, 1.5), id="q_1.5"),
        pytest.param(lambda s: rolling_rms(s, 0), id="rms_window_0"),
        pytest.param(lambda s: difference(s, lag=0), id="lag_0"),
        pytest.param(lambda s: difference(s, order=0), id="order_0"),
        pytest.param(lambda s: trailing_savitzky_golay(s, 3, -1), id="sg_degree_-1"),
        pytest.param(lambda s: trailing_savitzky_golay(s, 6, 4), id="sg_degree_4"),
        pytest.param(lambda s: trailing_savitzky_golay(s, 3, 2), id="sg_window_degree_plus_1"),
        pytest.param(lambda s: trailing_savitzky_golay(s, 3, True), id="sg_bool_degree"),
        pytest.param(
            lambda s: trailing_savitzky_golay(s, 3.0, 1),  # type: ignore[arg-type]
            id="sg_float_window",
        ),
    ],
)
def test_rejects_invalid_parameters(call: Callable[[HalfHourlySeries], object]) -> None:
    with pytest.raises(InvalidInputError):
        call(HalfHourlySeries(START, [1.0, 2.0, 3.0]))


def test_accepts_boundary_parameters() -> None:
    series = HalfHourlySeries(START, [3.0, 1.0, 2.0])
    assert exponential_moving_average(series, alpha=1.0).values == series.values
    assert rolling_quantile(series, 3, 0.0).values[-1] == 1.0
    assert rolling_quantile(series, 3, 1.0).values[-1] == 3.0
    # alpha 0.3 gives a default min_periods of round(2 / 0.3 - 1) = 6.
    rounded = exponential_moving_average(HalfHourlySeries(START, [1.0] * 7), alpha=0.3)
    assert [value is None for value in rounded.values] == [True] * 5 + [False] * 2


def test_orchid_failures_raise_calculation_error() -> None:
    with pytest.raises(CalculationError, match="non-finite"):  # Orchid's quantile returns NaN
        rolling_quantile(HalfHourlySeries(START, [-1.7e308, 1.7e308]), 2, 0.0)
    with pytest.raises(CalculationError) as failure:  # Orchid's variance overflows
        rolling_variance(HalfHourlySeries(START, [1e200, -1e200]), 2)
    assert isinstance(failure.value.__cause__, OverflowError)
