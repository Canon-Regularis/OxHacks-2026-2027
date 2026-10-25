from collections.abc import Callable
from datetime import UTC, datetime
from statistics import fmean

import pytest

from gridpulse.analysis.algorithms.decomposition import (
    ComponentStrength,
    classical_decomposition,
    component_strength,
    lowess,
    median_filter,
    savitzky_golay,
    stl_decomposition,
)
from gridpulse.analysis.algorithms.errors import (
    CalculationError,
    DegenerateDataError,
    InsufficientDataError,
    InvalidInputError,
    MissingDataError,
)
from gridpulse.analysis.algorithms.orchid import forecasting_baseline as fb
from gridpulse.analysis.algorithms.orchid import smoothing as sm
from gridpulse.analysis.algorithms.series import HalfHourlySeries

START = datetime(2026, 9, 1, tzinfo=UTC)
MakeSeries = Callable[..., HalfHourlySeries]
MakeValues = Callable[..., list[float]]
Smoother = Callable[[HalfHourlySeries], HalfHourlySeries]
Oracle = Callable[[list[float]], list[float]]

# Each centred smoother with its window and Orchid's output on one run of values. The runs
# they are tested on have power-of-two lengths or 9 values, so 9 / len(run) is exact and
# Orchid's LOWESS fits ceil(9 / len(run) * len(run)) = 9 points.
SMOOTHERS: dict[str, tuple[Smoother, int, Oracle]] = {
    "median_filter": (lambda s: median_filter(s, 5), 5, lambda run: sm.median_filter(run, 5)),
    "savitzky_golay": (
        lambda s: savitzky_golay(s, 7, 2),
        7,
        lambda run: sm.savitzky_golay(run, 7, 2),
    ),
    "lowess": (lambda s: lowess(s, 9), 9, lambda run: sm.lowess(run, 9 / len(run), iterations=4)),
    "lowess_without_reweighting": (
        lambda s: lowess(s, 9, robustness_iterations=0),
        9,
        lambda run: sm.lowess(run, 9 / len(run), iterations=1),
    ),
}


@pytest.mark.parametrize("period", [48, 7])
def test_classical_decomposition_is_orchids_with_the_trend_missing_at_the_ends(
    make_series: MakeSeries, period: int
) -> None:
    series = make_series(5 * period + 3, level=0.0)  # crosses zero, often negative
    expected = fb.classical_decomposition(series.complete_values(), period)
    result = classical_decomposition(series, period=period)
    half = period // 2

    assert (result.method, result.model, result.period) == ("classical", "additive", period)
    assert result.trend.values == tuple(expected.trend)
    assert result.trend.values[:half] == result.trend.values[-half:] == (None,) * half
    assert None not in result.trend.values[half:-half]
    assert result.seasonal.values == tuple(expected.seasonal)
    assert result.remainder.values == tuple(expected.residual)
    assert result.seasonal_indices == tuple(expected.seasonal_indices)
    components = (result.trend, result.seasonal, result.remainder)
    assert {(part.start, len(part)) for part in components} == {(series.start, len(series))}


@pytest.mark.parametrize("scale", [1.0, 1e-14])  # Orchid would drop every ratio at 1e-14
def test_multiplicative_decomposition_is_orchids_in_any_units(
    make_series: MakeSeries, scale: float
) -> None:
    values = make_series(200).complete_values()
    expected = fb.classical_decomposition(values, 48, model="multiplicative")
    series = HalfHourlySeries(START, [value * scale for value in values])
    result = classical_decomposition(series, period=48, model="multiplicative")
    trend = [None if value is None else value * scale for value in expected.trend]

    assert result.model == "multiplicative"
    assert result.trend.values == pytest.approx(trend, rel=1e-12)
    assert result.seasonal.values == pytest.approx(expected.seasonal, rel=1e-12)
    assert result.remainder.values == pytest.approx(expected.residual, rel=1e-12)
    assert result.seasonal_indices == pytest.approx(expected.seasonal_indices, rel=1e-12)


def test_classical_decomposition_needs_two_cycles(make_series: MakeSeries) -> None:
    classical_decomposition(make_series(96), period=48)
    with pytest.raises(InsufficientDataError) as excinfo:
        classical_decomposition(make_series(95), period=48)
    assert (excinfo.value.required, excinfo.value.available) == (96, 95)


@pytest.mark.parametrize(
    "value", [0.0, -5.0, 1e-11], ids=["zero", "negative", "within_1e-12_of_the_largest"]
)
def test_multiplicative_decomposition_needs_positive_values(
    make_series: MakeSeries, value: float
) -> None:
    values = make_series(96).complete_values()
    values[10] = value
    series = HalfHourlySeries(START, values)
    classical_decomposition(series, period=48)
    with pytest.raises(InvalidInputError) as excinfo:
        classical_decomposition(series, period=48, model="multiplicative")
    assert excinfo.type is InvalidInputError


def test_stl_is_orchids_with_windows_of_whole_cycles_and_periods(make_series: MakeSeries) -> None:
    # Orchid's LOWESS fits the ceil(frac * len) nearest points, so these exact fractions give
    # 7 of a phase's 7 or 8 values and 93 of 340, the default windows for period 48, where
    # 7 / 7 and 93 / 340 would give 8 and 94.
    series = make_series(340)
    values = series.complete_values()
    expected = sm.stl_decomposition(
        values, 48, seasonal_frac=55 / 64, trend_frac=35 / 128, iterations=2, robust=True
    )
    result = stl_decomposition(series, period=48)

    assert (result.method, result.model, result.period) == ("stl", "additive", 48)
    assert result.trend.values == pytest.approx(expected.trend, abs=1e-10)
    assert result.seasonal.values == pytest.approx(expected.seasonal, abs=1e-10)
    assert result.remainder.values == pytest.approx(expected.resid, abs=1e-10)
    indices = [fmean(expected.seasonal[phase::48]) for phase in range(48)]
    assert result.seasonal_indices == pytest.approx(indices, abs=1e-10)


@pytest.mark.parametrize(
    ("size", "seasonal_share"),
    [(30.0, 0.0), (1000.0, 1.0)],
    ids=["moderate_outlier_in_the_remainder", "gross_outlier_kept_in_the_seasonal"],
)
def test_stl_resists_moderate_outliers_but_can_keep_gross_ones(
    make_series: MakeSeries, size: float, seasonal_share: float
) -> None:
    # The remainder's standard deviation is about 2. As stl_decomposition warns, the gross
    # outlier also discredits the neighbours whose fits it pulled, and keeps its raw value.
    values = make_series(168).complete_values()  # 14 cycles of period 12
    base = stl_decomposition(HalfHourlySeries(START, values), period=12)
    values[90] += size
    result = stl_decomposition(HalfHourlySeries(START, values), period=12)

    def share(before: HalfHourlySeries, after: HalfHourlySeries) -> float:
        return (after.complete_values()[90] - before.complete_values()[90]) / size

    assert share(base.seasonal, result.seasonal) == pytest.approx(seasonal_share, abs=0.1)
    assert share(base.remainder, result.remainder) == pytest.approx(1 - seasonal_share, abs=0.15)


def test_stl_needs_seasonal_window_cycles_and_trend_window_values(
    make_series: MakeSeries,
) -> None:
    stl_decomposition(make_series(84), period=12)
    with pytest.raises(InsufficientDataError) as excinfo:
        stl_decomposition(make_series(83), period=12)
    assert (excinfo.value.required, excinfo.value.available) == (84, 83)
    with pytest.raises(InsufficientDataError) as excinfo:
        stl_decomposition(make_series(84), period=12, trend_window=85)
    assert (excinfo.value.required, excinfo.value.available) == (85, 84)


@pytest.mark.parametrize(
    ("offset", "scale"),
    [(0.0, 1e-13), (-50.0, 1.0), (1e3, 1e3)],
    ids=["tiny", "negative_prices", "large"],
)
def test_stl_components_follow_the_units(
    synthetic_values: MakeValues, offset: float, scale: float
) -> None:
    # Unscaled, Orchid would stop re-weighting tiny values at its 1e-12 residual floor.
    values = synthetic_values(96)
    base = stl_decomposition(HalfHourlySeries(START, values), period=12)
    series = HalfHourlySeries(START, [offset + scale * value for value in values])
    result = stl_decomposition(series, period=12)

    def moved(part: HalfHourlySeries, shift: float = 0.0) -> object:
        expected = [shift + scale * value for value in part.complete_values()]
        return pytest.approx(expected, rel=1e-9, abs=1e-9 * scale)

    assert result.trend.values == moved(base.trend, offset)
    assert result.seasonal.values == moved(base.seasonal)
    assert result.remainder.values == moved(base.remainder)


def test_stl_of_a_constant_series_is_all_trend() -> None:
    result = stl_decomposition(HalfHourlySeries(START, [42.0] * 84), period=12)
    assert result.trend.values == (42.0,) * 84
    assert result.seasonal.values == result.remainder.values == (0.0,) * 84
    assert result.seasonal_indices == (0.0,) * 12


def test_strengths_are_orchids_on_values_of_unit_range(make_series: MakeSeries) -> None:
    values = make_series(200).complete_values()
    low, high = min(values), max(values)
    unit = [(value - low) / (high - low) for value in values]  # the range is exactly 1
    expected = ComponentStrength(fb.trend_strength(unit, 48), fb.seasonal_strength(unit, 48))
    assert component_strength(HalfHourlySeries(START, unit), period=48) == expected


def test_strengths_are_low_for_white_noise_and_high_for_its_components(
    synthetic_values: MakeValues,
) -> None:
    noise = synthetic_values(672, amplitude=0.0, phi=0.0)  # 14 days
    daily = synthetic_values(672, phi=0.0, noise=1.0)
    trending = [0.05 * t + value for t, value in enumerate(noise)]
    white = component_strength(HalfHourlySeries(START, noise), period=48)
    cycle = component_strength(HalfHourlySeries(START, daily), period=48)
    trend = component_strength(HalfHourlySeries(START, trending), period=48)

    assert white.trend < 0.1
    assert white.seasonal < 0.15
    assert cycle.trend < 0.1
    assert cycle.seasonal > 0.95
    assert trend.trend > 0.9


@pytest.mark.parametrize(
    ("offset", "scale"),
    [(0.0, 1e-7), (-50.0, 1.0), (1e4, 1e3)],
    ids=["tiny", "negative_prices", "large"],
)
def test_strengths_do_not_depend_on_the_units(
    make_series: MakeSeries, offset: float, scale: float
) -> None:
    # Unscaled, tiny values fall below Orchid's absolute 1e-12 variance floor.
    values = make_series(200).complete_values()
    base = component_strength(HalfHourlySeries(START, values), period=48)
    moved = HalfHourlySeries(START, [offset + scale * value for value in values])
    result = component_strength(moved, period=48)
    assert (result.trend, result.seasonal) == pytest.approx((base.trend, base.seasonal), abs=1e-9)


def test_strengths_need_two_cycles_of_varying_values(make_series: MakeSeries) -> None:
    with pytest.raises(InsufficientDataError) as excinfo:
        component_strength(make_series(95), period=48)
    assert (excinfo.value.required, excinfo.value.available) == (96, 95)
    with pytest.raises(DegenerateDataError):
        component_strength(HalfHourlySeries(START, [42.0] * 96), period=48)


@pytest.mark.parametrize("name", list(SMOOTHERS))
def test_centred_smoothers_are_orchids_inside_each_run(
    synthetic_values: MakeValues, name: str
) -> None:
    smoother, window, orchid = SMOOTHERS[name]
    half = window // 2
    runs = [
        synthetic_values(64, level=0.0),
        synthetic_values(32, seed=1),
        synthetic_values(9, seed=2),
        synthetic_values(4, seed=3),  # shorter than every window
    ]

    def centred(run: list[float]) -> list[float | None]:
        if len(run) < window:
            return [None] * len(run)
        return [None] * half + orchid(run)[half:-half] + [None] * half

    values: list[float | None] = []
    expected: list[float | None] = []
    for run in runs:
        values += [*run, None]
        expected += [*centred(run), None]
    assert smoother(HalfHourlySeries(START, values)).values == pytest.approx(expected, abs=1e-10)


def test_median_filter_removes_a_spike() -> None:
    spiky = HalfHourlySeries(START, [1.0, 2.0, 100.0, 4.0, 5.0])
    assert median_filter(spiky, 3).values == (None, 2.0, 4.0, 5.0, None)


@pytest.mark.parametrize("degree", [0, 1])
def test_savitzky_golay_of_degree_0_or_1_is_the_centred_mean(degree: int) -> None:
    series = HalfHourlySeries(START, [1.0, 2.0, 6.0, 4.0, 5.0])
    assert savitzky_golay(series, 3, degree).values == pytest.approx((None, 3.0, 4.0, 5.0, None))


@pytest.mark.parametrize(("offset", "scale"), [(0.0, 1e-13), (-50.0, 1e3)], ids=["tiny", "large"])
def test_lowess_follows_the_units(
    synthetic_values: MakeValues, offset: float, scale: float
) -> None:
    # Unscaled, Orchid would stop re-weighting tiny values at its 1e-12 residual floor.
    values = synthetic_values(64)
    base = lowess(HalfHourlySeries(START, values), 9).values
    result = lowess(HalfHourlySeries(START, [offset + scale * value for value in values]), 9)
    expected = [None if value is None else offset + scale * value for value in base]
    assert result.values == pytest.approx(expected, rel=1e-9, abs=1e-9 * scale)


def test_lowess_of_a_constant_run_is_that_constant() -> None:
    result = lowess(HalfHourlySeries(START, [-7.5] * 9), 9)
    assert result.values == (None,) * 4 + (-7.5,) + (None,) * 4


def test_lowess_fits_exactly_window_points(synthetic_values: MakeValues) -> None:
    # Orchid fits ceil(frac * len) points: 7 at a quarter of 25, but 8 at 7 / 25, because
    # 7 / 25 * 25 rounds above 7.
    run = synthetic_values(25)
    expected = sm.lowess(run, 1 / 4, iterations=4)
    result = lowess(HalfHourlySeries(START, run), 7)
    assert result.values == pytest.approx([None] * 3 + expected[3:-3] + [None] * 3, abs=1e-10)


@pytest.mark.parametrize("robustness_iterations", range(4))
def test_lowess_keeps_a_gross_outlier_raw_at_odd_robustness_iterations(
    make_series: MakeSeries, robustness_iterations: int
) -> None:
    # As lowess warns, the outlier also discredits its neighbours, so the next fit keeps its
    # raw value, which then fits exactly and regains its weight.
    values = make_series(96).complete_values()
    values[50] += 100.0  # about 40 standard deviations
    series = HalfHourlySeries(START, values)
    smoothed = lowess(series, 7, robustness_iterations=robustness_iterations).values[50]
    kept = smoothed == pytest.approx(values[50])
    assert kept == (robustness_iterations % 2 == 1)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda s: classical_decomposition(s, period=1), id="classical_period_1"),
        pytest.param(
            lambda s: classical_decomposition(s, period=12.0),  # type: ignore[arg-type]
            id="classical_float_period",
        ),
        pytest.param(
            lambda s: classical_decomposition(s, period=12, model="log"),  # type: ignore[arg-type]
            id="classical_model",
        ),
        pytest.param(lambda s: stl_decomposition(s, period=True), id="stl_bool_period"),
        pytest.param(
            lambda s: stl_decomposition(s, period=12, seasonal_window=3), id="stl_seasonal_window_3"
        ),
        pytest.param(
            lambda s: stl_decomposition(s, period=12, seasonal_window=8), id="stl_even_seasonal"
        ),
        pytest.param(
            lambda s: stl_decomposition(s, period=12, trend_window=3), id="stl_trend_window_3"
        ),
        pytest.param(
            lambda s: stl_decomposition(s, period=12, trend_window=24), id="stl_even_trend"
        ),
        pytest.param(lambda s: component_strength(s, period=1), id="strength_period_1"),
        pytest.param(lambda s: median_filter(s, 1), id="median_window_1"),
        pytest.param(lambda s: median_filter(s, 4), id="median_even_window"),
        pytest.param(lambda s: savitzky_golay(s, 6, 2), id="sg_even_window"),
        pytest.param(lambda s: savitzky_golay(s, 5, -1), id="sg_negative_degree"),
        pytest.param(lambda s: savitzky_golay(s, 7, 4), id="sg_degree_4"),
        pytest.param(lambda s: savitzky_golay(s, 3, 2), id="sg_degree_window_minus_1"),
        pytest.param(lambda s: lowess(s, 3), id="lowess_window_3"),
        pytest.param(lambda s: lowess(s, 8), id="lowess_even_window"),
        pytest.param(
            lambda s: lowess(s, 9.0),  # type: ignore[arg-type]
            id="lowess_float_window",
        ),
        pytest.param(
            lambda s: lowess(s, 9, robustness_iterations=-1), id="lowess_negative_iterations"
        ),
    ],
)
def test_rejects_invalid_parameters(
    make_series: MakeSeries, call: Callable[[HalfHourlySeries], object]
) -> None:
    with pytest.raises(InvalidInputError) as excinfo:
        call(make_series(168))
    assert excinfo.type is InvalidInputError


@pytest.mark.parametrize(
    "call",
    [
        lambda s: classical_decomposition(s, period=12),
        lambda s: stl_decomposition(s, period=12),
        lambda s: component_strength(s, period=12),
    ],
    ids=["classical", "stl", "strength"],
)
def test_decompositions_reject_missing_values(
    make_series: MakeSeries, call: Callable[[HalfHourlySeries], object]
) -> None:
    values: list[float | None] = [*make_series(168).complete_values()]
    values[50] = None
    with pytest.raises(MissingDataError):
        call(HalfHourlySeries(START, values))


@pytest.mark.parametrize(
    "call",
    [
        lambda s: classical_decomposition(s, period=12),
        lambda s: stl_decomposition(s, period=12),
        lambda s: component_strength(s, period=12),
        lambda s: lowess(s, 9),
    ],
    ids=["classical", "stl", "strength", "lowess"],
)
def test_values_that_overflow_raise_calculation_error(
    call: Callable[[HalfHourlySeries], object],
) -> None:
    with pytest.raises(CalculationError):
        call(HalfHourlySeries(START, [1.7e308, -1.7e308] * 42))


def test_orchid_failures_raise_calculation_error(
    monkeypatch: pytest.MonkeyPatch, make_series: MakeSeries
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise ZeroDivisionError("boom")

    monkeypatch.setattr(sm, "stl_decomposition", broken)
    with pytest.raises(CalculationError) as excinfo:
        stl_decomposition(make_series(84), period=12)
    assert isinstance(excinfo.value.__cause__, ZeroDivisionError)

    decompose = fb.classical_decomposition

    def gappy(y: list[float], m: int, model: str) -> fb.DecompositionResult:
        result = decompose(y, m, model=model)
        result.trend[30] = None
        return result

    monkeypatch.setattr(fb, "classical_decomposition", gappy)
    with pytest.raises(CalculationError, match="missing"):
        classical_decomposition(make_series(96), period=12)
